# SPDX-License-Identifier: Apache-2.0
"""Benchmark Python and Rust raw-block checkpoint payload creation.

Run the full 1--128 TiB-equivalent sweep:

    python3 benchmarks/microbenchmark/raw_block_snapshot_benchmark.py

Compare the Python and Rust implementations:

    python3 benchmarks/microbenchmark/raw_block_snapshot_benchmark.py \
        --implementation both

Run a small smoke test:

    python3 benchmarks/microbenchmark/raw_block_snapshot_benchmark.py \
        --max-entries 1000 --implementation both
"""

# Standard
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any
import argparse
import gc
import importlib
import json
import os
import sys
import time
import traceback

KIB = 1024
MIB = 1024**2
GIB = 1024**3
TIB = 1024**4

SLOT_BYTES = 2 * MIB
PAYLOAD_BYTES = MIB
SHAPE = (2, 256, 8, 128)
CAPACITIES_TIB = (1, 2, 4, 8, 16, 32, 64, 128)
DTYPE_NAMES = {"torch.bfloat16": "bfloat16"}
RUST_SERIALIZER_NAME = "serialize_raw_block_checkpoint_payload"
RustSerializer = Callable[..., bytes]


class SyntheticMemoryFormat(Enum):
    """Memory format needed by the snapshot's ``fmt.name`` branch."""

    KV_T2D = auto()


@dataclass
class Metadata:
    """Synthetic equivalent of the DiskCacheMetadata fields being read."""

    path: str
    size: int
    shape: tuple[int, ...]
    dtype: str
    cached_positions: Any
    fmt: SyntheticMemoryFormat
    pin_count: int = 0


@dataclass
class Entry:
    """Synthetic equivalent of a raw-block index entry."""

    offset: int
    size: int
    meta: Metadata


def encoded_key(entry_number: int) -> str:
    """Return a realistic fixed-width encoded cache key."""
    return f"test_model@1@0@{entry_number:064x}@bfloat16"


def checkpoint_dtype_name(dtype: str | None) -> str | None:
    """Match RawBlockCore's dtype-name lookup."""
    if dtype is None:
        return None
    return DTYPE_NAMES.get(dtype, str(dtype))


def extend_index(index: dict[str, Entry], num_entries: int) -> None:
    """Grow the source index without rebuilding existing entries."""
    for entry_number in range(len(index), num_entries):
        offset = entry_number * SLOT_BYTES
        metadata = Metadata(
            path=f"/dev/synthetic@{offset}",
            size=PAYLOAD_BYTES,
            shape=SHAPE,
            dtype="torch.bfloat16",
            cached_positions=None,
            fmt=SyntheticMemoryFormat.KV_T2D,
        )
        index[encoded_key(entry_number)] = Entry(
            offset=offset,
            size=PAYLOAD_BYTES,
            meta=metadata,
        )


def snapshot_state(
    index: dict[str, Entry],
    capacity_bytes: int,
) -> tuple[dict[str, Any], int]:
    """Build the dictionary shape created by RawBlockCore._snapshot_state."""
    dirty_total = len(index)
    snapshot = {
        "version": 1,
        "device_path": "/dev/synthetic",
        "capacity_bytes": capacity_bytes,
        "block_align": 4096,
        "header_bytes": 4096,
        "slot_bytes": SLOT_BYTES,
        "meta_total_bytes": 0,
        "meta_magic": "LMCIDX01",
        "meta_version": 1,
        "data_base_offset": 0,
        "next_slot": len(index),
        "entries": {
            key: {
                "offset": entry.offset,
                "size": entry.meta.size,
                "shape": (
                    list(entry.meta.shape) if entry.meta.shape is not None else None
                ),
                "dtype": checkpoint_dtype_name(entry.meta.dtype),
                "fmt": (
                    entry.meta.fmt.name
                    if entry.meta.fmt is not None and hasattr(entry.meta.fmt, "name")
                    else str(entry.meta.fmt)
                    if entry.meta.fmt is not None
                    else None
                ),
                "cached_positions": (
                    entry.meta.cached_positions.tolist()
                    if entry.meta.cached_positions is not None
                    and hasattr(entry.meta.cached_positions, "tolist")
                    else None
                ),
            }
            for key, entry in index.items()
        },
    }
    return snapshot, dirty_total


def load_rust_serializer() -> RustSerializer:
    """Load the production Rust checkpoint serializer."""
    try:
        module = importlib.import_module("lmcache_rust_raw_block_io")
        serializer = getattr(module, RUST_SERIALIZER_NAME)
    except (AttributeError, ImportError) as error:
        raise RuntimeError(
            "The Rust checkpoint serializer is unavailable. Build it with:\n"
            "  cd rust/raw_block\n"
            "  python -m maturin develop --release"
        ) from error
    if not callable(serializer):
        raise RuntimeError(
            f"lmcache_rust_raw_block_io.{RUST_SERIALIZER_NAME} is not callable"
        )
    return serializer


def proc_status_gib(field: str) -> float:
    """Read a KiB-valued field from Linux ``/proc/self/status`` in GiB."""
    with open("/proc/self/status", encoding="ascii") as status_file:
        for line in status_file:
            if line.startswith(f"{field}:"):
                return int(line.split()[1]) * KIB / GIB
    raise RuntimeError(f"{field} is missing from /proc/self/status")


def validate_rust_payload(
    payload: bytes,
    capacity_bytes: int,
    num_entries: int,
    *,
    decode_json: bool,
) -> None:
    """Check Rust payload framing and optionally decode its complete JSON."""
    if not isinstance(payload, bytes):
        raise TypeError("Rust checkpoint serializer did not return bytes")
    if not payload.startswith(b'{"version":1,"device_path":'):
        raise RuntimeError("Rust checkpoint payload has an invalid prefix")
    if b'"entries":{' not in payload[:512] or not payload.endswith(b"}}"):
        raise RuntimeError("Rust checkpoint payload has invalid JSON framing")
    if not decode_json:
        return

    decoded = json.loads(payload)
    expected_fields = {
        "version": 1,
        "device_path": "/dev/synthetic",
        "capacity_bytes": capacity_bytes,
        "block_align": 4096,
        "header_bytes": 4096,
        "slot_bytes": SLOT_BYTES,
        "meta_total_bytes": 0,
        "meta_magic": "LMCIDX01",
        "meta_version": 1,
        "data_base_offset": 0,
        "next_slot": num_entries,
    }
    for field, expected in expected_fields.items():
        if decoded.get(field) != expected:
            raise RuntimeError(
                f"Rust checkpoint field {field!r} does not match the source"
            )

    entries = decoded.get("entries")
    if not isinstance(entries, dict) or len(entries) != num_entries:
        raise RuntimeError("Rust checkpoint entry count does not match source index")

    first_key = encoded_key(0)
    last_key = encoded_key(num_entries - 1)
    if entries.get(first_key, {}).get("offset") != 0:
        raise RuntimeError("Rust checkpoint first entry is invalid")
    if entries.get(last_key, {}).get("offset") != (num_entries - 1) * SLOT_BYTES:
        raise RuntimeError("Rust checkpoint last entry is invalid")


def run_case(
    index: dict[str, Entry],
    capacity_tib: int,
    index_size: float,
    implementation: str,
    rust_serializer: RustSerializer | None,
    validate_json: bool,
) -> None:
    """Time one capacity using an exact-size view of the shared index."""
    capacity_bytes = capacity_tib * TIB
    theoretical_entries = capacity_bytes // SLOT_BYTES
    num_entries = len(index)
    print(
        f"[{capacity_tib} TiB] timing {num_entries:,} entries with {implementation}...",
        file=sys.stderr,
        flush=True,
    )

    run_python = implementation in {"python", "both"}
    run_rust = implementation in {"rust", "both"}
    snapshot_seconds: float | None = None
    serialization_seconds: float | None = None
    encoding_seconds: float | None = None
    python_entries_per_second: float | None = None
    snapshot_size: float | None = None
    rust_capture_seconds: float | None = None
    rust_checkpoint_seconds: float | None = None
    rust_total_seconds: float | None = None
    rust_entries_per_second: float | None = None
    python_payload_size: int | None = None
    rust_payload_size: int | None = None
    expected_payload: bytes | None = None

    if run_python:
        rss_before = proc_status_gib("VmRSS")
        start = time.perf_counter()
        snapshot, dirty_total = snapshot_state(index, capacity_bytes)
        snapshot_seconds = time.perf_counter() - start

        rss_after = proc_status_gib("VmRSS")
        if len(snapshot["entries"]) != num_entries or dirty_total != num_entries:
            raise RuntimeError("snapshot entry count does not match source index")

        print(
            f"[{capacity_tib} TiB] snapshot ready; timing json.dumps...",
            file=sys.stderr,
            flush=True,
        )
        serialization_start = time.perf_counter()
        serialized = json.dumps(snapshot, separators=(",", ":"), ensure_ascii=True)
        serialization_seconds = time.perf_counter() - serialization_start
        if not serialized:
            raise RuntimeError("serialized snapshot is unexpectedly empty")

        print(
            f"[{capacity_tib} TiB] JSON ready; timing UTF-8 encoding...",
            file=sys.stderr,
            flush=True,
        )
        encoding_start = time.perf_counter()
        payload = serialized.encode("utf-8")
        encoding_seconds = time.perf_counter() - encoding_start
        if not payload:
            raise RuntimeError("encoded snapshot is unexpectedly empty")

        python_entries_per_second = num_entries / snapshot_seconds
        snapshot_size = rss_after - rss_before
        python_payload_size = len(payload)
        if validate_json and run_rust:
            expected_payload = payload

        del payload
        del serialized
        del snapshot
        gc.collect()

    if run_rust:
        if rust_serializer is None:
            raise RuntimeError("Rust implementation selected without a serializer")
        print(
            f"[{capacity_tib} TiB] copying the stable index for Rust...",
            file=sys.stderr,
            flush=True,
        )
        rust_capture_start = time.perf_counter()
        rust_index = index.copy()
        rust_capture_seconds = time.perf_counter() - rust_capture_start
        print(
            f"[{capacity_tib} TiB] timing Rust checkpoint payload creation...",
            file=sys.stderr,
            flush=True,
        )
        rust_start = time.perf_counter()
        rust_payload = rust_serializer(
            "/dev/synthetic",
            capacity_bytes,
            4096,
            4096,
            SLOT_BYTES,
            0,
            "LMCIDX01",
            1,
            0,
            num_entries,
            rust_index,
        )
        rust_checkpoint_seconds = time.perf_counter() - rust_start
        rust_total_seconds = rust_capture_seconds + rust_checkpoint_seconds
        validate_rust_payload(
            rust_payload,
            capacity_bytes,
            num_entries,
            decode_json=validate_json,
        )
        rust_payload_size = len(rust_payload)
        rust_entries_per_second = num_entries / rust_total_seconds
        if expected_payload is not None and rust_payload != expected_payload:
            raise RuntimeError("Python and Rust checkpoint payloads differ")

        del rust_payload
        del rust_index
        gc.collect()

    if (
        python_payload_size is not None
        and rust_payload_size is not None
        and python_payload_size != rust_payload_size
    ):
        raise RuntimeError("Python and Rust checkpoint payload sizes differ")
    payload_size = (
        rust_payload_size if rust_payload_size is not None else python_payload_size
    )
    if payload_size is None:
        raise RuntimeError("no checkpoint implementation was benchmarked")

    python_entries_text = (
        "" if python_entries_per_second is None else f"{python_entries_per_second:.3f}"
    )
    rust_entries_text = (
        "" if rust_entries_per_second is None else f"{rust_entries_per_second:.3f}"
    )
    snapshot_size_text = "" if snapshot_size is None else f"{snapshot_size:.3f}"
    snapshot_seconds_text = (
        "" if snapshot_seconds is None else f"{snapshot_seconds:.6f}"
    )
    serialization_seconds_text = (
        "" if serialization_seconds is None else f"{serialization_seconds:.6f}"
    )
    encoding_seconds_text = (
        "" if encoding_seconds is None else f"{encoding_seconds:.6f}"
    )
    rust_checkpoint_seconds_text = (
        "" if rust_checkpoint_seconds is None else f"{rust_checkpoint_seconds:.6f}"
    )
    rust_capture_seconds_text = (
        "" if rust_capture_seconds is None else f"{rust_capture_seconds:.6f}"
    )
    rust_total_seconds_text = (
        "" if rust_total_seconds is None else f"{rust_total_seconds:.6f}"
    )
    print(
        f"{capacity_tib},"
        f"{theoretical_entries},"
        f"{num_entries},"
        f"{python_entries_text},"
        f"{rust_entries_text},"
        f"{index_size:.3f},"
        f"{snapshot_size_text},"
        f"{payload_size / GIB:.3f},"
        f"{snapshot_seconds_text},"
        f"{serialization_seconds_text},"
        f"{encoding_seconds_text},"
        f"{rust_capture_seconds_text},"
        f"{rust_checkpoint_seconds_text},"
        f"{rust_total_seconds_text}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse benchmark command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Time synthetic RawBlockCore checkpoint creation with Python, "
            "the production Rust serializer, or both. Uses 2 MiB slots and "
            "does not require a block device."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--implementation",
        choices=("auto", "python", "rust", "both"),
        default="auto",
        help=(
            "Implementation to benchmark. Auto selects Rust when its extension "
            "is installed and otherwise uses Python."
        ),
    )
    parser.add_argument(
        "--capacities-tib",
        nargs="+",
        type=int,
        default=list(CAPACITIES_TIB),
        help="Device capacities to simulate.",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        help="Cap each case to this many entries for a small test.",
    )
    args = parser.parse_args()
    if any(capacity <= 0 for capacity in args.capacities_tib):
        parser.error("--capacities-tib values must be positive")
    if args.max_entries is not None and args.max_entries <= 0:
        parser.error("--max-entries must be positive")
    return args


def run_isolated(
    index: dict[str, Entry],
    capacity_tib: int,
    index_size: float,
    implementation: str,
    rust_serializer: RustSerializer | None,
    validate_json: bool,
) -> int:
    """Fork a clean snapshot worker that shares the current source index."""
    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:
        return_code = 0
        try:
            gc.enable()
            gc.collect()
            run_case(
                index,
                capacity_tib,
                index_size,
                implementation,
                rust_serializer,
                validate_json,
            )
        except BaseException:
            traceback.print_exc()
            return_code = 1
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
        os._exit(return_code)

    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


def main() -> int:
    """Run all selected capacities and print CSV results."""
    args = parse_args()
    rust_serializer: RustSerializer | None = None
    implementation = args.implementation
    if implementation == "auto":
        try:
            rust_serializer = load_rust_serializer()
        except RuntimeError as error:
            print(
                f"{error}\nFalling back to the Python implementation.",
                file=sys.stderr,
            )
            implementation = "python"
        else:
            implementation = "rust"
    elif implementation in {"rust", "both"}:
        try:
            rust_serializer = load_rust_serializer()
        except RuntimeError as error:
            print(error, file=sys.stderr)
            return 2

    print(f"Python: {sys.version.split()[0]}", file=sys.stderr)
    print(f"Implementation: {implementation}", file=sys.stderr)
    print(
        "Python timings: snapshot construction, json.dumps, and UTF-8 "
        "encoding. Rust timings: shallow stable-index capture, direct UTF-8 "
        "checkpoint payload creation, and their total. Excluded: source-index "
        "setup and device I/O.",
        file=sys.stderr,
    )
    print(
        "Sizes: index and Python snapshot use current-RSS deltas including "
        "allocator overhead; payload size is the exact byte length. Snapshot "
        "size does not include hidden fork copy-on-write pages.",
        file=sys.stderr,
    )
    print(
        "Index setup: one dictionary grows across capacities; entries are "
        "never rebuilt.",
        file=sys.stderr,
    )
    if args.max_entries is None and max(args.capacities_tib) >= 128:
        print(
            "Warning: the full 128 TiB case requires tens of GiB of RAM; "
            "comparison mode can require well over 100 GiB.",
            file=sys.stderr,
        )
    print(
        "device_tib,theoretical_entries,benchmarked_entries,"
        "python_entries_per_second,rust_entries_per_second,"
        "index_size_gib,snapshot_size_gib,payload_size_gib,"
        "snapshot_seconds,serialization_seconds,encoding_seconds,"
        "rust_capture_seconds,rust_checkpoint_seconds,rust_total_seconds",
        flush=True,
    )

    capacities_tib = sorted(args.capacities_tib)
    index: dict[str, Entry] = {}
    gc.collect()
    gc.disable()
    baseline_rss = proc_status_gib("VmRSS")
    total_setup_seconds = 0.0
    for capacity_tib in capacities_tib:
        target_entries = capacity_tib * TIB // SLOT_BYTES
        if args.max_entries is not None:
            target_entries = min(target_entries, args.max_entries)

        if target_entries > len(index):
            print(
                f"[setup] extending shared index from {len(index):,} to "
                f"{target_entries:,} entries...",
                file=sys.stderr,
                flush=True,
            )
            setup_start = time.perf_counter()
            extend_index(index, target_entries)
            total_setup_seconds += time.perf_counter() - setup_start
            gc.freeze()

        index_size = proc_status_gib("VmRSS") - baseline_rss
        return_code = run_isolated(
            index,
            capacity_tib,
            index_size,
            implementation,
            rust_serializer,
            args.max_entries is not None,
        )
        if return_code != 0:
            print(
                f"{capacity_tib} TiB worker failed with status {return_code}",
                file=sys.stderr,
            )
            return return_code

    print(
        f"[setup] total one-time index construction: {total_setup_seconds:.3f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
