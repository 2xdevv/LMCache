# SPDX-License-Identifier: Apache-2.0

"""Measure public RawBlock read calls against a temporary O_DIRECT regular file."""

# Standard
from pathlib import Path
import argparse
import json
import os
import platform
import statistics
import tempfile
import time

# Third Party
from test_read_tail import ALIGN, MIB, aligned_buffer, assert_guards
import lmcache_rust_raw_block_io as raw_block


def main() -> None:
    """Write reproducible latency samples for single and batched read scenarios."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--prime-allocator", action="store_true")
    args = parser.parse_args()
    if args.prime_allocator:
        # Model a process that has already allocated and freed larger objects.
        # glibc can then reuse heap storage for the 2 MiB bounce allocation.
        warmup = bytearray(4 * MIB)
        del warmup
    scenarios = (
        ("4KiB+4_tail", ALIGN + 4, ALIGN + 4, 0),
        ("64KiB+4_tail", 64 * 1024 + 4, 64 * 1024 + 4, 0),
        ("2MiB+4_tail", 2 * MIB + 4, 2 * MIB + 4, 0),
        ("2MiB_aligned", 2 * MIB, 2 * MIB, 0),
        ("2MiB+4_padded_capacity", 2 * MIB + 4, 2 * MIB + ALIGN, 0),
        ("2MiB+4_unaligned", 2 * MIB + 4, 2 * MIB + 4, 1),
    )
    results = []
    with tempfile.TemporaryDirectory(prefix="rawblock-tail-bench-") as temp:
        path = Path(temp) / "device.bin"
        contents = b"".join(bytes([page % 251]) * ALIGN for page in range(2048))
        with path.open("wb") as file:
            file.write(contents)
            file.flush()
            os.fsync(file.fileno())
        dev = raw_block.RawBlockDevice(
            str(path),
            writable=False,
            use_odirect=True,
            alignment=ALIGN,
            io_engine="io_uring",
            iouring_queue_depth=256,
        )
        try:
            for batch_size in (1, 8):
                for name, payload, capacity, misalignment in scenarios:
                    total = (payload + ALIGN - 1) // ALIGN * ALIGN
                    buffers = [
                        aligned_buffer(capacity, misalignment)
                        for _ in range(batch_size)
                    ]
                    views = [view for _, view in buffers]
                    offsets = [ALIGN] * batch_size
                    totals = [total] * batch_size
                    samples = []
                    count = (
                        args.samples if batch_size == 1 else max(32, args.samples // 4)
                    )
                    for iteration in range(count + 20):
                        start = time.perf_counter_ns()
                        if batch_size == 1:
                            dev.read_uring(ALIGN, views[0], payload, total)
                        else:
                            batch = dev.batched_read(offsets, views, totals)
                            result, errors = dev.wait_iouring(batch)
                            if errors or not all(result) or len(result) != batch_size:
                                raise AssertionError((result, errors))
                        elapsed = time.perf_counter_ns() - start
                        if iteration >= 20:
                            samples.append(elapsed / 1000)
                    for backing, view in buffers:
                        assert view[:payload] == contents[ALIGN : ALIGN + payload]
                        assert_guards(backing, view)
                    record = {
                        "scenario": name,
                        "batch_size": batch_size,
                        "median_us": statistics.median(samples),
                        "p95_us": sorted(samples)[int(0.95 * len(samples))],
                        "samples_us": samples,
                    }
                    results.append(record)
                    print(
                        f"{args.label} batch={batch_size} {name}: "
                        f"median={record['median_us']:.1f}us "
                        f"p95={record['p95_us']:.1f}us",
                        flush=True,
                    )
        finally:
            dev.close()
    args.output.write_text(
        json.dumps(
            {
                "label": args.label,
                "platform": platform.platform(),
                "extension": raw_block.__file__,
                "allocator_primed": args.prime_allocator,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
