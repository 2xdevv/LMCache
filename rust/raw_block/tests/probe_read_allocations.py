# SPDX-License-Identifier: Apache-2.0

"""Check bounce allocation sizes using the experiment's posix_memalign tracer."""

# Standard
from pathlib import Path
import argparse
import ctypes
import json
import tempfile

# Third Party
from test_read_tail import ALIGN, GUARD, MIB, aligned_buffer, assert_guards
import lmcache_rust_raw_block_io as raw_block


def main() -> None:
    """Assert allocation bounds, data integrity and preparation-failure cleanup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace = ctypes.CDLL(None)
    trace.trace_begin.argtypes = [ctypes.c_size_t]
    trace.trace_calls.restype = ctypes.c_size_t
    trace.trace_bytes.restype = ctypes.c_size_t
    records = []
    payload = 2 * MIB + 4
    total = 2 * MIB + ALIGN
    with tempfile.TemporaryDirectory(prefix="rawblock-tail-probe-") as temp:
        path = Path(temp) / "device.bin"
        contents = bytes(range(251)) * ((total + ALIGN) // 251 + 1)
        path.write_bytes(contents)
        dev = raw_block.RawBlockDevice(
            str(path),
            writable=False,
            use_odirect=True,
            alignment=ALIGN,
            io_engine="io_uring",
            iouring_queue_depth=8,
        )
        try:
            for batch_size in (1, 3):
                for misalignment, capacity in ((0, payload), (1, payload), (0, total)):
                    buffers = [
                        aligned_buffer(capacity, misalignment)
                        for _ in range(batch_size)
                    ]
                    views = [view for _, view in buffers]
                    trace.trace_begin(0)
                    try:
                        if batch_size == 1:
                            dev.read_uring(ALIGN, views[0], payload, total)
                        else:
                            batch = dev.batched_read(
                                [ALIGN] * batch_size, views, [total] * batch_size
                            )
                            assert dev.wait_iouring(batch) == ([True] * batch_size, [])
                    finally:
                        trace.trace_end()
                    allocated = trace.trace_bytes()
                    expected = (
                        0
                        if misalignment == 0 and capacity >= total
                        else ALIGN
                        if args.candidate and misalignment == 0
                        else total
                    ) * batch_size
                    assert allocated == expected, (allocated, expected)
                    for backing, view in buffers:
                        assert view[:payload] == contents[ALIGN : ALIGN + payload]
                        assert_guards(backing, view)
                    record = dict(
                        batch_size=batch_size,
                        misalignment=misalignment,
                        capacity=capacity,
                        calls=trace.trace_calls(),
                        allocated=allocated,
                    )
                    records.append(record)
                    print(record, flush=True)
            if args.candidate:
                # Fail preparation of the second logical read. The already
                # prepared first request must never become visible to the worker.
                buffers = [aligned_buffer(payload) for _ in range(2)]
                trace.trace_begin(2)
                try:
                    try:
                        dev.batched_read(
                            [ALIGN, ALIGN],
                            [view for _, view in buffers],
                            [total, total],
                        )
                    except RuntimeError as error:
                        assert "posix_memalign failed" in str(error)
                    else:
                        raise AssertionError("allocation failure was not propagated")
                finally:
                    trace.trace_end()
                for backing, view in buffers:
                    assert view == bytes([GUARD]) * len(view)
                    assert_guards(backing, view)
                out = bytearray(payload)
                trace.trace_begin(1)
                try:
                    try:
                        dev.read_uring(ALIGN, out, payload, total)
                    except RuntimeError as error:
                        assert "posix_memalign failed" in str(error)
                    else:
                        raise AssertionError("allocation failure was not propagated")
                finally:
                    trace.trace_end()
                out.extend(b"x")  # Must not retain an exported Python buffer.
                dev.read_uring(ALIGN, buffers[0][1], payload, total)
                assert buffers[0][1] == contents[ALIGN : ALIGN + payload]
                print(
                    "allocation-failure cleanup and subsequent read: passed",
                    flush=True,
                )
        finally:
            dev.close()
    args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
