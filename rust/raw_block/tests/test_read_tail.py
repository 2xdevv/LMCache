# SPDX-License-Identifier: Apache-2.0

"""Exercise RawBlockDevice read contracts on temporary regular files.

Run with the built extension on PYTHONPATH using ``python3 -m unittest discover
-s rust/raw_block/tests -v``. No LMCache Python or CUDA installation is needed.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
import ctypes
import tempfile
import unittest

# Third Party
import lmcache_rust_raw_block_io as raw_block

ALIGN = 4096
MIB = 1024 * 1024
GUARD = 0xA5


def aligned_buffer(
    capacity: int, misalignment: int = 0
) -> tuple[bytearray, memoryview]:
    """Return owned, guarded writable storage with a chosen address alignment."""
    backing = bytearray([GUARD]) * (capacity + 3 * ALIGN)
    address = ctypes.addressof(ctypes.c_ubyte.from_buffer(backing))
    start = ALIGN + (-address) % ALIGN + misalignment
    return backing, memoryview(backing)[start : start + capacity]


def assert_guards(backing: bytearray, view: memoryview) -> None:
    """Raise AssertionError if I/O changed bytes outside the exposed buffer."""
    base = ctypes.addressof(ctypes.c_ubyte.from_buffer(backing))
    start = ctypes.addressof(ctypes.c_ubyte.from_buffer(view)) - base
    assert backing[:start] == bytes([GUARD]) * start
    end = start + len(view)
    assert backing[end:] == bytes([GUARD]) * (len(backing) - end)


class ReadTailTests(unittest.TestCase):
    """Validate single and batched reads, including guarded padded destinations."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a temporary file and a real O_DIRECT io_uring device."""
        cls.temp = tempfile.TemporaryDirectory(prefix="rawblock-read-tail-")
        cls.path = Path(cls.temp.name) / "device.bin"
        cls.contents = b"".join(
            bytes([page % 251]) * ALIGN for page in range((4 * MIB) // ALIGN)
        )
        cls.path.write_bytes(cls.contents)
        cls.dev = cls.open_device()

    @classmethod
    def tearDownClass(cls) -> None:
        """Drain the device before deleting its temporary backing file."""
        cls.dev.close()
        cls.temp.cleanup()

    @classmethod
    def open_device(cls, odirect: bool = True, depth: int = 8) -> Any:
        """Open the test file through RawBlock's public constructor."""
        return raw_block.RawBlockDevice(
            str(cls.path),
            writable=False,
            use_odirect=odirect,
            alignment=ALIGN,
            io_engine="io_uring",
            iouring_queue_depth=depth,
        )

    def test_single_read_boundaries(self) -> None:
        """Return every requested byte without writing past destination capacity."""
        sizes = (1, 4, 4095, 4096, 4100, 8191, 8192, 2 * MIB, 2 * MIB + 4)
        for payload_len in sizes:
            total_len = (payload_len + ALIGN - 1) // ALIGN * ALIGN
            for capacity in sorted({payload_len, total_len, total_len + 17}):
                for misalignment in (0, 1, 128):
                    with self.subTest(
                        payload=payload_len, capacity=capacity, address=misalignment
                    ):
                        backing, view = aligned_buffer(capacity, misalignment)
                        self.dev.read_uring(ALIGN, view, payload_len, total_len)
                        self.assertEqual(
                            view[:payload_len],
                            self.contents[ALIGN : ALIGN + payload_len],
                        )
                        assert_guards(backing, view)

    def test_single_read_spare_capacity_and_extra_padding(self) -> None:
        """Handle spare capacity and transfers with multiple padding blocks."""
        for payload_len, capacity, total_len in (
            (4100, 5000, 8192),
            (8192, 8192, 12288),
            (4100, 4100, 16384),
        ):
            with self.subTest(payload=payload_len, capacity=capacity, total=total_len):
                backing, view = aligned_buffer(capacity)
                self.dev.read_uring(ALIGN, view, payload_len, total_len)
                self.assertEqual(
                    view[:payload_len], self.contents[ALIGN : ALIGN + payload_len]
                )
                assert_guards(backing, view)

    def test_batched_mixed_buffers_preserve_result_indices(self) -> None:
        """Return one ordered result per direct, padded or unaligned input."""
        cases = [(2 * MIB + 4, 0), (4100, 1), (8192, 0), (4, 0), (5000, 0)]
        buffers = [aligned_buffer(capacity, offset) for capacity, offset in cases]
        totals = [(capacity + ALIGN - 1) // ALIGN * ALIGN for capacity, _ in cases]
        batch = self.dev.batched_read(
            [ALIGN] * len(buffers), [view for _, view in buffers], totals
        )
        self.assertEqual(self.dev.wait_iouring(batch), ([True] * len(buffers), []))
        for backing, view in buffers:
            self.assertEqual(view, self.contents[ALIGN : ALIGN + len(view)])
            assert_guards(backing, view)

    def test_batch_larger_than_submission_queue(self) -> None:
        """Finish split reads even when their parts span multiple queue submissions."""
        buffers = [aligned_buffer(64 * 1024 + 4) for _ in range(33)]
        batch = self.dev.batched_read(
            [ALIGN] * len(buffers),
            [view for _, view in buffers],
            [68 * 1024] * len(buffers),
        )
        self.assertEqual(self.dev.wait_iouring(batch), ([True] * len(buffers), []))
        for backing, view in buffers:
            self.assertEqual(view, self.contents[ALIGN : ALIGN + len(view)])
            assert_guards(backing, view)

    def test_concurrent_single_reads(self) -> None:
        """Keep independent callers' destinations and completions separate."""

        def read_many(worker: int) -> None:
            """Repeatedly verify a distinct file range from one Python thread."""
            payload_len = 64 * 1024 + 4
            offset = (worker + 1) * ALIGN
            backing, view = aligned_buffer(payload_len)
            for _ in range(32):
                self.dev.read_uring(offset, view, payload_len, 68 * 1024)
                self.assertEqual(view, self.contents[offset : offset + payload_len])
            assert_guards(backing, view)

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(read_many, range(4)))

    def test_invalid_batch_publishes_no_reads(self) -> None:
        """A later invalid offset must not publish an earlier prepared read."""
        first_owner, first = aligned_buffer(4100)
        second_owner, second = aligned_buffer(4100)
        with self.assertRaisesRegex(ValueError, "aligned offset"):
            self.dev.batched_read([ALIGN, ALIGN + 1], [first, second], [8192, 8192])
        self.assertEqual(first, bytes([GUARD]) * len(first))
        self.assertEqual(second, bytes([GUARD]) * len(second))
        self.dev.read_uring(ALIGN, first, 4100, 8192)
        self.assertEqual(first, self.contents[ALIGN : ALIGN + 4100])
        assert_guards(first_owner, first)
        assert_guards(second_owner, second)

    def test_invalid_single_read_releases_destination(self) -> None:
        """Reject invalid requests and release their Python buffer exports."""
        out = bytearray(4100)
        cases = ((1, 4100, 8192), (0, 4101, 8192), (0, 4100, 4100))
        for offset, payload, total in cases:
            with self.subTest(offset=offset, payload=payload, total=total):
                with self.assertRaises(ValueError):
                    self.dev.read_uring(offset, out, payload, total)
                out.extend(b"x")
                out.pop()

    def test_single_kernel_error_then_reuse(self) -> None:
        """Propagate failed split reads and leave the device usable afterward."""
        backing, view = aligned_buffer(4100)
        # The offset passes the alignment check but is negative as a kernel
        # loff_t, so actual io_uring completions report EINVAL.
        with self.assertRaises(OSError):
            self.dev.read_uring(1 << 63, view, 4100, 8192)
        self.dev.read_uring(ALIGN, view, 4100, 8192)
        self.assertEqual(view, self.contents[ALIGN : ALIGN + len(view)])
        assert_guards(backing, view)

    def test_batched_kernel_error_preserves_logical_indices(self) -> None:
        """Report one failed logical read among successful split reads."""
        buffers = [aligned_buffer(4100) for _ in range(3)]
        offsets = [ALIGN, 1 << 63, 2 * ALIGN]
        batch = self.dev.batched_read(
            offsets, [view for _, view in buffers], [8192] * 3
        )
        results, errors = self.dev.wait_iouring(batch)
        self.assertEqual(results, [True, False, True])
        self.assertEqual([index for index, _ in errors], [1])
        for index in (0, 2):
            backing, view = buffers[index]
            offset = offsets[index]
            self.assertEqual(view, self.contents[offset : offset + len(view)])
            assert_guards(backing, view)

    def test_registered_buffer_read(self) -> None:
        """Read padded payloads correctly when the destination is registered."""
        dev = self.open_device()
        backing, view = aligned_buffer(64 * 1024 + 4)
        address = ctypes.addressof(ctypes.c_ubyte.from_buffer(view))
        try:
            dev.register_fixed_buffers([address], [len(view)])
            dev.read_uring(ALIGN, view, len(view), 68 * 1024)
            self.assertEqual(view, self.contents[ALIGN : ALIGN + len(view)])
            batch = dev.batched_read([ALIGN], [view], [68 * 1024])
            self.assertEqual(dev.wait_iouring(batch), ([True], []))
            self.assertEqual(view, self.contents[ALIGN : ALIGN + len(view)])
            assert_guards(backing, view)
        finally:
            dev.close()

    def test_partial_registration_allows_read_beyond_registered_range(self) -> None:
        """A valid larger Python buffer remains usable through ordinary reads."""
        dev = self.open_device()
        backing, view = aligned_buffer(64 * 1024 + 4)
        address = ctypes.addressof(ctypes.c_ubyte.from_buffer(view))
        try:
            dev.register_fixed_buffers([address], [ALIGN])
            dev.read_uring(ALIGN, view, len(view), 68 * 1024)
            self.assertEqual(view, self.contents[ALIGN : ALIGN + len(view)])
            assert_guards(backing, view)
        finally:
            dev.close()

    def test_buffered_read_padding(self) -> None:
        """Preserve padded-read behavior when O_DIRECT is disabled."""
        dev = self.open_device(odirect=False)
        try:
            backing, view = aligned_buffer(4100, 1)
            dev.read_uring(ALIGN, view, len(view), 8192)
            self.assertEqual(view, self.contents[ALIGN : ALIGN + len(view)])
            assert_guards(backing, view)
        finally:
            dev.close()

    def test_close_drains_split_batch(self) -> None:
        """Closing a device settles every requested read before releasing buffers."""
        dev = self.open_device(depth=2)
        buffers = [aligned_buffer(64 * 1024 + 4) for _ in range(33)]
        batch = dev.batched_read(
            [ALIGN] * len(buffers),
            [view for _, view in buffers],
            [68 * 1024] * len(buffers),
        )
        dev.close()
        results, errors = dev.wait_iouring(batch)
        self.assertEqual(len(results), len(buffers))
        self.assertEqual(
            {index for index, _ in errors},
            {i for i, ok in enumerate(results) if not ok},
        )
        for ok, (backing, view) in zip(results, buffers, strict=True):
            if ok:
                self.assertEqual(view, self.contents[ALIGN : ALIGN + len(view)])
            assert_guards(backing, view)


if __name__ == "__main__":
    unittest.main()
