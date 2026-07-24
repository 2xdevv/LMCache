# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from types import TracebackType
from typing import Any, cast
import json
import sys
import threading
import time
import types

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import DiskCacheMetadata
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend.raw_block.core import RawBlockCore, _Entry


class _TrackingLock:
    """Lock wrapper that exposes whether its context is currently held."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.held = False

    def __enter__(self) -> _TrackingLock:
        self._lock.acquire()
        self.held = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.held = False
        self._lock.release()


class _LockAssertingPositions:
    """Cached-position stand-in that requires conversion outside the lock."""

    def __init__(self, lock: _TrackingLock) -> None:
        self._lock = lock
        self.tolist_called = False

    def tolist(self) -> list[int]:
        """Return positions after asserting the core lock is not held."""
        assert not self._lock.held
        self.tolist_called = True
        return [1, 2, 3]


def _make_snapshot_core() -> tuple[
    RawBlockCore,
    _TrackingLock,
    _LockAssertingPositions,
]:
    """Build the minimal RawBlockCore state needed by checkpoint creation."""
    core = RawBlockCore.__new__(RawBlockCore)
    lock = _TrackingLock()
    positions = _LockAssertingPositions(lock)

    core.device_path = "/tmp/raw-block-snapshot-test"
    core.capacity_bytes = 1024 * 1024
    core.block_align = 4096
    core.header_bytes = 4096
    core.slot_bytes = 64 * 1024
    core.meta_total_bytes = 64 * 1024
    core.meta_magic_text = "LMCIDX01"
    core.meta_version = 1
    core.meta_idle_quiet_ms = 0
    core._data_base_offset = core.meta_total_bytes
    core._next_slot = 1
    core._meta_dirty_total = 1
    core._meta_persisted = 0
    core._inflight_io_count = 0
    core._last_io_ts = time.monotonic()
    core._lock = lock
    core._index = {
        "snapshot-key": _Entry(
            offset=core._data_base_offset,
            size=3,
            meta=DiskCacheMetadata(
                path=f"{core.device_path}@{core._data_base_offset}",
                size=3,
                shape=torch.Size([3]),
                dtype=torch.uint8,
                cached_positions=cast(torch.Tensor, positions),
                fmt=MemoryFormat.BINARY,
            ),
        )
    }
    return core, lock, positions


def test_raw_block_snapshot_converts_entry_metadata_outside_lock() -> None:
    """The detached Python snapshot must not hold the core mutation lock."""
    core, _, positions = _make_snapshot_core()

    snapshot, dirty_total = core._snapshot_state()

    entry = snapshot["entries"]["snapshot-key"]
    assert dirty_total == 1
    assert "free_slots" not in snapshot
    assert entry == {
        "offset": core.meta_total_bytes,
        "size": 3,
        "shape": [3],
        "dtype": "uint8",
        "fmt": "BINARY",
        "cached_positions": [1, 2, 3],
    }
    assert positions.tolist_called


def test_checkpoint_now_uses_rust_payload_serializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production checkpoint creation must pass a detached index to Rust."""
    core, lock, _ = _make_snapshot_core()
    captured: dict[str, Any] = {}
    writes: list[tuple[bytes, int]] = []

    def serialize_payload(
        device_path: str,
        capacity_bytes: int,
        block_align: int,
        header_bytes: int,
        slot_bytes: int,
        meta_total_bytes: int,
        meta_magic_text: str,
        meta_version: int,
        data_base_offset: int,
        next_slot: int,
        index: dict[str, _Entry],
    ) -> bytes:
        assert not lock.held
        captured.update(
            {
                "device_path": device_path,
                "capacity_bytes": capacity_bytes,
                "block_align": block_align,
                "header_bytes": header_bytes,
                "slot_bytes": slot_bytes,
                "meta_total_bytes": meta_total_bytes,
                "meta_magic_text": meta_magic_text,
                "meta_version": meta_version,
                "data_base_offset": data_base_offset,
                "next_slot": next_slot,
                "index": index,
            }
        )
        return b'{"source":"rust"}'

    def write_checkpoint(payload: bytes, dirty_total_snapshot: int) -> bool:
        writes.append((payload, dirty_total_snapshot))
        return True

    monkeypatch.setitem(
        sys.modules,
        "lmcache_rust_raw_block_io",
        types.SimpleNamespace(
            serialize_raw_block_checkpoint_payload=serialize_payload,
        ),
    )
    monkeypatch.setattr(core, "_write_checkpoint", write_checkpoint)

    core.checkpoint_now()

    assert captured == {
        "device_path": core.device_path,
        "capacity_bytes": core.capacity_bytes,
        "block_align": core.block_align,
        "header_bytes": core.header_bytes,
        "slot_bytes": core.slot_bytes,
        "meta_total_bytes": core.meta_total_bytes,
        "meta_magic_text": core.meta_magic_text,
        "meta_version": core.meta_version,
        "data_base_offset": core.meta_total_bytes,
        "next_slot": 1,
        "index": captured["index"],
    }
    assert writes == [(b'{"source":"rust"}', 1)]

    captured_index = cast(dict[str, _Entry], captured["index"])
    captured_index.clear()
    snapshot, _ = core._snapshot_state()
    assert list(snapshot["entries"]) == ["snapshot-key"]


def test_checkpoint_payload_falls_back_when_rust_function_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older extension without the serializer must use the Python path."""
    core, _, _ = _make_snapshot_core()
    monkeypatch.setitem(
        sys.modules,
        "lmcache_rust_raw_block_io",
        types.SimpleNamespace(),
    )

    payload, dirty_total = core._serialize_checkpoint_payload()
    snapshot = json.loads(payload)

    assert dirty_total == 1
    assert "free_slots" not in snapshot
    assert list(snapshot["entries"]) == ["snapshot-key"]
