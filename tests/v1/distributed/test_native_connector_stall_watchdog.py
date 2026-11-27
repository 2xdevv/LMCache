# SPDX-License-Identifier: Apache-2.0
"""Tests for the native connector stall watchdog.

Workers of the real native FS connector are parked on FIFOs: a FIFO open or
read blocks until the test opens or feeds the other end, which stands in for
a storage call that stops answering.
"""

# Standard
from collections.abc import Callable, Iterator
from pathlib import Path
import logging
import os
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.key_codec import encode_fs_filename, encode_storage_key
from lmcache.v1.distributed.l2_adapters import (
    create_l2_adapter,
    native_connector_l2_adapter,
)
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    FSNativeL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    NativeConnectorL2Adapter,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import create_event_notifier

try:
    # First Party
    from lmcache.lmcache_fs import LMCacheFSClient
except ImportError:
    LMCacheFSClient = None  # type: ignore[assignment,misc]

requires_fs_ext = pytest.mark.skipif(
    LMCacheFSClient is None,
    reason="native FS extension is not built",
)

# Larger than a pipe's buffer, so a writer parks in write() until drained.
_PIPE_FILL_BYTES = 512 * 1024
_STALL_FIELDS = {"stalled_workers", "oldest_busy_seconds"}


def _make_key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def _make_memory_obj(num_bytes: int) -> TensorMemoryObj:
    num_floats = num_bytes // 4
    metadata = MemoryObjMetadata(
        shape=torch.Size([num_floats]),
        dtype=torch.float32,
        address=0,
        phy_size=num_floats * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(
        torch.zeros(num_floats, dtype=torch.float32), metadata, parent_allocator=None
    )


def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not reached before the timeout")


def _find_worker(client: object, **fields: object) -> dict | None:
    for worker in client.worker_status()["workers"]:  # type: ignore[attr-defined]
        if all(worker[name] == value for name, value in fields.items()):
            return worker
    return None


def _queue(client: object, lane: str) -> dict:
    queues = client.worker_status()["queues"]  # type: ignore[attr-defined]
    return next(q for q in queues if q["lane"] == lane)


def _drain_fifo_and_close(rfd: int) -> None:
    try:
        while os.read(rfd, 65536):
            pass
    finally:
        os.close(rfd)


class _FifoReader:
    """Opens a FIFO for reading, which unblocks a writer's open(), and drains
    it on a thread once released."""

    def __init__(self, path: Path) -> None:
        self._rfd = os.open(path, os.O_RDONLY)
        self._thread: threading.Thread | None = None

    def release(self) -> None:
        self._thread = threading.Thread(
            target=_drain_fifo_and_close, args=(self._rfd,), daemon=True
        )
        self._thread.start()

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def _wait_completion(client: object, future_id: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for fid, ok, _err, _bits in client.drain_completions():  # type: ignore[attr-defined]
            if int(fid) == future_id:
                return bool(ok)
        time.sleep(0.02)
    raise AssertionError(f"future {future_id} did not complete")


@pytest.fixture
def adapter_warnings() -> Iterator[list[str]]:
    """Collect warning messages logged by the native connector adapter."""
    messages: list[str] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    handler = _Collector(level=logging.WARNING)
    adapter_logger = logging.getLogger(native_connector_l2_adapter.__name__)
    adapter_logger.addHandler(handler)
    yield messages
    adapter_logger.removeHandler(handler)


class _ClientWithoutWorkerStatus:
    """A native client built before worker_status() existed."""

    def __init__(self) -> None:
        self._notifier = create_event_notifier()

    def event_fd(self) -> int:
        return self._notifier.fileno()

    def drain_completions(self) -> list:
        return []

    def close(self) -> None:
        self._notifier.close()


# ---------------------------------------------------------------------------
# worker_status() on the real native FS connector
# ---------------------------------------------------------------------------


@requires_fs_ext
def test_worker_status_reports_parked_store_worker_and_its_queue(
    tmp_path: Path,
) -> None:
    """A worker parked inside a write shows as running on that key, its time
    keeps growing, and the tile queued behind it is counted; once released the
    worker returns to idle."""
    (tmp_path / "tmp").mkdir()
    client = LMCacheFSClient(str(tmp_path), 1, "tmp", False, 0, 0, 0)
    try:
        _wait_until(
            lambda: _find_worker(client, lane="shared", state="idle") is not None
        )

        parked_obj, queued_obj = _make_key(1), _make_key(2)
        parked_key = encode_storage_key(parked_obj)
        fifo = tmp_path / "tmp" / encode_fs_filename(parked_obj)
        os.mkfifo(fifo)

        parked_buf = bytearray(_PIPE_FILL_BYTES)
        parked_fid = client.submit_batch_set([parked_key], [memoryview(parked_buf)])
        reader = _FifoReader(fifo)  # returns once the worker opened the FIFO
        queued_buf = bytearray(4096)
        queued_fid = client.submit_batch_set(
            [encode_storage_key(queued_obj)], [memoryview(queued_buf)]
        )
        try:
            _wait_until(
                lambda: (
                    _find_worker(client, state="running", key=parked_key) is not None
                )
            )
            first = _find_worker(client, state="running", key=parked_key)
            assert first is not None
            assert first["lane"] == "shared"
            assert first["op"] == "set"
            assert first["future_id"] == parked_fid
            time.sleep(0.2)
            second = _find_worker(client, state="running", key=parked_key)
            assert second is not None
            assert second["activity"] == first["activity"]
            assert second["seconds"] > first["seconds"]

            queue = _queue(client, "shared")
            assert queue["depth"] == 1
            assert queue["oldest_seconds"] > 0
        finally:
            reader.release()

        assert _wait_completion(client, parked_fid)
        assert _wait_completion(client, queued_fid)
        reader.join()
        _wait_until(
            lambda: _find_worker(client, lane="shared", state="idle") is not None
        )
        idle = _find_worker(client, lane="shared", state="idle")
        assert idle is not None
        assert idle["activity"] > first["activity"]
        assert idle["future_id"] == 0
        assert _queue(client, "shared")["depth"] == 0
    finally:
        client.close()


@requires_fs_ext
def test_worker_status_reports_parked_read_pool_reader(tmp_path: Path) -> None:
    """A read-pool reader parked inside a read shows as lane read_pool, running
    a get on that key."""
    client = LMCacheFSClient(str(tmp_path), 1, "", False, 0, 2, 0)
    try:
        parked_obj = _make_key(3)
        parked_key = encode_storage_key(parked_obj)
        fifo = tmp_path / encode_fs_filename(parked_obj)
        os.mkfifo(fifo)

        payload = bytes(range(256)) * 16
        dst = bytearray(len(payload))
        fid = client.submit_batch_get([parked_key], [memoryview(dst)])
        wfd = os.open(fifo, os.O_WRONLY)  # returns once the reader opened it
        try:
            _wait_until(
                lambda: (
                    _find_worker(client, state="running", key=parked_key) is not None
                )
            )
            reader = _find_worker(client, state="running", key=parked_key)
            assert reader is not None
            assert reader["lane"] == "read_pool"
            assert reader["op"] == "get"
            assert reader["future_id"] == fid
            assert _queue(client, "read_pool")["depth"] == 0
        finally:
            view = memoryview(payload)
            sent = 0
            while sent < len(view):
                sent += os.write(wfd, view[sent:])
            os.close(wfd)

        assert _wait_completion(client, fid)
        assert bytes(dst) == payload
    finally:
        client.close()


# ---------------------------------------------------------------------------
# NativeConnectorL2Adapter warnings and status
# ---------------------------------------------------------------------------


@requires_fs_ext
def test_adapter_warns_once_per_stall_and_logs_when_the_worker_moves_on(
    tmp_path: Path, adapter_warnings: list[str]
) -> None:
    """A worker stuck past stall_warn_seconds is reported once, naming its key,
    and counted by report_status(); after release one more line says it moved
    on and the count returns to zero."""
    config = FSNativeL2AdapterConfig(
        str(tmp_path),
        num_workers=1,
        relative_tmp_dir="tmp",
        num_read_workers=0,
        stall_warn_seconds=0.5,
    )
    adapter = create_l2_adapter(config)
    try:
        stuck_obj = _make_key(4)
        stuck_key = encode_storage_key(stuck_obj)
        fifo = tmp_path / "tmp" / encode_fs_filename(stuck_obj)
        os.mkfifo(fifo)

        adapter.submit_store_task([stuck_obj], [_make_memory_obj(2 * _PIPE_FILL_BYTES)])
        reader = _FifoReader(fifo)
        try:
            _wait_until(
                lambda: any(stuck_key in m for m in adapter_warnings), timeout=5.0
            )
            status = adapter.report_status()
            assert status["stalled_workers"] == 1
            assert status["oldest_busy_seconds"] >= 0.5

            time.sleep(1.0)  # several more scans while still stuck
            assert sum(stuck_key in m for m in adapter_warnings) == 1
        finally:
            reader.release()

        _wait_until(lambda: bool(adapter.pop_completed_store_tasks()))
        reader.join()
        _wait_until(lambda: any("moved on" in m for m in adapter_warnings))
        _wait_until(lambda: adapter.report_status()["stalled_workers"] == 0)
    finally:
        adapter.close()


@requires_fs_ext
def test_adapter_omits_stall_fields_when_disabled(tmp_path: Path) -> None:
    """stall_warn_seconds=0 turns the watchdog off."""
    config = FSNativeL2AdapterConfig(str(tmp_path), stall_warn_seconds=0)
    adapter = create_l2_adapter(config)
    try:
        assert not _STALL_FIELDS & adapter.report_status().keys()
    finally:
        adapter.close()


@requires_fs_ext
def test_adapter_reports_stall_fields_by_default(tmp_path: Path) -> None:
    """With the default threshold and no stuck worker, both fields are present
    and report no stall."""
    adapter = create_l2_adapter(FSNativeL2AdapterConfig(str(tmp_path)))
    try:
        status = adapter.report_status()
        assert _STALL_FIELDS <= status.keys()
        assert status["stalled_workers"] == 0
    finally:
        adapter.close()


def test_adapter_runs_without_worker_status_support() -> None:
    """A client without worker_status() keeps working, with the watchdog off."""
    adapter = NativeConnectorL2Adapter(
        _ClientWithoutWorkerStatus(),
        MemoryTags(MemorySegment.STORAGE, MemoryResidency.REMOTE),
        stall_warn_seconds=0.5,
    )
    try:
        time.sleep(1.0)  # let the demux loop pass several scan points
        status = adapter.report_status()
        assert status["is_healthy"]
        assert not _STALL_FIELDS & status.keys()
    finally:
        adapter.close()
