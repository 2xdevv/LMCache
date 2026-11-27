# Native connector stall watchdog

`NativeConnectorL2Adapter` warns when a worker thread of its native (C++)
connector has spent too long on one key, so a storage backend that stops
answering shows up in the logs instead of as requests that silently never
finish.

## Why

Most native connectors make blocking calls on their worker threads that have
no deadline: USRBIO completion waits and FUSE metadata syscalls (`hf3fs`),
file syscalls on a network or FUSE filesystem (`fs_native`), TCP reads
without a socket timeout (`resp`), and NIXL transfer polling
(`nixl_native`). When such a call never returns, the worker is parked, the
batch it belongs to never completes, and nothing reports it: the adapter's
pending-op table has no timestamps and `report_status()` only checks the
demux thread.

The watchdog detects stalls; it does not cancel or time out any I/O. A call
that comes back late still completes normally.

## Split between C++ and Python

The C++ side only records state; the Python side decides what a stall is and
reports it.

```
worker thread (C++)                    demux thread (Python)
───────────────────                    ─────────────────────
create_connection()  state=starting    every min(5 s, threshold / 4):
take tile            state=running       snapshot = client.worker_status()
  per key: key=..., timer restarts       warn once per worker activity past
tile done            state=idle            stall_warn_seconds
                                         log once when that worker moves on
                                         keep counts for report_status()
```

- **C++ (`ConnectorBase`)**: every worker thread owns a `WorkerSlot` holding
  its lane, index, state, op, future id, current key, the time the state
  began, and an `activity` counter that increments on every change. Each
  queue records when its oldest request was enqueued. `worker_status()`
  returns a snapshot. Workers only update a slot under a per-slot mutex, so
  the cost is one short lock and a key copy per key.
- **Python (`NativeConnectorL2Adapter`)**: the demux loop, which already
  wakes every 500 ms, scans the snapshot. Thresholds, rate limiting and log
  text stay in Python and use the LMCache logger.

## `worker_status()`

Bound for every connector by `LMCACHE_BIND_CONNECTOR_METHODS`:

```python
{
    "workers": [
        {
            "lane": "store",       # shared, lookup, retrieve, store, delete,
                                   # or a connector lane such as read_pool
            "index": 2,            # worker number within the lane
            "state": "running",    # starting | idle | running | exited
            "op": "set",           # get | set | exists | delete | ""
            "future_id": 17,       # batch being served, 0 unless running
            "key": "model@0x0@0@3fa2...",  # current key, "" if unknown
            "seconds": 95.2,       # time in this state; while running,
                                   # time since the current key started
            "activity": 4031,      # increments on every state or key change
        },
    ],
    "queues": [
        {"lane": "store", "depth": 12, "oldest_seconds": 130.4},
    ],
}
```

- `starting` covers `create_connection()`, e.g. 3FS buffer registration.
- The default `do_batch_*` loops record each key before running it. A
  connector that overrides a batch method and hands the whole tile to its
  backend (Mooncake, NIXL) reports the tile only, with an empty `key`; it
  can call `set_current_key_()` to report progress.
- `fs_native`'s read pool threads, which run lookups and reads outside the
  shared workers, appear as lane `read_pool`; its queue `depth` counts keys
  not yet claimed.

A connector built without `worker_status()` (an older third-party plugin or
`lmcache-nixl` wheel) keeps working; the adapter turns the watchdog off. The
method is additive, so it does not change the NIXL connector's
`API_VERSION`.

## Configuration

Every native adapter type (`fs_native`, `hf3fs`, `resp`, `aerospike`,
`mooncake_store`, `nixl_native`, `native_plugin`) accepts:

- `stall_warn_seconds` (float, default `90`): how long a worker may spend on
  one key, or in `create_connection()`, before a warning. `0` disables the
  watchdog.

The default sits above the backends' own retry windows (the 3FS storage
client retries for up to 60 s), so a warning means the backend has stopped
answering rather than failing over.

```json
{"type": "hf3fs", "mount_point": "/mnt/3fs", "base_paths": "/mnt/3fs/kv",
 "stall_warn_seconds": 120}
```

## Output

One warning per stuck worker activity, then one line if it moves on:

```
WARNING ... Hf3fsL2Adapter: worker store#2 stuck in set on key
  model@0x0@0@3fa2... (future 17) for 95 s; lane store: 4/4 workers busy,
  12 queued (oldest 130 s)
WARNING ... Hf3fsL2Adapter: worker store#2 moved on after being stuck for at
  least 95 s
```

A worker stuck in `create_connection()` is reported as `still setting up its
connection after N s`. A connector that does not report keys shows
`on key (not reported)`. The queue part is left out for a lane without a
queue entry.

`report_status()` gains two fields while the watchdog is active:

- `stalled_workers`: workers past the threshold at the last scan.
- `oldest_busy_seconds`: the longest time any worker has spent on its
  current key or connection setup, at the last scan.

## What to do with a warning

The warning says which lane, worker, operation and key; a native stack dump
(`py-spy dump --native --pid <pid>`) shows which call is blocked. Fixes are
on the storage side (see the backend's page under
`docs/source/mp/l2_storage/`), or a restart when the backend client cannot
recover in place.
