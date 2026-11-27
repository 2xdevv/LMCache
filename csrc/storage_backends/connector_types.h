// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// the main "types" in this file:
// 1. Op: enum class for the batched operations (GET, SET, EXISTS, DELETE)
// 2. BatchState: shared communication state between threads executing a single
// batch operation
// 3. Request: the request data structure for a single operation (submitted to
// SQ)
// 4. Completion: the completion data structure for a single operation
// (collected from CQ)
// 5. WorkerSlot / ConnectorStatus: per-worker activity published for
// worker_status(), which the Python adapter scans for stalled workers

namespace lmcache {
namespace connector {

// we only support batched operations
// benefits are fewer submissions and fewer completions
enum class Op : uint8_t {
  BATCH_TILE_GET,
  BATCH_TILE_SET,
  BATCH_TILE_EXISTS,
  BATCH_TILE_DELETE
};

// Per-key result of a batch DELETE. Python reads any non-zero value as
// "the key is gone".
enum class DeleteResult : uint8_t {
  NOT_DELETED = 0,  // failed or unsupported: the key may still exist
  DELETED = 1,      // the key existed and was removed
  NOT_FOUND = 2     // the key was already absent
};

/*
shared communication state between threads executing a single batch operation.
all threads need to complete before the completion is sent.

tiling refers to dividing work for batched operations between threads
beforehand.
*/
struct BatchState {
  std::atomic<uint32_t> remaining_tiles{0};
  std::atomic<bool> any_failed{false};

  std::mutex err_mu;
  std::string first_error;

  // Per-key success/failure results used by both EXISTS and GET.
  // For EXISTS: 1 = key found, 0 = not found.
  // For GET: 1 = read succeeded, 0 = read failed (e.g. file
  //   not found).  This enables per-key error tolerance on loads.
  // For DELETE: a DeleteResult value.
  // IMPORTANT: not vector<bool> due to concurrent write data race
  std::vector<uint8_t> per_key_results;

  Op batch_op;
};

/*
LIFETIME GUARANTEE:
we have a strict assumption that Python will NOT clean up any buffer memory
before all C++ operations finish. This is guaranteed by the Python-side design
where the caller holds references to all buffers until drain_completions()
returns the corresponding future_id. Therefore, we do NOT need to track
buf_owner references or acquire the GIL to prevent premature cleanup.
we can safely use raw pointers extracted under the GIL without additional
lifetime management on the C++ side.
*/

struct Request {
  uint64_t future_id = 0;
  Op op;

  // all operations use the batched structure (even single-item operations
  // are treated as batches of size 1)
  std::vector<std::string> keys;
  std::vector<void*> buf_ptrs;
  std::vector<size_t> buf_lens;

  // shared batch state between threads executing a single batch operation
  // so that they can coordinate when to send the completion
  std::shared_ptr<BatchState> batch;

  // for batch exists tiles, track which indices this tile is responsible for
  size_t start_idx = 0;

  // batch_chunk_num_bytes for get/set operations (passed per-operation, not
  // per-connection)
  size_t batch_chunk_num_bytes = 0;

  // set when the tile is queued, so worker_status() can report queue age
  std::chrono::steady_clock::time_point enqueued_at{};
};

struct Completion {
  uint64_t future_id = 0;

  bool ok = true;

  // for EXISTS operations, store boolean results as
  // bytes (0/1). Single EXISTS will have 1 element, batch EXISTS will have N
  // elements. No result in the completion for SET and GET.
  std::vector<uint8_t> result_bytes;

  std::string error;
};

enum class WorkerState : uint8_t { STARTING, IDLE, RUNNING, EXITED };

inline const char* worker_state_name(WorkerState state) {
  switch (state) {
    case WorkerState::STARTING:
      return "starting";
    case WorkerState::IDLE:
      return "idle";
    case WorkerState::RUNNING:
      return "running";
    case WorkerState::EXITED:
      return "exited";
  }
  return "";
}

inline const char* op_name(Op op) {
  switch (op) {
    case Op::BATCH_TILE_GET:
      return "get";
    case Op::BATCH_TILE_SET:
      return "set";
    case Op::BATCH_TILE_EXISTS:
      return "exists";
    case Op::BATCH_TILE_DELETE:
      return "delete";
  }
  return "";
}

// One worker's view as returned by worker_status().
struct WorkerStatusEntry {
  std::string lane;
  int index = 0;
  std::string state;
  std::string op;  // empty unless running
  uint64_t future_id = 0;
  std::string key;  // empty when the connector does not report keys
  // time in the current state; while running, since the current key started
  double seconds = 0.0;
  uint64_t activity = 0;
};

// One work queue's view as returned by worker_status().
struct QueueStatusEntry {
  std::string lane;
  size_t depth = 0;  // queued work units (tiles, or keys for a key queue)
  double oldest_seconds = 0.0;
};

struct ConnectorStatus {
  std::vector<WorkerStatusEntry> workers;
  std::vector<QueueStatusEntry> queues;
};

inline double seconds_since(std::chrono::steady_clock::time_point start,
                            std::chrono::steady_clock::time_point now) {
  return std::chrono::duration<double>(now - start).count();
}

/*
Activity a worker thread publishes for worker_status().

The owning thread writes it through the methods below; worker_status() reads
it from another thread. Every change restarts the timer and bumps
`activity`, so a reader can tell "stuck on the same key" (activity unchanged)
from "busy on a new key" (activity changed). The mutex is a leaf lock held
only for the copy, never across I/O.
*/
class WorkerSlot {
 public:
  WorkerSlot(std::string lane, int index)
      : lane_(std::move(lane)),
        index_(index),
        since_(std::chrono::steady_clock::now()) {}

  WorkerSlot(const WorkerSlot&) = delete;
  WorkerSlot& operator=(const WorkerSlot&) = delete;

  void set_idle() { transition(WorkerState::IDLE, nullptr, 0); }
  void set_exited() { transition(WorkerState::EXITED, nullptr, 0); }

  // Start serving a tile of batch `future_id`; the key is unknown until
  // set_key() is called.
  void begin_tile(Op op, uint64_t future_id) {
    transition(WorkerState::RUNNING, &op, future_id);
  }

  // Record that the running worker moved on to `key`.
  void set_key(const std::string& key) {
    std::lock_guard<std::mutex> lk(mu_);
    key_ = key;
    since_ = std::chrono::steady_clock::now();
    ++activity_;
  }

  WorkerStatusEntry snapshot(std::chrono::steady_clock::time_point now) const {
    std::lock_guard<std::mutex> lk(mu_);
    WorkerStatusEntry entry;
    entry.lane = lane_;
    entry.index = index_;
    entry.state = worker_state_name(state_);
    if (state_ == WorkerState::RUNNING) {
      entry.op = op_name(op_);
      entry.future_id = future_id_;
      entry.key = key_;
    }
    entry.seconds = seconds_since(since_, now);
    entry.activity = activity_;
    return entry;
  }

 private:
  void transition(WorkerState state, const Op* op, uint64_t future_id) {
    std::lock_guard<std::mutex> lk(mu_);
    state_ = state;
    if (op != nullptr) {
      op_ = *op;
    }
    future_id_ = future_id;
    key_.clear();
    since_ = std::chrono::steady_clock::now();
    ++activity_;
  }

  const std::string lane_;
  const int index_;
  mutable std::mutex mu_;
  WorkerState state_ = WorkerState::STARTING;
  Op op_ = Op::BATCH_TILE_GET;
  uint64_t future_id_ = 0;
  std::string key_;
  std::chrono::steady_clock::time_point since_;
  uint64_t activity_ = 0;
};

}  // namespace connector
}  // namespace lmcache
