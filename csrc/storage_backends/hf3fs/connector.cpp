// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText:Copyright (c) 2026 Samsung Electronics Co., Ltd.

#include "connector.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits.h>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <thread>
#include <type_traits>
#include <utility>

#include "../keys.h"

namespace lmcache {
namespace connector {

namespace {

// io_depth for every ring. 0 makes the 3FS ring worker start prepared I/Os as
// soon as it sees them. A non-zero value holds back batches smaller than
// |io_depth|, and 3FS re-checks a held-back batch only when a later submit
// wakes its ring watcher, so the last request before the client goes idle
// would never complete.
constexpr int kIoDepth = 0;

// Ring timeout in ms. 3FS reads it only to flush a held-back batch when
// io_depth is negative, so with kIoDepth = 0 it has no effect.
constexpr int kIorTimeoutMs = 0;

/**
 * An open file registered with 3FS for USRBIO.
 *
 * Opens the file and registers its fd via hf3fs_reg_fd on construction;
 * deregisters and closes it on destruction. Move-only so a batch can keep
 * many files registered while their I/Os are in the ring.
 */
class RegisteredFile {
 public:
  /**
   * @param path File path
   * @param for_write true opens O_WRONLY|O_CREAT, false opens O_RDONLY
   * @throws std::runtime_error if open or hf3fs_reg_fd fails
   */
  RegisteredFile(const std::string& path, bool for_write) {
    int flags = for_write ? (O_WRONLY | O_CREAT) : O_RDONLY;
    fd_ = ::open(path.c_str(), flags, for_write ? 0644 : 0);
    if (fd_ < 0) {
      throw std::runtime_error("open failed: " + std::string(strerror(errno)));
    }
    int reg_result = hf3fs_reg_fd(fd_, 0);
    if (reg_result > 0) {
      ::close(fd_);
      throw std::runtime_error(
          "hf3fs_reg_fd failed: " + std::to_string(reg_result) +
          " (errno=" + strerror(reg_result) + ")");
    }
  }

  RegisteredFile(RegisteredFile&& other) noexcept
      : fd_(std::exchange(other.fd_, -1)) {}
  RegisteredFile& operator=(RegisteredFile&&) = delete;
  RegisteredFile(const RegisteredFile&) = delete;
  RegisteredFile& operator=(const RegisteredFile&) = delete;

  ~RegisteredFile() {
    if (fd_ >= 0) {
      hf3fs_dereg_fd(fd_);
      ::close(fd_);
    }
  }

  int fd() const { return fd_; }

 private:
  int fd_ = -1;
};

}  // namespace

/**
 * Construct a new Hf3fsConnector.
 *
 * Validates:
 * - mount_point is a valid 3FS path (via hf3fs_extract_mount_point)
 * - Each base_path is a subdirectory of mount_point
 * - ior_entries and single_buffer_size are positive
 */
Hf3fsConnector::Hf3fsConnector(std::string mount_point, std::string base_paths,
                               int num_workers, int ior_entries, int numa_id,
                               size_t single_buffer_size,
                               bool enable_key_buffer,
                               WorkerPoolConfig worker_pool_config)
    : ConnectorBase(num_workers, std::move(worker_pool_config)),
      mount_point_(std::move(mount_point)),
      ior_entries_(ior_entries),
      numa_id_(numa_id),
      single_buffer_size_(single_buffer_size),
      buffer_enabled_(enable_key_buffer) {
  if (ior_entries_ <= 0) {
    throw std::runtime_error("ior_entries must be positive, got " +
                             std::to_string(ior_entries_));
  }
  if (single_buffer_size_ == 0) {
    throw std::runtime_error("single_buffer_size must be positive");
  }

  // Debug print worker pool configuration
  fprintf(stderr,
          "[LMCache HF3FS] Worker pool: num_workers=%d, "
          "per_op_workers={",
          num_workers_);
  for (const auto& [key, count] : worker_pool_config_.per_op_workers) {
    fprintf(stderr, "%s=\"%d\"", key.c_str(), count);
  }
  fprintf(stderr, "}\n");

  // Validate mount_point using 3FS SDK
  char mount_point_buf[PATH_MAX];
  int ret = hf3fs_extract_mount_point(mount_point_buf, sizeof(mount_point_buf),
                                      mount_point_.c_str());
  if (ret < 0) {
    throw std::runtime_error("hf3fs_extract_mount_point failed: '" +
                             mount_point_ + "' is not a valid 3FS path");
  }
  if (ret > static_cast<int>(sizeof(mount_point_buf))) {
    throw std::runtime_error("Mount point path too long: '" + mount_point_ +
                             "'");
  }

  std::string extracted(mount_point_buf);
  if (extracted != mount_point_) {
    throw std::runtime_error("Mount point mismatch: config='" + mount_point_ +
                             "', extracted='" + extracted + "'");
  }

  // Parse comma-separated base_paths
  std::stringstream ss(base_paths);
  std::string path;
  while (std::getline(ss, path, ',')) {
    // Trim whitespace
    path.erase(0, path.find_first_not_of(" \t"));
    path.erase(path.find_last_not_of(" \t") + 1);
    if (!path.empty()) {
      base_paths_.push_back(path);
    }
  }

  if (base_paths_.empty()) {
    throw std::runtime_error("No valid base_paths provided");
  }

  // Validate each base_path is a subdirectory of mount_point
  for (const auto& bp : base_paths_) {
    if (bp.find(mount_point_) != 0) {
      throw std::runtime_error("base_path '" + bp +
                               "' is not a subdirectory of mount_point '" +
                               mount_point_ + "'");
    }
  }

  if (buffer_enabled_) {
    fprintf(stderr, "[LMCache HF3FS] Buffer enabled, begin scan...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    scan_and_build_buffer_();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    fprintf(stderr,
            "[LMCache HF3FS] End of scan, scanned %zu keys from %zu "
            "base_path(s), elapsed %ld ms\n",
            key_buffer_.size(), base_paths_.size(), elapsed_ms);
  } else {
    fprintf(stderr, "[LMCache HF3FS] Buffer disabled\n");
  }

  // IMPORTANT: start worker threads at the END of the constructor
  start_workers();
}

/**
 * Initialize read Ior (I/O ring).
 *
 * @param ior Ior structure to initialize
 * @throws std::runtime_error if hf3fs_iorcreate4 fails
 */
void Hf3fsConnector::init_read_ior(hf3fs_ior& ior) {
  int ret = hf3fs_iorcreate4(&ior, mount_point_.c_str(), ior_entries_,
                             true,  // for_read=true
                             kIoDepth, kIorTimeoutMs, numa_id_,
                             0);  // flags
  if (ret < 0) {
    throw std::runtime_error("hf3fs_iorcreate4 (read) failed: " +
                             std::to_string(-ret));
  }
}

/**
 * Initialize write Ior (I/O ring).
 *
 * @param ior Ior structure to initialize
 * @throws std::runtime_error if hf3fs_iorcreate4 fails
 */
void Hf3fsConnector::init_write_ior(hf3fs_ior& ior) {
  int ret = hf3fs_iorcreate4(&ior, mount_point_.c_str(), ior_entries_,
                             false,  // for_read=false
                             kIoDepth, kIorTimeoutMs, numa_id_,
                             0);  // flags
  if (ret < 0) {
    throw std::runtime_error("hf3fs_iorcreate4 (write) failed: " +
                             std::to_string(-ret));
  }
}

/**
 * Initialize read Iov (I/O buffer).
 *
 * @param iov Iov structure to initialize
 * @throws std::runtime_error if hf3fs_iovcreate fails
 */
void Hf3fsConnector::init_read_iov(hf3fs_iov& iov) {
  int ret = hf3fs_iovcreate(&iov, mount_point_.c_str(), iov_bytes(),
                            0,  // block_size
                            numa_id_);
  if (ret < 0) {
    throw std::runtime_error("hf3fs_iovcreate (read) failed: " +
                             std::to_string(-ret));
  }
}

/**
 * Initialize write Iov (I/O buffer).
 *
 * @param iov Iov structure to initialize
 * @throws std::runtime_error if hf3fs_iovcreate fails
 */
void Hf3fsConnector::init_write_iov(hf3fs_iov& iov) {
  int ret = hf3fs_iovcreate(&iov, mount_point_.c_str(), iov_bytes(),
                            0,  // block_size
                            numa_id_);
  if (ret < 0) {
    throw std::runtime_error("hf3fs_iovcreate (write) failed: " +
                             std::to_string(-ret));
  }
}

/**
 * Create a new per-thread connection with its read and write rings.
 *
 * @return WorkerHf3fsConn Initialized connection structure
 * @throws std::runtime_error if any Ior or Iov cannot be created
 */
WorkerHf3fsConn Hf3fsConnector::create_connection() {
  WorkerHf3fsConn conn;
  ensure_ring(conn, true);
  ensure_ring(conn, false);
  return conn;
}

/**
 * Create the Ior and Iov for one direction if they are not initialized.
 *
 * Rings are created eagerly in create_connection and recreated here after
 * ring_io discards one following a ring-level error.
 *
 * @param conn Connection to use
 * @param for_read true for the read ring, false for the write ring
 * @throws std::runtime_error if hf3fs_iorcreate4 or hf3fs_iovcreate fails
 */
void Hf3fsConnector::ensure_ring(WorkerHf3fsConn& conn, bool for_read) {
  if (for_read) {
    if (!conn.read_ior_initialized) {
      init_read_ior(conn.read_ior);
      conn.read_ior_initialized = true;
    }
    if (!conn.read_iov_initialized) {
      init_read_iov(conn.read_iov);
      conn.read_iov_initialized = true;
    }
  } else {
    if (!conn.write_ior_initialized) {
      init_write_ior(conn.write_ior);
      conn.write_ior_initialized = true;
    }
    if (!conn.write_iov_initialized) {
      init_write_iov(conn.write_iov);
      conn.write_iov_initialized = true;
    }
  }
}

/**
 * Select a base path based on the key's ``chunk_hash``.
 *
 * Wire format:
 *   <model>@<kv_rank_hex>@<ogid_hex>@<chunk_hash_hex>[@<cache_salt>]
 *
 * Algorithm:
 * 1. Extract the ``chunk_hash_hex`` field via
 * ``keys.h::get_chunk_hash_from_key``.
 * 2. Use the LAST 16 hex chars (64 bits) of that hash: sequential keys
 *    like ``...00000000``, ``...00000001`` share leading zeros but differ
 *    in their low bits, so the tail carries the entropy.
 * 3. Convert to uint64 and mod by number of paths.
 *
 * @param key Key string
 * @return Selected base path
 */
const std::string& Hf3fsConnector::select_base_path(
    const std::string& key) const {
  if (base_paths_.size() == 1) {
    return base_paths_[0];
  }

  std::string hash_str;
  try {
    hash_str = get_chunk_hash_from_key(key);
  } catch (const std::exception& e) {
    fprintf(stderr, "[LMCache HF3FS] get chunk hash failed: %s\n", e.what());
    return base_paths_[0];
  }

  // Use last 16 characters (64 bits) for better distribution.
  size_t hash_len = hash_str.length();
  size_t substr_start = (hash_len > 16) ? (hash_len - 16) : 0;
  std::string hash_substr = hash_str.substr(substr_start);

  uint64_t hash_val = std::stoull(hash_substr, nullptr, 16);
  size_t idx = hash_val % base_paths_.size();

  return base_paths_[idx];
}

/**
 * Convert a key to a full file path.
 *
 * The filename is produced by  ``keys.h::key_to_filename`` The base path is
 * selected by hashing the key's ``chunk_hash`` across ``base_paths_``.
 *
 * @param key Key string (wire format ``model@kv_rank@ogid@chunk_hash[@salt]``)
 * @return Full file path (base_path + filename)
 */
std::string Hf3fsConnector::key_to_path(const std::string& key) {
  const std::string& base_path = select_base_path(key);
  std::string filename = key_to_filename(key);
  return base_path + "/" + filename;
}

/**
 * Run reads or writes for many keys through one USRBIO ring.
 *
 * Each key's data is split into segments of at most single_buffer_size_
 * bytes. The call keeps up to ior_entries_ segments in the ring at once instead
 * of one round trip per key, and reaps completions as they arrive, so a read's
 * copy out of the Iov overlaps the I/Os that are still in flight rather than
 * waiting for the whole batch.
 *
 * The Iov is ior_entries_ fixed buffers of single_buffer_size_ bytes, and ring
 * slot s always uses buffer s. A reaped completion frees its slot and buffer
 * together, in any order, so a new segment never waits for Iov space.
 *
 * Writes copy the source data into the Iov before prepping; reads copy
 * completed segments out of the Iov as their completions are reaped. A file
 * stays registered until its last segment completes.
 *
 * If the ring itself fails (it cannot be created, or prep, submit, or wait
 * returns an error), it may still hold prepared or in-flight I/Os that
 * reference this call's files and Iov offsets. The ring and Iov for this
 * direction are then destroyed before the files are released, so no stale
 * completion or buffer write can leak into a later call; the next call
 * recreates them. Keys whose I/Os all completed keep their result; every
 * other key is failed with the ring error.
 *
 * Destroying the ring with I/Os in flight is safe: the 3FS fuse daemon holds
 * its own references to the ring and Iov memory while it processes them, and
 * I/Os it has not picked up yet fail instead of running. A write it already
 * picked up may still land after its key is reported failed.
 *
 * @tparam BufPtr void* to read into bufs, const void* to write from them;
 *   the write instantiation never writes through a caller buffer
 * @param conn Connection owning the rings
 * @param keys Keys to access
 * @param bufs Per-key user buffers
 * @param lens Per-key byte counts
 * @return Per-key error messages; an empty string means the key succeeded
 */
template <typename BufPtr>
std::vector<std::string> Hf3fsConnector::ring_io(
    WorkerHf3fsConn& conn, const std::vector<std::string>& keys,
    const std::vector<BufPtr>& bufs, const std::vector<size_t>& lens) {
  static_assert(
      std::is_same_v<BufPtr, void*> || std::is_same_v<BufPtr, const void*>,
      "ring_io reads into void* buffers or writes from const void* buffers");
  constexpr bool for_read = std::is_same_v<BufPtr, void*>;
  struct Segment {
    size_t key_idx;
    size_t file_off;
    size_t len;
  };

  std::vector<std::string> errors(keys.size());
  try {
    ensure_ring(conn, for_read);
  } catch (const std::exception& e) {
    errors.assign(keys.size(), e.what());
    return errors;
  }
  hf3fs_ior& ior = for_read ? conn.read_ior : conn.write_ior;
  hf3fs_iov& iov = for_read ? conn.read_iov : conn.write_iov;

  // In-flight bound: the ring capacity, one segment per slot.
  const size_t cap = static_cast<size_t>(ior_entries_);
  auto slot_buf = [&](size_t slot) {
    return iov.base + slot * single_buffer_size_;
  };

  // Slots map a completion's userdata back to its segment and its Iov buffer,
  // so segments can be retired and replaced out of order.
  std::vector<Segment> slots(cap);
  std::vector<size_t> free_slots;
  free_slots.reserve(cap);
  for (size_t s = cap; s > 0; --s) {
    free_slots.push_back(s - 1);
  }
  std::vector<hf3fs_cqe> cqes(cap);
  // Declared outside the try so files stay registered until the ring that may
  // still reference them has been destroyed.
  std::vector<std::optional<RegisteredFile>> files(keys.size());
  std::vector<size_t> key_pending(keys.size(), 0);
  std::vector<bool> completed(keys.size(), false);
  size_t inflight = 0;
  size_t unsubmitted = 0;

  auto submit = [&]() {
    int ret = hf3fs_submit_ios(&ior);
    if (ret < 0) {
      throw std::runtime_error("hf3fs_submit_ios failed: " +
                               std::to_string(-ret));
    }
    unsubmitted = 0;
  };

  // Reap at least min_results completions, copying reads out as they land.
  auto reap = [&](size_t min_results) {
    // A key with more segments than free slots waits on its own earlier
    // segments; waiting on segments that were prepped but never submitted
    // would block forever.
    if (unsubmitted > 0) {
      submit();
    }
    int got = 0;
    while (got == 0) {
      got = hf3fs_wait_for_ios(&ior, cqes.data(), static_cast<int>(cap),
                               static_cast<int>(min_results), nullptr);
      if (got < 0) {
        throw std::runtime_error("hf3fs_wait_for_ios failed: " +
                                 std::to_string(-got));
      }
    }
    for (int c = 0; c < got; ++c) {
      size_t slot = reinterpret_cast<uintptr_t>(cqes[c].userdata);
      const Segment& seg = slots[slot];
      std::string& err = errors[seg.key_idx];
      const int64_t result = cqes[c].result;
      if (result != static_cast<int64_t>(seg.len)) {
        if (err.empty()) {
          // A negative result is an errno from 3FS; a non-negative one is a
          // short transfer, e.g. a read past the end of a smaller file.
          err = std::string(for_read ? "read" : "write") + " (" +
                key_to_path(keys[seg.key_idx]) + ") ";
          if (result < 0) {
            int errnum = static_cast<int>(-result);
            err += "I/O error: " + std::string(strerror(errnum)) +
                   " (errno=" + std::to_string(errnum) + ")";
          } else {
            err += "short: requested " + std::to_string(seg.len) + " but got " +
                   std::to_string(result);
          }
        }
      } else if (err.empty()) {
        if constexpr (for_read) {
          memcpy(static_cast<char*>(bufs[seg.key_idx]) + seg.file_off,
                 slot_buf(slot), seg.len);
        }
      }
      free_slots.push_back(slot);
      --inflight;
      if (--key_pending[seg.key_idx] == 0) {
        completed[seg.key_idx] = true;
        files[seg.key_idx].reset();
      }
    }
  };

  try {
    for (size_t i = 0; i < keys.size(); ++i) {
      try {
        files[i].emplace(key_to_path(keys[i]), !for_read);
      } catch (const std::exception& e) {
        errors[i] = e.what();
        continue;
      }
      key_pending[i] =
          (lens[i] + single_buffer_size_ - 1) / single_buffer_size_;
      if (key_pending[i] == 0) {
        completed[i] = true;
        files[i].reset();
        continue;
      }
      for (size_t off = 0; off < lens[i];) {
        size_t sub_len = std::min(single_buffer_size_, lens[i] - off);
        // Retire a completion before prepping so at most cap segments, and
        // hence at most cap slot buffers, are in use.
        while (inflight == cap) {
          reap(1);
        }
        size_t slot = free_slots.back();
        free_slots.pop_back();
        uint8_t* ptr = slot_buf(slot);
        if constexpr (!for_read) {
          memcpy(ptr, static_cast<const char*>(bufs[i]) + off, sub_len);
        }
        // userdata carries the segment's slot so completions can be matched
        // back to it regardless of the order they are reaped in.
        auto prep = [&]() {
          return hf3fs_prep_io(&ior, &iov, for_read, ptr, files[i]->fd(), off,
                               sub_len, reinterpret_cast<const void*>(slot));
        };
        int ret = prep();
        // 3FS frees ring entries in order, so completions of later I/Os can be
        // reaped while an earlier I/O still holds the entries ahead of them,
        // and prep reports the ring full. Each reap retires a real completion,
        // so the earlier I/O eventually finishes and frees the entries.
        while (ret == -EAGAIN && inflight > 0) {
          reap(1);
          ret = prep();
        }
        if (ret < 0) {
          throw std::runtime_error("hf3fs_prep_io failed: " +
                                   std::to_string(-ret));
        }
        slots[slot] = {i, off, sub_len};
        ++inflight;
        ++unsubmitted;
        off += sub_len;
      }
      // Start this key's I/O now so it overlaps opening the next files,
      // instead of idling until the whole batch is prepped.
      submit();
    }
    while (inflight > 0) {
      reap(1);
    }
  } catch (const std::exception& e) {
    conn.destroy_ring(for_read);
    for (size_t k = 0; k < keys.size(); ++k) {
      if (!completed[k] && errors[k].empty()) {
        errors[k] = e.what();
      }
    }
  }
  return errors;
}

/**
 * Retrieve data for a single key through the read ring.
 *
 * @param conn Connection to use
 * @param key Key to retrieve
 * @param buf Output buffer
 * @param len Number of bytes to read
 * @param chunk_size Chunk size (unused for 3FS)
 * @throws std::runtime_error if the read fails
 */
void Hf3fsConnector::do_single_get(WorkerHf3fsConn& conn,
                                   const std::string& key, void* buf,
                                   size_t len, size_t chunk_size) {
  (void)chunk_size;  // Unused for 3FS
  auto errors = ring_io<void*>(conn, {key}, {buf}, {len});
  if (!errors[0].empty()) {
    throw std::runtime_error(errors[0]);
  }
}

/**
 * Store data for a single key through the write ring.
 *
 * @param conn Connection to use
 * @param key Key to store
 * @param buf Input buffer
 * @param len Number of bytes to write
 * @param chunk_size Chunk size (unused for 3FS)
 * @throws std::runtime_error if the write fails
 */
void Hf3fsConnector::do_single_set(WorkerHf3fsConn& conn,
                                   const std::string& key, const void* buf,
                                   size_t len, size_t chunk_size) {
  (void)chunk_size;  // Unused for 3FS
  auto errors = ring_io<const void*>(conn, {key}, {buf}, {len});
  if (!errors[0].empty()) {
    throw std::runtime_error(errors[0]);
  }
  if (buffer_enabled_) {
    buffer_add_(key);
  }
}

/**
 * Retrieve a tile of keys with batched ring I/O.
 *
 * Per-key failures (e.g. missing file, short read) only zero that key's
 * result, matching ConnectorBase's load error tolerance. A ring failure zeroes
 * only the keys whose I/Os had not completed by then.
 *
 * @param conn Connection to use
 * @param req Tile request
 */
void Hf3fsConnector::do_batch_get(WorkerHf3fsConn& conn, const Request& req) {
  auto errors = ring_io<void*>(conn, req.keys, req.buf_ptrs, req.buf_lens);
  for (size_t i = 0; i < errors.size(); ++i) {
    bool ok = errors[i].empty();
    req.batch->per_key_results[req.start_idx + i] = ok ? 1 : 0;
    if (!ok) {
      fprintf(stderr, "[LMCache GET] key %s failed: %s\n", req.keys[i].c_str(),
              errors[i].c_str());
    }
  }
}

/**
 * Store a tile of keys with batched ring I/O.
 *
 * A failed key does not stop the others; after a ring failure, keys not yet
 * written are failed. Successfully written keys are added to the key buffer
 * even when others in the tile fail.
 *
 * @param conn Connection to use
 * @param req Tile request
 * @throws std::runtime_error if any key fails, naming the first failure
 */
void Hf3fsConnector::do_batch_set(WorkerHf3fsConn& conn, const Request& req) {
  // Request carries mutable pointers for both directions; view them as const
  // so the write path cannot write into the caller's source buffers.
  const std::vector<const void*> srcs(req.buf_ptrs.begin(), req.buf_ptrs.end());
  auto errors = ring_io<const void*>(conn, req.keys, srcs, req.buf_lens);
  size_t num_failed = 0;
  const std::string* first_error = nullptr;
  for (size_t i = 0; i < errors.size(); ++i) {
    if (errors[i].empty()) {
      if (buffer_enabled_) {
        buffer_add_(req.keys[i]);
      }
    } else if (num_failed++ == 0) {
      first_error = &errors[i];
    }
  }
  if (num_failed > 0) {
    throw std::runtime_error(std::to_string(num_failed) + " of " +
                             std::to_string(errors.size()) +
                             " keys failed to store, first: " + *first_error);
  }
}

/**
 * Check if a key exists.
 *
 * @param conn Connection to use
 * @param key Key to check
 * @return true if key exists, false otherwise
 */
bool Hf3fsConnector::do_single_exists(WorkerHf3fsConn& conn,
                                      const std::string& key) {
  (void)conn;  // Unused
  if (!buffer_enabled_) {
    std::string file_path = key_to_path(key);
    return std::filesystem::exists(file_path);
  }
  return buffer_contains_(key);
}

/**
 * Delete a key.
 *
 * @param conn Connection to use
 * @param key Key to delete
 * @return true if deleted, false if not found
 */
bool Hf3fsConnector::do_single_delete(WorkerHf3fsConn& conn,
                                      const std::string& key) {
  (void)conn;  // Unused
  std::string file_path = key_to_path(key);
  try {
    bool removed = std::filesystem::remove(file_path);
    if (removed && buffer_enabled_) {
      buffer_remove_(key);
    }
    return removed;
  } catch (const std::filesystem::filesystem_error& e) {
    fprintf(stderr, "[LMCache HF3FS] Delete file %s failed: %s\n",
            file_path.c_str(), e.what());
    return false;
  }
}

/**
 * Scan all base_paths in parallel and populate the key buffer.
 *
 * Splits base_paths_ into num_workers_ contiguous slices (bounded by the
 * number of base_paths). Each thread scans one slice of directories into a
 * private local_set to avoid contention, then all local_sets are merged into
 * key_buffer_ in a single-threaded pass.
 *
 * Called from the constructor (single-threaded, before workers start).
 */
void Hf3fsConnector::scan_and_build_buffer_() {
  if (base_paths_.empty()) {
    return;
  }

  int num_workers =
      std::min(num_workers_, static_cast<int>(base_paths_.size()));
  if (num_workers <= 0) {
    num_workers = 1;
  }

  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(num_workers));
  std::vector<std::unordered_set<std::string>> local_sets(num_workers);

  for (int i = 0; i < num_workers; ++i) {
    // slice sizes differ by at most one.
    size_t start = static_cast<size_t>(i) * base_paths_.size() /
                   static_cast<size_t>(num_workers);
    size_t end = static_cast<size_t>(i + 1) * base_paths_.size() /
                 static_cast<size_t>(num_workers);

    threads.emplace_back(
        &Hf3fsConnector::buffer_scan_worker_, this,
        std::vector<std::string>(
            base_paths_.begin() + static_cast<std::ptrdiff_t>(start),
            base_paths_.begin() + static_cast<std::ptrdiff_t>(end)),
        std::ref(local_sets[i]));
  }

  for (auto& t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  // Single-threaded merge -- no contention
  for (int i = 0; i < num_workers; ++i) {
    for (const auto& key : local_sets[i]) {
      key_buffer_.Insert(key);
    }
  }
}

/**
 * Scan the given base_path directories and recover keys from .data filenames.
 *
 * This keeps directory traversal local to the connector; the per-file mapping
 * from filename to the original wire key is delegated to the shared
 * ``keys.h::filename_to_key`` (the inverse of ``keys.h::key_to_filename``),
 * so the on-disk encoding stays consistent with the fs_native connector.
 *
 * Each path in base_paths is iterated in turn; within one directory, every
 * regular ".data" file that decodes to a well-formed key is inserted into the
 * provided local set.
 *
 * Called from worker threads spawned in scan_and_build_buffer_().
 *
 * @param base_paths Directories to scan for .data files
 * @param local_set Set to populate with recovered keys
 */
void Hf3fsConnector::buffer_scan_worker_(
    const std::vector<std::string>& base_paths,
    std::unordered_set<std::string>& local_set) {
  std::error_code ec;
  for (const auto& base_path : base_paths) {
    for (const auto& entry :
         std::filesystem::directory_iterator(base_path, ec)) {
      if (!entry.is_regular_file(ec)) {
        continue;
      }

      const std::string filename = entry.path().filename().string();
      std::string recovered = filename_to_key(filename);
      if (!recovered.empty()) {
        local_set.insert(std::move(recovered));
      }
    }
  }
}

/**
 * Add a key to the buffer.
 *
 * Thread-safe via concurrent_flat_hash_set (sharded locks).
 * Called after a successful do_single_set().
 */
void Hf3fsConnector::buffer_add_(const std::string& key) {
  key_buffer_.Insert(key);
}

/**
 * Remove a key from the buffer.
 *
 * Thread-safe via concurrent_flat_hash_set (sharded locks).
 * Called after a successful do_single_delete().
 */
void Hf3fsConnector::buffer_remove_(const std::string& key) {
  key_buffer_.Erase(key);
}

/**
 * Check if a key exists in the buffer.
 *
 * Thread-safe read using sharded_flat_hash_set::Contains().
 * Each shard has its own lock, reducing contention.
 *
 * @param key Key to check
 * @return true if key is in the buffer, false otherwise
 */
bool Hf3fsConnector::buffer_contains_(const std::string& key) const {
  return key_buffer_.Contains(key);
}

/**
 * Optimized batch exists using the in-memory key buffer.
 *
 * When the buffer is enabled, all lookups are answered from the hash
 * set in microseconds. When disabled, falls back to the base class
 * default which iterates do_single_exists (filesystem).
 */
void Hf3fsConnector::do_batch_exists(WorkerHf3fsConn& conn,
                                     const Request& req) {
  (void)conn;
  if (!buffer_enabled_) {
    ConnectorBase::do_batch_exists(conn, req);
    return;
  }
  // Buffer path: check cache for all keys
  for (size_t i = 0; i < req.keys.size(); ++i) {
    req.batch->per_key_results[req.start_idx + i] =
        buffer_contains_(req.keys[i]) ? 1 : 0;
  }
}

/**
 * Shutdown all connections.
 *
 * Worker destructors handle cleanup automatically,
 * so this method is a no-op.
 */
void Hf3fsConnector::shutdown_connections() {
  // Worker destructors handle cleanup
}

}  // namespace connector
}  // namespace lmcache
