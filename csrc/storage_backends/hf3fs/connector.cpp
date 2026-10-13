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
#include <utility>

#include "../keys.h"

namespace lmcache {
namespace connector {

namespace {

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
 * - Parameters are within valid ranges
 */
Hf3fsConnector::Hf3fsConnector(std::string mount_point, std::string base_paths,
                               int num_workers, int ior_entries, int io_depth,
                               int numa_id, size_t iov_size, int time_out,
                               bool enable_key_buffer,
                               WorkerPoolConfig worker_pool_config)
    : ConnectorBase(num_workers, std::move(worker_pool_config)),
      mount_point_(std::move(mount_point)),
      ior_entries_(ior_entries),
      io_depth_(io_depth),
      numa_id_(numa_id),
      iov_size_(iov_size),
      time_out_(time_out),
      buffer_enabled_(enable_key_buffer) {
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
                             io_depth_,
                             time_out_,  // timeout (ms)
                             numa_id_,
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
                             io_depth_,
                             time_out_,  // timeout (ms)
                             numa_id_,
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
  int ret = hf3fs_iovcreate(&iov, mount_point_.c_str(), iov_size_,
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
  int ret = hf3fs_iovcreate(&iov, mount_point_.c_str(), iov_size_,
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
 * Each key's data is split into segments of at most iov_size_ bytes. Segments
 * are packed into the Iov at increasing offsets and prepared on the ring
 * until either the ring (ior_entries_) or the Iov (iov_size_) is full; then
 * the whole wave is submitted with a single hf3fs_submit_ios and reaped
 * together. This lets 3FS work on many I/Os concurrently instead of one
 * round trip per key.
 *
 * Writes copy the source data into the Iov before prepping; reads copy
 * completed segments out of the Iov after the wave is reaped. A file stays
 * registered until the wave holding its last segment completes, so the number
 * of open fds is bounded by the wave, not the batch.
 *
 * If the ring itself fails (it cannot be created, or prep, submit, or wait
 * returns an error), it may still hold prepared or in-flight I/Os that
 * reference this call's files and Iov offsets. The ring and Iov for this
 * direction are then destroyed before the files are released, so no stale
 * completion or buffer write can leak into a later call; the next call
 * recreates them. Keys whose I/Os all completed in earlier waves keep their
 * result; every other key is failed with the ring error.
 *
 * @param conn Connection owning the rings
 * @param for_read true to read into bufs, false to write from bufs
 * @param keys Keys to access
 * @param bufs Per-key user buffers
 * @param lens Per-key byte counts
 * @return Per-key error messages; an empty string means the key succeeded
 */
std::vector<std::string> Hf3fsConnector::ring_io(
    WorkerHf3fsConn& conn, bool for_read, const std::vector<std::string>& keys,
    const std::vector<void*>& bufs, const std::vector<size_t>& lens) {
  struct Segment {
    size_t key_idx;
    size_t file_off;
    size_t len;
    size_t iov_off;
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

  const size_t max_entries = static_cast<size_t>(ior_entries_);
  // Declared outside the try below so files stay registered until the ring
  // that may still reference them has been destroyed.
  std::vector<std::optional<RegisteredFile>> files(keys.size());
  std::vector<Segment> wave;
  wave.reserve(max_entries);
  std::vector<hf3fs_cqe> cqes(max_entries);
  size_t iov_used = 0;
  // Keys before this index have all their I/Os reaped.
  size_t first_open_key = 0;

  auto flush_wave = [&](size_t next_key_idx) {
    if (!wave.empty()) {
      int ret = hf3fs_submit_ios(&ior);
      if (ret < 0) {
        throw std::runtime_error("hf3fs_submit_ios failed: " +
                                 std::to_string(-ret));
      }
      size_t reaped = 0;
      while (reaped < wave.size()) {
        int remaining = static_cast<int>(wave.size() - reaped);
        ret = hf3fs_wait_for_ios(&ior, cqes.data(), remaining, remaining,
                                 nullptr);
        if (ret < 0) {
          throw std::runtime_error("hf3fs_wait_for_ios failed: " +
                                   std::to_string(-ret));
        }
        for (int c = 0; c < ret; ++c) {
          const Segment& seg =
              wave[reinterpret_cast<uintptr_t>(cqes[c].userdata)];
          std::string& err = errors[seg.key_idx];
          if (cqes[c].result != static_cast<int64_t>(seg.len)) {
            if (err.empty()) {
              err = std::string(for_read ? "read" : "write") + " (" +
                    key_to_path(keys[seg.key_idx]) + ") failed, requested " +
                    std::to_string(seg.len) + " but got " +
                    std::to_string(cqes[c].result);
            }
          } else if (for_read && err.empty()) {
            memcpy(static_cast<char*>(bufs[seg.key_idx]) + seg.file_off,
                   iov.base + seg.iov_off, seg.len);
          }
        }
        reaped += static_cast<size_t>(ret);
      }
      wave.clear();
      iov_used = 0;
    }
    for (; first_open_key < next_key_idx; ++first_open_key) {
      files[first_open_key].reset();
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
      for (size_t off = 0; off < lens[i];) {
        size_t sub_len = std::min(iov_size_, lens[i] - off);
        if (wave.size() == max_entries || iov_used + sub_len > iov_size_) {
          // Earlier segments of key i may be in this wave; keep it open.
          flush_wave(i);
        }
        uint8_t* ptr = iov.base + iov_used;
        if (!for_read) {
          memcpy(ptr, static_cast<const char*>(bufs[i]) + off, sub_len);
        }
        // userdata carries the segment's index in the wave.
        int ret =
            hf3fs_prep_io(&ior, &iov, for_read, ptr, files[i]->fd(), off,
                          sub_len, reinterpret_cast<const void*>(wave.size()));
        if (ret < 0) {
          throw std::runtime_error("hf3fs_prep_io failed: " +
                                   std::to_string(-ret));
        }
        wave.push_back({i, off, sub_len, iov_used});
        iov_used += sub_len;
        off += sub_len;
      }
    }
    flush_wave(keys.size());
  } catch (const std::exception& e) {
    conn.destroy_ring(for_read);
    for (size_t k = first_open_key; k < keys.size(); ++k) {
      if (errors[k].empty()) {
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
  auto errors = ring_io(conn, true, {key}, {buf}, {len});
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
  auto errors = ring_io(conn, false, {key}, {const_cast<void*>(buf)}, {len});
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
  auto errors = ring_io(conn, true, req.keys, req.buf_ptrs, req.buf_lens);
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
  auto errors = ring_io(conn, false, req.keys, req.buf_ptrs, req.buf_lens);
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
