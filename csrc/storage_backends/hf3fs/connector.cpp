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
#include <random>
#include <stdexcept>
#include <sstream>
#include <thread>

#include "../keys.h"

namespace lmcache {
namespace connector {

namespace {

/**
 * A unique temporary file next to a final path, opened for writing and
 * registered with 3FS, released on destruction.
 *
 * The caller renames temp_path() to the final path once every I/O to it
 * succeeded, or unlinks it, so readers never see a partial file.
 */
class RegisteredFile {
 public:
  explicit RegisteredFile(const std::string& final_path) {
    std::random_device random;
    do {
      // O_EXCL prevents collisions across workers, processes, and hosts.
      temp_path_ = final_path + "." + std::to_string(random());
      fd_ = ::open(temp_path_.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    } while (fd_ < 0 && errno == EEXIST);
    if (fd_ < 0) {
      throw std::runtime_error("open failed: " + std::string(strerror(errno)));
    }
    int reg_result = hf3fs_reg_fd(fd_, 0);
    if (reg_result > 0) {
      ::close(fd_);
      if (::unlink(temp_path_.c_str()) < 0) {
        fprintf(stderr, "[LMCache HF3FS] Remove temporary file %s failed: %s\n",
                temp_path_.c_str(), strerror(errno));
      }
      throw std::runtime_error(
          "hf3fs_reg_fd failed: " + std::to_string(reg_result) +
          " (errno=" + strerror(reg_result) + ")");
    }
  }
  ~RegisteredFile() {
    hf3fs_dereg_fd(fd_);
    ::close(fd_);
  }
  RegisteredFile(const RegisteredFile&) = delete;
  RegisteredFile& operator=(const RegisteredFile&) = delete;

  int fd() const { return fd_; }
  const std::string& temp_path() const { return temp_path_; }

 private:
  int fd_ = -1;
  std::string temp_path_;
};

/**
 * One write of a wave: len bytes at file_off of keys[key_idx].
 */
struct Segment {
  size_t key_idx;
  size_t file_off;
  size_t len;
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
 * Create a new per-thread connection.
 *
 * Initializes:
 * - Read Ior and Iov
 * - Write Ior and Iov
 * - File descriptors (initialized to -1)
 *
 * @return WorkerHf3fsConn Initialized connection structure
 */
WorkerHf3fsConn Hf3fsConnector::create_connection() {
  WorkerHf3fsConn conn;

  // Initialize read Ior
  init_read_ior(conn.read_ior);
  conn.read_ior_initialized = true;

  // Initialize write Ior
  init_write_ior(conn.write_ior);
  conn.write_ior_initialized = true;

  // Initialize read Iov
  init_read_iov(conn.read_iov);
  conn.read_iov_initialized = true;

  // Initialize write Iov
  init_write_iov(conn.write_iov);
  conn.write_iov_initialized = true;

  // File descriptors start uninitialized
  conn.fd = -1;

  return conn;
}

/**
 * Open a file for read or write.
 *
 * Steps:
 * 1. Open file with appropriate flags (O_RDONLY or O_WRONLY|O_CREAT)
 * 2. Register FD with 3FS via hf3fs_reg_fd
 * 3. Store file path in connection
 *
 * @param conn Connection to use
 * @param file_path Path to file
 * @param for_write true for write, false for read
 * @throws std::runtime_error if open or hf3fs_reg_fd fails
 */
void Hf3fsConnector::open_file(WorkerHf3fsConn& conn,
                               const std::string& file_path, bool for_write) {
  std::string opened_path = file_path;
  if (for_write) {
    std::random_device random;
    do {
      // O_EXCL prevents collisions across workers, processes, and hosts.
      opened_path = file_path + "." + std::to_string(random());
      conn.fd = ::open(opened_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    } while (conn.fd < 0 && errno == EEXIST);
  } else {
    conn.fd = ::open(opened_path.c_str(), O_RDONLY);
  }
  if (conn.fd < 0) {
    throw std::runtime_error("open failed: " + std::string(strerror(errno)));
  }

  int reg_result = hf3fs_reg_fd(conn.fd, 0);
  if (reg_result > 0) {
    ::close(conn.fd);
    conn.fd = -1;
    if (for_write && ::unlink(opened_path.c_str()) < 0) {
      fprintf(stderr, "[LMCache HF3FS] Remove temporary file %s failed: %s\n",
              opened_path.c_str(), strerror(errno));
    }
    throw std::runtime_error(
        "hf3fs_reg_fd failed: " + std::to_string(reg_result) +
        " (errno=" + strerror(reg_result) + ")");
  }
  conn.registered = true;
  conn.file_path = std::move(opened_path);
}

/**
 * Close a file and deregister from 3FS.
 *
 * Steps:
 * 1. Deregister FD from 3FS via hf3fs_dereg_fd
 * 2. Close Linux FD
 * 3. Clear file path
 *
 * @param conn Connection to use
 */
void Hf3fsConnector::close_file(WorkerHf3fsConn& conn) {
  // Step 1: Deregister FD from 3FS first (only if registration succeeded)
  if (conn.registered) {
    hf3fs_dereg_fd(conn.fd);
    conn.registered = false;
  }
  // Step 2: Close Linux FD
  if (conn.fd >= 0) {
    ::close(conn.fd);
    conn.fd = -1;
  }
  conn.file_path.clear();
}

/**
 * Read data from a file using 3FS I/O.
 *
 * Steps (per chunk):
 * 1. Prepare I/O request via hf3fs_prep_io
 * 2. Submit I/O via hf3fs_submit_ios
 * 3. Wait for completion via hf3fs_wait_for_ios
 * 4. Copy data from Iov buffer to output buffer
 *
 * If len > iov_size_, data is read in multiple chunks.
 *
 * @param conn Connection to use
 * @param ior Ior to use for I/O
 * @param iov Iov buffer for data
 * @param buf Output buffer
 * @param len Number of bytes to read
 * @throws std::runtime_error if any 3FS operation fails
 */
void Hf3fsConnector::read_file(WorkerHf3fsConn& conn, hf3fs_ior& ior,
                               hf3fs_iov& iov, void* buf, size_t len) {
  size_t file_offset = 0;
  size_t total_read = 0;

  while (file_offset < len) {
    size_t sub_len = std::min(iov_size_, len - file_offset);

    int ret = hf3fs_prep_io(&ior, &iov, true, iov.base, conn.fd, file_offset,
                            sub_len, nullptr);
    if (ret < 0) {
      throw std::runtime_error("hf3fs_prep_io failed: " + std::to_string(-ret));
    }

    ret = hf3fs_submit_ios(&ior);
    if (ret < 0) {
      throw std::runtime_error("hf3fs_submit_ios failed: " +
                               std::to_string(-ret));
    }

    hf3fs_cqe cqe;
    ret = hf3fs_wait_for_ios(&ior, &cqe, 1, 1, nullptr);
    if (ret < 0) {
      throw std::runtime_error("hf3fs_wait_for_ios failed: " +
                               std::to_string(-ret));
    }
    if (cqe.result < 0) {
      throw std::runtime_error("I/O failed: " + std::to_string(-cqe.result));
    }
    if (cqe.result != sub_len) {
      throw std::runtime_error(
          "read_file (" + conn.file_path + ") failed, requested " +
          std::to_string(sub_len) + " but got " + std::to_string(cqe.result));
    }
    memcpy(static_cast<char*>(buf) + total_read, iov.base, sub_len);
    total_read += sub_len;
    file_offset += sub_len;
  }
}

/**
 * Write data to a file using 3FS I/O.
 *
 * Steps (per chunk):
 * 1. Copy data to Iov buffer
 * 2. Prepare I/O request via hf3fs_prep_io
 * 3. Submit I/O via hf3fs_submit_ios
 * 4. Wait for completion via hf3fs_wait_for_ios
 *
 * If len > iov_size_, data is written in multiple chunks.
 *
 * @param conn Connection to use
 * @param ior Ior to use for I/O
 * @param iov Iov buffer for data
 * @param buf Input buffer
 * @param len Number of bytes to write
 * @throws std::runtime_error if any 3FS operation fails
 */
void Hf3fsConnector::write_file(WorkerHf3fsConn& conn, hf3fs_ior& ior,
                                hf3fs_iov& iov, const void* buf, size_t len) {
  size_t file_offset = 0;
  size_t total_written = 0;

  while (file_offset < len) {
    size_t sub_len = std::min(iov_size_, len - file_offset);

    // Copy data to Iov buffer
    memcpy(iov.base, static_cast<const char*>(buf) + file_offset, sub_len);

    int ret = hf3fs_prep_io(&ior, &iov, false, iov.base, conn.fd, file_offset,
                            sub_len, nullptr);
    if (ret < 0) {
      throw std::runtime_error("hf3fs_prep_io failed: " + std::to_string(-ret));
    }

    ret = hf3fs_submit_ios(&ior);
    if (ret < 0) {
      throw std::runtime_error("hf3fs_submit_ios failed: " +
                               std::to_string(-ret));
    }

    hf3fs_cqe cqe;
    ret = hf3fs_wait_for_ios(&ior, &cqe, 1, 1, nullptr);
    if (ret < 0) {
      throw std::runtime_error("hf3fs_wait_for_ios failed: " +
                               std::to_string(-ret));
    }
    if (cqe.result < 0) {
      throw std::runtime_error("I/O failed: " + std::to_string(-cqe.result));
    }

    if (cqe.result != sub_len) {
      throw std::runtime_error(
          "write_file (" + conn.file_path + ") failed, requested " +
          std::to_string(sub_len) + " but got " + std::to_string(cqe.result));
    }
    total_written += sub_len;
    file_offset += sub_len;
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
 * Retrieve data for a key.
 *
 * Steps:
 * 1. Convert key to file path
 * 2. Open file for reading
 * 3. Read data using 3FS I/O
 * 4. Close file
 *
 * @param conn Connection to use
 * @param key Key to retrieve
 * @param buf Output buffer
 * @param len Number of bytes to read
 * @param chunk_size Chunk size (unused for 3FS)
 */
void Hf3fsConnector::do_single_get(WorkerHf3fsConn& conn,
                                   const std::string& key, void* buf,
                                   size_t len, size_t chunk_size) {
  (void)chunk_size;  // Unused for 3FS
  std::string file_path = key_to_path(key);
  open_file(conn, file_path, false);
  try {
    read_file(conn, conn.read_ior, conn.read_iov, buf, len);
  } catch (const std::exception& e) {
    close_file(conn);
    throw;
  }
  close_file(conn);
}

/**
 * Store data for a key.
 *
 * Steps:
 * 1. Convert key to file path
 * 2. Open a unique temporary file for writing
 * 3. Write data using 3FS I/O
 * 4. Close file and atomically rename to final path
 * 5. Update key buffer
 *
 * @param conn Connection to use
 * @param key Key to store
 * @param buf Input buffer
 * @param len Number of bytes to write
 * @param chunk_size Chunk size (unused for 3FS)
 */
void Hf3fsConnector::do_single_set(WorkerHf3fsConn& conn,
                                   const std::string& key, const void* buf,
                                   size_t len, size_t chunk_size) {
  (void)chunk_size;  // Unused for 3FS

  std::string file_path = key_to_path(key);
  open_file(conn, file_path, true);
  const std::string temporary_path = conn.file_path;
  try {
    write_file(conn, conn.write_ior, conn.write_iov, buf, len);
    close_file(conn);
    std::filesystem::rename(temporary_path, file_path);
  } catch (const std::exception& e) {
    close_file(conn);
    if (::unlink(temporary_path.c_str()) < 0) {
      fprintf(stderr, "[LMCache HF3FS] Remove temporary file %s failed: %s\n",
              temporary_path.c_str(), strerror(errno));
    }
    throw;
  }
  if (buffer_enabled_) {
    buffer_add_(key);
  }
}

/**
 * Store a tile of keys in write waves.
 *
 * A wave packs consecutive keys' data back to back into the write Iov, up to
 * ior_entries_ segments or iov_size_ bytes, preps them all, submits them
 * together, and waits for all of them before the next wave starts. A key that
 * does not fit in the rest of a wave continues in the next one, and its file
 * stays registered until the wave holding its last segment completes. Each
 * key is written to a temporary file that is renamed to its final path once
 * all of its I/Os succeed, so readers never see a partial file. This
 * replaces one open/submit/wait/close round trip per key.
 *
 * A key that fails to open or whose I/O fails does not stop the others. If the
 * ring itself fails (prep, submit, or wait returns an error), keys whose last
 * wave completed keep their result and every other key is failed with the
 * ring error. Successfully written keys are added to the key buffer even when
 * others in the tile fail.
 *
 * @param conn Connection to use
 * @param req Tile request
 * @throws std::runtime_error if any key fails, naming the first failure
 */
void Hf3fsConnector::do_batch_set(WorkerHf3fsConn& conn, const Request& req) {
  const size_t n = req.keys.size();
  std::vector<std::string> errors(n);
  const size_t cap = static_cast<size_t>(ior_entries_);
  // A segment's index in the wave is its userdata.
  std::vector<Segment> wave;
  wave.reserve(cap);
  std::vector<hf3fs_cqe> cqes(cap);
  // Declared outside the try so files stay registered until the ring error
  // path has finished.
  std::vector<std::optional<RegisteredFile>> files(n);
  size_t next_key = 0;
  size_t next_off = 0;
  // Keys before done_keys have had every segment complete.
  size_t done_keys = 0;

  // Close a key's file, then publish it by renaming the temporary file to the
  // final path if every I/O succeeded, or remove it otherwise.
  auto finish_key = [&](size_t k) {
    if (!files[k]) {
      return;
    }
    const std::string temp_path = files[k]->temp_path();
    files[k].reset();
    if (errors[k].empty()) {
      try {
        std::filesystem::rename(temp_path, key_to_path(req.keys[k]));
        return;
      } catch (const std::exception& e) {
        errors[k] = e.what();
      }
    }
    if (::unlink(temp_path.c_str()) < 0) {
      fprintf(stderr, "[LMCache HF3FS] Remove temporary file %s failed: %s\n",
              temp_path.c_str(), strerror(errno));
    }
  };

  try {
    while (next_key < n) {
      wave.clear();
      size_t used = 0;
      while (wave.size() < cap && used < iov_size_ && next_key < n) {
        const size_t i = next_key;
        const size_t len = req.buf_lens[i];
        if (next_off == 0) {
          try {
            files[i].emplace(key_to_path(req.keys[i]));
          } catch (const std::exception& e) {
            errors[i] = e.what();
            ++next_key;
            continue;
          }
          if (len == 0) {
            ++next_key;
            continue;
          }
        }
        const size_t sub_len = std::min(len - next_off, iov_size_ - used);
        uint8_t* ptr = conn.write_iov.base + used;
        memcpy(ptr, static_cast<const char*>(req.buf_ptrs[i]) + next_off,
               sub_len);
        int ret = hf3fs_prep_io(&conn.write_ior, &conn.write_iov, false, ptr,
                                files[i]->fd(), next_off, sub_len,
                                reinterpret_cast<const void*>(wave.size()));
        if (ret < 0) {
          throw std::runtime_error("hf3fs_prep_io failed: " +
                                   std::to_string(-ret));
        }
        wave.push_back({i, next_off, sub_len});
        used += sub_len;
        next_off += sub_len;
        if (next_off == len) {
          ++next_key;
          next_off = 0;
        }
      }

      if (!wave.empty()) {
        int ret = hf3fs_submit_ios(&conn.write_ior);
        if (ret < 0) {
          throw std::runtime_error("hf3fs_submit_ios failed: " +
                                   std::to_string(-ret));
        }
        // Completions arrive in any order; userdata maps each to its segment.
        size_t reaped = 0;
        while (reaped < wave.size()) {
          int got = hf3fs_wait_for_ios(
              &conn.write_ior, cqes.data(), static_cast<int>(cap),
              static_cast<int>(wave.size() - reaped), nullptr);
          if (got < 0) {
            throw std::runtime_error("hf3fs_wait_for_ios failed: " +
                                     std::to_string(-got));
          }
          for (int c = 0; c < got; ++c) {
            const Segment& seg =
                wave[reinterpret_cast<uintptr_t>(cqes[c].userdata)];
            std::string& err = errors[seg.key_idx];
            const int64_t result = cqes[c].result;
            if (result != static_cast<int64_t>(seg.len) && err.empty()) {
              err = "write (" + key_to_path(req.keys[seg.key_idx]) + ") ";
              if (result < 0) {
                err += "I/O failed: " + std::to_string(-result);
              } else {
                err += "short: requested " + std::to_string(seg.len) +
                       " but got " + std::to_string(result);
              }
            }
          }
          reaped += static_cast<size_t>(got);
        }
      }

      // Every key before the cursor has now had all of its segments complete.
      for (; done_keys < next_key; ++done_keys) {
        finish_key(done_keys);
      }
    }
  } catch (const std::exception& e) {
    for (size_t k = done_keys; k < n; ++k) {
      if (errors[k].empty()) {
        errors[k] = e.what();
      }
      finish_key(k);
    }
  }

  size_t num_failed = 0;
  const std::string* first_error = nullptr;
  for (size_t i = 0; i < n; ++i) {
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
                             std::to_string(n) +
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
