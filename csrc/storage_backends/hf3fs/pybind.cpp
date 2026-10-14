// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText:Copyright (c) 2026 Samsung Electronics Co., Ltd.

#include <pybind11/pybind11.h>
#include "../connector_pybind_utils.h"
#include "connector.h"

namespace py = pybind11;

/**
 * Python bindings for Hf3fsConnector.
 *
 * Exposes the native C++ 3FS connector to Python as LMCacheHf3fsClient.
 * Uses the LMCACHE_BIND_CONNECTOR_METHODS macro to bind standard connector
 * methods.
 *
 * Example usage:
 * @code
 *   from lmcache.lmcache_hf3fs import LMCacheHf3fsClient
 *
 *   client = LMCacheHf3fsClient(
 *       mount_point="/mnt/3fs",
 *       base_paths="/mnt/3fs/path1,/mnt/3fs/path2",
 *       num_workers=8,
 *       ior_entries=16,
 *       numa_id=-1,
 *       single_buffer_size=8388608,
 *   )
 * @endcode
 */
PYBIND11_MODULE(lmcache_hf3fs, m) {
  py::class_<lmcache::connector::Hf3fsConnector>(m, "LMCacheHf3fsClient")
      .def(py::init([](std::string mount_point, std::string base_paths,
                       int num_workers, int ior_entries, int numa_id,
                       size_t single_buffer_size, bool enable_key_buffer,
                       py::object per_op_workers) {
             return new lmcache::connector::Hf3fsConnector(
                 std::move(mount_point), std::move(base_paths), num_workers,
                 ior_entries, numa_id, single_buffer_size, enable_key_buffer,
                 lmcache::connector::pybind_utils::parse_per_op_workers(
                     per_op_workers));
           }),
           py::arg("mount_point"), py::arg("base_paths"),
           py::arg("num_workers") = 8, py::arg("ior_entries") = 16,
           py::arg("numa_id") = -1,
           py::arg("single_buffer_size") = 8 * 1024 * 1024,
           py::arg("enable_key_buffer") = true,
           py::arg("per_op_workers") = py::none())
          LMCACHE_BIND_CONNECTOR_METHODS(lmcache::connector::Hf3fsConnector);
}
