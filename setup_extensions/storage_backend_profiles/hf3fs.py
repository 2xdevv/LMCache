# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText:Copyright (c) 2026 Samsung Electronics Co., Ltd.

"""HF3FS native L2 storage backend profile.

Builds the ``lmcache.lmcache_hf3fs`` extension when the HF3FS SDK
is available.  Enabled via ``BUILD_WITH_HF3FS=1`` or auto-detected
through ``HF3FS_INCLUDE_DIR`` / ``HF3FS_LIB_DIR`` env vars.

The HF3FS backend depends on the system Abseil library (libabsl-dev)
for hash set implementations (``sharded_flat_hash_set``, ``flat_hash_set``).
"""

# Standard
from pathlib import Path
from typing import TYPE_CHECKING
import os

# First Party
from setup_extensions.storage_backend_profiles import StorageBackendProfile

ROOT_DIR = Path(__file__).parent.parent.parent

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension


class Hf3fsStorageBackend(StorageBackendProfile):
    """HF3FS native 3FS L2 storage backend."""

    name = "hf3fs"
    env_var = "BUILD_WITH_HF3FS"

    #: Compiler default include dirs searched for Abseil (libabsl-dev
    #: installs into ``/usr/include``) after ``HF3FS_INCLUDE_DIR``.
    system_include_dirs: tuple[str, ...] = ("/usr/local/include", "/usr/include")

    def detect(self) -> bool:
        """Detect HF3FS SDK via ``BUILD_HF3FS`` / ``HF3FS_INCLUDE_DIR``.

        ``BUILD_HF3FS=0`` disables the build. ``BUILD_HF3FS=1``, or a
        non-empty ``HF3FS_INCLUDE_DIR`` when ``BUILD_HF3FS`` is unset,
        enables it if the Abseil header and shared library are present.

        Returns:
            ``True`` if the HF3FS extension should be built.
        """
        hf3fs_env = os.environ.get("BUILD_HF3FS")
        if hf3fs_env == "0":
            return False

        if hf3fs_env is None and not os.environ.get("HF3FS_INCLUDE_DIR"):
            return False

        if self._check_absl_available():
            return True

        print("[HF3FS] WARNING: HF3FS build requested but Abseil not available !!!")
        print(
            "[HF3FS] WARNING: unset BUILD_HF3FS or install libabsl-dev:"
            " sudo apt-get install -y libabsl-dev "
        )
        return False

    def _check_absl_available(self) -> bool:
        """Check if Abseil headers and shared libraries are available.

        Returns:
            ``True`` if ``absl_hash`` shared library is findable via
            :func:`ctypes.util.find_library` AND the
            ``flat_hash_set.h`` header is present under one of the
            ``;``-separated ``HF3FS_INCLUDE_DIR`` entries or
            :attr:`system_include_dirs`.
        """
        # Standard
        import ctypes.util

        # 1. Check Abseil shared library via ctypes
        if not ctypes.util.find_library("absl_hash"):
            print("[HF3FS] WARNING: Abseil library (libabsl-dev) not found !!!")
            return False

        # 2. Check header file. HF3FS_INCLUDE_DIR usually points at the 3FS
        # usrbio headers, so the system dirs must be searched as well.
        hf3fs_include = os.environ.get("HF3FS_INCLUDE_DIR", "")
        include_dirs = [d for d in hf3fs_include.split(";") if d]
        include_dirs.extend(self.system_include_dirs)
        for include_dir in include_dirs:
            header_path = os.path.join(
                include_dir, "absl", "container", "flat_hash_set.h"
            )
            if os.path.isfile(header_path):
                return True

        print(
            f"[HF3FS] WARNING: Abseil header not found in {', '.join(include_dirs)} !!!"
        )
        return False

    def build(self, extra_cxx_flags: list[str]) -> list["Extension"]:
        """Build the ``lmcache.lmcache_hf3fs`` CppExtension.

        The extension uses system Abseil (libabsl-dev) for header-only
        inline symbols.

        Args:
            extra_cxx_flags: Additional C++ compiler flags from the
                selected GPU backend profile.

        Returns:
            List with the ``lmcache.lmcache_hf3fs`` Extension.
        """
        # Third Party
        from torch.utils import cpp_extension

        hf3fs_include = os.environ.get("HF3FS_INCLUDE_DIR", "")
        hf3fs_lib = os.environ.get("HF3FS_LIB_DIR", "")
        include_dirs = [
            str(ROOT_DIR / "csrc/storage_backends"),
            str(ROOT_DIR / "csrc/storage_backends/hf3fs"),
        ]
        if hf3fs_include:
            include_dirs.extend(hf3fs_include.split(";"))
        library_dirs: list[str] = []
        if hf3fs_lib:
            library_dirs.extend(hf3fs_lib.split(";"))
        return [
            cpp_extension.CppExtension(
                "lmcache.lmcache_hf3fs",
                sources=[
                    "csrc/storage_backends/hf3fs/pybind.cpp",
                    "csrc/storage_backends/hf3fs/connector.cpp",
                    "csrc/storage_backends/hf3fs/hf3fs_absl_init.cpp",
                ],
                include_dirs=include_dirs,
                library_dirs=library_dirs + ["/usr/lib/x86_64-linux-gnu"],
                libraries=[
                    "hf3fs_api_shared",
                    "absl_hash",
                    "absl_city",
                    "absl_int128",
                    "absl_raw_hash_set",
                ],
                runtime_library_dirs=library_dirs + ["/usr/lib/x86_64-linux-gnu"],
                extra_compile_args={
                    "cxx": extra_cxx_flags + ["-O3", "-std=c++17"],
                },
            ),
        ]
