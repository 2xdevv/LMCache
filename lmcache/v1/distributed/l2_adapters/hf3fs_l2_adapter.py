# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText:Copyright (c) 2026 Samsung Electronics Co., Ltd.

"""
3FS native L2 adapter config and factory.

Backed by the native C++ 3FS connector wrapped with
``NativeConnectorL2Adapter``.
"""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
    )

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)

logger = init_logger(__name__)

# Keys consumed only by LMCache (never forwarded to 3FS).
_LMCACHE_ONLY_KEYS = {
    "type",
    "num_workers",
    "per_op_workers",
}


class Hf3fsL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for an L2 adapter backed by the native C++ 3FS connector.

    Fields:
    - mount_point: 3FS mount point directory (required).
    - base_paths: Comma-separated subdirectories under mount_point (required).
    - num_workers: C++ worker threads for I/O (default 4).
    - ior_entries: Positive maximum concurrent requests per Ior (default 8).
    - io_depth: Batch control parameter (default 0, range [-128, 128]).
    - numa_id: NUMA node ID (default -1 = current node).
    - single_buffer_size: Positive bytes per ring entry (default 16 MiB).
      Each worker allocates this size times ior_entries for each direction.
    - enable_key_buffer (bool): Enable in-memory key buffer for accelerated
        exists lookups. When enabled, do_batch_exists() uses a pre-scanned
        hash set instead of filesystem calls. Default: True.
    - per_op_workers: Optional dict mapping lane keys
                (``"lookup"``, ``"retrieve"``, ``"store"``,
                ``"delete"``) to dedicated worker counts.  Ops
                not mentioned use the shared ``num_workers`` pool.
    """

    def __init__(
        self,
        mount_point: str,
        base_paths: str,
        num_workers: int = 4,
        ior_entries: int = 8,
        io_depth: int = 0,
        numa_id: int = -1,
        single_buffer_size: int = 16777216,
        enable_key_buffer: bool = True,
        per_op_workers: dict[str, int] | None = None,
    ) -> None:
        """Initialize the native HF3FS configuration.

        Args:
            mount_point: 3FS mount point directory.
            base_paths: Comma-separated storage directories under mount_point.
            num_workers: Number of shared C++ I/O workers.
            ior_entries: Positive number of requests supported by each ring.
            io_depth: 3FS worker batching control.
            numa_id: NUMA node, or -1 for the current node.
            single_buffer_size: Positive bytes budgeted per ring entry. Each
                read/write Iov has single_buffer_size * ior_entries bytes.
            enable_key_buffer: Whether to cache keys for existence lookups.
            per_op_workers: Dedicated worker counts by operation lane.

        Raises:
            ValueError: If a buffer size, ring capacity, or worker count is
                invalid.
        """
        if type(ior_entries) is not int or ior_entries <= 0:
            raise ValueError("ior_entries must be a positive integer")
        if type(single_buffer_size) is not int or single_buffer_size <= 0:
            raise ValueError("single_buffer_size must be a positive integer")
        self.mount_point = mount_point
        self.base_paths = base_paths
        self.num_workers = L2AdapterConfigBase._validate_num_workers(num_workers)
        self.ior_entries = ior_entries
        self.io_depth = io_depth
        self.numa_id = numa_id
        self.single_buffer_size = single_buffer_size
        self.enable_key_buffer = enable_key_buffer
        self.per_op_workers = L2AdapterConfigBase._validate_per_op_workers(
            per_op_workers
        )

    @property
    def iov_size(self) -> int:
        """Return the total Iov size in bytes per worker per direction."""
        return self.single_buffer_size * self.ior_entries

    @classmethod
    def from_dict(cls, d: dict) -> "Hf3fsL2AdapterConfig":
        """Parse native HF3FS fields and validate the storage directories.

        Args:
            d: Configuration fields accepted by the constructor. The obsolete
                iov_size field is rejected; use single_buffer_size instead.

        Returns:
            A configuration with validated paths, worker counts, and sizes.

        Raises:
            ValueError: If a field or storage directory is invalid.
        """
        if "iov_size" in d:
            raise ValueError(
                "iov_size is no longer configurable; use single_buffer_size "
                "(total Iov size = single_buffer_size * ior_entries)"
            )
        mount_point = d.get("mount_point")
        if not isinstance(mount_point, str) or not mount_point:
            raise ValueError("mount_point must be a non-empty string")

        base_paths = d.get("base_paths")
        if not isinstance(base_paths, str) or not base_paths:
            raise ValueError("base_paths must be a non-empty string")

        # Validate paths exist and are directories
        try:
            mp_path = Path(mount_point).resolve()
            if not mp_path.exists():
                raise ValueError(f"mount_point does not exist: {mount_point}")
            if not mp_path.is_dir():
                raise ValueError(f"mount_point is not a directory: {mount_point}")

            # Validate each base_path is a subdirectory of mount_point
            paths = [p.strip() for p in base_paths.split(",")]
            for path_str in paths:
                if not path_str:
                    raise ValueError("Empty path in base_paths")
                path = Path(path_str).resolve()
                try:
                    path.relative_to(mp_path)
                except ValueError as err:
                    raise ValueError(
                        f"base_path '{path_str}' is not a subdirectory of "
                        f"mount_point '{mount_point}'"
                    ) from err
                if not path.exists():
                    raise ValueError(f"base_path does not exist: {path_str}")
                if not path.is_dir():
                    raise ValueError(f"base_path is not a directory: {path_str}")
        except Exception as e:
            raise ValueError(f"Invalid path configuration: {e}") from e

        num_workers = d.get("num_workers", 4)
        if not isinstance(num_workers, int) or num_workers <= 0:
            raise ValueError("num_workers must be a positive integer")

        ior_entries = d.get("ior_entries", 8)

        io_depth = d.get("io_depth", 0)
        if not isinstance(io_depth, int) or not (-128 <= io_depth <= 128):
            raise ValueError("io_depth must be in range [-128, 128]")

        numa_id = d.get("numa_id", -1)
        if not isinstance(numa_id, int):
            raise ValueError("numa_id must be an integer")

        single_buffer_size = d.get("single_buffer_size", 16777216)

        enable_key_buffer = d.get("enable_key_buffer", True)
        if not isinstance(enable_key_buffer, bool):
            raise ValueError("enable_key_buffer must be a boolean")

        per_op_workers = L2AdapterConfigBase._parse_per_op_workers_from_dict(d)

        return cls(
            mount_point=mount_point,
            base_paths=base_paths,
            num_workers=num_workers,
            ior_entries=ior_entries,
            io_depth=io_depth,
            numa_id=numa_id,
            single_buffer_size=single_buffer_size,
            enable_key_buffer=enable_key_buffer,
            per_op_workers=per_op_workers,
        )

    @classmethod
    def help(cls) -> str:
        """Return descriptions and defaults for native HF3FS config fields."""
        return (
            "Hf3fs L2 adapter config fields:\n"
            "- mount_point (str): 3FS mount point directory (required)\n"
            "- base_paths (str): Comma-separated subdirectories under "
            "mount_point (required)\n"
            "- num_workers (int): C++ worker threads (default 4, >0)\n"
            "- ior_entries (int): Max concurrent requests per Ior "
            "(default 8, positive integer)\n"
            "- io_depth (int): Batch control parameter "
            "(default 0, range [-128, 128])\n"
            "- numa_id (int): NUMA node ID (default -1 = current node)\n"
            "- single_buffer_size (int): Bytes per ring entry "
            "(default 16777216 = 16 MiB, positive integer). Each read/write "
            "Iov has single_buffer_size * ior_entries bytes per worker.\n"
            "- enable_key_buffer (bool): Enable in-memory key buffer for "
            "accelerated exists lookups (default True)\n"
            "- per_op_workers (dict[str, int] | None): Optional dict mapping "
            "lane keys to dedicated worker counts. Valid keys: lookup, "
            "retrieve, store, delete. Ops not mentioned use the shared "
            "num_workers pool."
        )


def _create_hf3fs_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: Optional[L1MemoryDesc] = None,
) -> L2AdapterInterface:
    """Create Hf3FS native L2 adapter."""
    try:
        # First Party
        from lmcache.lmcache_hf3fs import (
            LMCacheHf3fsClient,
        )
    except ImportError as e:
        raise RuntimeError(
            "Hf3fs native L2 adapter requires the C++ extension. "
            "Build with: pip install -e ."
        ) from e

    # Lazy import to avoid circular dependency
    # First Party
    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
        NativeConnectorL2Adapter,
    )

    assert isinstance(config, Hf3fsL2AdapterConfig)
    native_client = LMCacheHf3fsClient(
        config.mount_point,
        config.base_paths,
        config.num_workers,
        config.ior_entries,
        config.io_depth,
        config.numa_id,
        config.single_buffer_size,
        time_out=200,
        enable_key_buffer=config.enable_key_buffer,
        per_op_workers=config.per_op_workers,
    )
    logger.info(
        "Created Hf3fs native L2 adapter: %s (workers=%d, paths=%d)",
        config.base_paths,
        config.num_workers,
        len(config.base_paths.split(",")),
    )
    return NativeConnectorL2Adapter(
        native_client,
        type_name="Hf3fsL2Adapter",
        extra_status={
            "mount_point": config.mount_point,
            "base_paths": config.base_paths,
            "num_workers": config.num_workers,
            "ior_entries": config.ior_entries,
            "single_buffer_size": config.single_buffer_size,
            "iov_size": config.iov_size,
            "io_depth": config.io_depth,
            "per_op_workers": config.per_op_workers,
        },
    )


register_l2_adapter_type("hf3fs", Hf3fsL2AdapterConfig)
register_l2_adapter_factory("hf3fs", _create_hf3fs_l2_adapter)
