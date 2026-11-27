# SPDX-License-Identifier: Apache-2.0
"""Tests for the stall_warn_seconds field shared by native adapter configs."""

# Standard
from pathlib import Path
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.aerospike_l2_adapter import (
    AerospikeL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdapterConfigBase
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    FSNativeL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.hf3fs_l2_adapter import (
    Hf3fsL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.mooncake_store_l2_adapter import (
    MooncakeStoreL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.native_plugin_l2_adapter import (
    NativePluginL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.nixl_native_l2_adapter import (
    NixlNativeL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.resp_l2_adapter import RESPL2AdapterConfig

_CONFIG_CLASSES: dict[str, type[L2AdapterConfigBase]] = {
    "fs_native": FSNativeL2AdapterConfig,
    "hf3fs": Hf3fsL2AdapterConfig,
    "resp": RESPL2AdapterConfig,
    "aerospike": AerospikeL2AdapterConfig,
    "mooncake_store": MooncakeStoreL2AdapterConfig,
    "nixl_native": NixlNativeL2AdapterConfig,
    "native_plugin": NativePluginL2AdapterConfig,
}
_ADAPTER_TYPES = list(_CONFIG_CLASSES)


def _minimal_dict(tmp_path: Path, adapter_type: str) -> dict[str, object]:
    """The smallest dict each adapter type's from_dict() accepts."""
    base_path = tmp_path / "kv"
    base_path.mkdir(exist_ok=True)
    dicts: dict[str, dict[str, object]] = {
        "fs_native": {"base_path": str(base_path)},
        "hf3fs": {"mount_point": str(tmp_path), "base_paths": str(base_path)},
        "resp": {"host": "localhost", "port": 6379},
        "aerospike": {"hosts": "127.0.0.1:3000"},
        "mooncake_store": {},
        "nixl_native": {"backend": "POSIX"},
        "native_plugin": {"module_path": "example.module", "class_name": "Client"},
    }
    return dicts[adapter_type]


def _from_dict(tmp_path: Path, adapter_type: str, **extra: object) -> Any:
    """Parse a minimal config of ``adapter_type`` with ``extra`` keys added."""
    d = {"type": adapter_type, **_minimal_dict(tmp_path, adapter_type), **extra}
    return _CONFIG_CLASSES[adapter_type].from_dict(d)


@pytest.mark.parametrize("adapter_type", _ADAPTER_TYPES)
def test_stall_warn_seconds_defaults_to_90(tmp_path: Path, adapter_type: str) -> None:
    config = _from_dict(tmp_path, adapter_type)
    assert config.stall_warn_seconds == 90.0


@pytest.mark.parametrize("adapter_type", _ADAPTER_TYPES)
@pytest.mark.parametrize("value", [0, 12.5, 300])
def test_stall_warn_seconds_accepts_non_negative_numbers(
    tmp_path: Path, adapter_type: str, value: float
) -> None:
    config = _from_dict(tmp_path, adapter_type, stall_warn_seconds=value)
    assert config.stall_warn_seconds == float(value)


@pytest.mark.parametrize("adapter_type", _ADAPTER_TYPES)
@pytest.mark.parametrize("value", [-1, True, "90", None])
def test_stall_warn_seconds_rejects_invalid_values(
    tmp_path: Path, adapter_type: str, value: object
) -> None:
    with pytest.raises(ValueError, match="stall_warn_seconds"):
        _from_dict(tmp_path, adapter_type, stall_warn_seconds=value)


@pytest.mark.parametrize("adapter_type", _ADAPTER_TYPES)
def test_stall_warn_seconds_is_described_in_help(adapter_type: str) -> None:
    assert "stall_warn_seconds" in _CONFIG_CLASSES[adapter_type].help()


def test_mooncake_does_not_forward_stall_warn_seconds(tmp_path: Path) -> None:
    """The field is LMCache's own, so it never reaches the Mooncake SDK."""
    config = _from_dict(tmp_path, "mooncake_store", stall_warn_seconds=30)
    assert isinstance(config, MooncakeStoreL2AdapterConfig)
    assert "stall_warn_seconds" not in config.setup_config
