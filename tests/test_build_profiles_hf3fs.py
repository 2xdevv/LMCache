# SPDX-License-Identifier: Apache-2.0
"""Tests for the HF3FS storage backend profile's detection contract."""

# Standard
from pathlib import Path
import ctypes.util

# Third Party
import pytest

# First Party
from setup_extensions.storage_backend_profiles.hf3fs import Hf3fsStorageBackend


def _make_include_dir(root: Path, header: str) -> Path:
    """Create an include dir under ``root`` containing ``header``."""
    header_path = root / header
    header_path.parent.mkdir(parents=True)
    header_path.write_text("")
    return root


@pytest.fixture
def system_absl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Make Abseil available only in a fake system include dir."""
    for env_var in ("BUILD_HF3FS", "BUILD_WITH_HF3FS", "HF3FS_INCLUDE_DIR"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: f"lib{name}.so")
    system_dir = _make_include_dir(
        tmp_path / "system", "absl/container/flat_hash_set.h"
    )
    monkeypatch.setattr(
        Hf3fsStorageBackend,
        "system_include_dirs",
        (str(system_dir),),
        raising=False,
    )


@pytest.fixture
def usrbio_include(tmp_path: Path) -> Path:
    """An include dir holding only the 3FS usrbio header."""
    return _make_include_dir(tmp_path / "usrbio", "hf3fs_usrbio.h")


@pytest.mark.usefixtures("system_absl")
def test_build_hf3fs_with_usrbio_include_dir_finds_system_absl(
    monkeypatch: pytest.MonkeyPatch, usrbio_include: Path
) -> None:
    """The documented recipe detects Abseil outside ``HF3FS_INCLUDE_DIR``."""
    monkeypatch.setenv("BUILD_HF3FS", "1")
    monkeypatch.setenv("HF3FS_INCLUDE_DIR", str(usrbio_include))

    assert Hf3fsStorageBackend().detect()


@pytest.mark.usefixtures("system_absl")
def test_include_dir_alone_auto_detects(
    monkeypatch: pytest.MonkeyPatch, usrbio_include: Path
) -> None:
    """With ``BUILD_HF3FS`` unset, ``HF3FS_INCLUDE_DIR`` enables the build."""
    monkeypatch.setenv("HF3FS_INCLUDE_DIR", str(usrbio_include))

    assert Hf3fsStorageBackend().detect()


@pytest.mark.usefixtures("system_absl")
def test_build_hf3fs_zero_overrides_include_dir(
    monkeypatch: pytest.MonkeyPatch, usrbio_include: Path
) -> None:
    """``BUILD_HF3FS=0`` disables the build even with an include dir."""
    monkeypatch.setenv("BUILD_HF3FS", "0")
    monkeypatch.setenv("HF3FS_INCLUDE_DIR", str(usrbio_include))

    assert not Hf3fsStorageBackend().detect()


@pytest.mark.usefixtures("system_absl")
def test_nothing_set_skips_build() -> None:
    """Without ``BUILD_HF3FS`` or ``HF3FS_INCLUDE_DIR`` nothing is built."""
    assert not Hf3fsStorageBackend().detect()


def test_absl_found_in_any_include_dir_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, usrbio_include: Path
) -> None:
    """Each ``;``-separated ``HF3FS_INCLUDE_DIR`` entry is searched."""
    monkeypatch.delenv("BUILD_WITH_HF3FS", raising=False)
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: f"lib{name}.so")
    monkeypatch.setattr(Hf3fsStorageBackend, "system_include_dirs", (), raising=False)
    absl_include = _make_include_dir(
        tmp_path / "absl", "absl/container/flat_hash_set.h"
    )
    monkeypatch.setenv("BUILD_HF3FS", "1")
    monkeypatch.setenv("HF3FS_INCLUDE_DIR", f"{usrbio_include};{absl_include}")

    assert Hf3fsStorageBackend().detect()


def test_missing_absl_header_skips_build(
    monkeypatch: pytest.MonkeyPatch, usrbio_include: Path
) -> None:
    """Without the Abseil header anywhere, the build is skipped."""
    monkeypatch.delenv("BUILD_WITH_HF3FS", raising=False)
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: f"lib{name}.so")
    monkeypatch.setattr(Hf3fsStorageBackend, "system_include_dirs", (), raising=False)
    monkeypatch.setenv("BUILD_HF3FS", "1")
    monkeypatch.setenv("HF3FS_INCLUDE_DIR", str(usrbio_include))

    assert not Hf3fsStorageBackend().detect()
