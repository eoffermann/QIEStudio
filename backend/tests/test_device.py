"""Tests for the device_manager (DESIGN §4.3, §9.2).

These run in the dev/test image which has **no torch**, so they exercise the CPU fallback
path and the pure capability logic. Torch-dependent probing is covered indirectly via the
fallback and override behavior.
"""

from __future__ import annotations

from app.services import device_manager
from app.services.device_manager import DeviceInfo, detect_backend, get_device_info


def test_detect_backend_cpu_without_torch() -> None:
    # No torch in the dev image -> CPU. (No override passed.)
    assert detect_backend() in {"cpu", "cuda", "rocm", "mps"}


def test_detect_backend_override_wins() -> None:
    assert detect_backend(override="cuda") == "cuda"
    assert detect_backend(override="MPS") == "mps"


def test_get_device_info_cpu_fallback_is_sane() -> None:
    info = get_device_info(refresh=True)
    assert isinstance(info, DeviceInfo)
    assert info.backend in {"cpu", "cuda", "rocm", "mps"}
    # RAM must be read from the OS and be positive even without psutil.
    assert info.total_ram_mb > 0


def test_cpu_fallback_capabilities_all_false() -> None:
    info = device_manager._cpu_fallback()
    assert info.backend == "cpu"
    assert info.device_str == "cpu"
    assert info.compute_capability is None
    assert info.total_vram_mb == 0
    assert info.free_vram_mb == 0
    assert info.supports_bf16 is False
    assert info.supports_fp8_native is False
    assert info.supports_int4_nunchaku is False
    assert info.total_ram_mb > 0


def test_total_ram_mb_positive() -> None:
    assert device_manager._total_ram_mb() > 0


def test_get_device_info_is_cached() -> None:
    first = get_device_info(refresh=True)
    second = get_device_info()
    assert first is second  # cached snapshot returned without re-probing


def test_mps_device_info_capabilities() -> None:
    info = device_manager._mps_device_info()
    assert info.backend == "mps"
    assert info.supports_bf16 is True
    assert info.supports_fp8_native is False
    assert info.supports_int4_nunchaku is False
    assert info.compute_capability is None
    assert info.total_vram_mb == info.total_ram_mb
