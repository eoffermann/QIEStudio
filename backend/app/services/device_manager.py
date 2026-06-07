"""Accelerator detection and capability reporting (DESIGN §4.3, §9.2).

The ``device_manager`` detects the available accelerator (CUDA / ROCm / MPS / CPU) and
reports the capabilities the model advisor (§9) needs: backend, compute capability / SM,
total & free VRAM, system RAM, and which precision/quant paths the hardware supports.

**torch is imported lazily** (inside functions) and every torch access is wrapped so this
module imports cleanly even when torch is absent — the fast dev/test image ships without
torch. When torch is unavailable, :func:`get_device_info` returns a sensible CPU fallback
(``backend="cpu"``, all accelerator capabilities ``False``, RAM read from the OS without
pulling in ``psutil``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from app.config import get_settings
from app.logging_utils import phase

log = logging.getLogger(__name__)

# fp8_e4m3fn runs natively (with a speedup) only on Ada/Hopper, SM >= 8.9. Below that it
# can be emulated on CUDA but with no speedup (DESIGN §9.1).
_FP8_NATIVE_MIN_SM: tuple[int, int] = (8, 9)
# Nunchaku SVDQuant int4 is CUDA-only and supported on Ampere (SM 8.0) and newer.
_INT4_MIN_SM: tuple[int, int] = (8, 0)


@dataclass(frozen=True)
class DeviceInfo:
    """A snapshot of the active accelerator and its capabilities (DESIGN §4.3)."""

    backend: str  # "cuda" | "rocm" | "mps" | "cpu"
    device_str: str  # e.g. "cuda:0", "mps", "cpu"
    name: str  # e.g. "NVIDIA RTX A6000"
    compute_capability: tuple[int, int] | None  # (8, 6) for Ampere; None for non-CUDA
    total_vram_mb: int
    free_vram_mb: int
    total_ram_mb: int
    supports_bf16: bool
    supports_fp8_native: bool  # CUDA & SM >= 8.9 (Ada/Hopper). A6000 is SM 8.6 => False
    supports_int4_nunchaku: bool  # CUDA only, Ampere+ supported => A6000 True


# Process-wide cache; refreshed on demand (free VRAM in particular drifts during a run).
_cached: DeviceInfo | None = None


def detect_backend(override: str | None = None) -> str:
    """Detect the active accelerator backend, honoring an explicit override.

    Args:
        override: Forces a backend ("cuda" / "rocm" / "mps" / "cpu") when provided; falls
            back to ``settings.device_override`` if this is ``None``.

    Returns:
        One of ``"cuda"`` (incl. ROCm reported as a HIP build => ``"rocm"``), ``"rocm"``,
        ``"mps"``, or ``"cpu"``. Returns ``"cpu"`` when torch is unavailable.
    """
    chosen = override if override is not None else get_settings().device_override
    if chosen:
        return chosen.lower()

    try:
        import torch
    except Exception:  # noqa: BLE001 — torch absent in the dev/test image
        return "cpu"

    try:
        if torch.cuda.is_available():
            # ROCm builds expose CUDA-style APIs but set torch.version.hip.
            if getattr(torch.version, "hip", None):
                return "rocm"
            return "cuda"
    except Exception:  # noqa: BLE001
        pass

    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass

    return "cpu"


def _total_ram_mb() -> int:
    """Read total system RAM in MB without depending on ``psutil``.

    Uses ``os.sysconf`` (POSIX) where available, with a safe conservative fallback so the
    value is always positive even on platforms lacking these knobs.
    """
    # Linux / most POSIX: pages * page size.
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages * page_size // (1024 * 1024))
    except (ValueError, OSError, AttributeError):
        pass

    # Fallback: parse /proc/meminfo on Linux.
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb // 1024
    except (OSError, ValueError, IndexError):
        pass

    # Last-resort conservative default (8 GB) so downstream math stays sane.
    return 8 * 1024


def _cpu_fallback() -> DeviceInfo:
    """Build a CPU-only DeviceInfo (torch absent or no accelerator)."""
    return DeviceInfo(
        backend="cpu",
        device_str="cpu",
        name="CPU",
        compute_capability=None,
        total_vram_mb=0,
        free_vram_mb=0,
        total_ram_mb=_total_ram_mb(),
        supports_bf16=False,
        supports_fp8_native=False,
        supports_int4_nunchaku=False,
    )


def _cuda_device_info(backend: str) -> DeviceInfo:
    """Build DeviceInfo for a CUDA or ROCm device (caller verified availability)."""
    import torch

    index = torch.cuda.current_device()
    device_str = f"cuda:{index}"
    props = torch.cuda.get_device_properties(index)
    name = getattr(props, "name", "CUDA device")

    major = getattr(props, "major", None)
    minor = getattr(props, "minor", None)
    cc: tuple[int, int] | None = (
        (int(major), int(minor)) if major is not None and minor is not None else None
    )

    total_vram = int(getattr(props, "total_memory", 0)) // (1024 * 1024)
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(index)
        free_vram = int(free_bytes) // (1024 * 1024)
    except Exception:  # noqa: BLE001 — mem_get_info may be unavailable on some builds
        free_vram = total_vram

    is_cuda = backend == "cuda"
    fp8_native = bool(is_cuda and cc is not None and cc >= _FP8_NATIVE_MIN_SM)
    # Nunchaku int4 is CUDA-only (never ROCm) and Ampere+.
    int4 = bool(is_cuda and cc is not None and cc >= _INT4_MIN_SM)
    bf16 = _cuda_supports_bf16()

    return DeviceInfo(
        backend=backend,
        device_str=device_str,
        name=str(name),
        compute_capability=cc,
        total_vram_mb=total_vram,
        free_vram_mb=free_vram,
        total_ram_mb=_total_ram_mb(),
        supports_bf16=bf16,
        supports_fp8_native=fp8_native,
        supports_int4_nunchaku=int4,
    )


def _cuda_supports_bf16() -> bool:
    """Whether the current CUDA/ROCm device supports bf16."""
    try:
        import torch

        return bool(torch.cuda.is_bf16_supported())
    except Exception:  # noqa: BLE001
        return True  # Modern accelerators support bf16; assume yes if the probe fails.


def _mps_device_info() -> DeviceInfo:
    """Build DeviceInfo for Apple Silicon (MPS) — unified memory, bf16/fp16 only."""
    ram = _total_ram_mb()
    # MPS shares unified memory with the system; report it as the VRAM budget.
    return DeviceInfo(
        backend="mps",
        device_str="mps",
        name="Apple Silicon (MPS)",
        compute_capability=None,
        total_vram_mb=ram,
        free_vram_mb=ram,
        total_ram_mb=ram,
        supports_bf16=True,
        supports_fp8_native=False,  # no native fp8_e4m3fn path on MPS
        supports_int4_nunchaku=False,  # Nunchaku is CUDA-only
    )


def get_device_info(refresh: bool = False) -> DeviceInfo:
    """Return the active :class:`DeviceInfo`, cached after the first probe.

    Args:
        refresh: When ``True``, re-probe the device (free VRAM in particular changes as
            jobs run, so callers planning a run should refresh).

    Returns:
        A populated :class:`DeviceInfo`. Falls back to a CPU-only descriptor when torch is
        unavailable or no accelerator is present.
    """
    global _cached
    if _cached is not None and not refresh:
        return _cached

    with phase(log, "Probing accelerator (device_manager)"):
        backend = detect_backend()
        try:
            if backend in ("cuda", "rocm"):
                info = _cuda_device_info(backend)
            elif backend == "mps":
                info = _mps_device_info()
            else:
                info = _cpu_fallback()
        except Exception:  # noqa: BLE001 — any probe failure degrades to CPU fallback
            log.exception("Device probe failed; falling back to CPU descriptor")
            info = _cpu_fallback()

    log.info(
        "Device: backend=%s name=%r cc=%s VRAM=%d/%dMB RAM=%dMB "
        "bf16=%s fp8_native=%s int4_nunchaku=%s",
        info.backend,
        info.name,
        info.compute_capability,
        info.free_vram_mb,
        info.total_vram_mb,
        info.total_ram_mb,
        info.supports_bf16,
        info.supports_fp8_native,
        info.supports_int4_nunchaku,
    )
    _cached = info
    return info
