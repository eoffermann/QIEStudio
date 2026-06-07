"""Tests for the model_advisor (DESIGN §9).

The advisor is deterministic given a synthetic :class:`DeviceInfo`, so these tests build
representative devices (A6000 Ampere, Ada 24 GB, 16 GB Ampere, H100, Apple MPS, ROCm) and
assert backend filtering, fit scoring, and the recommendation policy.
"""

from __future__ import annotations

from app.services.device_manager import DeviceInfo
from app.services.model_advisor import advise


def _dev(
    *,
    backend: str,
    total_vram_mb: int,
    cc: tuple[int, int] | None = (8, 6),
    bf16: bool = True,
    fp8_native: bool = False,
    int4: bool = False,
    ram_mb: int = 64_000,
) -> DeviceInfo:
    return DeviceInfo(
        backend=backend,
        device_str=f"{backend}:0",
        name=f"synthetic {backend}",
        compute_capability=cc,
        total_vram_mb=total_vram_mb,
        free_vram_mb=total_vram_mb,
        total_ram_mb=ram_mb,
        supports_bf16=bf16,
        supports_fp8_native=fp8_native,
        supports_int4_nunchaku=int4,
    )


def _opt(advice, precision):  # noqa: ANN001
    return next(o for o in advice.options if o.precision == precision)


def test_a6000_offers_int4_and_fp8_emulated_caveat() -> None:
    # A6000: SM 8.6 -> int4 available, fp8 available but emulated (no speedup).
    dev = _dev(backend="cuda", total_vram_mb=48_000, cc=(8, 6), int4=True, fp8_native=False)
    advice = advise(mode="edit", longer_edge=1024, device=dev)

    assert _opt(advice, "int4").available is True
    fp8 = _opt(advice, "fp8")
    assert fp8.available is True
    assert any("emulated" in c for c in fp8.caveats)


def test_h100_recommends_bf16() -> None:
    dev = _dev(backend="cuda", total_vram_mb=80_000, cc=(9, 0), fp8_native=True, int4=True)
    advice = advise(mode="generate", longer_edge=1024, device=dev)
    assert advice.recommended == "bf16"
    assert _opt(advice, "bf16").status == "recommended"


def test_24gb_ada_recommends_fp8_when_bf16_wont_fit() -> None:
    # 24 GB Ada (SM 8.9): bf16 (~46 GB peak) won't fit; fp8 fits comfortably.
    dev = _dev(backend="cuda", total_vram_mb=24_000, cc=(8, 9), fp8_native=True, int4=True)
    advice = advise(mode="edit", longer_edge=1024, num_loras=2, device=dev)
    assert _opt(advice, "bf16").status == "wont_fit"
    assert advice.recommended == "fp8"


def test_16gb_recommends_int4() -> None:
    # 16 GB Ampere: bf16 + fp8 won't fit comfortably; int4 (~13.5 GB peak) recommended.
    dev = _dev(backend="cuda", total_vram_mb=16_000, cc=(8, 6), fp8_native=False, int4=True)
    advice = advise(mode="generate", longer_edge=1024, device=dev)
    assert advice.recommended == "int4"
    assert _opt(advice, "bf16").status == "wont_fit"


def test_mps_only_offers_bf16() -> None:
    dev = _dev(
        backend="mps",
        total_vram_mb=96_000,
        cc=None,
        bf16=True,
        fp8_native=False,
        int4=False,
        ram_mb=96_000,
    )
    advice = advise(mode="generate", longer_edge=1024, device=dev)
    assert _opt(advice, "bf16").available is True
    assert _opt(advice, "fp8").available is False
    assert _opt(advice, "int4").available is False
    assert "Apple Silicon" in _opt(advice, "int4").rationale


def test_rocm_hides_int4() -> None:
    dev = _dev(backend="rocm", total_vram_mb=48_000, cc=(9, 4), int4=False, fp8_native=False)
    advice = advise(mode="edit", longer_edge=1024, device=dev)
    assert _opt(advice, "int4").available is False
    assert "CUDA-only" in _opt(advice, "int4").rationale
    assert _opt(advice, "fp8").available is True


def test_int4_never_offered_without_capability() -> None:
    dev = _dev(backend="cuda", total_vram_mb=48_000, cc=(7, 5), int4=False)
    advice = advise(mode="edit", longer_edge=1024, device=dev)
    assert _opt(advice, "int4").available is False


def test_higher_resolution_increases_peak_vram() -> None:
    dev = _dev(backend="cuda", total_vram_mb=48_000, cc=(8, 6), int4=True)
    low = advise(mode="generate", longer_edge=1024, device=dev)
    high = advise(mode="generate", longer_edge=2048, device=dev)
    assert _opt(high, "bf16").est_peak_vram_mb > _opt(low, "bf16").est_peak_vram_mb


def test_batch_and_loras_increase_peak_vram() -> None:
    dev = _dev(backend="cuda", total_vram_mb=48_000, cc=(8, 6), int4=True)
    base = advise(mode="edit", longer_edge=1024, batch=1, num_loras=0, device=dev)
    more = advise(mode="edit", longer_edge=1024, batch=4, num_loras=3, device=dev)
    assert _opt(more, "fp8").est_peak_vram_mb > _opt(base, "fp8").est_peak_vram_mb


def test_cpu_device_nothing_fits() -> None:
    dev = _dev(backend="cpu", total_vram_mb=0, cc=None, bf16=False)
    advice = advise(mode="generate", longer_edge=1024, device=dev)
    assert advice.recommended is None
    # bf16 is unavailable on a bf16-incapable CPU fallback.
    assert _opt(advice, "bf16").available is False


def test_recommended_option_marked_recommended() -> None:
    dev = _dev(backend="cuda", total_vram_mb=80_000, cc=(9, 0), fp8_native=True, int4=True)
    advice = advise(mode="generate", longer_edge=1024, device=dev)
    assert advice.recommended is not None
    rec = _opt(advice, advice.recommended)
    assert rec.status == "recommended"
