"""VRAM-aware precision/quantization advisor (DESIGN §9).

The ``model_advisor`` combines the :class:`~app.services.device_manager.DeviceInfo` with the
planned run (mode, target resolution, batch size, number of active LoRAs) to rank the
precision options — **bf16 / fp8_e4m3fn (scaled) / Nunchaku SVDQuant int4** — and recommend
the best fit, with a per-option estimated peak VRAM, headroom, rationale, and caveats
(DESIGN §9.1 / §9.2).

The estimate is deliberately a transparent analytic model (no torch, no probing) so it is
deterministic and unit-testable: pass a synthetic :class:`DeviceInfo` and assert the ranking.
The figures follow the spec's order-of-magnitude guidance for the ~20B Qwen-Image MMDiT.

Backend filtering (DESIGN §4.3 / §9.2):

- **int4 (Nunchaku)** is never offered unless ``device.supports_int4_nunchaku`` (CUDA only).
- **fp8** is offered on CUDA even below SM 8.9, but flagged "emulated — no speedup" when
  ``not device.supports_fp8_native``. On MPS only bf16 is offered.
- The recommended option is the smallest-footprint precision with comfortable headroom,
  preferring bf16 when it fits comfortably, else fp8, else int4.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.services.device_manager import DeviceInfo, get_device_info

log = logging.getLogger(__name__)

# --- VRAM model (MB). Order-of-magnitude figures calibrated so estimated *peaks* track
# the spec's guidance examples (DESIGN §9.1 weight table + §9.2 peak strings). bf16 is the
# heavy full-precision DiT (~40 GB weights); fp8/int4 footprints are set so a 1024px batch-1
# run lands near the spec's "fp8 ~15 GB" / "int4 ~11 GB" peak figures once aux + latents are
# folded in. ---
_DIT_WEIGHTS_MB: dict[str, int] = {
    "bf16": 40_000,  # ~40 GB full bf16 (DESIGN §9.1)
    "fp8": 13_000,  # scaled fp8_e4m3fn — ~half-ish; peak lands ~15 GB at 1024 (DESIGN §9.2)
    "int4": 9_000,  # SVDQuant int4 — peak lands ~11 GB at 1024 (DESIGN §9.2)
}
# VAE + text encoder add a couple GB on top, folded into every option (DESIGN §9.1).
_AUX_MB: int = 2_000
# Per-LoRA adapter resident cost (a few hundred MB each; DESIGN §5.3 / §9.2).
_LORA_MB: int = 350
# Reference latent/activation working set at 1024px longer edge, batch 1. Scales with the
# pixel area (W*H) and linearly with batch. A rough but monotonic proxy for the advisor.
_LATENT_REF_MB: int = 1_500
_REF_LONG_EDGE: int = 1024

# Headroom thresholds (fraction of total VRAM left free after the estimated peak).
_COMFORTABLE_HEADROOM_FRAC: float = 0.15  # >= 15% free => "recommended"-eligible
_TIGHT_HEADROOM_MB: int = 1_500  # fits but < this much spare => "tight"


@dataclass(frozen=True)
class PrecisionOption:
    """One precision/quant option, scored against the planned run (DESIGN §9.2)."""

    precision: str  # "bf16" | "fp8" | "int4"
    available: bool
    status: str  # "recommended" | "fits" | "tight" | "wont_fit" | "unavailable"
    est_peak_vram_mb: int
    headroom_mb: int
    rationale: str
    caveats: list[str]


@dataclass(frozen=True)
class Advice:
    """The ranked advice for a planned run (DESIGN §9.2)."""

    device: DeviceInfo
    options: list[PrecisionOption]
    recommended: str | None


def _latent_mb(longer_edge: int, batch: int) -> int:
    """Estimate the latent/activation working-set MB for a run.

    Scales with pixel area relative to the 1024px reference and linearly with batch.
    """
    area_scale = (max(longer_edge, 16) / _REF_LONG_EDGE) ** 2
    return int(_LATENT_REF_MB * area_scale * max(batch, 1))


def _est_peak_vram_mb(precision: str, longer_edge: int, batch: int, num_loras: int) -> int:
    """Estimated peak VRAM (MB) for a precision option under the planned run."""
    weights = _DIT_WEIGHTS_MB[precision]
    return weights + _AUX_MB + _latent_mb(longer_edge, batch) + num_loras * _LORA_MB


def _backend_precisions(device: DeviceInfo) -> list[str]:
    """The precision options to even consider for this backend (before fit scoring)."""
    if device.backend == "mps":
        return ["bf16"]
    if device.backend in ("cuda", "rocm"):
        opts = ["bf16", "fp8"]
        # int4 (Nunchaku) is CUDA-only and gated on capability.
        if device.supports_int4_nunchaku:
            opts.append("int4")
        return opts
    # CPU / unknown: bf16 is the only nominal option (no acceleration — for completeness).
    return ["bf16"]


def _caveats_for(precision: str, device: DeviceInfo) -> list[str]:
    """Backend-specific caveats for a precision on this device (DESIGN §9.1/§9.2)."""
    caveats: list[str] = []
    if precision == "fp8":
        if device.backend in ("cuda", "rocm") and not device.supports_fp8_native:
            caveats.append(
                "fp8 is emulated on this architecture (SM < 8.9) — saves memory but no "
                "compute speedup."
            )
        if device.backend == "rocm":
            caveats.append("fp8 support on ROCm depends on the GPU architecture.")
    if precision == "int4":
        caveats.append(
            "Nunchaku SVDQuant int4 is CUDA-only and constrains how/whether LoRAs fuse."
        )
    if device.backend == "mps" and precision == "bf16":
        caveats.append("Large latents rely on unified memory on Apple Silicon.")
    return caveats


def advise(
    *,
    mode: str,
    longer_edge: int,
    batch: int = 1,
    num_loras: int = 0,
    device: DeviceInfo | None = None,
) -> Advice:
    """Rank precision options for a planned run and recommend the best fit (DESIGN §9.2).

    Args:
        mode: ``"generate"`` or ``"edit"`` (recorded for context; the VRAM model is shared).
        longer_edge: The target resolution's longer edge in pixels (latent VRAM scales with
            its square). Compute it via :func:`app.services.resolution.resolve` if needed.
        batch: Number of images per prompt (latent VRAM scales linearly with this).
        num_loras: Count of active LoRA adapters (each adds resident VRAM).
        device: A :class:`DeviceInfo` to score against; defaults to the detected device
            (refreshed so free-VRAM is current). Pass a synthetic one for deterministic
            tests.

    Returns:
        An :class:`Advice` with one :class:`PrecisionOption` per considered precision and the
        recommended precision string (or ``None`` if nothing fits).
    """
    dev = device if device is not None else get_device_info(refresh=True)
    longer_edge = max(int(longer_edge), 16)
    batch = max(int(batch), 1)
    num_loras = max(int(num_loras), 0)

    budget = dev.total_vram_mb
    considered = _backend_precisions(dev)

    options: list[PrecisionOption] = []
    # Track which precisions comfortably fit, to pick a recommendation afterwards.
    fits_comfortably: dict[str, int] = {}  # precision -> headroom_mb

    for precision in ("bf16", "fp8", "int4"):
        available = precision in considered
        peak = _est_peak_vram_mb(precision, longer_edge, batch, num_loras)
        headroom = budget - peak
        caveats = _caveats_for(precision, dev)

        if not available:
            reason = _unavailable_reason(precision, dev)
            options.append(
                PrecisionOption(
                    precision=precision,
                    available=False,
                    status="unavailable",
                    est_peak_vram_mb=peak,
                    headroom_mb=headroom,
                    rationale=reason,
                    caveats=caveats,
                )
            )
            continue

        # bf16 on bf16-incapable hardware (e.g. CPU fallback) is effectively unavailable.
        if precision == "bf16" and not dev.supports_bf16:
            options.append(
                PrecisionOption(
                    precision=precision,
                    available=False,
                    status="unavailable",
                    est_peak_vram_mb=peak,
                    headroom_mb=headroom,
                    rationale="bf16 is not supported by this device.",
                    caveats=caveats,
                )
            )
            continue

        comfortable_threshold = int(budget * _COMFORTABLE_HEADROOM_FRAC)
        if budget <= 0:
            # No measurable VRAM budget (CPU fallback): nothing can be sized to fit.
            status = "wont_fit"
            rationale = "No accelerator VRAM available on this device."
        elif headroom < 0:
            status = "wont_fit"
            rationale = (
                f"{_pretty(precision)} needs ~{peak} MB but only {budget} MB total VRAM "
                f"is available."
            )
        elif headroom < _TIGHT_HEADROOM_MB or headroom < comfortable_threshold:
            status = "tight"
            rationale = (
                f"{_pretty(precision)} fits at ~{peak} MB peak but headroom is tight "
                f"(~{headroom} MB) — consider offloading / VAE tiling."
            )
        else:
            status = "fits"
            rationale = (
                f"{_pretty(precision)} fits at ~{peak} MB peak with comfortable headroom "
                f"(~{headroom} MB)."
            )
            fits_comfortably[precision] = headroom

        options.append(
            PrecisionOption(
                precision=precision,
                available=True,
                status=status,
                est_peak_vram_mb=peak,
                headroom_mb=headroom,
                rationale=rationale,
                caveats=caveats,
            )
        )

    recommended = _choose_recommended(fits_comfortably, options)
    options = _apply_recommended_status(options, recommended)

    log.info(
        "Advisor (mode=%s long_edge=%d batch=%d loras=%d backend=%s): recommended=%s",
        mode,
        longer_edge,
        batch,
        num_loras,
        dev.backend,
        recommended,
    )
    return Advice(device=dev, options=options, recommended=recommended)


def _choose_recommended(
    fits_comfortably: dict[str, int],
    options: list[PrecisionOption],
) -> str | None:
    """Pick the recommended precision (DESIGN §9.2).

    Prefer bf16 when it has comfortable headroom, else fp8, else int4. If nothing fits
    comfortably, fall back to the *tight* option with the largest headroom (best effort).
    """
    for precision in ("bf16", "fp8", "int4"):
        if precision in fits_comfortably:
            return precision

    # Nothing comfortable — recommend the tightest-fitting (>= 0 headroom) option, if any,
    # preferring smaller footprints which are likelier to fit.
    tight: list[PrecisionOption] = [
        o for o in options if o.available and o.status == "tight"
    ]
    if tight:
        # Prefer the smallest peak (most likely to actually fit at runtime).
        tight.sort(key=lambda o: o.est_peak_vram_mb)
        return tight[0].precision
    return None


def _apply_recommended_status(
    options: list[PrecisionOption], recommended: str | None
) -> list[PrecisionOption]:
    """Promote the recommended option's status to ``"recommended"``."""
    if recommended is None:
        return options
    promoted: list[PrecisionOption] = []
    for o in options:
        if o.precision == recommended and o.available:
            promoted.append(
                PrecisionOption(
                    precision=o.precision,
                    available=o.available,
                    status="recommended",
                    est_peak_vram_mb=o.est_peak_vram_mb,
                    headroom_mb=o.headroom_mb,
                    rationale=o.rationale,
                    caveats=o.caveats,
                )
            )
        else:
            promoted.append(o)
    return promoted


def _unavailable_reason(precision: str, dev: DeviceInfo) -> str:
    """Explain why a precision isn't offered on this backend (DESIGN §9.2)."""
    if precision == "int4":
        if dev.backend == "mps":
            return "Nunchaku SVDQuant int4 is CUDA-only and unavailable on Apple Silicon."
        if dev.backend == "rocm":
            return "Nunchaku SVDQuant int4 is CUDA-only and unavailable on ROCm."
        return "Nunchaku SVDQuant int4 requires a supported CUDA GPU (Ampere or newer)."
    if precision == "fp8" and dev.backend == "mps":
        return "fp8_e4m3fn has no native path on Apple Silicon (MPS)."
    return f"{_pretty(precision)} is not available on the {dev.backend} backend."


def _pretty(precision: str) -> str:
    """Human-friendly precision label for rationale strings."""
    return {
        "bf16": "bf16",
        "fp8": "fp8_e4m3fn",
        "int4": "SVDQuant int4",
    }.get(precision, precision)
