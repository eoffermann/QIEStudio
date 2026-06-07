"""Output upscaling / refine service (DESIGN §13 #2).

An optional post-process that turns 1–2 MP results into deliverables. DESIGN §13 #2 allows
either Real-ESRGAN **or** a Qwen-Image img2img refine pass; to avoid pulling a heavy new
dependency the v1 **baseline is a high-quality Lanczos upscale via Pillow** — it works on
every backend with zero extra deps. A clearly-documented :func:`refine_with_model` hook is
left in place for a future Qwen-Image img2img refinement pass (which would route through the
single-accelerator job queue like any other inference).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.logging_utils import phase

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

log = logging.getLogger(__name__)

# Snap the maximum scale so a stray request can't ask for a multi-gigapixel canvas.
MAX_SCALE = 8.0


def upscale_image(
    *,
    image: PILImage,
    scale: float = 2.0,
    max_long_edge: int | None = None,
) -> PILImage:
    """Upscale ``image`` with a high-quality Lanczos resample (DESIGN §13 #2 baseline).

    Computes the target size from ``scale`` (linear magnification), then — if
    ``max_long_edge`` is given and the target's longest edge would exceed it — rescales
    proportionally down to that cap (so the cap always wins, protecting downstream VRAM).
    Dimensions are clamped to at least 1px.

    Args:
        image: The source PIL image.
        scale: Linear magnification factor (``> 0``; clamped to :data:`MAX_SCALE`).
        max_long_edge: Optional cap on the output's longest edge, in pixels.

    Returns:
        A new resampled :class:`PIL.Image.Image` (same mode as the input).

    Raises:
        ValueError: If ``scale`` is not positive.
    """
    from PIL import Image  # local import keeps the module light at import time

    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    scale = min(scale, MAX_SCALE)

    target_w = max(1, round(image.width * scale))
    target_h = max(1, round(image.height * scale))

    if max_long_edge is not None and max(target_w, target_h) > max_long_edge:
        ratio = max_long_edge / max(target_w, target_h)
        target_w = max(1, round(target_w * ratio))
        target_h = max(1, round(target_h * ratio))

    with phase(
        log,
        f"Upscaling {image.width}x{image.height} -> {target_w}x{target_h} (Lanczos)",
    ):
        return image.resize((target_w, target_h), Image.LANCZOS)


def refine_with_model(*, image: PILImage, prompt: str = "", strength: float = 0.3) -> PILImage:
    """Refine an output with a Qwen-Image img2img pass (DESIGN §13 #2 — **future work**).

    This is the documented hook for the higher-quality refine option: a low-strength
    img2img pass over the (already-upscaled) image to recover detail. It is intentionally
    **not implemented** in v1 — wiring it up means loading the Generate pipeline through the
    single-accelerator job queue (DESIGN §4.1/§9), which the lightweight Lanczos baseline
    deliberately avoids. Until then :func:`upscale_image` is the supported path.

    Args:
        image: The image to refine.
        prompt: Guidance prompt for the refine pass.
        strength: img2img denoising strength (lower preserves more of the input).

    Raises:
        NotImplementedError: Always, in v1.
    """
    raise NotImplementedError(
        "Model-based refine (Qwen-Image img2img) is not implemented in v1. "
        "Use upscale_image() for the Lanczos baseline; a future release will route a "
        "low-strength img2img refine pass through the single-accelerator job queue."
    )
