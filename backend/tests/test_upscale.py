"""Tests for the upscale/refine service (DESIGN §13 #2).

Pillow-only and fully deterministic — no torch, no DB. Covers the Lanczos baseline (scale
math, the ``max_long_edge`` cap, validation) and the documented model-refine stub.
"""

from __future__ import annotations

import pytest
from app.services import upscale
from PIL import Image


def test_upscale_doubles_dimensions() -> None:
    src = Image.new("RGB", (100, 50), (1, 2, 3))
    out = upscale.upscale_image(image=src, scale=2.0)
    assert out.size == (200, 100)
    assert out.mode == "RGB"


def test_upscale_fractional_scale_rounds() -> None:
    src = Image.new("RGB", (101, 51))
    out = upscale.upscale_image(image=src, scale=1.5)
    assert out.size == (round(101 * 1.5), round(51 * 1.5))


def test_max_long_edge_caps_output() -> None:
    src = Image.new("RGB", (100, 50))
    # scale would give 400x200, but the cap of 300 forces a proportional shrink.
    out = upscale.upscale_image(image=src, scale=4.0, max_long_edge=300)
    assert max(out.size) == 300
    # Aspect ratio preserved (2:1).
    assert out.size == (300, 150)


def test_max_long_edge_inactive_when_below_cap() -> None:
    src = Image.new("RGB", (100, 50))
    out = upscale.upscale_image(image=src, scale=2.0, max_long_edge=1000)
    assert out.size == (200, 100)


def test_scale_clamped_to_max() -> None:
    src = Image.new("RGB", (10, 10))
    out = upscale.upscale_image(image=src, scale=999.0)
    assert max(out.size) == int(10 * upscale.MAX_SCALE)


def test_preserves_mode_rgba() -> None:
    src = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    out = upscale.upscale_image(image=src, scale=2.0)
    assert out.mode == "RGBA"


def test_invalid_scale_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        upscale.upscale_image(image=Image.new("RGB", (4, 4)), scale=0)


def test_refine_with_model_is_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="img2img"):
        upscale.refine_with_model(image=Image.new("RGB", (4, 4)))
