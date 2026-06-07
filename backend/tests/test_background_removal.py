"""Tests for the background-removal service (DESIGN §13 #1).

The unit tests **monkeypatch** ``rembg`` so they run in the torch-free dev image without the
real matting model. A separate ``@pytest.mark.gpu`` test exercises the real dependency and
**skips cleanly** when ``rembg`` is absent (it lives in the [inference] extra).
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest
from app.services import background_removal
from PIL import Image


def _make_fake_rembg(monkeypatch: pytest.MonkeyPatch) -> list[Image.Image]:
    """Install a fake ``rembg`` module whose ``remove`` returns an RGBA passthrough."""
    seen: list[Image.Image] = []

    def _remove(img: Image.Image):  # noqa: ANN202
        seen.append(img)
        return img.convert("RGBA")

    fake = types.ModuleType("rembg")
    fake.remove = _remove  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rembg", fake)
    return seen


def test_module_imports_without_rembg() -> None:
    """The service module must import in the dev image (rembg is lazy-imported)."""
    mod = importlib.import_module("app.services.background_removal")
    assert hasattr(mod, "remove_background")


def test_remove_background_returns_rgba(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _make_fake_rembg(monkeypatch)
    src = Image.new("RGB", (32, 24), (10, 20, 30))

    out = background_removal.remove_background(image=src)

    assert out.mode == "RGBA"
    assert out.size == (32, 24)
    assert seen and seen[0] is src  # the source image was passed straight to rembg


def test_remove_background_handles_bytes_return(monkeypatch: pytest.MonkeyPatch) -> None:
    """rembg can return raw bytes; the service must normalize to an RGBA PIL image."""
    import io

    def _remove(_img: Image.Image) -> bytes:
        buf = io.BytesIO()
        Image.new("RGBA", (8, 8), (0, 0, 0, 0)).save(buf, format="PNG")
        return buf.getvalue()

    fake = types.ModuleType("rembg")
    fake.remove = _remove  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rembg", fake)

    out = background_removal.remove_background(image=Image.new("RGB", (8, 8)))
    assert out.mode == "RGBA"
    assert out.size == (8, 8)


def test_remove_background_raises_when_rembg_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clear RuntimeError when rembg cannot be imported (dev image)."""
    monkeypatch.setitem(sys.modules, "rembg", None)  # forces ImportError on `from rembg ...`
    with pytest.raises(RuntimeError, match="rembg"):
        background_removal.remove_background(image=Image.new("RGB", (8, 8)))


@pytest.mark.gpu
def test_remove_background_real_model() -> None:
    """Exercise the real rembg model; skip cleanly if it is not installed."""
    pytest.importorskip("rembg")
    src = Image.new("RGB", (64, 64), (200, 50, 50))
    out = background_removal.remove_background(image=src)
    assert out.mode == "RGBA"
    assert out.size == (64, 64)
