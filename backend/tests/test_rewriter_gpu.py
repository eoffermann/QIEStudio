"""Real-GPU integration tests for the Qwen-VL prompt enhancer (DESIGN §5.8).

Every test here is ``@pytest.mark.gpu`` and is run later by the orchestrator inside the CUDA
image (torch + transformers + downloaded VL weights present). In the fast dev image (no torch)
they must **collect and SKIP cleanly** — never error — so a module-level guard skips the whole
file when torch/CUDA are unavailable and no torch/transformers import happens at import time.

These exercise the heavy path the pure tests cannot: a real multimodal Edit rewrite (the VL
model *sees* the image), a real text Generate rewrite, and resident-model reuse + swap/unload.
"""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.gpu


def _torch_cuda_available() -> bool:
    """True only if torch is importable AND a CUDA device is present (no import at collect)."""
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


# Skip the entire module cleanly (not error) when there is no CUDA GPU / no torch.
if not _torch_cuda_available():
    pytest.skip(
        "No CUDA GPU / torch available — GPU rewriter tests run in the CUDA image only.",
        allow_module_level=True,
    )


# --- fixtures (only reached when torch+CUDA are present) -------------------------------

# Use the configured default rewriter (8B) for the live runs — it fits a single GPU easily.
_VL_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


@pytest.fixture()
def service():  # noqa: ANN201 — RewriterService; annotating needs the heavy import
    from app.services.rewriter import get_rewriter_service

    svc = get_rewriter_service()
    yield svc
    svc.unload()


def _solid_image(size: int = 512):  # noqa: ANN202
    from PIL import Image

    return Image.new("RGB", (size, size), (200, 120, 60))


# --- Generate-mode text rewrite --------------------------------------------------------


def test_generate_rewrite_returns_text(service) -> None:  # noqa: ANN001
    out = service.enhance(
        mode="generate",
        prompt="a cat on a sofa",
        vl_model=_VL_MODEL,
    )
    assert isinstance(out, str)
    assert out.strip() != ""


# --- Edit-mode multimodal rewrite (the VL model sees the image) ------------------------


def test_edit_rewrite_is_multimodal(service) -> None:  # noqa: ANN001
    out = service.enhance(
        mode="edit",
        prompt="make the background a beach at sunset",
        images=[_solid_image()],
        vl_model=_VL_MODEL,
        edit_model_id="Qwen/Qwen-Image-Edit-2511",
    )
    assert isinstance(out, str)
    assert out.strip() != ""


# --- resident reuse + swap/unload ------------------------------------------------------


def test_resident_model_is_reused(service) -> None:  # noqa: ANN001
    service.load(vl_model=_VL_MODEL)
    first = service._resident
    service.load(vl_model=_VL_MODEL)
    assert service._resident is first  # same object reused, no reload


def test_unload_frees_resident(service) -> None:  # noqa: ANN001
    service.load(vl_model=_VL_MODEL)
    assert service._resident is not None
    service.unload()
    assert service._resident is None
