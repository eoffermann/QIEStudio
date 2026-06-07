"""Real-GPU integration tests for the pipeline service (DESIGN §2, §5.5, §9).

Every test here is marked ``@pytest.mark.gpu`` and is run later by the orchestrator inside
the CUDA Docker image (torch + diffusers + downloaded weights present). In the fast dev
image (no torch) they must **collect and SKIP cleanly** — never error — so a module-level
guard skips the whole file when torch/CUDA are unavailable, and no torch/diffusers import
happens at import time.

These exercise the heavy paths the pure tests cannot: a real bf16 Generate run with live
preview + cancellation, a real Edit run with an ordered image list, fp8 quantization on
Ampere (emulated), and the int4 (Nunchaku) availability guard.
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
        "No CUDA GPU / torch available — GPU pipeline tests run in the CUDA image only.",
        allow_module_level=True,
    )


# --- fixtures (only reached when torch+CUDA are present) -------------------------------


@pytest.fixture(scope="module")
def device() -> str:
    return "cuda:0"


@pytest.fixture()
def service():  # noqa: ANN201 — PipelineService; annotating needs the heavy import
    from app.services.pipeline_service import get_pipeline_service

    return get_pipeline_service()


def _solid_image(size: int = 512):  # noqa: ANN202
    from PIL import Image

    return Image.new("RGB", (size, size), (127, 127, 127))


# --- bf16 Generate: progress + live preview --------------------------------------------


def test_generate_bf16_with_preview_and_progress(service, device: str) -> None:  # noqa: ANN001
    from app.services.pipeline_service import RunRequest, StepProgress

    seen: list[StepProgress] = []
    req = RunRequest(
        mode="generate",
        model_id="Qwen/Qwen-Image",
        model_revision=None,
        precision="bf16",
        device=device,
        prompt="a small red cube on a white table, studio lighting",
        width=512,
        height=512,
        num_inference_steps=8,
        seed=1234,
        preview_every_n_steps=4,
    )
    images = service.run(req, on_step=seen.append)
    assert len(images) == 1
    assert seen and seen[-1].step == seen[-1].total
    # At least one throttled preview should have been decoded.
    assert any(p.preview_png for p in seen)


# --- Edit: ordered multi-image input + reference params --------------------------------


def test_edit_bf16_ordered_images(service, device: str) -> None:  # noqa: ANN001
    from app.services.pipeline_service import RunRequest

    req = RunRequest(
        mode="edit",
        model_id="Qwen/Qwen-Image-Edit-2511",
        model_revision=None,
        precision="bf16",
        device=device,
        prompt="combine the two references into one scene",
        images=[_solid_image(), _solid_image()],  # order is model-significant (DESIGN §2.1)
        num_inference_steps=8,
        seed=7,
        preview_every_n_steps=0,  # previews off for this run
    )
    images = service.run(req)
    assert len(images) == 1


def test_edit_requires_input_image(service, device: str) -> None:  # noqa: ANN001
    from app.services.pipeline_service import RunRequest

    req = RunRequest(
        mode="edit",
        model_id="Qwen/Qwen-Image-Edit-2511",
        model_revision=None,
        precision="bf16",
        device=device,
        prompt="no images supplied",
        images=[],
        num_inference_steps=4,
    )
    with pytest.raises(ValueError, match="at least one input image"):
        service.run(req)


# --- Cancellation ----------------------------------------------------------------------


def test_run_cancels_cleanly(service, device: str) -> None:  # noqa: ANN001
    from app.services.pipeline_service import CanceledError, RunRequest

    req = RunRequest(
        mode="generate",
        model_id="Qwen/Qwen-Image",
        model_revision=None,
        precision="bf16",
        device=device,
        prompt="a blue sphere",
        width=512,
        height=512,
        num_inference_steps=20,
        seed=99,
        preview_every_n_steps=0,
    )
    # Cancel after the first step.
    with pytest.raises(CanceledError):
        service.run(req, is_canceled=lambda: True)


# --- fp8 on Ampere (emulated) ----------------------------------------------------------


def test_fp8_generate_runs_emulated_on_ampere(service, device: str) -> None:  # noqa: ANN001
    """On A6000 (SM 8.6) fp8 is emulated — validate it *runs*, not that it is faster."""
    from app.services.pipeline_service import RunRequest

    req = RunRequest(
        mode="generate",
        model_id="Qwen/Qwen-Image",
        model_revision=None,
        precision="fp8",
        device=device,
        prompt="a green apple",
        width=512,
        height=512,
        num_inference_steps=6,
        seed=3,
        preview_every_n_steps=0,
    )
    images = service.run(req)
    assert len(images) == 1


# --- int4 (Nunchaku) availability guard ------------------------------------------------


def test_int4_load_clear_error_when_nunchaku_absent(service, device: str) -> None:  # noqa: ANN001
    """If nunchaku / SVDQuant weights are missing, loading raises a clear NotImplementedError."""
    if importlib.util.find_spec("nunchaku") is not None:
        pytest.skip("nunchaku is installed; int4 load is expected to succeed elsewhere.")
    with pytest.raises(NotImplementedError, match="Nunchaku"):
        service.load(
            mode="generate",
            model_id="Qwen/Qwen-Image",
            model_revision=None,
            precision="int4",
            device=device,
        )
