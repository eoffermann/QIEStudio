"""Pure-logic unit tests for the pipeline service + reproducibility (no torch required).

These exercise the precision→dtype mapping, backend precision guards, pipeline-kwarg
assembly, preview-throttle decision, the resident-pipeline cache key, and the full
reproducibility metadata / PNG-text path. They must run in the fast dev image, which ships
**without** torch/diffusers — so the modules under test import lazily and these tests touch
no model code.
"""

from __future__ import annotations

import json

import pytest
from app.services import reproducibility as repro
from app.services.pipeline_service import (
    MODE_EDIT,
    MODE_GENERATE,
    PRECISION_BF16,
    PRECISION_FP8,
    PRECISION_INT4,
    RunRequest,
    StepProgress,
    build_pipeline_kwargs,
    make_load_key,
    precision_to_dtype_name,
    should_emit_preview,
    validate_precision_for_backend,
)


def test_module_imports_without_torch() -> None:
    """The service + reproducibility modules import even when torch is absent."""
    import importlib

    import app.services.pipeline_service as svc

    importlib.reload(svc)  # re-import path must not pull torch at module top level
    assert svc.get_pipeline_service() is svc.get_pipeline_service()  # singleton


# --- precision_to_dtype_name -----------------------------------------------------------


@pytest.mark.parametrize("precision", [PRECISION_BF16, PRECISION_FP8, PRECISION_INT4])
def test_precision_to_dtype_name_all_bf16(precision: str) -> None:
    assert precision_to_dtype_name(precision) == "bfloat16"


def test_precision_to_dtype_name_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown precision"):
        precision_to_dtype_name("fp16")


# --- validate_precision_for_backend ----------------------------------------------------


def test_bf16_allowed_everywhere() -> None:
    for backend in ("cuda", "rocm", "mps", "cpu"):
        validate_precision_for_backend(PRECISION_BF16, backend=backend, fp8_native=False)


def test_int4_cuda_only() -> None:
    validate_precision_for_backend(PRECISION_INT4, backend="cuda", fp8_native=True)
    for backend in ("rocm", "mps", "cpu"):
        with pytest.raises(ValueError, match="CUDA-only"):
            validate_precision_for_backend(PRECISION_INT4, backend=backend, fp8_native=False)


def test_fp8_requires_cuda_or_rocm() -> None:
    validate_precision_for_backend(PRECISION_FP8, backend="cuda", fp8_native=True)
    validate_precision_for_backend(PRECISION_FP8, backend="rocm", fp8_native=False)
    for backend in ("mps", "cpu"):
        with pytest.raises(ValueError, match="fp8"):
            validate_precision_for_backend(PRECISION_FP8, backend=backend, fp8_native=False)


def test_fp8_on_ampere_emulated_warns_not_blocked(caplog: pytest.LogCaptureFixture) -> None:
    """fp8 on CUDA without native SM>=8.9 (e.g. A6000 SM 8.6) is allowed but warns."""
    import logging

    with caplog.at_level(logging.WARNING):
        validate_precision_for_backend(PRECISION_FP8, backend="cuda", fp8_native=False)
    assert any("EMULATED" in r.message for r in caplog.records)


# --- build_pipeline_kwargs -------------------------------------------------------------


def _edit_req(**over: object) -> RunRequest:
    base: dict[str, object] = {
        "mode": MODE_EDIT,
        "model_id": "Qwen/Qwen-Image-Edit-2511",
        "model_revision": None,
        "precision": PRECISION_BF16,
        "device": "cuda:0",
        "prompt": "make it night",
    }
    base.update(over)
    return RunRequest(**base)  # type: ignore[arg-type]


def test_edit_defaults_match_reference_params() -> None:
    """Edit RunRequest defaults are the verbatim model-card reference params (DESIGN §2.1)."""
    req = _edit_req()
    assert req.num_inference_steps == 40
    assert req.true_cfg_scale == 4.0
    assert req.guidance_scale == 1.0
    assert req.negative_prompt == " "  # a single space
    assert req.batch == 1


def test_build_kwargs_omits_size_when_unset() -> None:
    """Edit mode without explicit size omits width/height (match-source path)."""
    kwargs = build_pipeline_kwargs(_edit_req())
    assert "width" not in kwargs and "height" not in kwargs
    assert kwargs["num_inference_steps"] == 40
    assert kwargs["true_cfg_scale"] == 4.0
    assert kwargs["guidance_scale"] == 1.0
    assert kwargs["negative_prompt"] == " "
    assert kwargs["num_images_per_prompt"] == 1
    assert "generator" not in kwargs  # built at run time with torch
    assert "image" not in kwargs  # added separately for edit


def test_build_kwargs_includes_size_when_both_set() -> None:
    kwargs = build_pipeline_kwargs(_edit_req(width=1024, height=688))
    assert kwargs["width"] == 1024 and kwargs["height"] == 688


def test_build_kwargs_batch_maps_to_num_images() -> None:
    kwargs = build_pipeline_kwargs(_edit_req(batch=4))
    assert kwargs["num_images_per_prompt"] == 4


# --- should_emit_preview ---------------------------------------------------------------


def test_preview_disabled_when_every_n_zero() -> None:
    assert should_emit_preview(5, 40, 0) is False
    assert should_emit_preview(40, 40, 0) is False  # not even the final step


def test_preview_on_cadence_and_final_step() -> None:
    assert should_emit_preview(5, 40, 5) is True
    assert should_emit_preview(10, 40, 5) is True
    assert should_emit_preview(7, 40, 5) is False
    assert should_emit_preview(40, 40, 5) is True  # always on the last step
    assert should_emit_preview(38, 40, 5) is False


# --- make_load_key ---------------------------------------------------------------------


def test_load_key_distinguishes_configs() -> None:
    a = make_load_key(mode=MODE_EDIT, model_id="m", model_revision=None,
                      precision=PRECISION_BF16, device="cuda:0")
    b = make_load_key(mode=MODE_GENERATE, model_id="m", model_revision=None,
                      precision=PRECISION_BF16, device="cuda:0")
    c = make_load_key(mode=MODE_EDIT, model_id="m", model_revision=None,
                      precision=PRECISION_FP8, device="cuda:0")
    same = make_load_key(mode=MODE_EDIT, model_id="m", model_revision=None,
                         precision=PRECISION_BF16, device="cuda:0")
    assert a == same
    assert a != b and a != c and b != c


# --- StepProgress ----------------------------------------------------------------------


def test_step_progress_dataclass() -> None:
    sp = StepProgress(step=5, total=40, preview_png=b"PNG")
    assert sp.step == 5 and sp.total == 40 and sp.preview_png == b"PNG"


# --- reproducibility.build_metadata ----------------------------------------------------


def _full_meta() -> dict:
    return repro.build_metadata(
        mode="edit",
        model_id="Qwen/Qwen-Image-Edit-2511",
        model_revision="abc123",
        precision="fp8",
        device="cuda:0 (NVIDIA RTX A6000, SM 8.6)",
        prompt="put the chair in the room",
        enhanced_prompt="A photorealistic chair placed in the living room ...",
        negative_prompt=" ",
        width=1024,
        height=688,
        steps=40,
        true_cfg_scale=4.0,
        guidance_scale=1.0,
        seed=12345,
        loras=[{"path": "/data/loras/a.safetensors", "weight": 0.8, "adapter_name": "a"}],
        input_hashes=["aaa", "bbb"],
        rewriter_model="Qwen/Qwen3-VL-8B-Instruct",
        output_format="png",
    )


def test_build_metadata_captures_design_5_5_fields() -> None:
    meta = _full_meta()
    assert meta["mode"] == "edit"
    assert meta["model_id"] == "Qwen/Qwen-Image-Edit-2511"
    assert meta["model_revision"] == "abc123"
    assert meta["precision"] == "fp8"
    assert "A6000" in meta["device"]
    assert meta["prompt"].startswith("put the chair")
    assert meta["enhanced_prompt"].startswith("A photorealistic")
    assert meta["negative_prompt"] == " "
    assert meta["rewriter_model"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert meta["resolution"] == {"width": 1024, "height": 688}
    assert meta["sampler"]["num_inference_steps"] == 40
    assert meta["sampler"]["true_cfg_scale"] == 4.0
    assert meta["sampler"]["guidance_scale"] == 1.0
    assert meta["seed"] == 12345
    assert meta["input_hashes"] == ["aaa", "bbb"]  # order preserved (model-significant)
    assert meta["output_format"] == "png"
    assert meta["schema_version"] == repro.METADATA_SCHEMA_VERSION


def test_build_metadata_normalizes_loras() -> None:
    meta = repro.build_metadata(
        mode="generate", model_id="Qwen/Qwen-Image", model_revision=None, precision="bf16",
        device="cuda:0", prompt="p", enhanced_prompt=None, negative_prompt=" ",
        width=None, height=None, steps=40, true_cfg_scale=4.0, guidance_scale=1.0, seed=None,
        loras=[{"lora_id": "L1", "weight": 1}, {"path": "/x.safetensors", "weight": "0.5"}],
        input_hashes=None, rewriter_model=None, output_format="webp",
    )
    assert meta["loras"][0] == {"lora_id": "L1", "weight": 1.0}
    assert meta["loras"][1] == {"path": "/x.safetensors", "weight": 0.5}
    assert meta["input_hashes"] == []
    assert meta["resolution"] == {"width": None, "height": None}
    assert meta["seed"] is None


def test_build_metadata_is_json_serializable() -> None:
    json.dumps(_full_meta())  # must not raise


# --- reproducibility.metadata_to_png_text ----------------------------------------------


def test_png_text_roundtrips_full_blob() -> None:
    meta = _full_meta()
    chunks = repro.metadata_to_png_text(meta)
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in chunks.items())
    recovered = json.loads(chunks[repro.PNG_PARAMETERS_KEY])
    assert recovered == meta


def test_png_text_flattens_key_fields() -> None:
    chunks = repro.metadata_to_png_text(_full_meta())
    assert chunks["qie_mode"] == "edit"
    assert chunks["qie_precision"] == "fp8"
    assert chunks["qie_seed"] == "12345"
    assert chunks["qie_width"] == "1024"
    assert chunks["qie_steps"] == "40"
    assert chunks["qie_input_hashes"] == "aaa,bbb"
    assert json.loads(chunks["qie_loras"])[0]["adapter_name"] == "a"


def test_png_text_omits_none_fields() -> None:
    meta = repro.build_metadata(
        mode="generate", model_id="Qwen/Qwen-Image", model_revision=None, precision="bf16",
        device="cuda:0", prompt="p", enhanced_prompt=None, negative_prompt=" ",
        width=None, height=None, steps=40, true_cfg_scale=4.0, guidance_scale=1.0, seed=None,
        loras=[], input_hashes=[], rewriter_model=None, output_format="png",
    )
    chunks = repro.metadata_to_png_text(meta)
    assert "qie_enhanced_prompt" not in chunks
    assert "qie_model_revision" not in chunks
    assert "qie_seed" not in chunks
    assert "qie_width" not in chunks
    assert "qie_loras" not in chunks  # empty list -> omitted
    assert "qie_input_hashes" not in chunks
