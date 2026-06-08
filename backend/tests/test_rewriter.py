"""Pure-logic tests for the rewriter service (DESIGN §5.8).

The model registry, backend filtering, the co-reside-vs-swap advisor, and the prompt-assembly
helpers are deterministic and torch-free, so these run anywhere (no GPU, no torch). The real VL
``generate`` is exercised separately in ``test_rewriter_gpu.py`` (``@pytest.mark.gpu``).
"""

from __future__ import annotations

import importlib.util

from app.services.device_manager import DeviceInfo
from app.services.rewriter import (
    MODE_EDIT,
    MODE_GENERATE,
    advise_rewriter,
    build_system_prompt,
    build_user_text,
    clean_rewrite,
    list_rewriter_models,
)


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


# --- module imports without torch ------------------------------------------------------


def test_module_imports_without_torch() -> None:
    # The whole point of the lazy-import structure: importing the service must not require
    # torch/transformers. (They are imported only inside load/generate.)
    assert importlib.util.find_spec("app.services.rewriter") is not None


# --- model registry + backend filtering ------------------------------------------------


def test_registry_lists_known_models_on_cuda() -> None:
    models = list_rewriter_models(device=_dev(backend="cuda", total_vram_mb=48_000, int4=True))
    ids = {m["model_id"] for m in models}
    assert "Qwen/Qwen3-VL-30B-A3B-Instruct" in ids
    assert "Qwen/Qwen3-VL-8B-Instruct" in ids


def test_cuda_offers_awq_fp8_and_gguf() -> None:
    models = list_rewriter_models(device=_dev(backend="cuda", total_vram_mb=48_000))
    by_id = {m["model_id"]: m for m in models}
    quants_30b = {q["quant"] for q in by_id["Qwen/Qwen3-VL-30B-A3B-Instruct"]["quant_options"]}
    quants_8b = {q["quant"] for q in by_id["Qwen/Qwen3-VL-8B-Instruct"]["quant_options"]}
    assert "awq" in quants_30b and "gguf" in quants_30b
    assert "fp8" in quants_8b and "gguf" in quants_8b


def test_mps_hides_cuda_only_quants_keeps_gguf() -> None:
    models = list_rewriter_models(device=_dev(backend="mps", total_vram_mb=64_000, cc=None))
    for m in models:
        quants = {q["quant"] for q in m["quant_options"]}
        assert "awq" not in quants
        assert "fp8" not in quants
        # GGUF is the portable path and must remain available.
        assert "gguf" in quants


def test_rocm_hides_awq_and_fp8_keeps_gguf() -> None:
    models = list_rewriter_models(device=_dev(backend="rocm", total_vram_mb=48_000))
    for m in models:
        quants = {q["quant"] for q in m["quant_options"]}
        assert "awq" not in quants and "fp8" not in quants
        assert "gguf" in quants


def test_default_model_flagged() -> None:
    # The configured default (Qwen3-VL-8B-Instruct per config.py) is flagged is_default.
    models = list_rewriter_models(device=_dev(backend="cuda", total_vram_mb=48_000))
    defaults = [m for m in models if m["is_default"]]
    assert len(defaults) == 1
    assert defaults[0]["model_id"] == "Qwen/Qwen3-VL-8B-Instruct"


# --- co-reside vs swap advisor ---------------------------------------------------------


def test_8b_co_resides_on_a6000_with_image_model_loaded() -> None:
    # A6000 48GB with a ~20GB image model loaded: the 8B (~9GB) should co-reside.
    dev = _dev(backend="cuda", total_vram_mb=48_000)
    advice = advise_rewriter(
        vl_model="Qwen/Qwen3-VL-8B-Instruct",
        image_model_resident_mb=20_000,
        device=dev,
    )
    assert advice.can_co_reside is True
    assert advice.decision == "co_reside"
    assert "co-resident" in advice.rationale


def test_30b_needs_swap_with_bf16_image_model_resident() -> None:
    # With a bf16 image model resident (~42GB, the §9 footprint) on a 48GB card, the 30B
    # (~22GB) can't fit alongside -> swap, matching the spec's "usually swaps" guidance.
    dev = _dev(backend="cuda", total_vram_mb=48_000)
    advice = advise_rewriter(
        vl_model="Qwen/Qwen3-VL-30B-A3B-Instruct",
        image_model_resident_mb=42_000,
        device=dev,
    )
    assert advice.can_co_reside is False
    assert advice.decision == "swap"
    assert "swap" in advice.rationale


def test_30b_co_resides_on_huge_card() -> None:
    # On an 80GB card with a 20GB image model, even the 30B fits alongside.
    dev = _dev(backend="cuda", total_vram_mb=80_000)
    advice = advise_rewriter(
        vl_model="Qwen/Qwen3-VL-30B-A3B-Instruct",
        image_model_resident_mb=20_000,
        device=dev,
    )
    assert advice.can_co_reside is True
    assert advice.decision == "co_reside"


def test_no_vram_never_co_resides() -> None:
    dev = _dev(backend="cpu", total_vram_mb=0, cc=None, bf16=False)
    advice = advise_rewriter(vl_model="Qwen/Qwen3-VL-8B-Instruct", device=dev)
    assert advice.can_co_reside is False
    assert advice.decision == "swap"


def test_unknown_model_id_gets_default_footprint() -> None:
    dev = _dev(backend="cuda", total_vram_mb=48_000)
    advice = advise_rewriter(vl_model="some/Custom-VL", image_model_resident_mb=0, device=dev)
    assert advice.est_vl_vram_mb > 0
    assert advice.vl_model == "some/Custom-VL"


def test_advice_is_deterministic() -> None:
    dev = _dev(backend="cuda", total_vram_mb=48_000)
    a = advise_rewriter(vl_model="Qwen/Qwen3-VL-8B-Instruct", image_model_resident_mb=15_000,
                        device=dev)
    b = advise_rewriter(vl_model="Qwen/Qwen3-VL-8B-Instruct", image_model_resident_mb=15_000,
                        device=dev)
    assert a == b


# --- prompt assembly -------------------------------------------------------------------


def test_build_system_prompt_edit_uses_edit_template() -> None:
    sp = build_system_prompt(mode=MODE_EDIT, edit_model_id="Qwen/Qwen-Image-Edit-2511")
    assert "Edit Prompt Enhancer" in sp


def test_build_system_prompt_generate_uses_generate_template() -> None:
    sp = build_system_prompt(mode=MODE_GENERATE, edit_model_id=None)
    assert "You are a Prompt optimizer" in sp


def test_build_system_prompt_edit_2512_variant_selected() -> None:
    # 2512 edit base selects the 2512 variant (which falls back to the default edit template,
    # since 2512 ships no EDIT_SYSTEM_PROMPT) — still a valid edit template.
    sp = build_system_prompt(mode=MODE_EDIT, edit_model_id="Qwen/Qwen-Image-Edit-2512")
    assert "Edit Prompt Enhancer" in sp


def test_build_user_text_frames_instruction() -> None:
    text = build_user_text(prompt="  make the sky blue  ")
    assert "User Input: make the sky blue" in text
    assert text.rstrip().endswith("Rewritten Prompt:")


def test_clean_rewrite_strips_fences_and_collapses_newlines() -> None:
    raw = '```json\n{"Rewritten": "..."}\n```'
    cleaned = clean_rewrite(raw)
    assert "```" not in cleaned
    assert "\n" not in cleaned


def test_clean_rewrite_collapses_whitespace() -> None:
    assert clean_rewrite("  a   prompt\nwith\nlines  ") == "a prompt with lines"


def test_clean_rewrite_unwraps_rewritten_json() -> None:
    # The EDIT template makes the VL model answer as {"Rewritten": "..."}; unwrap it.
    assert clean_rewrite('{"Rewritten": "a red apple on a sunny counter"}') == (
        "a red apple on a sunny counter"
    )
    # Also when fenced.
    assert clean_rewrite('```json\n{"Rewritten": "x y z"}\n```') == "x y z"
