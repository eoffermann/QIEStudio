"""Reproducibility metadata assembly (DESIGN §5.5 / §6, RUN §6).

Every run records the full set of facts needed to re-create it: mode, model id/revision,
precision/quant, device, resolution, seed, LoRAs + weights, input image hashes, and **both
the original and the enhanced prompt**. The :class:`PipelineService` produces the run; this
module turns the run parameters into a plain, JSON-serializable ``dict`` plus a set of PNG
text chunks. The ``asset_store`` (a separate service) embeds these chunks in the output PNG
and persists the dict alongside the output — this module stays **pure** so it is fully
unit-testable without torch, diffusers, or any I/O.
"""

from __future__ import annotations

import json
from typing import Any

# Schema version for the metadata blob, so a future reader can branch on layout changes.
METADATA_SCHEMA_VERSION = 1

# The PNG text-chunk key under which the full JSON metadata blob is embedded. The
# individual flattened keys below are written too, for human-readable inspection in tools
# that surface PNG text chunks (e.g. exiftool).
PNG_PARAMETERS_KEY = "qie_metadata"


def _normalize_loras(loras: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Return a clean, ordered list of ``{lora_id?, adapter_name?, weight, path?}`` entries.

    Accepts the run-time LoRA descriptors used by the pipeline (``{"path", "weight",
    "adapter_name"}``) as well as the saved-prompt form (``{"lora_id", "weight"}``) and
    keeps whichever identifying fields are present. ``weight`` is always coerced to float.
    """
    out: list[dict[str, Any]] = []
    for entry in loras or []:
        item: dict[str, Any] = {}
        for key in ("lora_id", "adapter_name", "path"):
            value = entry.get(key)
            if value is not None:
                item[key] = value
        item["weight"] = float(entry.get("weight", 1.0))
        out.append(item)
    return out


def build_metadata(
    *,
    mode: str,
    model_id: str,
    model_revision: str | None,
    precision: str,
    device: str,
    prompt: str,
    enhanced_prompt: str | None,
    negative_prompt: str,
    width: int | None,
    height: int | None,
    steps: int,
    true_cfg_scale: float,
    guidance_scale: float,
    seed: int | None,
    loras: list[dict[str, Any]] | None,
    input_hashes: list[str] | None,
    rewriter_model: str | None,
    output_format: str,
) -> dict[str, Any]:
    """Assemble the full reproducibility metadata dict for a single output (DESIGN §5.5).

    The returned dict is JSON-serializable and captures everything required to re-run the
    job: mode, model id/revision, precision/quant, device, both prompts, negative prompt,
    resolution, sampler settings, seed, LoRAs + weights, ordered input image hashes (Edit
    mode), the rewriter model used for enhancement (if any), and the output format.

    Args:
        mode: ``"generate"`` or ``"edit"``.
        model_id: HF repo id of the image model that produced the output.
        model_revision: Pinned revision/commit of the image model, or ``None``.
        precision: ``"bf16"`` | ``"fp8"`` | ``"int4"`` — the precision/quant path used.
        device: Human-readable device string (e.g. ``"cuda:0 (NVIDIA RTX A6000, SM 8.6)"``).
        prompt: The original, user-authored prompt.
        enhanced_prompt: The Qwen-VL rewritten prompt actually fed to the model, or ``None``
            when enhancement was not used.
        negative_prompt: Negative prompt (``" "`` — a single space — by default).
        width: Output width in pixels, or ``None`` for "match source" / model default.
        height: Output height in pixels, or ``None``.
        steps: ``num_inference_steps``.
        true_cfg_scale: ``true_cfg_scale`` used.
        guidance_scale: ``guidance_scale`` used.
        seed: The integer seed, or ``None`` if a random seed was used and not captured.
        loras: Active LoRA descriptors with weights.
        input_hashes: Ordered sha256 hashes of the input images (Edit mode); order is
            model-significant and preserved here.
        rewriter_model: The Qwen-VL model id used to enhance the prompt, or ``None``.
        output_format: ``"png"`` | ``"webp"`` | ``"jpeg"``.

    Returns:
        A JSON-serializable metadata dict.
    """
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "app": "QIE Studio",
        "mode": mode,
        "model_id": model_id,
        "model_revision": model_revision,
        "precision": precision,
        "device": device,
        "prompt": prompt,
        "enhanced_prompt": enhanced_prompt,
        "negative_prompt": negative_prompt,
        "rewriter_model": rewriter_model,
        "resolution": {"width": width, "height": height},
        "sampler": {
            "num_inference_steps": int(steps),
            "true_cfg_scale": float(true_cfg_scale),
            "guidance_scale": float(guidance_scale),
        },
        "seed": seed,
        "loras": _normalize_loras(loras),
        "input_hashes": list(input_hashes or []),
        "output_format": output_format,
    }


def metadata_to_png_text(meta: dict[str, Any]) -> dict[str, str]:
    """Render a metadata dict into PngInfo-ready text chunks (all values are ``str``).

    The full blob is JSON-encoded under :data:`PNG_PARAMETERS_KEY` so a reader can recover
    the exact structured metadata. A handful of high-value fields are *also* flattened into
    their own human-readable chunks for tools that display PNG text without parsing JSON.

    Args:
        meta: A dict as returned by :func:`build_metadata`.

    Returns:
        A ``dict[str, str]`` suitable for ``PIL.PngImagePlugin.PngInfo.add_text``.
    """
    chunks: dict[str, str] = {PNG_PARAMETERS_KEY: json.dumps(meta, ensure_ascii=False)}

    resolution = meta.get("resolution") or {}
    sampler = meta.get("sampler") or {}

    flat: dict[str, Any] = {
        "qie_mode": meta.get("mode"),
        "qie_model_id": meta.get("model_id"),
        "qie_model_revision": meta.get("model_revision"),
        "qie_precision": meta.get("precision"),
        "qie_device": meta.get("device"),
        "qie_prompt": meta.get("prompt"),
        "qie_enhanced_prompt": meta.get("enhanced_prompt"),
        "qie_negative_prompt": meta.get("negative_prompt"),
        "qie_rewriter_model": meta.get("rewriter_model"),
        "qie_width": resolution.get("width"),
        "qie_height": resolution.get("height"),
        "qie_steps": sampler.get("num_inference_steps"),
        "qie_true_cfg_scale": sampler.get("true_cfg_scale"),
        "qie_guidance_scale": sampler.get("guidance_scale"),
        "qie_seed": meta.get("seed"),
        "qie_output_format": meta.get("output_format"),
    }
    for key, value in flat.items():
        if value is not None:
            chunks[key] = str(value)

    loras = meta.get("loras") or []
    if loras:
        chunks["qie_loras"] = json.dumps(loras, ensure_ascii=False)
    input_hashes = meta.get("input_hashes") or []
    if input_hashes:
        chunks["qie_input_hashes"] = ",".join(str(h) for h in input_hashes)

    return chunks
