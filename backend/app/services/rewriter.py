"""Qwen-VL prompt enhancer service (DESIGN §5.8).

This service owns the **multimodal prompt-rewriting** path that the Qwen-Image team strongly
recommends for stable, high-quality editing. It mirrors :mod:`app.services.pipeline_service`:

- A **config-driven registry** of selectable Qwen3-VL rewriter models with their quant/backend
  options (AWQ / fp8 are CUDA-centric; GGUF is the portable path for ROCm/MPS), filtered to
  what the detected backend supports. The default comes from ``settings.default_rewriter_model``
  and arbitrary HF repo ids are allowed (DESIGN §5.8).
- A **rewriter advisor** (analogous to the image-model advisor in §9) that estimates whether a
  chosen VL model can **co-reside** with the loaded image pipeline given the VRAM the image
  model already occupies, or whether it needs a **runtime swap**, with a one-line trade-off
  rationale and an estimated VRAM figure.
- :meth:`RewriterService.enhance` — genuinely **multimodal** in Edit mode (the VL model *sees*
  the input image(s) alongside the text, which is the whole point of §5.8); enrichment of the
  text prompt in Generate mode. The system prompt is the verbatim official template from
  :mod:`app.services.rewriter_templates`, the variant selected from the chosen edit base.

**torch / transformers are imported lazily inside methods** so this module imports cleanly in
the fast dev/test image (which ships without the inference stack). All pure, deterministic
logic — the model registry, backend filtering, and the co-reside-vs-swap advisor — lives in
module-level functions and is unit-tested without torch; the real VL ``generate`` is exercised
only by the ``@pytest.mark.gpu`` tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.config import get_settings
from app.logging_utils import phase
from app.services.device_manager import DeviceInfo, get_device_info
from app.services.rewriter_templates import (
    get_edit_system_prompt,
    get_generate_system_prompt,
    select_template_variant,
)

if TYPE_CHECKING:  # imports used only for type hints; never executed at runtime
    from PIL.Image import Image as PILImage

log = logging.getLogger(__name__)

# --- Mode constants (mirror pipeline_service; DESIGN §2/§5.8) ---------------------------

MODE_GENERATE = "generate"
MODE_EDIT = "edit"

# Backend-portability classes for a quant option (DESIGN §5.8 table / §4.3).
_BACKENDS_CUDA = ("cuda",)
_BACKENDS_CUDA_ROCM = ("cuda", "rocm")
_BACKENDS_ALL = ("cuda", "rocm", "mps", "cpu")

# Generation budget for the rewrite. A rewrite is short prose, so a modest cap keeps it fast.
_MAX_NEW_TOKENS = 512


# --- Quant / backend registry (DESIGN §5.8) --------------------------------------------


@dataclass(frozen=True)
class QuantOption:
    """One quant/backend combo for a rewriter model (DESIGN §5.8 table)."""

    quant: str  # "awq" | "fp8" | "gguf" | "bf16"
    backends: tuple[str, ...]  # backends this combo runs on
    label: str  # human-friendly description
    requires_runtime: str | None = None  # extra runtime needed (e.g. "gguf"), else None

    def supported_on(self, backend: str) -> bool:
        """Whether this quant runs on the given device backend."""
        return backend in self.backends


@dataclass(frozen=True)
class RewriterModel:
    """A selectable Qwen-VL rewriter model + its quant/backend options (DESIGN §5.8)."""

    model_id: str
    label: str
    # Rough total resident weight footprint in MB *at its lightest portable quant* — used by
    # the co-reside advisor. A3B MoE keeps active params modest but total weights are large.
    est_vram_mb: int
    quant_options: tuple[QuantOption, ...]
    co_resides_typically: bool  # spec guidance: 8B often co-resides; 30B usually swaps
    notes: str = ""

    def options_for_backend(self, backend: str) -> list[QuantOption]:
        """Quant options usable on this backend (DESIGN §5.8 — GGUF portable, AWQ/fp8 CUDA)."""
        return [q for q in self.quant_options if q.supported_on(backend)]


# GGUF is the portable path (CUDA/ROCm/MPS) via a GGUF runtime; AWQ/fp8 are CUDA-centric.
_GGUF = QuantOption(
    quant="gguf",
    backends=_BACKENDS_ALL,
    label="Q4/Q5 GGUF (portable; needs a GGUF runtime)",
    requires_runtime="gguf",
)
_AWQ = QuantOption(quant="awq", backends=_BACKENDS_CUDA, label="AWQ (CUDA)", requires_runtime="awq")
_FP8 = QuantOption(quant="fp8", backends=_BACKENDS_CUDA, label="fp8 (CUDA Ada/Hopper)")

# The known/default rewriter models from the DESIGN §5.8 table. Config-driven default +
# arbitrary HF repo ids are still allowed (handled in :func:`list_rewriter_models`).
_REGISTRY: tuple[RewriterModel, ...] = (
    RewriterModel(
        model_id="Qwen/Qwen3-VL-30B-A3B-Instruct",
        label="Qwen3-VL-30B-A3B-Instruct",
        est_vram_mb=22_000,  # ~Q4 GGUF / AWQ resident footprint of the large MoE weights
        quant_options=(_AWQ, _GGUF),
        co_resides_typically=False,
        notes=(
            "Best-quality rewrites; A3B MoE keeps active params modest but total weights are "
            "large — usually requires a runtime swap on a single GPU."
        ),
    ),
    RewriterModel(
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        label="Qwen3-VL-8B-Instruct",
        est_vram_mb=9_000,  # fp8 / GGUF resident footprint of the 8B dense weights
        quant_options=(_FP8, _GGUF),
        co_resides_typically=True,
        notes=(
            "Strong fallback that often co-resides alongside the loaded image model → no swap, "
            "low latency for instant enhancement."
        ),
    ),
)

_REGISTRY_BY_ID: dict[str, RewriterModel] = {m.model_id: m for m in _REGISTRY}

# Fallback footprint for an arbitrary (non-registry) HF repo id the advisor is asked about.
_UNKNOWN_MODEL_VRAM_MB = 16_000


# --- Public registry helpers (pure; unit-tested) ---------------------------------------


def list_rewriter_models(*, device: DeviceInfo | None = None) -> list[dict[str, Any]]:
    """List selectable VL rewriter models + quant/backend options, filtered by device (§5.8).

    Each entry exposes the model id/label, its estimated footprint, whether it is the default
    (``settings.default_rewriter_model``), the spec's co-reside guidance, and the quant options
    available *on the detected backend* (AWQ/fp8 hidden off CUDA; GGUF kept as the portable
    path). Arbitrary HF repo ids are allowed at run time; this lists the known/configured set.

    Args:
        device: Device to filter quant options against; defaults to the detected device.

    Returns:
        A list of plain dicts (router-serializable), default model first.
    """
    dev = device if device is not None else get_device_info()
    default_id = get_settings().default_rewriter_model

    entries: list[dict[str, Any]] = []
    for model in _REGISTRY:
        opts = model.options_for_backend(dev.backend)
        entries.append(
            {
                "model_id": model.model_id,
                "label": model.label,
                "est_vram_mb": model.est_vram_mb,
                "is_default": model.model_id == default_id,
                "co_resides_typically": model.co_resides_typically,
                "notes": model.notes,
                "quant_options": [
                    {
                        "quant": q.quant,
                        "label": q.label,
                        "backends": list(q.backends),
                        "requires_runtime": q.requires_runtime,
                    }
                    for q in opts
                ],
            }
        )

    # Surface the configured default first even if it isn't in the static registry (arbitrary
    # repo id allowed): synthesize a minimal entry so the UI can preselect it.
    if default_id not in _REGISTRY_BY_ID:
        entries.insert(
            0,
            {
                "model_id": default_id,
                "label": default_id,
                "est_vram_mb": _UNKNOWN_MODEL_VRAM_MB,
                "is_default": True,
                "co_resides_typically": False,
                "notes": "Custom rewriter repo id (not in the known registry).",
                "quant_options": [
                    {
                        "quant": _GGUF.quant,
                        "label": _GGUF.label,
                        "backends": list(_GGUF.backends),
                        "requires_runtime": _GGUF.requires_runtime,
                    }
                ],
            },
        )
    else:
        entries.sort(key=lambda e: (not e["is_default"], e["model_id"]))
    return entries


def _registry_vram_mb(vl_model: str) -> int:
    """Estimated resident VRAM (MB) for a rewriter model id (registry or arbitrary)."""
    model = _REGISTRY_BY_ID.get(vl_model)
    return model.est_vram_mb if model is not None else _UNKNOWN_MODEL_VRAM_MB


# --- Rewriter advisor: co-reside vs swap (DESIGN §5.8, analogous to §9) ------------------


@dataclass(frozen=True)
class RewriterAdvice:
    """Co-reside-vs-swap advice for a rewriter model on the current device (DESIGN §5.8)."""

    vl_model: str
    can_co_reside: bool
    decision: str  # "co_reside" | "swap"
    est_vl_vram_mb: int
    image_model_resident_mb: int
    free_after_image_mb: int
    rationale: str


def advise_rewriter(
    *,
    vl_model: str,
    image_model_resident_mb: int = 0,
    device: DeviceInfo | None = None,
) -> RewriterAdvice:
    """Estimate whether ``vl_model`` can co-reside with the loaded image pipeline (DESIGN §5.8).

    The decision is a transparent analytic model (no torch, no probing) so it is deterministic
    and unit-testable: the VL model **co-resides** when its estimated footprint fits in the VRAM
    left free *after* the image model (plus a small safety margin); otherwise it needs a
    **runtime swap** (unload the image model → load VL → reload), serialized through the GPU
    queue. The rationale surfaces the trade-off in the spec's terms (better rewrites vs. an
    added load/unload).

    Args:
        vl_model: The chosen rewriter HF repo id (registry footprint used, else a default).
        image_model_resident_mb: VRAM the loaded image pipeline currently occupies (its
            precision + LoRAs folded in by the caller, e.g. from the §9 advisor's peak).
        device: Device to score against; defaults to the detected device (refreshed for
            current free-VRAM). Pass a synthetic :class:`DeviceInfo` for deterministic tests.

    Returns:
        A :class:`RewriterAdvice` with the decision, estimated figures, and a one-line rationale.
    """
    dev = device if device is not None else get_device_info(refresh=True)
    vl_mb = _registry_vram_mb(vl_model)
    resident = max(int(image_model_resident_mb), 0)

    # Budget: total VRAM minus what the image model already holds, with a small safety margin
    # for activations/KV-cache so a "co-reside" verdict isn't razor-thin.
    safety_margin_mb = 2_000
    free_after_image = dev.total_vram_mb - resident
    can_co_reside = free_after_image - vl_mb >= safety_margin_mb and dev.total_vram_mb > 0

    if can_co_reside:
        decision = "co_reside"
        rationale = (
            f"{vl_model} (~{vl_mb} MB) fits alongside the loaded image model "
            f"(~{resident} MB of {dev.total_vram_mb} MB) — co-resident, instant rewrites, no "
            f"load/unload churn."
        )
    else:
        decision = "swap"
        rationale = (
            f"{vl_model} (~{vl_mb} MB) does not fit alongside the image model "
            f"(~{resident} MB of {dev.total_vram_mb} MB free VRAM) — a runtime swap "
            f"(unload image model → load VL → reload) is needed each run; better rewrites at "
            f"the cost of added load/unload latency."
        )

    advice = RewriterAdvice(
        vl_model=vl_model,
        can_co_reside=can_co_reside,
        decision=decision,
        est_vl_vram_mb=vl_mb,
        image_model_resident_mb=resident,
        free_after_image_mb=free_after_image,
        rationale=rationale,
    )
    log.info(
        "Rewriter advisor: model=%s backend=%s total_vram=%dMB image_resident=%dMB vl=%dMB "
        "-> %s",
        vl_model,
        dev.backend,
        dev.total_vram_mb,
        resident,
        vl_mb,
        decision,
    )
    return advice


# --- Pure prompt assembly (unit-tested without torch) ----------------------------------


def build_system_prompt(*, mode: str, edit_model_id: str | None) -> str:
    """Return the verbatim official system prompt for the run (DESIGN §5.8).

    Edit mode uses ``EDIT_SYSTEM_PROMPT`` (variant matched to the chosen edit base);
    Generate mode uses ``SYSTEM_PROMPT`` (the default text-to-image enrichment template).

    Args:
        mode: ``"generate"`` or ``"edit"``.
        edit_model_id: The chosen edit base model id (selects the 2512 vs default variant).

    Returns:
        The exact upstream template string.
    """
    if mode == MODE_EDIT:
        variant = select_template_variant(edit_model_id=edit_model_id or "")
        return get_edit_system_prompt(variant=variant)
    return get_generate_system_prompt()


def build_user_text(*, prompt: str) -> str:
    """Wrap the user's instruction the way the upstream tool frames it (DESIGN §5.8).

    Mirrors the vendored tools' ``User Input: ... \\n\\n Rewritten Prompt:`` framing so the
    model's continuation is the rewrite, not a conversational reply.
    """
    return f"User Input: {prompt.strip()}\n\nRewritten Prompt:"


def clean_rewrite(text: str) -> str:
    """Normalize the model's raw rewrite into a single clean prompt string (DESIGN §5.8).

    Strips whitespace and any stray ```json fences / "Rewritten" wrappers the templates can
    elicit, collapsing internal newlines into spaces as the upstream tools do.
    """
    out = text.strip()
    # Drop code fences the EDIT template's JSON example can elicit.
    out = out.replace("```json", "").replace("```", "").strip()
    out = out.replace("\r\n", "\n").replace("\n", " ")
    return " ".join(out.split())


# --- The service -----------------------------------------------------------------------


@dataclass
class _ResidentVL:
    """A loaded VL model + its processor, keyed for reuse (DESIGN §5.8)."""

    model_id: str
    model: Any
    processor: Any


class RewriterService:
    """Loads and runs the Qwen-VL rewriter, caching one resident VL model between calls.

    Single-accelerator: model loads/unloads and the runtime swap with the image pipeline are
    **serialized through the GPU queue** by the job manager (this service only loads/unloads the
    VL model and runs ``generate``; deciding *when* to swap, and unloading the image pipeline,
    is the job manager's concern). The resident VL model is reused while the model id is
    unchanged, mirroring :class:`~app.services.pipeline_service.PipelineService`.
    """

    def __init__(self) -> None:
        self._resident: _ResidentVL | None = None

    # -- loading ------------------------------------------------------------------------

    def load(self, *, vl_model: str, device: str | None = None) -> None:
        """Load (or reuse) the resident VL model + processor for ``vl_model`` (DESIGN §5.8).

        Reuses the resident model when the id is unchanged. torch/transformers are imported
        lazily here so the module stays importable without the inference stack.

        Args:
            vl_model: The rewriter HF repo id.
            device: Optional target device string; defaults to the detected device's
                ``device_str`` (``device_map="auto"`` handles placement when omitted).
        """
        if self._resident is not None and self._resident.model_id == vl_model:
            log.info("Reusing resident VL rewriter %s", vl_model)
            return

        self._unload()
        with phase(
            log,
            f"Loading VL rewriter {vl_model} — first load downloads/initializes weights and "
            "can take a while",
        ):
            model, processor = self._build_vl(vl_model=vl_model, device=device)
        self._resident = _ResidentVL(model_id=vl_model, model=model, processor=processor)

    def _build_vl(self, *, vl_model: str, device: str | None) -> tuple[Any, Any]:
        """Construct the VL model + processor via transformers (lazy import)."""
        import torch  # lazy: torch absent in the dev/test image
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(vl_model, trust_remote_code=True)
        with phase(log, f"from_pretrained({vl_model}) VL weights"):
            model = AutoModelForImageTextToText.from_pretrained(
                vl_model,
                torch_dtype=torch.bfloat16,
                device_map="auto" if device is None else None,
                trust_remote_code=True,
            )
        if device is not None:
            model = model.to(device)
        model.eval()
        return model, processor

    def unload(self) -> None:
        """Release the resident VL model and free accelerator memory (DESIGN §5.8 swap path)."""
        self._unload()

    def _unload(self) -> None:
        if self._resident is None:
            return
        log.info("Unloading resident VL rewriter %s", self._resident.model_id)
        self._resident = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — best-effort; torch may be absent
            pass

    # -- enhancement --------------------------------------------------------------------

    def enhance(
        self,
        *,
        mode: str,
        prompt: str,
        images: list[PILImage] | None = None,
        vl_model: str | None = None,
        edit_model_id: str | None = None,
    ) -> str:
        """Rewrite ``prompt`` via the VL model, multimodally in Edit mode (DESIGN §5.8).

        Builds the chat input as ``system`` = the verbatim official template (variant matched
        to ``edit_model_id``) and ``user`` content = the input image(s) **+** the user's
        instruction, applies the processor's Qwen3-VL chat template, and runs ``generate``. In
        Edit mode the image(s) are passed to the VL model (the whole point of §5.8 — the rewrite
        is grounded in what's actually in the picture). The raw output is cleaned into a single
        prompt string and returned; the caller shows it as an editable diff (never silent) and
        stores both prompts in job metadata.

        Args:
            mode: ``"generate"`` or ``"edit"``.
            prompt: The user's original instruction.
            images: Ordered input PIL images (Edit mode); order is model-significant.
            vl_model: Rewriter repo id; defaults to ``settings.default_rewriter_model``.
            edit_model_id: The chosen edit base id (selects the template variant).

        Returns:
            The rewritten prompt (stripped/cleaned).

        Raises:
            ValueError: If the mode is unknown.
        """
        if mode not in (MODE_GENERATE, MODE_EDIT):
            raise ValueError(f"Unknown mode: {mode!r}")

        chosen = vl_model or get_settings().default_rewriter_model
        system_prompt = build_system_prompt(mode=mode, edit_model_id=edit_model_id)
        user_text = build_user_text(prompt=prompt)
        pil_images = list(images or [])

        self.load(vl_model=chosen)
        assert self._resident is not None  # load() guarantees this
        model = self._resident.model
        processor = self._resident.processor

        messages = self._build_messages(
            system_prompt=system_prompt,
            user_text=user_text,
            images=pil_images,
            include_images=mode == MODE_EDIT,
        )

        with phase(log, f"VL rewrite ({mode}, {chosen}, {len(pil_images)} image(s))"):
            raw = self._generate(model=model, processor=processor, messages=messages,
                                 images=pil_images if mode == MODE_EDIT else [])

        rewritten = clean_rewrite(raw)
        log.info("Rewrite produced %d chars (mode=%s)", len(rewritten), mode)
        return rewritten

    def _build_messages(
        self,
        *,
        system_prompt: str,
        user_text: str,
        images: list[PILImage],
        include_images: bool,
    ) -> list[dict[str, Any]]:
        """Assemble the Qwen3-VL chat messages (system + multimodal user content).

        In Edit mode each input image is added as an ``{"type": "image"}`` content part ahead
        of the instruction text (order preserved), so the VL model sees the picture(s) it is
        rewriting against (DESIGN §5.8). Kept torch-free for reuse/testing of the structure.
        """
        user_content: list[dict[str, Any]] = []
        if include_images:
            for image in images:
                user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": user_text})
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _generate(
        self,
        *,
        model: Any,
        processor: Any,
        messages: list[dict[str, Any]],
        images: list[PILImage],
    ) -> str:
        """Run the VL chat template + ``generate`` and decode the rewrite (lazy torch)."""
        import torch

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            images=images or None,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS, do_sample=False)

        # Strip the prompt tokens so only the continuation (the rewrite) is decoded.
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated, strict=False)]
        decoded = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return decoded[0] if decoded else ""


# --- Process singleton -----------------------------------------------------------------

_service: RewriterService | None = None


def get_rewriter_service() -> RewriterService:
    """Return the process-wide :class:`RewriterService` (created lazily)."""
    global _service
    if _service is None:
        _service = RewriterService()
    return _service
