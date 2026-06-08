"""Pipeline service — load & run the Generate and Edit pipelines (DESIGN §2, §5.5, §9).

This service owns the heavy inference path:

- Loads :class:`~diffusers.QwenImagePipeline` (Generate, text-to-image) and
  :class:`~diffusers.QwenImageEditPlusPipeline` (Edit, ordered multi-image + instruction).
- Implements the three precision paths — **bf16**, **fp8_e4m3fn (scaled)** via
  optimum-quanto, and **Nunchaku SVDQuant int4** (CUDA-only) — with backend guards
  (DESIGN §9.1).
- Stacks external LoRAs with per-adapter weights via ``load_lora_weights`` + PEFT
  ``set_adapters`` and unloads them between jobs (DESIGN §5.3).
- Streams per-step progress and **throttled latent→RGB live previews** through a
  ``callback_on_step_end`` hook (DESIGN §5.5).
- Keeps the active pipeline **resident** between jobs and reuses it when the load key is
  unchanged (DESIGN §9.3).

**torch / diffusers are imported lazily inside methods** so this module imports cleanly in
the fast dev/test image, which ships without the inference stack. All pure, testable logic
(precision→dtype mapping, pipeline-kwarg assembly, preview-throttle decision) lives in
module-level functions with no torch dependency; the heavy model execution is exercised only
by the ``@pytest.mark.gpu`` tests.
"""

from __future__ import annotations

import contextlib
import io
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.logging_utils import phase

if TYPE_CHECKING:  # imports used only for type hints; never executed at runtime
    from collections.abc import Callable

log = logging.getLogger(__name__)

# --- Model / mode constants (DESIGN §2) -----------------------------------------------

MODE_GENERATE = "generate"
MODE_EDIT = "edit"

PRECISION_BF16 = "bf16"
PRECISION_FP8 = "fp8"
PRECISION_INT4 = "int4"

# Reference inference parameters for the Edit model (CLAUDE.md / DESIGN §2.1). Used as the
# verbatim defaults for an Edit run unless a RunRequest overrides them.
EDIT_DEFAULT_STEPS = 40
EDIT_DEFAULT_TRUE_CFG_SCALE = 4.0
EDIT_DEFAULT_GUIDANCE_SCALE = 1.0
EDIT_DEFAULT_NEGATIVE_PROMPT = " "  # a single space — the documented default
EDIT_DEFAULT_NUM_IMAGES_PER_PROMPT = 1

# Decode the live preview at a reduced size to keep the throttled VAE decode cheap.
_PREVIEW_MAX_EDGE = 256

# Default Nunchaku SVDQuant sources per base model (DESIGN §9.1 / §9.4). A single
# pre-quantized transformer safetensors replaces the base DiT. ``{prec}`` is filled from
# nunchaku's get_precision() (int4 on Ampere/Ada, fp4 on Blackwell); ``{rank}`` is the
# SVDQuant rank (128=quality, 32=speed). Override/extend via QIE_NUNCHAKU_QUANT_SOURCES.
_DEFAULT_NUNCHAKU_SOURCES: dict[str, str] = {
    "Qwen/Qwen-Image": (
        "nunchaku-tech/nunchaku-qwen-image/svdq-{prec}_r{rank}-qwen-image.safetensors"
    ),
    "Qwen/Qwen-Image-Edit": (
        "nunchaku-tech/nunchaku-qwen-image-edit/"
        "svdq-{prec}_r{rank}-qwen-image-edit.safetensors"
    ),
    "Qwen/Qwen-Image-Edit-2509": (
        "nunchaku-tech/nunchaku-qwen-image-edit-2509/"
        "svdq-{prec}_r{rank}-qwen-image-edit-2509.safetensors"
    ),
    # Community SVDQuant of Edit-2511 (official weights pending upstream issue #858). The
    # 2511 filenames use a tier suffix rather than rank, so {rank} is unused here.
    "Qwen/Qwen-Image-Edit-2511": (
        "QuantFunc/Nunchaku-Qwen-Image-EDIT-2511/"
        "nunchaku_qwen_image_edit_2511_balance_{prec}.safetensors"
    ),
}


# --- Data transfer objects -------------------------------------------------------------


@dataclass
class RunRequest:
    """A fully-resolved request for one inference run.

    The job service resolves prompt slots, LoRA paths, and the concrete resolution before
    constructing this; the pipeline service does not touch the DB or storage.
    """

    mode: str  # "generate" | "edit"
    model_id: str
    model_revision: str | None
    precision: str  # "bf16" | "fp8" | "int4"
    device: str  # "cuda" | "cuda:0" | "mps" | "cpu"
    prompt: str
    negative_prompt: str = EDIT_DEFAULT_NEGATIVE_PROMPT
    width: int | None = None
    height: int | None = None
    num_inference_steps: int = EDIT_DEFAULT_STEPS
    true_cfg_scale: float = EDIT_DEFAULT_TRUE_CFG_SCALE
    guidance_scale: float = EDIT_DEFAULT_GUIDANCE_SCALE
    seed: int | None = None
    batch: int = 1
    images: list[Any] = field(default_factory=list)  # ordered PIL images (Edit mode)
    loras: list[dict[str, Any]] = field(default_factory=list)  # {path, weight, adapter_name}
    preview_every_n_steps: int = 5


@dataclass
class StepProgress:
    """Per-step progress payload streamed to the WebSocket layer (DESIGN §5.5)."""

    step: int  # 1-based completed step index
    total: int  # total scheduled steps
    preview_png: bytes | None  # small RGB preview PNG on the throttled cadence, else None


@dataclass
class LoadOptions:
    """Optional offloading / memory toggles applied at load time (DESIGN §9.3)."""

    enable_model_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    enable_attention_slicing: bool = False
    enable_vae_tiling: bool = False


# --- Pure logic (unit-tested without torch) --------------------------------------------


def precision_to_dtype_name(precision: str) -> str:
    """Map a precision label to the torch dtype name used for ``torch_dtype``.

    All three precision paths load the base weights in **bfloat16**; fp8 and int4 then apply
    quantization on top (so the compute/storage dtype is bf16 in every case). Returns the
    dtype *name* (not the torch object) so this stays torch-free and unit-testable.

    Args:
        precision: ``"bf16"`` | ``"fp8"`` | ``"int4"``.

    Returns:
        ``"bfloat16"``.

    Raises:
        ValueError: If the precision label is unknown.
    """
    if precision not in (PRECISION_BF16, PRECISION_FP8, PRECISION_INT4):
        raise ValueError(f"Unknown precision: {precision!r}")
    return "bfloat16"


def validate_precision_for_backend(precision: str, *, backend: str, fp8_native: bool) -> None:
    """Validate that a precision path is usable on the detected backend (DESIGN §9.1).

    The advisor (§9) should already filter unavailable options, but the pipeline guards
    again so a bad request fails fast with a clear, actionable error rather than mid-run.

    Args:
        precision: ``"bf16"`` | ``"fp8"`` | ``"int4"``.
        backend: ``"cuda"`` | ``"rocm"`` | ``"mps"`` | ``"cpu"``.
        fp8_native: Whether the device supports native fp8_e4m3fn (SM >= 8.9). When CUDA but
            not native, fp8 is allowed but runs **emulated** (no speedup) — logged, not blocked.

    Raises:
        ValueError: If the precision is incompatible with the backend (e.g. int4 off CUDA).
    """
    if precision == PRECISION_BF16:
        return
    if precision == PRECISION_INT4:
        if backend != "cuda":
            raise ValueError(
                "int4 (Nunchaku SVDQuant) is CUDA-only and unavailable on "
                f"backend={backend!r}."
            )
        return
    if precision == PRECISION_FP8:
        if backend not in ("cuda", "rocm"):
            raise ValueError(
                f"fp8_e4m3fn requires a CUDA/ROCm backend; got backend={backend!r}."
            )
        if backend == "cuda" and not fp8_native:
            log.warning(
                "fp8_e4m3fn requested on CUDA without native SM>=8.9 support — it will run "
                "EMULATED (functional, no speedup)."
            )
        return
    raise ValueError(f"Unknown precision: {precision!r}")


def build_pipeline_kwargs(req: RunRequest) -> dict[str, Any]:
    """Assemble the keyword arguments passed to the diffusers pipeline ``__call__``.

    Excludes the ``generator`` and ``callback_on_step_end`` (constructed at run time with
    torch) and the ``image=`` list (added separately for Edit mode). Width/height are only
    included when both are set — the Edit pipeline derives size from the input images when
    they are omitted (DESIGN §5.4 "match source").

    Args:
        req: The resolved run request.

    Returns:
        A kwargs dict ready to be merged with the run-time torch objects.
    """
    kwargs: dict[str, Any] = {
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt,
        "num_inference_steps": int(req.num_inference_steps),
        "true_cfg_scale": float(req.true_cfg_scale),
        "guidance_scale": float(req.guidance_scale),
        "num_images_per_prompt": int(req.batch),
    }
    if req.width is not None and req.height is not None:
        kwargs["width"] = int(req.width)
        kwargs["height"] = int(req.height)
    return kwargs


def should_emit_preview(step: int, total: int, every_n: int) -> bool:
    """Decide whether to decode a live preview after a completed step (DESIGN §5.5).

    Emits on the throttled cadence and always on the final step, and never when previews are
    disabled (``every_n <= 0``).

    Args:
        step: 1-based index of the just-completed step.
        total: Total scheduled steps.
        every_n: Throttle cadence; ``<= 0`` disables previews entirely.

    Returns:
        ``True`` to decode + emit a preview for this step.
    """
    if every_n <= 0:
        return False
    if step >= total:
        return True
    return step % every_n == 0


def make_load_key(*, mode: str, model_id: str, model_revision: str | None, precision: str,
                  device: str) -> tuple[str, str, str | None, str, str]:
    """Build the cache key identifying a resident pipeline (DESIGN §9.3).

    Two loads with the same key reuse the resident pipeline; a change in any component
    triggers a reload.
    """
    return (mode, model_id, model_revision, precision, device)


# --- The service -----------------------------------------------------------------------


class PipelineService:
    """Loads and runs the Generate/Edit pipelines, keeping one resident between jobs.

    Single-accelerator: the job queue guarantees one run at a time (DESIGN §9.3), so this
    holds a single resident pipeline keyed by ``(mode, model_id, revision, precision,
    device)`` and reloads only when the key changes.
    """

    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._load_key: tuple[str, str, str | None, str, str] | None = None
        self._active_adapters: list[str] = []
        # Forward-hook handles for the manual (SVDQuant/int4) LoRA path; removed on reset.
        self._lora_hooks: list[Any] = []

    # -- loading ------------------------------------------------------------------------

    def load(
        self,
        *,
        mode: str,
        model_id: str,
        model_revision: str | None,
        precision: str,
        device: str,
        options: LoadOptions | None = None,
    ) -> None:
        """Load (or reuse) the resident pipeline for the given configuration.

        Reuses the resident pipeline when the load key is unchanged. Otherwise it constructs
        the appropriate pipeline for the mode, applies the precision path (bf16 / fp8 / int4),
        moves it to the device, and applies any offloading toggles.

        Args:
            mode: ``"generate"`` or ``"edit"``.
            model_id: HF repo id of the image model.
            model_revision: Pinned revision/commit, or ``None`` for the default.
            precision: ``"bf16"`` | ``"fp8"`` | ``"int4"``.
            device: Target device string (``"cuda"`` / ``"cuda:0"`` / ``"mps"`` / ``"cpu"``).
            options: Optional offloading / memory toggles.

        Raises:
            ValueError: If the mode is unknown or the precision is unusable on the backend.
            NotImplementedError: If an int4 (Nunchaku) load is requested but unavailable.
        """
        key = make_load_key(
            mode=mode,
            model_id=model_id,
            model_revision=model_revision,
            precision=precision,
            device=device,
        )
        if self._load_key == key and self._pipeline is not None:
            log.info("Reusing resident %s pipeline (%s, %s)", mode, model_id, precision)
            return

        if mode not in (MODE_GENERATE, MODE_EDIT):
            raise ValueError(f"Unknown mode: {mode!r}")

        backend = _backend_of_device(device)
        fp8_native = _fp8_native_for(device)
        validate_precision_for_backend(precision, backend=backend, fp8_native=fp8_native)

        # Drop any prior pipeline first so we don't briefly hold two in VRAM.
        self._unload()

        opts = options or LoadOptions()
        with phase(
            log,
            f"Loading {mode} pipeline {model_id} ({precision}) on {device} — "
            "first load downloads/initializes weights and can take a while",
        ):
            pipeline = self._build_pipeline(
                mode=mode,
                model_id=model_id,
                model_revision=model_revision,
                precision=precision,
                device=device,
                options=opts,
            )

        self._apply_load_options(pipeline, device=device, options=opts)

        self._pipeline = pipeline
        self._load_key = key
        self._active_adapters = []

    def _build_pipeline(
        self,
        *,
        mode: str,
        model_id: str,
        model_revision: str | None,
        precision: str,
        device: str,
        options: LoadOptions,
    ) -> Any:
        """Construct the diffusers pipeline for ``mode`` with the chosen precision path.

        Device placement is deferred to the caller's offload decision: when CPU offload is
        enabled, the pipeline is left on CPU so accelerate can manage per-module movement
        (calling ``.to(cuda)`` first would defeat offload and double the peak memory).
        """
        import torch  # lazy: torch absent in the dev/test image
        from diffusers import QwenImageEditPlusPipeline, QwenImagePipeline

        dtype = getattr(torch, precision_to_dtype_name(precision))
        pipeline_cls = QwenImageEditPlusPipeline if mode == MODE_EDIT else QwenImagePipeline
        offload = options.enable_model_cpu_offload or options.enable_sequential_cpu_offload

        if precision == PRECISION_INT4:
            return self._build_int4_pipeline(
                pipeline_cls=pipeline_cls,
                model_id=model_id,
                model_revision=model_revision,
                dtype=dtype,
                device=device,
                offload=offload,
            )

        with phase(log, f"from_pretrained({model_id}, {precision})"):
            pipeline = pipeline_cls.from_pretrained(
                model_id,
                revision=model_revision,
                torch_dtype=dtype,
            )

        if precision == PRECISION_FP8:
            self._quantize_fp8(pipeline)

        # Only place on the accelerator when NOT offloading (offload manages placement).
        if not offload and device != "cpu":
            pipeline = pipeline.to(device)
        return pipeline

    def _quantize_fp8(self, pipeline: Any) -> None:
        """Quantize the transformer (and text encoder if present) to fp8_e4m3fn (scaled).

        Uses optimum-quanto's ``quantize`` + ``freeze`` (scaled fp8). On Ampere (SM 8.6) this
        is emulated — functional but with no speedup (already warned in
        :func:`validate_precision_for_backend`).
        """
        with phase(log, "Quantizing transformer to fp8_e4m3fn (optimum-quanto, scaled)"):
            from optimum.quanto import freeze, qfloat8, quantize

            transformer = getattr(pipeline, "transformer", None)
            if transformer is None:
                raise ValueError("Pipeline exposes no `transformer` to quantize for fp8.")
            quantize(transformer, weights=qfloat8)
            freeze(transformer)

    def _build_int4_pipeline(
        self,
        *,
        pipeline_cls: Any,
        model_id: str,
        model_revision: str | None,
        dtype: Any,
        device: str,
        offload: bool = False,
    ) -> Any:
        """Load a Nunchaku **SVDQuant int4** pipeline (CUDA-only) (DESIGN §9.1).

        Nunchaku ships a pre-quantized SVDQuant transformer (a single ``.safetensors``) that
        replaces the base DiT; only the text encoder + VAE are fetched from the base repo, so
        the heavy ~40 GB bf16 transformer is never downloaded/loaded. ``get_precision()``
        picks **int4** (Ampere/Ada/most GPUs) vs **fp4** (Blackwell/RTX 50xx). On low-VRAM
        cards the transformer enables block offload + sequential CPU offload so it runs on
        gaming/workstation GPUs — the whole point of this path.

        Raises a clear :class:`NotImplementedError` only when nunchaku itself is unavailable
        or no quant source is mapped for ``model_id`` (so the advisor/UI degrade cleanly).
        """
        try:
            from nunchaku import NunchakuQwenImageTransformer2DModel
            from nunchaku.utils import get_precision
        except Exception as exc:  # noqa: BLE001 — nunchaku is an optional CUDA-only dep
            raise NotImplementedError(
                "Nunchaku (SVDQuant int4) is not installed. It is CUDA-only; the CUDA image "
                "installs a torch/python-matched wheel. See DESIGN §9.4."
            ) from exc

        ref = self._nunchaku_quant_ref(model_id, get_precision)
        free_vram_mb = self._free_vram_mb()

        with phase(log, f"Loading Nunchaku SVDQuant transformer {ref}"):
            transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(ref)

        # Low-VRAM path (gaming/workstation cards): per-block GPU residency + sequential
        # offload. Forced when the caller requested offload or free VRAM is tight.
        from app.config import get_settings

        low_vram = offload or (0 < free_vram_mb < get_settings().int4_offload_below_vram_mb)
        if low_vram and hasattr(transformer, "set_offload"):
            log.info("int4: enabling Nunchaku block offload (low-VRAM path)")
            transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)

        with phase(log, f"from_pretrained({model_id}) with int4 transformer"):
            pipeline = pipeline_cls.from_pretrained(
                model_id,
                revision=model_revision,
                transformer=transformer,
                torch_dtype=dtype,
            )

        if low_vram and hasattr(pipeline, "enable_sequential_cpu_offload"):
            pipeline.enable_sequential_cpu_offload()
        elif device != "cpu":
            pipeline = pipeline.to(device)
        return pipeline

    def _nunchaku_quant_ref(self, model_id: str, get_precision: Any) -> str:
        """Resolve the Nunchaku pre-quantized ``repo/file.safetensors`` ref for a base model.

        Sources are config-overridable (``nunchaku_quant_sources``); the defaults cover the
        shipped Generate/Edit bases. ``{prec}`` <- get_precision(), ``{rank}`` <- int4_rank.
        """
        from app.config import get_settings

        settings = get_settings()
        sources = {**_DEFAULT_NUNCHAKU_SOURCES, **(settings.nunchaku_quant_sources or {})}
        template = sources.get(model_id)
        if template is None:
            raise NotImplementedError(
                f"No Nunchaku int4 quant source mapped for {model_id!r}. Map one via "
                "QIE_NUNCHAKU_QUANT_SOURCES, or pick a model with published SVDQuant weights "
                "(DESIGN §9.4)."
            )
        return template.format(prec=get_precision(), rank=settings.int4_rank)

    def _free_vram_mb(self) -> int:
        """Best-effort free VRAM in MB (0 if torch/CUDA unavailable)."""
        try:
            import torch

            if torch.cuda.is_available():
                free, _total = torch.cuda.mem_get_info()
                return int(free // (1024 * 1024))
        except Exception:  # noqa: BLE001
            pass
        return 0

    def _apply_load_options(self, pipeline: Any, *, device: str, options: LoadOptions) -> None:
        """Apply offloading / memory toggles, ignoring ones the pipeline lacks (DESIGN §9.3)."""
        if options.enable_sequential_cpu_offload and hasattr(
            pipeline, "enable_sequential_cpu_offload"
        ):
            log.info("Enabling sequential CPU offload")
            pipeline.enable_sequential_cpu_offload()
        elif options.enable_model_cpu_offload and hasattr(pipeline, "enable_model_cpu_offload"):
            log.info("Enabling model CPU offload")
            pipeline.enable_model_cpu_offload()
        if options.enable_attention_slicing and hasattr(pipeline, "enable_attention_slicing"):
            log.info("Enabling attention slicing")
            pipeline.enable_attention_slicing()
        if options.enable_vae_tiling and hasattr(pipeline, "enable_vae_tiling"):
            log.info("Enabling VAE tiling")
            pipeline.enable_vae_tiling()

    def _unload(self) -> None:
        """Release the resident pipeline and free accelerator memory."""
        if self._pipeline is None:
            return
        log.info("Unloading resident pipeline")
        self._pipeline = None
        self._load_key = None
        self._active_adapters = []
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — best-effort; torch may be absent
            pass

    # -- LoRA stacking ------------------------------------------------------------------

    def apply_loras(self, loras: list[dict[str, Any]]) -> None:
        """Load + activate the given LoRAs with per-adapter weights — **precision-aware**.

        Standard bf16/fp8 transformers use the diffusers PEFT path
        (``load_lora_weights`` + ``set_adapters``). For **Nunchaku SVDQuant int4**
        transformers — whose ``SVDQW4A4Linear`` layers PEFT cannot wrap — we apply each LoRA
        ourselves as a parallel **fp16 low-rank correction** via forward hooks:
        ``y = svdquant_int4_linear(x) + scale · up(down(x))``. That is mathematically exactly
        LoRA, leaves the int4 base path untouched, and works for any external diffusers/kohya
        LoRA — so int4 (the workstation/gaming-card path) gets full LoRA support too (§5.3).

        Each descriptor is ``{"path": <local safetensors>, "weight": float,
        "adapter_name": str}``.

        Raises:
            RuntimeError: If called before a pipeline is loaded.
        """
        if self._pipeline is None:
            raise RuntimeError("apply_loras called before load(); no pipeline resident.")

        self._reset_loras()
        if not loras:
            return

        transformer = getattr(self._pipeline, "transformer", None)
        if self._is_svdquant(transformer):
            self._apply_loras_manual(transformer, loras)
        else:
            self._apply_loras_peft(self._pipeline, loras)

    @staticmethod
    def _is_svdquant(transformer: Any) -> bool:
        """True if the transformer uses Nunchaku SVDQuant layers (PEFT can't wrap them)."""
        if transformer is None:
            return False
        if type(transformer).__module__.lower().startswith("nunchaku"):
            return True
        return any(type(m).__name__.startswith("SVDQ") for m in transformer.modules())

    def _apply_loras_peft(self, pipeline: Any, loras: list[dict[str, Any]]) -> None:
        """Standard diffusers PEFT LoRA path (bf16/fp8)."""
        names: list[str] = []
        weights: list[float] = []
        with phase(log, f"Loading {len(loras)} LoRA adapter(s) [PEFT]"):
            for i, entry in enumerate(loras):
                name = str(entry.get("adapter_name") or f"lora_{i}")
                weight = float(entry.get("weight", 1.0))
                log.info("  LoRA %s <- %s (weight=%.3f)", name, entry["path"], weight)
                pipeline.load_lora_weights(entry["path"], adapter_name=name)
                names.append(name)
                weights.append(weight)
        pipeline.set_adapters(names, adapter_weights=weights)
        self._active_adapters = names

    def _apply_loras_manual(self, transformer: Any, loras: list[dict[str, Any]]) -> None:
        """Precision-aware manual LoRA for SVDQuant int4 via parallel fp16 forward hooks."""
        import torch

        with phase(log, f"Applying {len(loras)} LoRA(s) to SVDQuant int4 (manual fp16 hooks)"):
            for i, entry in enumerate(loras):
                name = str(entry.get("adapter_name") or f"lora_{i}")
                weight = float(entry.get("weight", 1.0))
                pairs = _load_lora_pairs(entry["path"])
                matched = 0
                for module_path, down, up, alpha in pairs:
                    module = _resolve_submodule(transformer, module_path)
                    if module is None:
                        continue
                    rank = down.shape[0]
                    scale = weight * (float(alpha) / rank if alpha is not None else 1.0)
                    handle = module.register_forward_hook(
                        _make_lora_hook(down, up, scale, torch)
                    )
                    self._lora_hooks.append(handle)
                    matched += 1
                log.info(
                    "  LoRA %s <- %s (weight=%.3f): %d/%d target modules matched",
                    name, entry["path"], weight, matched, len(pairs),
                )
                if matched == 0:
                    raise RuntimeError(
                        f"LoRA {entry['path']!r} matched no modules on the SVDQuant "
                        "transformer — key/layout mismatch."
                    )
                self._active_adapters.append(name)

    def _reset_loras(self) -> None:
        """Unload any active LoRA adapters (PEFT) and remove manual int4 hooks."""
        for handle in self._lora_hooks:
            with contextlib.suppress(Exception):
                handle.remove()
        self._lora_hooks = []
        if self._active_adapters and self._pipeline is not None:
            pipeline = self._pipeline
            try:
                if hasattr(pipeline, "delete_adapters"):
                    pipeline.delete_adapters(self._active_adapters)
                elif hasattr(pipeline, "unload_lora_weights"):
                    pipeline.unload_lora_weights()
            except Exception:  # noqa: BLE001 — never let LoRA cleanup abort the next job
                log.debug("PEFT LoRA reset skipped/failed; continuing", exc_info=True)
        self._active_adapters = []

    # -- running ------------------------------------------------------------------------

    def run(
        self,
        req: RunRequest,
        *,
        on_step: Callable[[StepProgress], None] | None = None,
        is_canceled: Callable[[], bool] | None = None,
    ) -> list[Any]:
        """Run one inference, streaming throttled progress + previews (DESIGN §5.5).

        Loads/reuses the pipeline for ``req``, applies its LoRAs, and runs the pipeline with a
        ``callback_on_step_end`` hook that (a) checks cancellation each step and aborts
        cleanly, and (b) decodes a small live preview on the throttled cadence. A failed
        preview decode is logged and skipped rather than failing the run.

        Args:
            req: The resolved run request.
            on_step: Optional callback invoked with a :class:`StepProgress` each step.
            is_canceled: Optional predicate polled each step; returning ``True`` aborts the
                run (raising :class:`CanceledError`).

        Returns:
            The output images as a list of ``PIL.Image.Image``.

        Raises:
            CanceledError: If ``is_canceled`` returned ``True`` during the run.
        """
        import torch  # lazy

        self.load(
            mode=req.mode,
            model_id=req.model_id,
            model_revision=req.model_revision,
            precision=req.precision,
            device=req.device,
        )
        self.apply_loras(req.loras)
        pipeline = self._pipeline
        assert pipeline is not None  # load() guarantees this

        kwargs = build_pipeline_kwargs(req)
        if req.mode == MODE_EDIT:
            if not req.images:
                raise ValueError("Edit mode requires at least one input image.")
            kwargs["image"] = list(req.images)  # order is model-significant (DESIGN §2.1)

        if req.seed is not None:
            kwargs["generator"] = torch.Generator(device=_generator_device(req.device))
            kwargs["generator"].manual_seed(int(req.seed))

        total = int(req.num_inference_steps)
        callback = self._make_step_callback(
            pipeline=pipeline,
            total=total,
            every_n=req.preview_every_n_steps,
            on_step=on_step,
            is_canceled=is_canceled,
            width=req.width,
            height=req.height,
        )
        kwargs["callback_on_step_end"] = callback
        kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

        with phase(log, f"Running {req.mode} inference ({total} steps, {req.precision})"):
            output = pipeline(**kwargs)

        images = list(getattr(output, "images", []))
        log.info("Inference produced %d image(s)", len(images))
        return images

    def _make_step_callback(
        self,
        *,
        pipeline: Any,
        total: int,
        every_n: int,
        on_step: Callable[[StepProgress], None] | None,
        is_canceled: Callable[[], bool] | None,
        width: int | None,
        height: int | None,
    ) -> Callable[..., dict[str, Any]]:
        """Build the diffusers ``callback_on_step_end`` closure (cancel + preview)."""

        def _callback(pipe: Any, step_index: int, _timestep: Any,
                      callback_kwargs: dict[str, Any]) -> dict[str, Any]:
            step = int(step_index) + 1  # diffusers passes a 0-based index
            if is_canceled is not None and is_canceled():
                log.info("Cancellation observed at step %d/%d — aborting run", step, total)
                raise CanceledError(step=step, total=total)

            preview: bytes | None = None
            if should_emit_preview(step, total, every_n):
                latents = callback_kwargs.get("latents")
                if latents is not None:
                    preview = self._decode_preview(pipe, latents, width, height)

            if on_step is not None:
                on_step(StepProgress(step=step, total=total, preview_png=preview))
            return callback_kwargs

        return _callback

    def _decode_preview(
        self, pipeline: Any, latents: Any, width: int | None, height: int | None
    ) -> bytes | None:
        """Decode latents to a small preview PNG; return ``None`` (skip) on any failure."""
        try:
            import torch

            with torch.no_grad():
                image = _latents_to_preview_image(
                    pipeline, latents, width=width, height=height, max_edge=_PREVIEW_MAX_EDGE
                )
            if image is None:
                return None
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception:  # noqa: BLE001 — a preview must never fail the job (DESIGN §5.5)
            log.warning("Live preview decode failed; skipping this preview", exc_info=True)
            return None


class CanceledError(RuntimeError):
    """Raised inside the step callback when a run is cooperatively canceled (DESIGN §4.1)."""

    def __init__(self, *, step: int, total: int) -> None:
        super().__init__(f"Run canceled at step {step}/{total}")
        self.step = step
        self.total = total


# --- Run-time helpers (touch torch; not part of the pure-tested surface) ---------------


def _backend_of_device(device: str) -> str:
    """Map a device string to a backend label for precision validation."""
    base = device.split(":", 1)[0].lower()
    if base == "cuda":
        return "cuda"
    if base in ("mps", "cpu", "rocm"):
        return base
    return base


def _fp8_native_for(device: str) -> bool:
    """Whether the device natively supports fp8_e4m3fn (SM >= 8.9). Best-effort."""
    if _backend_of_device(device) != "cuda":
        return False
    try:
        import torch

        index = 0 if ":" not in device else int(device.split(":", 1)[1])
        major, minor = torch.cuda.get_device_capability(index)
        return (major, minor) >= (8, 9)
    except Exception:  # noqa: BLE001 — probe failure: assume emulated to stay safe
        return False


def _generator_device(device: str) -> str:
    """The device a seeded ``torch.Generator`` should live on (CPU for MPS)."""
    backend = _backend_of_device(device)
    if backend == "mps":
        return "cpu"  # torch.Generator(device="mps") is unsupported; seed on CPU
    return device


def _latents_to_preview_image(
    pipeline: Any, latents: Any, *, width: int | None, height: int | None, max_edge: int
) -> Any | None:
    """Decode in-progress Qwen latents to a small ``PIL.Image`` preview (DESIGN §5.5).

    Qwen packs latents and the (3D) VAE uses per-channel ``latents_mean``/``latents_std`` — so
    a generic ``vae.decode`` fails. This mirrors the diffusers ``QwenImagePipeline`` decode
    path exactly: ``_unpack_latents`` → de-normalize → ``vae.decode(...)[:, :, 0]`` →
    ``image_processor.postprocess``. Needs the target ``width``/``height`` to unpack; returns
    ``None`` (skip) if they're unknown. Downscales to ``max_edge`` to keep the cost modest.
    """
    import torch

    if width is None or height is None:
        return None
    vae = pipeline.vae
    lat = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
    lat = lat.to(dtype=vae.dtype, device=vae.device)

    z_dim = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(lat.device, lat.dtype)
    inv_std = (
        1.0 / torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(lat.device, lat.dtype)
    )
    lat = lat / inv_std + mean

    with torch.no_grad():
        decoded = vae.decode(lat, return_dict=False)[0][:, :, 0]  # 3D VAE -> first frame
    image = pipeline.image_processor.postprocess(decoded, output_type="pil")[0]

    w, h = image.size
    longest = max(w, h)
    if longest > max_edge:
        scale = max_edge / float(longest)
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return image


# --- Manual LoRA application for SVDQuant int4 (precision-aware; DESIGN §5.3) -----------

# LoRA prefixes seen across diffusers / ComfyUI / kohya exports, stripped to match the
# transformer's own module paths (e.g. "transformer_blocks.0.attn.to_q").
_LORA_PREFIXES = ("transformer.", "diffusion_model.", "lora_unet_", "base_model.model.")


def _strip_lora_prefix(base: str) -> str:
    for pfx in _LORA_PREFIXES:
        if base.startswith(pfx):
            return base[len(pfx) :]
    return base


def _load_lora_pairs(path: str) -> list[tuple[str, Any, Any, float | None]]:
    """Parse a LoRA safetensors into ``(module_path, down, up, alpha)`` tuples.

    Handles both kohya (``lora_down``/``lora_up``/``alpha``) and diffusers
    (``lora_A``/``lora_B``) naming. ``down`` is ``[rank, in]``, ``up`` is ``[out, rank]``.
    """
    from safetensors.torch import load_file

    sd = load_file(path)
    downs: dict[str, Any] = {}
    ups: dict[str, Any] = {}
    alphas: dict[str, float] = {}
    for key, tensor in sd.items():
        if key.endswith(".alpha"):
            alphas[_strip_lora_prefix(key[: -len(".alpha")])] = float(tensor.reshape(-1)[0])
        elif key.endswith(".lora_down.weight") or key.endswith(".lora_A.weight"):
            base = key.rsplit(".lora_down.weight", 1)[0].rsplit(".lora_A.weight", 1)[0]
            downs[_strip_lora_prefix(base)] = tensor
        elif key.endswith(".lora_up.weight") or key.endswith(".lora_B.weight"):
            base = key.rsplit(".lora_up.weight", 1)[0].rsplit(".lora_B.weight", 1)[0]
            ups[_strip_lora_prefix(base)] = tensor
    pairs: list[tuple[str, Any, Any, float | None]] = []
    for base, down in downs.items():
        up = ups.get(base)
        if up is not None:
            pairs.append((base, down, up, alphas.get(base)))
    return pairs


def _resolve_submodule(transformer: Any, path: str) -> Any | None:
    """Resolve a dotted module path on the transformer, tolerating a ``transformer.`` prefix."""
    candidates = [path]
    if path.startswith("transformer."):
        candidates.append(path[len("transformer.") :])
    for candidate in candidates:
        try:
            return transformer.get_submodule(candidate)
        except AttributeError:
            continue
    return None


def _make_lora_hook(down: Any, up: Any, scale: float, torch: Any):  # noqa: ANN201
    """Build a forward hook adding ``scale · up(down(x))`` (fp16) to a layer's output."""

    def _hook(module: Any, inputs: tuple, output: Any) -> Any:
        x = inputs[0]
        d = down.to(device=x.device, dtype=x.dtype)
        u = up.to(device=x.device, dtype=x.dtype)
        delta = torch.nn.functional.linear(torch.nn.functional.linear(x, d), u) * scale
        if isinstance(output, tuple):
            return (output[0] + delta, *output[1:])
        return output + delta

    return _hook


# --- Process singleton -----------------------------------------------------------------

_service: PipelineService | None = None


def get_pipeline_service() -> PipelineService:
    """Return the process-wide :class:`PipelineService` (created lazily)."""
    global _service
    if _service is None:
        _service = PipelineService()
    return _service
