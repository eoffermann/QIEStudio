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
        """Load a Nunchaku SVDQuant int4 pipeline (CUDA-only) (DESIGN §9.1).

        Nunchaku ships a pre-quantized SVDQuant transformer that is swapped into the base
        pipeline. The pre-quantized weights are a separate artifact; if nunchaku or the
        weights are unavailable, raise a clear :class:`NotImplementedError` so the GPU test
        can *skip with a reason* rather than crash mid-run.
        """
        try:
            from nunchaku import NunchakuQwenImageTransformer2DModel
        except Exception as exc:  # noqa: BLE001 — nunchaku is an optional CUDA-only dep
            raise NotImplementedError(
                "Nunchaku (SVDQuant int4) is not installed. It is CUDA-only and installed "
                "separately in the CUDA Docker image; install `nunchaku` and provide the "
                "pre-quantized transformer to enable int4."
            ) from exc

        with phase(log, f"Loading Nunchaku SVDQuant int4 transformer for {model_id}"):
            try:
                transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
                    model_id,
                    revision=model_revision,
                    torch_dtype=dtype,
                )
            except Exception as exc:  # noqa: BLE001 — missing/incompatible quantized weights
                raise NotImplementedError(
                    "Could not load a Nunchaku SVDQuant int4 transformer for "
                    f"{model_id!r}: {exc}. A pre-quantized SVDQuant artifact is required "
                    "for int4."
                ) from exc

        with phase(log, f"from_pretrained({model_id}) with int4 transformer"):
            pipeline = pipeline_cls.from_pretrained(
                model_id,
                revision=model_revision,
                transformer=transformer,
                torch_dtype=dtype,
            )
        if not offload and device != "cpu":
            pipeline = pipeline.to(device)
        return pipeline

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
        """Load + activate the given LoRAs with per-adapter weights (DESIGN §5.3).

        Any previously-applied adapters are unloaded first so each job starts clean. Each
        descriptor is ``{"path": <local safetensors path>, "weight": float,
        "adapter_name": str}``; ``adapter_name`` defaults to a stable name derived from the
        list position when omitted.

        Args:
            loras: Ordered LoRA descriptors to apply.

        Raises:
            RuntimeError: If called before a pipeline is loaded.
        """
        if self._pipeline is None:
            raise RuntimeError("apply_loras called before load(); no pipeline resident.")

        self._reset_loras()
        if not loras:
            return

        pipeline = self._pipeline
        names: list[str] = []
        weights: list[float] = []
        with phase(log, f"Loading {len(loras)} LoRA adapter(s)"):
            for i, entry in enumerate(loras):
                path = entry["path"]
                name = str(entry.get("adapter_name") or f"lora_{i}")
                weight = float(entry.get("weight", 1.0))
                log.info("  LoRA %s <- %s (weight=%.3f)", name, path, weight)
                pipeline.load_lora_weights(path, adapter_name=name)
                names.append(name)
                weights.append(weight)

        # Activate all adapters with their per-adapter weights (PEFT set_adapters).
        pipeline.set_adapters(names, adapter_weights=weights)
        self._active_adapters = names

    def _reset_loras(self) -> None:
        """Unload any active LoRA adapters so the next job starts from the base weights."""
        if not self._active_adapters or self._pipeline is None:
            self._active_adapters = []
            return
        pipeline = self._pipeline
        try:
            if hasattr(pipeline, "delete_adapters"):
                pipeline.delete_adapters(self._active_adapters)
            elif hasattr(pipeline, "unload_lora_weights"):
                pipeline.unload_lora_weights()
        except Exception:  # noqa: BLE001 — never let LoRA cleanup abort the next job
            log.exception("Failed to cleanly reset LoRA adapters; continuing")
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
                    preview = self._decode_preview(pipe, latents)

            if on_step is not None:
                on_step(StepProgress(step=step, total=total, preview_png=preview))
            return callback_kwargs

        return _callback

    def _decode_preview(self, pipeline: Any, latents: Any) -> bytes | None:
        """Decode latents to a small preview PNG; return ``None`` (skip) on any failure."""
        try:
            import torch

            with torch.no_grad():
                image = _latents_to_preview_image(pipeline, latents, max_edge=_PREVIEW_MAX_EDGE)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception:  # noqa: BLE001 — a preview must never fail the job (DESIGN §5.5)
            log.debug("Live preview decode failed; skipping this preview", exc_info=True)
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


def _latents_to_preview_image(pipeline: Any, latents: Any, *, max_edge: int) -> Any:
    """Decode latents to a small ``PIL.Image`` via the pipeline's VAE (cheap preview).

    Uses the pipeline's own VAE-scaling constants and image processor when available so the
    preview matches the model's color space, then downscales to ``max_edge`` on the long
    side to keep the decode + encode cost modest.
    """
    import torch

    vae = pipeline.vae
    latents = latents.to(dtype=vae.dtype, device=vae.device)

    # Undo the diffusers latent scaling/shift where the VAE config provides it.
    scaling = float(getattr(vae.config, "scaling_factor", 1.0) or 1.0)
    shift = getattr(vae.config, "shift_factor", None)
    decode_in = latents / scaling if scaling else latents
    if shift is not None:
        decode_in = decode_in + float(shift)

    with torch.no_grad():
        decoded = vae.decode(decode_in).sample

    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    # Take the first sample in the batch for the preview.
    tensor = decoded[0].detach().to(dtype=torch.float32, device="cpu")
    array = (tensor.permute(1, 2, 0).numpy() * 255).round().astype("uint8")

    from PIL import Image

    image = Image.fromarray(array)
    width, height = image.size
    longest = max(width, height)
    if longest > max_edge:
        scale = max_edge / float(longest)
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    return image


# --- Process singleton -----------------------------------------------------------------

_service: PipelineService | None = None


def get_pipeline_service() -> PipelineService:
    """Return the process-wide :class:`PipelineService` (created lazily)."""
    global _service
    if _service is None:
        _service = PipelineService()
    return _service
