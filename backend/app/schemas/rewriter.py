"""Pydantic request/response models for the Qwen-VL prompt-enhancer API (DESIGN §5.8, §7).

These shape the JSON surface of ``/api/rewriter`` (models / advise / enhance) and provide thin
serialization adapters over the dataclasses returned by
:mod:`app.services.rewriter` (:class:`~app.services.rewriter.RewriterAdvice`).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.services.rewriter import RewriterAdvice


class QuantOptionRead(BaseModel):
    """One quant/backend combo for a rewriter model (DESIGN §5.8)."""

    quant: str
    label: str
    backends: list[str]
    requires_runtime: str | None = None


class RewriterModelRead(BaseModel):
    """A selectable VL rewriter model + its device-filtered quant options (DESIGN §5.8)."""

    model_id: str
    label: str
    est_vram_mb: int
    is_default: bool
    co_resides_typically: bool
    notes: str
    quant_options: list[QuantOptionRead]


class RewriterModelsResponse(BaseModel):
    """Response for ``GET /api/rewriter/models`` (DESIGN §7)."""

    models: list[RewriterModelRead]
    default_model: str


class AdviseRewriterRequest(BaseModel):
    """Body for ``POST /api/rewriter/advise`` (DESIGN §5.8, §7).

    ``image_model_resident_mb`` is the VRAM the loaded image pipeline currently occupies (its
    precision + LoRAs folded in by the caller, e.g. from the §9 advisor's estimated peak). The
    advisor uses it to decide whether the VL model can co-reside or needs a swap.
    """

    vl_model: str
    image_model_resident_mb: int = Field(
        default=0, ge=0, description="VRAM the loaded image model currently occupies (MB)."
    )


class RewriterAdviceRead(BaseModel):
    """Co-reside-vs-swap advice for a rewriter model (DESIGN §5.8)."""

    vl_model: str
    can_co_reside: bool
    decision: str
    est_vl_vram_mb: int
    image_model_resident_mb: int
    free_after_image_mb: int
    rationale: str

    @classmethod
    def from_advice(cls, advice: RewriterAdvice) -> RewriterAdviceRead:
        """Build a read model from a :class:`RewriterAdvice` dataclass."""
        return cls(
            vl_model=advice.vl_model,
            can_co_reside=advice.can_co_reside,
            decision=advice.decision,
            est_vl_vram_mb=advice.est_vl_vram_mb,
            image_model_resident_mb=advice.image_model_resident_mb,
            free_after_image_mb=advice.free_after_image_mb,
            rationale=advice.rationale,
        )


class EnhanceRequest(BaseModel):
    """Body for ``POST /api/rewriter/enhance`` (DESIGN §5.8, §7).

    In Edit mode ``image_ids`` are resolved to PIL images and passed to the VL model alongside
    the text (the multimodal rewrite). Order is preserved (model-significant). ``vl_model``
    defaults to ``settings.default_rewriter_model``; ``edit_model_id`` selects the template
    variant (2512 vs default).
    """

    mode: str = "edit"
    prompt: str
    image_ids: list[str] = Field(default_factory=list)
    vl_model: str | None = None
    edit_model_id: str | None = None


class EnhanceResponse(BaseModel):
    """Response for ``POST /api/rewriter/enhance`` (DESIGN §5.8).

    Returns both prompts so the caller can render an editable diff (never a silent rewrite)
    and persist both in job metadata for reproducibility.
    """

    enhanced_prompt: str
    original_prompt: str
