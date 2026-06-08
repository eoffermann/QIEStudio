"""Request/response schemas for the jobs API (DESIGN §5.5, §5.6, §7)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class LoraSelection(BaseModel):
    lora_id: str
    weight: float = 1.0


class ResolutionSpec(BaseModel):
    """Either a preset (base/orientation/aspect) or an explicit width/height."""

    base: int | str | None = None  # int | "match"
    orientation: str | None = None
    aspect: str | None = None
    width: int | None = None
    height: int | None = None
    max_long_edge: int | None = None


class JobSubmit(BaseModel):
    mode: Literal["generate", "edit"]
    prompt: str = ""
    enhanced_prompt: str | None = None
    rewriter_model: str | None = None
    negative_prompt: str = " "

    model_id: str | None = None  # defaults from settings per mode
    model_revision: str | None = None
    precision: Literal["bf16", "fp8", "int4"] = "bf16"

    # Inputs (Edit mode): explicit ordered asset ids, and/or a prompt with slot fills.
    prompt_id: str | None = None
    input_asset_ids: list[str] = Field(default_factory=list)
    slot_fills: dict[str, list[str]] = Field(default_factory=dict)

    loras: list[LoraSelection] = Field(default_factory=list)
    resolution: ResolutionSpec = Field(default_factory=ResolutionSpec)

    num_inference_steps: int = 40
    true_cfg_scale: float = 4.0
    guidance_scale: float = 1.0
    seed: int | None = None
    batch: int = 1

    output_format: Literal["png", "webp", "jpeg"] = "png"
    output_quality: int = 95
    preview_every_n_steps: int | None = None  # None -> settings default

    # Offloading toggles (DESIGN §9.3).
    enable_model_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    enable_attention_slicing: bool = False
    enable_vae_tiling: bool = False


class SweepSpec(BaseModel):
    """Batch/sweep expansion (DESIGN §5.6). Exactly one sweep kind per request."""

    kind: Literal["slot", "seed", "param"]
    # slot sweep: fill one slot with N assets -> N jobs
    slot_name: str | None = None
    slot_asset_ids: list[str] = Field(default_factory=list)
    # seed sweep: N seeds
    seeds: list[int] = Field(default_factory=list)
    # param sweep: grid over steps / cfg
    steps_grid: list[int] = Field(default_factory=list)
    cfg_grid: list[float] = Field(default_factory=list)


class BatchSubmit(BaseModel):
    base: JobSubmit
    sweep: SweepSpec


class JobOutputRead(BaseModel):
    id: str
    position: int
    seed: int | None
    asset_id: str | None
    file_url: str
    thumb_url: str | None
    metadata: dict[str, Any]


class JobRead(BaseModel):
    id: str
    batch_id: str | None
    mode: str
    status: str
    precision: str
    device: str
    model_id: str
    model_revision: str | None
    prompt: str
    enhanced_prompt: str | None
    rewriter_model: str | None
    params: dict[str, Any]
    progress: float
    progress_step: int
    progress_total: int
    error: str | None
    created_at: str | None
    started_at: str | None
    ended_at: str | None
    outputs: list[JobOutputRead] = Field(default_factory=list)


class ReorderRequest(BaseModel):
    """Body for ``POST /api/jobs/reorder`` — the desired execution order of pending jobs."""

    job_ids: list[str]


class QueueState(BaseModel):
    """Current queue snapshot (DESIGN §13 queue management)."""

    running: str | None
    pending: list[str]


class JobSubmitResponse(BaseModel):
    job_id: str


class BatchSubmitResponse(BaseModel):
    batch_id: str
    job_ids: list[str]
