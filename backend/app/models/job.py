"""Job, JobInput, JobOutput models (DESIGN §5.5, §6).

A Job persists the full **reproducibility metadata** (DESIGN §5.5 / RUN §6): mode, model
id/revision, precision/quant, device, resolution, seed, LoRAs+weights, input image hashes,
and **both the original and enhanced prompts**. Status is persisted so jobs survive reloads.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlmodel import Field, SQLModel

from app.models.base import (
    created_at_column,
    id_field,
    jsonb_column,
    owner_field,
    utcnow,
    workspace_field,
)

# status: queued | running | done | error | canceled
# mode: generate | edit
# precision: bf16 | fp8 | int4


class Job(SQLModel, table=True):
    __tablename__ = "job"

    id: str = id_field()
    owner_id: str | None = owner_field()
    workspace_id: str | None = workspace_field()

    batch_id: str | None = Field(default=None, index=True)
    mode: str = Field(index=True)
    status: str = Field(default="queued", index=True)

    precision: str = "bf16"
    device: str = ""  # e.g. "cuda:0 (NVIDIA RTX A6000, SM 8.6)"
    model_id: str = ""
    model_revision: str | None = None

    prompt: str = ""
    enhanced_prompt: str | None = None
    rewriter_model: str | None = None

    # Full run params (resolution, steps, true_cfg_scale, guidance_scale, negative,
    # seed, loras[{lora_id,weight}], offload flags, output format/quality, ...).
    params_json: dict[str, Any] = Field(default_factory=dict, sa_column=jsonb_column())

    progress: float = 0.0  # 0..1
    progress_step: int = 0
    progress_total: int = 0
    error: str | None = None

    created_at: datetime = Field(default_factory=utcnow, sa_column=created_at_column())
    started_at: datetime | None = None
    ended_at: datetime | None = None


class JobInput(SQLModel, table=True):
    """Resolved, ordered input images for an Edit job (order is model-significant)."""

    __tablename__ = "job_input"

    id: str = id_field()
    job_id: str = Field(index=True, foreign_key="job.id")
    position: int = 0
    asset_id: str = Field(foreign_key="asset.id")
    sha256: str = ""  # snapshot of the input hash for reproducibility


class JobOutput(SQLModel, table=True):
    __tablename__ = "job_output"

    id: str = id_field()
    job_id: str = Field(index=True, foreign_key="job.id")
    position: int = 0
    # The ephemeral Asset created for this output, so the UI can promote-to-library or
    # send-to-input the result (DESIGN §5.5 "send output to input" / promote).
    asset_id: str | None = Field(default=None, foreign_key="asset.id")
    storage_key: str
    thumb_key: str | None = None
    seed: int | None = None
    # The complete sidecar metadata embedded in the PNG and stored alongside the output.
    metadata_json: dict[str, Any] = Field(default_factory=dict, sa_column=jsonb_column())
