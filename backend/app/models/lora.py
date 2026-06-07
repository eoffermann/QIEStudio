"""LoRA registry model (DESIGN §5.3, §6)."""

from __future__ import annotations

from datetime import datetime

from sqlmodel import Field, SQLModel

from app.models.base import (
    created_at_column,
    id_field,
    owner_field,
    str_array_column,
    utcnow,
    workspace_field,
)


class Lora(SQLModel, table=True):
    __tablename__ = "lora"

    id: str = id_field()
    owner_id: str | None = owner_field()
    workspace_id: str | None = workspace_field()

    name: str = Field(index=True)
    storage_key: str  # .safetensors under loras/ via StorageProvider
    thumb_key: str | None = None
    description: str = ""
    trigger_words: list[str] = Field(default_factory=list, sa_column=str_array_column())
    recommended_weight: float = 1.0

    # Which base the adapter targets, and which run modes it applies to.
    base_compat: str = Field(default="qwen-image", index=True)  # qwen-image | qwen-image-edit
    modes: list[str] = Field(default_factory=list, sa_column=str_array_column())

    source: str = "upload"  # upload | hf | civitai | url
    source_ref: str = ""  # repo id / model:version / url
    license: str = ""
    sha256: str = Field(default="", index=True)
    bytes: int = 0
    enabled: bool = Field(default=True)

    created_at: datetime = Field(default_factory=utcnow, sa_column=created_at_column())
