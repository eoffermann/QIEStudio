"""Prompt model with image slots/bindings and LoRA bindings (DESIGN §5.2, §6)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlmodel import Field, SQLModel

from app.models.base import (
    created_at_column,
    id_field,
    jsonb_column,
    owner_field,
    str_array_column,
    updated_at_column,
    utcnow,
    workspace_field,
)


class Prompt(SQLModel, table=True):
    __tablename__ = "prompt"

    id: str = id_field()
    owner_id: str | None = owner_field()
    workspace_id: str | None = workspace_field()

    name: str = Field(index=True)
    text: str = ""
    tags: list[str] = Field(default_factory=list, sa_column=str_array_column())
    mode: str = Field(default="any", index=True)  # generate | edit | any
    favorite: bool = Field(default=False)

    # Optional generation defaults (resolution preset, steps, cfg, seed, ...).
    defaults_json: dict[str, Any] = Field(default_factory=dict, sa_column=jsonb_column())

    created_at: datetime = Field(default_factory=utcnow, sa_column=created_at_column())
    updated_at: datetime = Field(default_factory=utcnow, sa_column=updated_at_column())
    last_used_at: datetime | None = Field(default=None)


class PromptImage(SQLModel, table=True):
    """An ordered entry in a prompt's image list: a pinned asset or an open slot.

    ``position`` preserves the model-significant ordering (DESIGN §5.2).
    """

    __tablename__ = "prompt_image"

    id: str = id_field()
    prompt_id: str = Field(index=True, foreign_key="prompt.id")
    position: int = 0

    role: str = "pinned"  # pinned | slot
    asset_id: str | None = Field(default=None, foreign_key="asset.id")  # for role=pinned

    slot_name: str | None = None  # for role=slot
    slot_min: int | None = None
    slot_max: int | None = None
    hint: str | None = None


class PromptLora(SQLModel, table=True):
    """A LoRA binding carried by a saved prompt (auto-applied on load; overridable)."""

    __tablename__ = "prompt_lora"

    id: str = id_field()
    prompt_id: str = Field(index=True, foreign_key="prompt.id")
    lora_id: str = Field(foreign_key="lora.id")
    weight: float = 1.0
