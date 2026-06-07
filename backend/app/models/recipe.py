"""Recipe model (DESIGN §13 round-out feature #4).

A recipe is a saved, named **composition** — the full run config (mode, prompt, LoRAs,
resolution, settings, and slot layout) — so a whole setup can be re-run in one click. Stored
as a JSON snapshot of a job-submit-shaped payload so it stays forward-compatible as the run
schema grows.
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
    str_array_column,
    updated_at_column,
    utcnow,
    workspace_field,
)


class Recipe(SQLModel, table=True):
    __tablename__ = "recipe"

    id: str = id_field()
    owner_id: str | None = owner_field()
    workspace_id: str | None = workspace_field()

    name: str = Field(index=True)
    description: str = ""
    tags: list[str] = Field(default_factory=list, sa_column=str_array_column())
    mode: str = Field(default="generate", index=True)  # generate | edit

    # A JobSubmit-shaped snapshot of the full composition (prompt, loras, resolution,
    # settings, slot layout, ...).
    config_json: dict[str, Any] = Field(default_factory=dict, sa_column=jsonb_column())

    created_at: datetime = Field(default_factory=utcnow, sa_column=created_at_column())
    updated_at: datetime = Field(default_factory=utcnow, sa_column=updated_at_column())
