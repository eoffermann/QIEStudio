"""Pydantic request/response models for the recipes API (DESIGN §13 #4, §6).

A *recipe* is a saved, named **composition** — the full run config (prompt + LoRAs +
resolution + settings + slot layout) — stored as a :class:`~app.schemas.jobs.JobSubmit`-
shaped snapshot in ``config_json`` so it stays forward-compatible as the run schema grows
and can be re-run in one click (``POST /api/recipes/{id}/instantiate`` → a JobSubmit payload
ready to POST to ``/api/jobs``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.models.recipe import Recipe


class RecipeCreate(BaseModel):
    """Body for ``POST /api/recipes`` — create / save a recipe (DESIGN §13 #4).

    ``config_json`` carries a JobSubmit-shaped run config; ``mode`` mirrors that config's
    mode so recipes can be listed/filtered without unpacking the JSON blob.
    """

    name: str
    description: str = ""
    tags: list[str] = []
    mode: str = "generate"  # generate | edit
    config_json: dict[str, Any] = {}


class RecipeUpdate(BaseModel):
    """Body for ``PATCH /api/recipes/{id}`` (DESIGN §7).

    Every field is optional; ``None`` means "leave unchanged" so partial updates are
    unambiguous.
    """

    name: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    mode: str | None = None
    config_json: dict[str, Any] | None = None


class RecipeRead(BaseModel):
    """Full public view of a recipe (DESIGN §6, §13 #4)."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: str
    tags: list[str]
    mode: str
    config_json: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_recipe(cls, recipe: Recipe) -> RecipeRead:
        """Build a read model from an ORM ``Recipe`` row."""
        return cls.model_validate(recipe)
