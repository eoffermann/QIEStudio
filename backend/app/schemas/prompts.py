"""Pydantic request/response models for the prompts API (DESIGN §5.2, §7).

These shape the JSON surface of ``/api/prompts`` — list/read responses plus the
create/patch/duplicate bodies. A prompt carries two *ordered* child collections
(images + LoRA bindings); the ordering of ``images`` is model-significant (Edit
mode passes the composed input list to the pipeline in this exact order) and is
preserved end-to-end.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.models.prompt import Prompt, PromptImage, PromptLora


class PromptImageIn(BaseModel):
    """One entry in a prompt's ordered image list (a pinned asset or an open slot).

    For ``role="pinned"`` an ``asset_id`` is required. For ``role="slot"`` a
    ``slot_name`` is required and ``slot_min``/``slot_max``/``hint`` describe how
    the user fills it at run time (DESIGN §5.2).
    """

    role: str  # pinned | slot
    asset_id: str | None = None
    slot_name: str | None = None
    slot_min: int | None = None
    slot_max: int | None = None
    hint: str | None = None


class PromptImageRead(PromptImageIn):
    """Public view of a :class:`~app.models.prompt.PromptImage` row."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    position: int


class PromptLoraIn(BaseModel):
    """A LoRA binding carried by a saved prompt (auto-applied on load)."""

    lora_id: str
    weight: float = 1.0


class PromptLoraRead(PromptLoraIn):
    """Public view of a :class:`~app.models.prompt.PromptLora` row."""

    model_config = ConfigDict(from_attributes=True)

    id: str


class PromptCreate(BaseModel):
    """Body for ``POST /api/prompts`` — create / save-with-name (DESIGN §7).

    ``images`` and ``loras`` are optional; when supplied they replace the (empty)
    child collections in declared order.
    """

    name: str
    text: str = ""
    tags: list[str] = []
    mode: str = "any"  # generate | edit | any
    favorite: bool = False
    defaults_json: dict[str, Any] = {}
    images: list[PromptImageIn] | None = None
    loras: list[PromptLoraIn] | None = None


class PromptUpdate(BaseModel):
    """Body for ``PATCH /api/prompts/{id}`` — rename / edit / bindings (DESIGN §7).

    Every field is optional; ``None`` means "leave unchanged". Supplying ``images``
    or ``loras`` atomically *replaces* that ordered collection (pass ``[]`` to clear).
    """

    name: str | None = None
    text: str | None = None
    tags: list[str] | None = None
    mode: str | None = None
    favorite: bool | None = None
    defaults_json: dict[str, Any] | None = None
    images: list[PromptImageIn] | None = None
    loras: list[PromptLoraIn] | None = None


class PromptRead(BaseModel):
    """Full public view of a prompt with its ordered children (DESIGN §5.2, §6)."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    text: str
    tags: list[str]
    mode: str
    favorite: bool
    defaults_json: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    last_used_at: datetime | None = None
    images: list[PromptImageRead] = []
    loras: list[PromptLoraRead] = []

    @classmethod
    def build(
        cls,
        prompt: Prompt,
        images: list[PromptImage],
        loras: list[PromptLora],
    ) -> PromptRead:
        """Assemble a read model from a prompt row and its ordered children."""
        return cls(
            id=prompt.id,
            name=prompt.name,
            text=prompt.text,
            tags=list(prompt.tags),
            mode=prompt.mode,
            favorite=prompt.favorite,
            defaults_json=dict(prompt.defaults_json),
            created_at=prompt.created_at,
            updated_at=prompt.updated_at,
            last_used_at=prompt.last_used_at,
            images=[PromptImageRead.model_validate(i) for i in images],
            loras=[PromptLoraRead.model_validate(loro) for loro in loras],
        )
