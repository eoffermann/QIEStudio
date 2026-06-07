"""Pydantic request/response models for the LoRA API (DESIGN §5.3, §7).

These shape ``/api/loras`` — list/read responses plus the import / patch bodies. The
``.safetensors`` blob is uploaded as multipart form-data on ``/upload`` (handled in the
router), not modelled here.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.lora import Lora


class LoraRead(BaseModel):
    """Public view of a :class:`~app.models.lora.Lora` row (DESIGN §5.3, §6)."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    storage_key: str
    thumb_key: str | None = None
    description: str
    trigger_words: list[str]
    recommended_weight: float
    base_compat: str
    modes: list[str]
    source: str
    source_ref: str
    license: str
    sha256: str
    bytes: int
    enabled: bool
    created_at: datetime

    @classmethod
    def from_lora(cls, lora: Lora) -> LoraRead:
        """Build a read model from an ORM ``Lora`` row."""
        return cls.model_validate(lora)


class LoraImport(BaseModel):
    """Body for ``POST /api/loras/import`` — install via HF / CivitAI / URL (DESIGN §7).

    ``source`` selects the importer; ``ref`` is the repo id, CivitAI URL/id, or direct URL.
    ``filename`` optionally pins the HF weight file when auto-detection is not desired.
    """

    source: str  # hf | civitai | url
    ref: str
    name: str | None = None
    filename: str | None = None


class LoraUpdate(BaseModel):
    """Body for ``PATCH /api/loras/{id}`` — edit metadata (DESIGN §7).

    Every field is optional; ``None`` means "leave unchanged".
    """

    name: str | None = None
    description: str | None = None
    trigger_words: list[str] | None = None
    recommended_weight: float | None = None
    base_compat: str | None = None
    modes: list[str] | None = None
    license: str | None = None
    enabled: bool | None = None


class LoraSearchResult(BaseModel):
    """One entry from ``GET /api/loras/search`` (HF Hub or CivitAI)."""

    source: str  # hf | civitai
    name: str
    ref: str  # repo id (hf) or "model_id" / "model_id@version_id" (civitai)
    base_model: str = ""
    trigger_words: list[str] = []
