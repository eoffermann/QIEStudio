"""Pydantic request/response models for the assets API (DESIGN §5.1, §7).

These shape the JSON surface of ``/api/assets`` — list/read responses and the
PATCH update body. Binaries themselves are streamed by dedicated ``/file`` and
``/thumb`` endpoints, not embedded here.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.asset import Asset


class AssetRead(BaseModel):
    """Public view of an :class:`~app.models.asset.Asset` (DESIGN §5.1, §6).

    Storage keys are intentionally exposed so the frontend can build ``/file`` /
    ``/thumb`` URLs, but absolute paths never leave the storage provider.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    scope: str
    source: str
    source_job_id: str | None = None
    storage_key: str
    thumb_key: str | None = None
    name: str
    description: str
    tags: list[str]
    collection: str | None = None
    width: int
    height: int
    format: str
    bytes: int
    sha256: str
    created_at: datetime
    last_used_at: datetime | None = None

    @classmethod
    def from_asset(cls, asset: Asset) -> AssetRead:
        """Build a read model from an ORM ``Asset`` row."""
        return cls.model_validate(asset)


class AssetUpdate(BaseModel):
    """Editable fields for ``PATCH /api/assets/{id}`` — rename/tag/move (DESIGN §7).

    Every field is optional; ``None`` means "leave unchanged" so partial updates
    are unambiguous.
    """

    name: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    collection: str | None = None


class UploadResponse(BaseModel):
    """Result of ``POST /api/assets/upload`` — the created (ephemeral) assets."""

    assets: list[AssetRead]
