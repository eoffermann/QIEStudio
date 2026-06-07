"""Image asset model (DESIGN §5.1, §6).

Covers all three image sources — ephemeral uploads, generated outputs, and persistent
library assets — distinguished by ``scope`` + ``source``. Binaries are addressed by
``storage_key``/``thumb_key`` via the StorageProvider (never absolute paths).
"""

from __future__ import annotations

from datetime import datetime

from sqlmodel import Field, SQLModel

from app.models.base import (
    created_at_column,
    id_field,
    new_id,
    owner_field,
    str_array_column,
    utcnow,
    workspace_field,
)

# Scope: "ephemeral" (per-run upload/output, GC'd) | "library" (persistent).
# Source: "upload" | "output" | "import".


class Asset(SQLModel, table=True):
    __tablename__ = "asset"

    id: str = id_field()
    owner_id: str | None = owner_field()
    workspace_id: str | None = workspace_field()

    scope: str = Field(default="ephemeral", index=True)
    source: str = Field(default="upload", index=True)
    source_job_id: str | None = Field(default=None, index=True)

    storage_key: str
    thumb_key: str | None = None

    name: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list, sa_column=str_array_column())
    collection: str | None = Field(default=None, index=True)

    width: int = 0
    height: int = 0
    format: str = ""  # PNG | JPEG | WEBP | GIF | ...
    bytes: int = 0
    sha256: str = Field(default="", index=True)  # content hash for de-dup (DESIGN §5.1)

    created_at: datetime = Field(default_factory=utcnow, sa_column=created_at_column())
    last_used_at: datetime | None = Field(default=None)


def make_asset_id() -> str:
    return new_id()
