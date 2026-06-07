"""Integration credential model (DESIGN §5.7, §6).

API keys (HF, CivitAI) are stored **encrypted at rest** (Fernet, key derived from the app
secret) and never returned to the client in plaintext.
"""

from __future__ import annotations

from datetime import datetime

from sqlmodel import Field, SQLModel

from app.models.base import id_field, owner_field, utcnow, workspace_field


class Integration(SQLModel, table=True):
    __tablename__ = "integration"

    id: str = id_field()
    owner_id: str | None = owner_field()
    workspace_id: str | None = workspace_field()

    provider: str = Field(index=True)  # huggingface | civitai
    encrypted_key: str = ""  # Fernet ciphertext; never exposed to clients
    label: str = ""
    status: str = "unconfigured"  # unconfigured | valid | invalid
    last_validated_at: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
