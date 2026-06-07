"""Shared model mixins and column helpers (DESIGN §6)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import ARRAY, Column, DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field


def new_id() -> str:
    """Generate a compact unique primary key."""
    return uuid4().hex


def utcnow() -> datetime:
    """Timezone-aware current UTC timestamp."""
    return datetime.now(UTC)


def id_field() -> str:
    """A primary-key string id with a generated default."""
    return Field(default_factory=new_id, primary_key=True)


def owner_field() -> str:
    """A nullable owner id defaulted to the implicit local owner (DESIGN §4.1a).

    Imported lazily to avoid a circular import with the interfaces package.
    """
    from app.interfaces.auth import IMPLICIT_OWNER_ID

    return Field(default=IMPLICIT_OWNER_ID, index=True, nullable=True)


def workspace_field() -> str:
    from app.interfaces.auth import IMPLICIT_WORKSPACE_ID

    return Field(default=IMPLICIT_WORKSPACE_ID, index=True, nullable=True)


def str_array_column() -> Column:
    """A PostgreSQL ``text[]`` column for tag/trigger-word lists."""
    return Column(ARRAY(String), nullable=False, server_default="{}")


def jsonb_column(nullable: bool = False) -> Column:
    """A PostgreSQL ``jsonb`` column for free-form metadata blobs."""
    return Column(JSONB, nullable=nullable)


def created_at_column() -> Column:
    return Column(DateTime(timezone=True), nullable=False, default=utcnow)


def updated_at_column() -> Column:
    return Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)
