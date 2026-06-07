"""Key/value settings store (DESIGN §6): default model/precision, theme/accent, etc."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlmodel import Field, SQLModel

from app.models.base import jsonb_column, utcnow


class Setting(SQLModel, table=True):
    __tablename__ = "setting"

    key: str = Field(primary_key=True)
    value_json: dict[str, Any] = Field(default_factory=dict, sa_column=jsonb_column())
    updated_at: datetime = Field(default_factory=utcnow)
