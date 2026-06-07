"""Pydantic request/response models for the integrations API (DESIGN §5.7, §7).

These shape ``/api/integrations``. The API key is **write-only** — it is accepted on
``PUT`` but never echoed back; responses carry only a masked hint plus validation status.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class IntegrationSet(BaseModel):
    """Body for ``PUT /api/integrations/{provider}`` — set/validate a credential."""

    key: str
    label: str = ""


class IntegrationStatus(BaseModel):
    """Masked public view of a provider credential (never the key)."""

    provider: str
    label: str
    status: str  # unconfigured | valid | invalid
    configured: bool
    last_validated_at: datetime | None = None
    masked_hint: str | None = None
