"""Integrations API (DESIGN §5.7, §7) — encrypted external-service credentials.

    GET    /api/integrations             # providers + status (keys masked)
    PUT    /api/integrations/{provider}  # set/validate API key
    DELETE /api/integrations/{provider}

Keys are write-only: accepted on ``PUT`` but never returned. Responses carry only masked
status, sourced from :mod:`app.services.integrations`.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.deps import CurrentPrincipal, DbSession
from app.schemas.integrations import IntegrationSet, IntegrationStatus
from app.services import integrations

router = APIRouter(prefix="/api/integrations", tags=["integrations"])


@router.get("", response_model=list[IntegrationStatus])
def list_integrations(
    session: DbSession,
    principal: CurrentPrincipal,
) -> list[IntegrationStatus]:
    """List every supported provider with masked status (never the key)."""
    rows = integrations.list_integrations(session=session, principal=principal)
    return [IntegrationStatus(**row) for row in rows]


@router.put("/{provider}", response_model=IntegrationStatus)
def set_integration(
    provider: str,
    body: IntegrationSet,
    session: DbSession,
    principal: CurrentPrincipal,
) -> IntegrationStatus:
    """Set/validate a provider API key. Returns masked status, not the key."""
    try:
        integrations.set_key(
            provider=provider,
            key=body.key,
            label=body.label,
            session=session,
            principal=principal,
        )
    except integrations.UnknownProviderError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    rows = integrations.list_integrations(session=session, principal=principal)
    match = next(r for r in rows if r["provider"] == provider.lower().strip())
    return IntegrationStatus(**match)


@router.delete("/{provider}", status_code=status.HTTP_204_NO_CONTENT)
def delete_integration(
    provider: str,
    session: DbSession,
    principal: CurrentPrincipal,
) -> None:
    """Delete a provider credential."""
    try:
        removed = integrations.delete_integration(
            provider=provider, session=session, principal=principal
        )
    except integrations.UnknownProviderError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No credential configured for {provider!r}",
        )
