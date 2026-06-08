"""Integration credential store (DESIGN §5.7).

Encrypted-at-rest API keys for external providers (``huggingface``, ``civitai``). Keys are
encrypted via :mod:`app.security` (Fernet) and **never returned to the client in plaintext**
— only a masked hint + validation status. The plaintext is exposed solely to backend
services (LoRA imports / HF downloads) through :func:`get_key_plaintext`.

On save the key is validated with a lightweight authenticated call. That network call is
isolated in :func:`_validate_key` (which dispatches to per-provider validators using a
single :func:`_get` seam) so unit tests monkeypatch it and never touch the network.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import httpx
from sqlmodel import Session, select

from app.interfaces.auth import Principal
from app.models.base import utcnow
from app.models.integration import Integration
from app.security import decrypt_secret, encrypt_secret, mask_secret, try_decrypt_secret

log = logging.getLogger(__name__)

# Providers we manage. Extensible (DESIGN §5.7).
SUPPORTED_PROVIDERS: tuple[str, ...] = ("huggingface", "civitai")

# Validation endpoints (lightweight authenticated calls).
_HF_WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
_CIVITAI_PROBE_URL = "https://civitai.com/api/v1/models"


class UnknownProviderError(ValueError):
    """Raised when an unsupported provider name is supplied."""


def _require_provider(provider: str) -> str:
    provider = provider.lower().strip()
    if provider not in SUPPORTED_PROVIDERS:
        raise UnknownProviderError(
            f"Unknown provider {provider!r}; supported: {', '.join(SUPPORTED_PROVIDERS)}"
        )
    return provider


def _get(url: str, *, headers: dict[str, str], params: dict | None = None) -> int:
    """Issue a GET and return the HTTP status (the single network seam to monkeypatch)."""
    with httpx.Client(follow_redirects=True, timeout=15.0) as client:
        resp = client.get(url, headers=headers, params=params)
        return resp.status_code


def _validate_key(provider: str, key: str) -> bool:
    """Validate a credential with a lightweight authenticated call.

    Returns ``True`` if the provider accepts the key. Network errors are treated as
    "invalid" (the key save still succeeds, but status is recorded as ``invalid``).
    """
    try:
        if provider == "huggingface":
            status = _get(_HF_WHOAMI_URL, headers={"Authorization": f"Bearer {key}"})
            return status == 200
        if provider == "civitai":
            status = _get(
                _CIVITAI_PROBE_URL,
                headers={"Authorization": f"Bearer {key}"},
                params={"limit": 1},
            )
            return status == 200
    except httpx.HTTPError as exc:  # network/timeout — cannot confirm validity
        log.warning("Validation call for %s failed: %s", provider, exc)
        return False
    return False


def _find(session: Session, *, provider: str, principal: Principal) -> Integration | None:
    stmt = select(Integration).where(
        Integration.provider == provider,
        Integration.owner_id == principal.owner_id,
    )
    return session.exec(stmt).first()


def set_key(
    *,
    provider: str,
    key: str,
    label: str = "",
    session: Session,
    principal: Principal,
) -> Integration:
    """Create or update a provider credential, encrypting it and validating it on save.

    The key is validated (status ``valid``/``invalid``) and stored as Fernet ciphertext.
    The plaintext is never persisted or returned. Upserts on ``(provider, owner_id)``.
    """
    provider = _require_provider(provider)
    if not key or not key.strip():
        raise ValueError("API key must not be empty")
    key = key.strip()

    valid = _validate_key(provider, key)
    now: datetime = utcnow()

    integration = _find(session, provider=provider, principal=principal)
    if integration is None:
        integration = Integration(
            provider=provider,
            owner_id=principal.owner_id,
            workspace_id=principal.workspace_id,
            created_at=now,
        )
        session.add(integration)

    integration.encrypted_key = encrypt_secret(key)
    integration.label = label
    integration.status = "valid" if valid else "invalid"
    integration.last_validated_at = now
    integration.updated_at = now

    session.add(integration)
    session.commit()
    session.refresh(integration)
    log.info("Stored %s credential (status=%s)", provider, integration.status)
    # Make a HuggingFace token usable by every download path immediately (no restart).
    if provider == "huggingface":
        apply_hf_token_to_env(session=session, principal=principal)
    return integration


# Env vars huggingface_hub / transformers / diffusers read for the auth token. Setting both
# covers the current (`HF_TOKEN`) and legacy (`HUGGING_FACE_HUB_TOKEN`) names.
_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def apply_hf_token_to_env(*, session: Session, principal: Principal) -> bool:
    """Sync the stored HuggingFace token into the process environment.

    All weight downloads — the image pipelines, the Qwen-VL rewriter, the Nunchaku int4
    sources, and HF LoRA imports — ultimately call ``from_pretrained`` / ``hf_hub_download``,
    which read the token from ``HF_TOKEN`` (via ``huggingface_hub.get_token()``). Without
    this, the stored credential was only used by the explicit LoRA-import path, so every
    other download ran anonymously and the console logged "accessing HuggingFace without a
    token" (and was subject to anonymous rate limits / gated-repo denials).

    Sets both env vars when a token is present, clears them otherwise (so a deleted token
    stops authenticating). v1 is single-user, so this is called for the implicit owner.
    Returns ``True`` if a token was applied, ``False`` if cleared/absent.
    """
    token = get_key_plaintext(provider="huggingface", session=session, principal=principal)
    if token:
        for var in _HF_TOKEN_ENV_VARS:
            os.environ[var] = token
        log.info("Applied stored HuggingFace token to environment for weight downloads")
        return True
    for var in _HF_TOKEN_ENV_VARS:
        os.environ.pop(var, None)
    return False


def get_key_plaintext(
    *,
    provider: str,
    session: Session,
    principal: Principal,
) -> str | None:
    """Return the decrypted API key for backend use, or ``None`` if absent/undecryptable.

    Backend-only (LoRA imports, HF downloads). Never expose the result to clients.
    """
    provider = _require_provider(provider)
    integration = _find(session, provider=provider, principal=principal)
    if integration is None or not integration.encrypted_key:
        return None
    return try_decrypt_secret(integration.encrypted_key)


def list_integrations(*, session: Session, principal: Principal) -> list[dict]:
    """Return masked status for every supported provider (DESIGN §7 ``GET /api/integrations``).

    One entry per supported provider (configured or not). Never includes the key — only a
    masked hint derived from the decrypted value plus validation status.
    """
    rows = {
        i.provider: i
        for i in session.exec(
            select(Integration).where(Integration.owner_id == principal.owner_id)
        ).all()
    }
    out: list[dict] = []
    for provider in SUPPORTED_PROVIDERS:
        integration = rows.get(provider)
        if integration is None:
            out.append(
                {
                    "provider": provider,
                    "label": "",
                    "status": "unconfigured",
                    "configured": False,
                    "last_validated_at": None,
                    "masked_hint": None,
                }
            )
            continue
        hint: str | None = None
        if integration.encrypted_key:
            plaintext = try_decrypt_secret(integration.encrypted_key)
            hint = mask_secret(plaintext) if plaintext is not None else "••••"
        out.append(
            {
                "provider": provider,
                "label": integration.label,
                "status": integration.status,
                "configured": bool(integration.encrypted_key),
                "last_validated_at": integration.last_validated_at,
                "masked_hint": hint,
            }
        )
    return out


def delete_integration(*, provider: str, session: Session, principal: Principal) -> bool:
    """Delete a provider credential. Returns ``True`` if a row was removed."""
    provider = _require_provider(provider)
    integration = _find(session, provider=provider, principal=principal)
    if integration is None:
        return False
    session.delete(integration)
    session.commit()
    log.info("Deleted %s credential", provider)
    # Stop authenticating downloads once the HF token is gone.
    if provider == "huggingface":
        apply_hf_token_to_env(session=session, principal=principal)
    return True


# Re-export for symmetry / explicit backend use.
__all__ = [
    "SUPPORTED_PROVIDERS",
    "UnknownProviderError",
    "apply_hf_token_to_env",
    "decrypt_secret",
    "delete_integration",
    "get_key_plaintext",
    "list_integrations",
    "set_key",
]
