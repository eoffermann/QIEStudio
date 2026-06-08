"""Tests for the integration credential store (DESIGN §5.7).

DB-backed (uses the ``session`` fixture, which skips when no PostgreSQL is reachable). The
provider-validation network call is monkeypatched via the ``_get`` seam so tests never hit
the network. Asserts keys are encrypted at rest and never echoed in plaintext.
"""

from __future__ import annotations

import pytest
from app.interfaces.auth import Principal
from app.models.integration import Integration
from app.security import decrypt_secret
from app.services import integrations

pytestmark = pytest.mark.db


@pytest.fixture
def principal() -> Principal:
    return Principal()


@pytest.fixture
def stub_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(integrations, "_get", lambda url, headers, params=None: 200)


@pytest.fixture
def stub_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(integrations, "_get", lambda url, headers, params=None: 401)


def test_set_key_encrypts_and_validates(session, principal, stub_valid) -> None:
    integration = integrations.set_key(
        provider="huggingface", key="hf_secret_token", label="my hf",
        session=session, principal=principal,
    )
    assert integration.status == "valid"
    assert integration.label == "my hf"
    # Stored ciphertext must not be the plaintext, but must decrypt back to it.
    assert integration.encrypted_key != "hf_secret_token"
    assert decrypt_secret(integration.encrypted_key) == "hf_secret_token"
    assert integration.last_validated_at is not None


def test_invalid_key_marked_invalid_but_stored(session, principal, stub_invalid) -> None:
    integration = integrations.set_key(
        provider="civitai", key="bad", session=session, principal=principal
    )
    assert integration.status == "invalid"
    assert integration.encrypted_key  # still stored


def test_set_key_upserts(session, principal, stub_valid) -> None:
    from sqlmodel import select

    integrations.set_key(provider="civitai", key="first", session=session, principal=principal)
    integrations.set_key(provider="civitai", key="second", session=session, principal=principal)
    # Exactly one row, and it holds the latest value.
    rows = session.exec(
        select(Integration).where(Integration.provider == "civitai")
    ).all()
    assert len(rows) == 1
    assert integrations.get_key_plaintext(
        provider="civitai", session=session, principal=principal
    ) == "second"


def test_get_key_plaintext_roundtrip(session, principal, stub_valid) -> None:
    integrations.set_key(provider="huggingface", key="tok123", session=session, principal=principal)
    assert integrations.get_key_plaintext(
        provider="huggingface", session=session, principal=principal
    ) == "tok123"


def test_get_key_plaintext_absent(session, principal) -> None:
    assert integrations.get_key_plaintext(
        provider="civitai", session=session, principal=principal
    ) is None


def test_list_integrations_masks_key(session, principal, stub_valid) -> None:
    integrations.set_key(
        provider="huggingface", key="hf_abcd1234", label="lbl",
        session=session, principal=principal,
    )
    rows = integrations.list_integrations(session=session, principal=principal)
    by_provider = {r["provider"]: r for r in rows}

    # Both supported providers always appear.
    assert set(by_provider) == set(integrations.SUPPORTED_PROVIDERS)

    hf = by_provider["huggingface"]
    assert hf["status"] == "valid"
    assert hf["configured"] is True
    assert hf["masked_hint"] == "••••1234"
    # No field should leak the plaintext.
    assert "hf_abcd1234" not in str(hf)

    civ = by_provider["civitai"]
    assert civ["status"] == "unconfigured"
    assert civ["configured"] is False
    assert civ["masked_hint"] is None


def test_delete_integration(session, principal, stub_valid) -> None:
    integrations.set_key(provider="civitai", key="k", session=session, principal=principal)
    assert integrations.delete_integration(
        provider="civitai", session=session, principal=principal
    ) is True
    assert integrations.delete_integration(
        provider="civitai", session=session, principal=principal
    ) is False


@pytest.fixture
def clean_hf_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # monkeypatch restores os.environ after the test, so direct writes by the code under
    # test are rolled back too.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)


def test_set_hf_key_applies_token_to_env(session, principal, stub_valid, clean_hf_env) -> None:
    import os

    integrations.set_key(
        provider="huggingface", key="hf_secret", session=session, principal=principal
    )
    assert os.environ["HF_TOKEN"] == "hf_secret"
    assert os.environ["HUGGING_FACE_HUB_TOKEN"] == "hf_secret"


def test_delete_hf_key_clears_token_env(session, principal, stub_valid, clean_hf_env) -> None:
    import os

    integrations.set_key(
        provider="huggingface", key="hf_secret", session=session, principal=principal
    )
    integrations.delete_integration(
        provider="huggingface", session=session, principal=principal
    )
    assert "HF_TOKEN" not in os.environ
    assert "HUGGING_FACE_HUB_TOKEN" not in os.environ


def test_civitai_key_does_not_touch_hf_env(session, principal, stub_valid, clean_hf_env) -> None:
    import os

    integrations.set_key(provider="civitai", key="cv", session=session, principal=principal)
    assert "HF_TOKEN" not in os.environ


def test_unknown_provider_rejected(session, principal) -> None:
    with pytest.raises(integrations.UnknownProviderError):
        integrations.set_key(provider="dropbox", key="x", session=session, principal=principal)


def test_empty_key_rejected(session, principal, stub_valid) -> None:
    with pytest.raises(ValueError):
        integrations.set_key(provider="civitai", key="   ", session=session, principal=principal)
