"""Foundation unit tests: imports, health, interfaces, crypto, model registration."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_app_imports_and_models_register() -> None:
    import app.models  # noqa: F401
    from sqlmodel import SQLModel

    tables = set(SQLModel.metadata.tables)
    assert {
        "asset",
        "prompt",
        "prompt_image",
        "prompt_lora",
        "lora",
        "job",
        "job_input",
        "job_output",
        "integration",
        "setting",
    } <= tables


def test_healthz(client) -> None:  # noqa: ANN001
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_no_auth_provider_returns_implicit_owner() -> None:
    from app.interfaces.auth import IMPLICIT_OWNER_ID, NoAuthProvider

    p = NoAuthProvider().principal_for()
    assert p.owner_id == IMPLICIT_OWNER_ID


def test_local_storage_roundtrip_and_traversal_guard(tmp_path: Path) -> None:
    from app.interfaces.storage import LocalFsStorage, normalize_key

    store = LocalFsStorage(tmp_path)
    store.put_bytes("assets/ab/cd/x.bin", b"hello")
    assert store.exists("assets/ab/cd/x.bin")
    assert store.get_bytes("assets/ab/cd/x.bin") == b"hello"
    assert store.size("assets/ab/cd/x.bin") == 5
    assert store.local_path("assets/ab/cd/x.bin") is not None

    with pytest.raises(ValueError):
        normalize_key("../escape")
    with pytest.raises(ValueError):
        normalize_key("/abs/path")


def test_secret_encryption_roundtrip_and_mask() -> None:
    from app.security import decrypt_secret, encrypt_secret, mask_secret

    secret = "hf_abcdef, civitai_1234567890"
    token = encrypt_secret(secret)
    assert token != secret
    assert decrypt_secret(token) == secret
    assert mask_secret("hf_supersecretkey").endswith("tkey")
    assert "supersecret" not in mask_secret("hf_supersecretkey")


def test_inprocess_queue_runs_and_cancels() -> None:
    import time

    from app.interfaces.queue import InProcessJobQueue, QueuedTask

    q = InProcessJobQueue()
    q.start()
    results: list[str] = []
    q.submit(QueuedTask(job_id="j1", fn=lambda: results.append("ran"), label="t"))
    for _ in range(50):
        if results:
            break
        time.sleep(0.05)
    assert results == ["ran"]

    q.request_cancel("j2")
    assert q.is_canceled("j2")
    q.shutdown()


def test_settings_cors_origins_parses_csv_star_and_json(monkeypatch) -> None:  # noqa: ANN001
    """Regression: QIE_CORS_ORIGINS=* (or CSV) from .env must not crash startup (NoDecode)."""
    from app.config import Settings

    monkeypatch.setenv("QIE_CORS_ORIGINS", "*")
    assert Settings(_env_file=None).cors_origins == ["*"]
    monkeypatch.setenv("QIE_CORS_ORIGINS", "http://a.com,http://b.com")
    assert Settings(_env_file=None).cors_origins == ["http://a.com", "http://b.com"]
    monkeypatch.setenv("QIE_CORS_ORIGINS", '["x", "y"]')
    assert Settings(_env_file=None).cors_origins == ["x", "y"]
