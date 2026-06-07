"""Shared pytest fixtures.

Test layers:
- **Pure-logic tests** (resolution math, advisor, crypto, storage keys, civitai URL
  parsing, prompt-slot resolution) need no DB and run anywhere.
- **DB tests** (``@pytest.mark.db``) need a live PostgreSQL. Point ``QIE_DATABASE_URL`` at a
  throwaway DB (the compose ``db`` service works). If unreachable, these tests *skip* rather
  than fail, so the pure-logic suite stays green on a bare host.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

# Configure a temp data root + dev secret BEFORE importing app modules (settings cache).
_TMP_DATA = Path(tempfile.mkdtemp(prefix="qie-test-data-"))
os.environ.setdefault("QIE_DATA_ROOT", str(_TMP_DATA))
os.environ.setdefault("QIE_SECRET_KEY", "test-secret-key-deterministic-0123456789")
os.environ.setdefault("QIE_RUN_MIGRATIONS_ON_STARTUP", "false")


@pytest.fixture(scope="session")
def data_root() -> Path:
    return _TMP_DATA


def _db_reachable() -> bool:
    try:
        from app.db import get_engine
        from sqlalchemy import text

        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="session")
def db_ready() -> bool:
    return _db_reachable()


@pytest.fixture
def session(db_ready: bool) -> Iterator[object]:
    """A DB session against a freshly-created schema; skips if no DB is reachable."""
    if not db_ready:
        pytest.skip("No PostgreSQL reachable (set QIE_DATABASE_URL)")
    from app.db import create_all, get_engine
    from sqlmodel import Session, SQLModel

    create_all()
    with Session(get_engine()) as s:
        yield s
    # Drop all tables to isolate tests.
    SQLModel.metadata.drop_all(get_engine())


@pytest.fixture
def client() -> Iterator[object]:
    """A FastAPI TestClient with lifespan run (migrations disabled in tests)."""
    from app.main import create_app
    from fastapi.testclient import TestClient

    with TestClient(create_app()) as c:
        yield c
