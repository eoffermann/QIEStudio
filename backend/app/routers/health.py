"""Health/readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import text

from app import __version__
from app.db import get_engine

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness — the process is up. Does not touch the DB."""
    return {"status": "ok", "version": __version__}


@router.get("/readyz")
def readyz() -> dict[str, object]:
    """Readiness — verifies the DB is reachable."""
    db_ok = True
    detail = "ok"
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        db_ok = False
        detail = str(exc)
    return {"status": "ok" if db_ok else "degraded", "database": db_ok, "detail": detail}
