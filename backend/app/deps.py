"""Shared FastAPI dependencies (request-scoped principal, storage, queue, DB session).

Routers depend on these so the infrastructure seams (auth/storage/queue) stay swappable and
request handlers stay stateless (DESIGN §4.1a).
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Header
from sqlmodel import Session

from app.db import get_session
from app.interfaces.auth import AuthProvider, NoAuthProvider, Principal
from app.interfaces.queue import InProcessJobQueue, get_job_queue
from app.interfaces.storage import StorageProvider, build_storage_provider


@lru_cache
def get_auth_provider() -> AuthProvider:
    """The configured auth provider (NoAuth in v1)."""
    return NoAuthProvider()


@lru_cache
def get_storage() -> StorageProvider:
    """The configured storage provider (LocalFs in v1)."""
    return build_storage_provider()


def get_principal(
    authorization: Annotated[str | None, Header()] = None,
) -> Principal:
    """Resolve the request principal via the auth provider."""
    return get_auth_provider().principal_for(authorization=authorization)


def db_session() -> Iterator[Session]:
    yield from get_session()


# Convenience annotated aliases for router signatures.
DbSession = Annotated[Session, Depends(db_session)]
CurrentPrincipal = Annotated[Principal, Depends(get_principal)]
Storage = Annotated[StorageProvider, Depends(get_storage)]
Queue = Annotated[InProcessJobQueue, Depends(get_job_queue)]
