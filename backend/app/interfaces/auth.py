"""Auth provider seam (DESIGN §4.1a).

v1 is single-user with no auth: :class:`NoAuthProvider` returns one implicit owner. Every
user-owned row carries a nullable ``owner_id`` defaulted to this principal, so enabling
multi-user later is a matter of *filtering*, not a schema migration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

# The single implicit owner used throughout v1. Stored on every user-owned row so a future
# multi-user mode only has to start populating + filtering these columns.
IMPLICIT_OWNER_ID = "local"
IMPLICIT_WORKSPACE_ID = "local"


@dataclass(frozen=True)
class Principal:
    """The authenticated actor for a request (or the implicit local owner in v1)."""

    owner_id: str = IMPLICIT_OWNER_ID
    workspace_id: str = IMPLICIT_WORKSPACE_ID
    display_name: str = "Local User"


class AuthProvider(ABC):
    """Resolves the :class:`Principal` for an incoming request."""

    @abstractmethod
    def principal_for(self, *, authorization: str | None = None) -> Principal:
        """Return the principal for the request given its Authorization header (if any)."""


class NoAuthProvider(AuthProvider):
    """v1 provider: always returns the single implicit local owner."""

    _PRINCIPAL = Principal()

    def principal_for(self, *, authorization: str | None = None) -> Principal:  # noqa: ARG002
        return self._PRINCIPAL
