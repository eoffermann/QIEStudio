"""Storage provider seam (DESIGN §4.1a).

All binary access goes through a :class:`StorageProvider`; the DB stores **storage keys**,
never absolute paths. v1 ships :class:`LocalFsStorage`; an S3/object-store provider can be
dropped in later without touching callers or the schema.

A *storage key* is a POSIX-style relative path like ``assets/ab/cd/<sha>.png``. Keys are
sanitized so they can never escape the storage root.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import BinaryIO


class StorageProvider(ABC):
    """Abstract binary store addressed by opaque string keys."""

    @abstractmethod
    def put_bytes(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def put_file(self, key: str, src_path: Path) -> None: ...

    @abstractmethod
    def get_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def open_read(self, key: str) -> BinaryIO: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def size(self, key: str) -> int: ...

    @abstractmethod
    def local_path(self, key: str) -> Path | None:
        """Return a filesystem path for the key if one exists (local provider), else None.

        Inference and image libraries (Pillow, safetensors) want a real path; object stores
        return ``None`` and callers fall back to streaming via :meth:`open_read`.
        """


def normalize_key(key: str) -> PurePosixPath:
    """Validate and normalize a storage key, rejecting traversal/absolute paths."""
    p = PurePosixPath(key)
    if p.is_absolute() or any(part == ".." for part in p.parts):
        raise ValueError(f"Unsafe storage key: {key!r}")
    if not p.parts:
        raise ValueError("Empty storage key")
    return p


class LocalFsStorage(StorageProvider):
    """Local-filesystem storage rooted at a single directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / Path(*normalize_key(key).parts)

    def put_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def put_file(self, key: str, src_path: Path) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_path, path)

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def open_read(self, key: str) -> BinaryIO:
        return self._path(key).open("rb")

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()

    def size(self, key: str) -> int:
        return self._path(key).stat().st_size

    def local_path(self, key: str) -> Path | None:
        return self._path(key)

    def iter_keys(self, prefix: str = "") -> Iterator[str]:
        """Yield all keys under an optional prefix (local-only convenience)."""
        base = self.root / Path(*normalize_key(prefix).parts) if prefix else self.root
        if not base.exists():
            return
        for p in base.rglob("*"):
            if p.is_file():
                yield p.relative_to(self.root).as_posix()


def build_storage_provider() -> StorageProvider:
    """Factory selecting the configured storage backend."""
    from app.config import get_settings

    settings = get_settings()
    if settings.storage_backend == "local":
        return LocalFsStorage(settings.data_root)
    raise ValueError(f"Unsupported storage backend: {settings.storage_backend!r}")
