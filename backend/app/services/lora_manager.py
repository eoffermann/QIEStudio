"""LoRA registry + multi-source importer (DESIGN §5.3).

A registry over the :class:`~app.models.lora.Lora` row backed by the
:class:`~app.interfaces.storage.StorageProvider` (``.safetensors`` stored under ``loras/``).
Imports are supported from **local upload**, **direct URL**, **CivitAI**, and **Hugging
Face**, all of which compute a sha256 checksum and persist editable metadata.

**Download strategy (v1):** imports run **synchronously** with flushed progress logging
(per the user's long-running-phase convention). The interface is broker-ready — a router
may instead enqueue an import as a background job via :mod:`app.interfaces.queue` — but the
synchronous path keeps v1 simple and lets unit tests exercise the full flow. All network
access goes through injectable hooks (``download_fn`` / the ``huggingface_hub`` import and
:mod:`app.services.civitai`'s seam) so unit tests never touch the network.

There are **no size caps** (DESIGN: "No size/license caps").
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path

import httpx
from sqlmodel import Session, select

from app.interfaces.auth import Principal
from app.interfaces.storage import StorageProvider
from app.logging_utils import phase
from app.models.lora import Lora
from app.services import civitai, integrations

log = logging.getLogger(__name__)

# A function that fetches the bytes at a URL (injectable so tests don't hit the network).
DownloadFn = Callable[[str, str | None], bytes]

# Base-model tags (DESIGN §6: base_compat).
BASE_QWEN_IMAGE = "qwen-image"
BASE_QWEN_IMAGE_EDIT = "qwen-image-edit"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _storage_key(sha256: str, filename: str) -> str:
    """Content-addressed key under ``loras/`` (sharded by hash prefix to avoid huge dirs)."""
    suffix = Path(filename).suffix.lower() or ".safetensors"
    return f"loras/{sha256[:2]}/{sha256[2:4]}/{sha256}{suffix}"


def is_safetensors(*, data: bytes | None = None, filename: str | None = None) -> bool:
    """Cheaply verify a blob is a safetensors file by header structure and/or extension.

    The safetensors format begins with an 8-byte little-endian header length followed by a
    JSON header object, so the first non-whitespace header byte is ``{``. We accept a file
    if the extension is ``.safetensors`` *or* the header parses as a plausible safetensors
    header.
    """
    if filename and filename.lower().endswith(".safetensors"):
        return True
    if data is not None and len(data) >= 9:
        header_len = int.from_bytes(data[:8], "little")
        if 0 < header_len <= len(data) - 8:
            head = data[8 : 8 + min(header_len, 64)].lstrip()
            return head[:1] == b"{"
    return False


def _http_download(url: str, token: str | None = None) -> bytes:
    """Default download implementation with flushed progress logging."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    chunks: list[bytes] = []
    received = 0
    with (
        phase(log, f"Downloading LoRA from {url}"),
        httpx.Client(follow_redirects=True, timeout=None) as client,
        client.stream("GET", url, headers=headers) as resp,
    ):
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        for chunk in resp.iter_bytes(chunk_size=1 << 20):
            chunks.append(chunk)
            received += len(chunk)
            if total:
                log.info(
                    "  ... %d/%d MiB (%.0f%%)",
                    received >> 20,
                    total >> 20,
                    100.0 * received / total,
                )
            else:
                log.info("  ... %d MiB", received >> 20)
    return b"".join(chunks)


def _persist(
    *,
    data: bytes,
    filename: str,
    name: str,
    source: str,
    source_ref: str,
    session: Session,
    storage: StorageProvider,
    principal: Principal,
    description: str = "",
    trigger_words: list[str] | None = None,
    recommended_weight: float = 1.0,
    base_compat: str = BASE_QWEN_IMAGE,
    modes: list[str] | None = None,
    license: str = "",
    thumb_key: str | None = None,
) -> Lora:
    """Store the blob via the storage provider and create the registry row."""
    if not is_safetensors(data=data, filename=filename):
        raise ValueError(f"{filename!r} does not look like a safetensors file")

    sha256 = _sha256(data)
    key = _storage_key(sha256, filename)
    if not storage.exists(key):
        storage.put_bytes(key, data)

    lora = Lora(
        name=name,
        storage_key=key,
        thumb_key=thumb_key,
        description=description,
        trigger_words=trigger_words or [],
        recommended_weight=recommended_weight,
        base_compat=base_compat,
        modes=modes or [],
        source=source,
        source_ref=source_ref,
        license=license,
        sha256=sha256,
        bytes=len(data),
        owner_id=principal.owner_id,
        workspace_id=principal.workspace_id,
    )
    session.add(lora)
    session.commit()
    session.refresh(lora)
    log.info("Registered LoRA %s (%s, %d bytes)", lora.name, source, lora.bytes)
    return lora


def import_from_upload(
    *,
    data: bytes,
    filename: str,
    name: str | None = None,
    session: Session,
    storage: StorageProvider,
    principal: Principal,
) -> Lora:
    """Install a LoRA from an uploaded ``.safetensors`` blob (DESIGN §5.3)."""
    return _persist(
        data=data,
        filename=filename,
        name=name or Path(filename).stem,
        source="upload",
        source_ref=filename,
        session=session,
        storage=storage,
        principal=principal,
    )


def import_from_url(
    *,
    url: str,
    name: str | None = None,
    session: Session,
    storage: StorageProvider,
    principal: Principal,
    download_fn: DownloadFn = _http_download,
) -> Lora:
    """Install a LoRA from a direct ``.safetensors`` URL (DESIGN §5.3).

    ``download_fn`` is injectable so unit tests supply bytes without network access.
    """
    data = download_fn(url, None)
    filename = url.rsplit("/", 1)[-1].split("?", 1)[0] or "lora.safetensors"
    return _persist(
        data=data,
        filename=filename,
        name=name or Path(filename).stem,
        source="url",
        source_ref=url,
        session=session,
        storage=storage,
        principal=principal,
    )


def import_from_civitai(
    *,
    url_or_id: str,
    name: str | None = None,
    session: Session,
    storage: StorageProvider,
    principal: Principal,
    download_fn: DownloadFn = _http_download,
) -> Lora:
    """Install a LoRA from CivitAI by URL or id (DESIGN §5.3).

    Normalizes the colour-domain URL → canonical model/version ids, resolves metadata
    (trigger words, preview, download url, base model, license) via
    :mod:`app.services.civitai`, then downloads the ``.safetensors``. The CivitAI API key
    is read from the encrypted integration store (authorizes mature content).
    """
    model_id, version_id = civitai.normalize_civitai_url(url_or_id)
    api_key = integrations.get_key_plaintext(
        provider="civitai", session=session, principal=principal
    )
    meta = civitai.fetch_model_metadata(
        model_id=model_id, version_id=version_id, api_key=api_key
    )
    if not meta.download_url:
        raise ValueError(f"CivitAI model {meta.model_id} has no downloadable .safetensors file")

    data = download_fn(meta.download_url, api_key)
    filename = f"{meta.name}.safetensors"
    base_compat = _civitai_base_compat(meta.base_model)
    return _persist(
        data=data,
        filename=filename,
        name=name or meta.name,
        source="civitai",
        source_ref=f"{meta.model_id}@{meta.version_id}" if meta.version_id else str(meta.model_id),
        session=session,
        storage=storage,
        principal=principal,
        trigger_words=meta.trigger_words,
        recommended_weight=meta.recommended_weight or 1.0,
        base_compat=base_compat,
        license=meta.license,
    )


def _civitai_base_compat(base_model: str) -> str:
    """Map a CivitAI ``baseModel`` tag to our ``base_compat`` enum (best-effort)."""
    low = (base_model or "").lower()
    if "edit" in low:
        return BASE_QWEN_IMAGE_EDIT
    return BASE_QWEN_IMAGE


# A function matching huggingface_hub.hf_hub_download's relevant signature (injectable).
HfDownloadFn = Callable[..., str]
HfListFn = Callable[[str, str | None], list[str]]


def _hf_list_files(repo_id: str, token: str | None) -> list[str]:
    """List files in an HF repo (isolated so tests can monkeypatch)."""
    from huggingface_hub import list_repo_files

    return list(list_repo_files(repo_id, token=token))


def _hf_download(repo_id: str, filename: str, token: str | None, cache_dir: str | None) -> str:
    """Download one file from HF into the cache, returning a local path (isolated seam)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id, filename=filename, token=token, cache_dir=cache_dir
    )


def _pick_hf_weight_file(files: list[str]) -> str | None:
    """Auto-detect the LoRA weight file among an HF repo's files."""
    safes = [f for f in files if f.lower().endswith(".safetensors")]
    if not safes:
        return None
    # Prefer a file whose name hints at a LoRA adapter; else the first safetensors.
    for f in safes:
        low = f.lower()
        if "lora" in low or "adapter" in low or "pytorch_lora" in low:
            return f
    return safes[0]


def import_from_hf(
    *,
    repo_id: str,
    filename: str | None = None,
    name: str | None = None,
    session: Session,
    storage: StorageProvider,
    principal: Principal,
    list_files_fn: HfListFn = _hf_list_files,
    download_fn: HfDownloadFn = _hf_download,
    cache_dir: str | None = None,
) -> Lora:
    """Install a LoRA from a Hugging Face repo (DESIGN §5.3).

    Auto-detects the weight file when ``filename`` is omitted, using the HF token from the
    encrypted integration store for gated/private repos. The ``list_files_fn`` /
    ``download_fn`` seams are injectable so unit tests avoid the Hub.
    """
    token = integrations.get_key_plaintext(
        provider="huggingface", session=session, principal=principal
    )
    if filename is None:
        files = list_files_fn(repo_id, token)
        filename = _pick_hf_weight_file(files)
        if filename is None:
            raise ValueError(f"No .safetensors weight file found in HF repo {repo_id!r}")

    with phase(log, f"Downloading {repo_id}:{filename} from Hugging Face"):
        local_path = download_fn(
            repo_id=repo_id, filename=filename, token=token, cache_dir=cache_dir
        )
    data = Path(local_path).read_bytes()
    return _persist(
        data=data,
        filename=filename,
        name=name or repo_id.rsplit("/", 1)[-1],
        source="hf",
        source_ref=f"{repo_id}:{filename}",
        session=session,
        storage=storage,
        principal=principal,
    )


def list_loras(*, session: Session, principal: Principal) -> list[Lora]:
    """List registered LoRAs for the principal (newest first)."""
    stmt = (
        select(Lora)
        .where(Lora.owner_id == principal.owner_id)
        .order_by(Lora.created_at.desc())  # type: ignore[attr-defined]
    )
    return list(session.exec(stmt).all())


def get_lora(*, lora_id: str, session: Session, principal: Principal) -> Lora | None:
    """Fetch one LoRA scoped to the principal."""
    lora = session.get(Lora, lora_id)
    if lora is None or lora.owner_id != principal.owner_id:
        return None
    return lora


def update_lora(
    *,
    lora_id: str,
    session: Session,
    principal: Principal,
    name: str | None = None,
    description: str | None = None,
    trigger_words: list[str] | None = None,
    recommended_weight: float | None = None,
    base_compat: str | None = None,
    modes: list[str] | None = None,
    license: str | None = None,
    enabled: bool | None = None,
) -> Lora | None:
    """Patch editable metadata on a LoRA (``None`` fields are left unchanged)."""
    lora = get_lora(lora_id=lora_id, session=session, principal=principal)
    if lora is None:
        return None
    if name is not None:
        lora.name = name
    if description is not None:
        lora.description = description
    if trigger_words is not None:
        lora.trigger_words = trigger_words
    if recommended_weight is not None:
        lora.recommended_weight = recommended_weight
    if base_compat is not None:
        lora.base_compat = base_compat
    if modes is not None:
        lora.modes = modes
    if license is not None:
        lora.license = license
    if enabled is not None:
        lora.enabled = enabled
    session.add(lora)
    session.commit()
    session.refresh(lora)
    return lora


def set_enabled(
    *, lora_id: str, enabled: bool, session: Session, principal: Principal
) -> Lora | None:
    """Toggle the enabled flag on a LoRA."""
    return update_lora(
        lora_id=lora_id, enabled=enabled, session=session, principal=principal
    )


def delete_lora(
    *, lora_id: str, session: Session, storage: StorageProvider, principal: Principal
) -> bool:
    """Delete a LoRA row and its stored blob (unless another row shares the blob)."""
    lora = get_lora(lora_id=lora_id, session=session, principal=principal)
    if lora is None:
        return False
    key = lora.storage_key
    session.delete(lora)
    session.commit()
    # Only drop the blob if no other row references the same content-addressed key.
    shared = session.exec(select(Lora).where(Lora.storage_key == key)).first()
    if shared is None:
        storage.delete(key)
    log.info("Deleted LoRA %s", lora_id)
    return True


def check_compat(*, lora: Lora, mode: str, base: str) -> list[str]:
    """Validate a LoRA against a planned run, returning human-readable warnings (DESIGN §5.3).

    The jobs subsystem calls this before applying adapters. Empty list means "no concerns".

    Args:
        lora: The LoRA being considered.
        mode: The run mode, ``"edit"`` or ``"generate"``.
        base: The target base-model family, ``"qwen-image"`` or ``"qwen-image-edit"``.
    """
    warnings: list[str] = []

    if lora.base_compat and base and lora.base_compat != base:
        warnings.append(
            f"LoRA '{lora.name}' targets {lora.base_compat} but the run uses {base}; "
            "it may not apply cleanly."
        )

    if lora.modes and mode not in lora.modes:
        warnings.append(
            f"LoRA '{lora.name}' is tagged for modes {lora.modes} but this run is '{mode}'."
        )

    if not lora.enabled:
        warnings.append(f"LoRA '{lora.name}' is disabled and will be skipped.")

    return warnings


def search_hf(*, query: str, limit: int = 20) -> list[dict]:
    """Search the HF Hub for LoRA models (DESIGN §7). Isolated so callers can degrade to []."""
    from huggingface_hub import HfApi

    api = HfApi()
    results: list[dict] = []
    for model in api.list_models(search=query, limit=limit, library="diffusers"):
        results.append({"repo_id": model.id, "name": model.id})
    return results
