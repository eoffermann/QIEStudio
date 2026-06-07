"""Asset store service (DESIGN §5.1, §6).

Handles every image source the app deals with — **ephemeral uploads**, **generated
outputs**, and the **persistent library** — plus promote-to-library, thumbnails,
content de-duplication, and format normalization.

Pipeline (on import):

1. Open via Pillow (HEIC/HEIF via ``pillow_heif``; first frame for animated GIF).
2. Apply EXIF orientation (:func:`PIL.ImageOps.exif_transpose`).
3. Normalize to RGB (palette/alpha/CMYK handled).
4. Persist a normalized PNG working copy + a WebP thumbnail (max 384px, aspect kept).
5. Record dimensions, format, byte size, and a content sha256.

De-duplication is by content sha256 *within an owner*: re-importing identical bytes
returns the existing :class:`~app.models.asset.Asset` instead of storing it twice.

All binaries are addressed by **storage keys** through the
:class:`~app.interfaces.storage.StorageProvider` — never absolute paths (DESIGN §4.1a).
Key scheme: ``assets/<sha[:2]>/<sha>.<ext>`` and ``thumbs/<sha[:2]>/<sha>.webp``;
output sidecars live at ``assets/<sha[:2]>/<sha>.json``.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageOps, PngImagePlugin
from sqlalchemy import ARRAY, String, cast
from sqlalchemy.dialects.postgresql import array
from sqlmodel import Session, select

from app.interfaces.auth import Principal
from app.interfaces.storage import StorageProvider
from app.logging_utils import phase
from app.models.asset import Asset
from app.models.job import JobOutput

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

log = logging.getLogger(__name__)

# Thumbnails: longest edge, aspect preserved (DESIGN §5.1).
THUMB_MAX_EDGE = 384

# Pillow format name -> normalized short format label stored on the Asset.
_FORMAT_LABELS = {
    "PNG": "PNG",
    "JPEG": "JPEG",
    "WEBP": "WEBP",
    "GIF": "GIF",
    "HEIF": "HEIF",
    "HEIC": "HEIF",
    "TIFF": "TIFF",
    "MPO": "JPEG",
}

_heif_registered = False


def _ensure_heif() -> None:
    """Lazily register the HEIF/HEIC opener with Pillow (DESIGN §5.1)."""
    global _heif_registered
    if _heif_registered:
        return
    try:
        import pillow_heif  # noqa: PLC0415

        pillow_heif.register_heif_opener()
    except Exception:  # noqa: BLE001
        # HEIC support is optional; non-HEIC formats still import fine without it.
        log.warning("pillow-heif unavailable; HEIC/HEIF imports will fail")
    _heif_registered = True


def _normalize_image(data: bytes) -> PILImage:
    """Open, EXIF-orient, and normalize an image to RGB (DESIGN §5.1).

    Handles palette/alpha/CMYK and takes the first frame of an animated GIF.
    """
    _ensure_heif()
    img = Image.open(io.BytesIO(data))
    # Animated GIF / multi-frame: first frame only.
    if getattr(img, "is_animated", False):
        img.seek(0)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        # Flatten alpha onto white; convert palette/CMYK/grayscale uniformly via RGB.
        if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            img = img.convert("RGB")
    return img


def _detect_format(data: bytes) -> str:
    """Return the normalized source-format label for the raw bytes."""
    _ensure_heif()
    with Image.open(io.BytesIO(data)) as probe:
        raw = (probe.format or "").upper()
    return _FORMAT_LABELS.get(raw, raw or "UNKNOWN")


def _png_bytes(img: PILImage, metadata: dict[str, Any] | None = None) -> bytes:
    """Encode an RGB image as PNG, optionally embedding ``metadata`` via PngInfo."""
    buf = io.BytesIO()
    pnginfo: PngImagePlugin.PngInfo | None = None
    if metadata:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("qie-metadata", json.dumps(metadata, default=str))
    img.save(buf, format="PNG", pnginfo=pnginfo)
    return buf.getvalue()


def _thumb_bytes(img: PILImage) -> bytes:
    """Produce a WebP thumbnail (max edge ``THUMB_MAX_EDGE``, aspect preserved)."""
    thumb = img.copy()
    thumb.thumbnail((THUMB_MAX_EDGE, THUMB_MAX_EDGE), Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="WEBP", quality=85, method=4)
    return buf.getvalue()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _asset_key(sha: str, ext: str) -> str:
    return f"assets/{sha[:2]}/{sha}.{ext}"


def _thumb_key(sha: str) -> str:
    return f"thumbs/{sha[:2]}/{sha}.webp"


def _sidecar_key(sha: str) -> str:
    return f"assets/{sha[:2]}/{sha}.json"


def _find_dupe(*, session: Session, principal: Principal, sha: str, scope: str) -> Asset | None:
    """Return an existing asset with the same content hash for this owner + scope."""
    stmt = select(Asset).where(
        Asset.sha256 == sha,
        Asset.owner_id == principal.owner_id,
        Asset.scope == scope,
    )
    return session.exec(stmt).first()


def ingest_upload(
    *,
    data: bytes,
    filename: str,
    session: Session,
    storage: StorageProvider,
    principal: Principal,
    scope: str = "ephemeral",
) -> Asset:
    """Import uploaded image bytes into an :class:`Asset` (DESIGN §5.1).

    Normalizes to a PNG working copy + WebP thumbnail, records metadata, and
    de-dupes by content sha256 within the owner + scope. Returns the existing asset
    on a hash hit instead of storing twice.
    """
    with phase(log, f"Normalizing upload {filename!r} ({len(data)} bytes)"):
        source_format = _detect_format(data)
        img = _normalize_image(data)
        png = _png_bytes(img)
        sha = _sha256(png)

    existing = _find_dupe(session=session, principal=principal, sha=sha, scope=scope)
    if existing is not None:
        log.info("Dedup hit for %s (scope=%s) -> asset %s", sha[:12], scope, existing.id)
        return existing

    key = _asset_key(sha, "png")
    thumb_key = _thumb_key(sha)
    storage.put_bytes(key, png)
    storage.put_bytes(thumb_key, _thumb_bytes(img))

    asset = Asset(
        owner_id=principal.owner_id,
        workspace_id=principal.workspace_id,
        scope=scope,
        source="upload",
        storage_key=key,
        thumb_key=thumb_key,
        name=filename,
        format="PNG",
        width=img.width,
        height=img.height,
        bytes=len(png),
        sha256=sha,
    )
    session.add(asset)
    session.commit()
    session.refresh(asset)
    log.info(
        "Ingested upload %s (%dx%d, src=%s) -> asset %s",
        filename,
        img.width,
        img.height,
        source_format,
        asset.id,
    )
    return asset


def save_output_image(
    *,
    image: PILImage,
    job_id: str,
    position: int,
    seed: int | None,
    metadata: dict[str, Any],
    session: Session,
    storage: StorageProvider,
    principal: Principal,
) -> tuple[Asset, JobOutput]:
    """Persist a generated output image (DESIGN §5.1, §5.5, §6).

    Stores the output binary (PNG with ``metadata`` embedded via PngInfo) + a WebP
    thumbnail + a JSON sidecar, then creates an ephemeral ``Asset(source="output",
    source_job_id=job_id)`` and a matching ``JobOutput`` row. Returns both.
    """
    with phase(log, f"Saving output for job {job_id} (pos {position})"):
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        png = _png_bytes(rgb, metadata=metadata)
        sha = _sha256(png)
        key = _asset_key(sha, "png")
        thumb_key = _thumb_key(sha)
        storage.put_bytes(key, png)
        storage.put_bytes(thumb_key, _thumb_bytes(rgb))
        storage.put_bytes(_sidecar_key(sha), json.dumps(metadata, default=str).encode("utf-8"))

    asset = Asset(
        owner_id=principal.owner_id,
        workspace_id=principal.workspace_id,
        scope="ephemeral",
        source="output",
        source_job_id=job_id,
        storage_key=key,
        thumb_key=thumb_key,
        name=f"{job_id}-{position}",
        format="PNG",
        width=rgb.width,
        height=rgb.height,
        bytes=len(png),
        sha256=sha,
    )
    session.add(asset)

    output = JobOutput(
        job_id=job_id,
        position=position,
        storage_key=key,
        thumb_key=thumb_key,
        seed=seed,
        metadata_json=metadata,
    )
    session.add(output)
    session.commit()
    session.refresh(asset)
    session.refresh(output)
    log.info("Saved output asset %s + job_output %s for job %s", asset.id, output.id, job_id)
    return asset, output


def promote_to_library(*, asset_id: str, session: Session, principal: Principal) -> Asset:
    """Promote an ephemeral upload or output to a persistent library asset (DESIGN §5.1).

    Works on both uploads and outputs, carrying along source metadata (``source`` and
    ``source_job_id`` are preserved; only ``scope`` flips to ``"library"``).
    """
    asset = _get_owned(asset_id=asset_id, session=session, principal=principal)
    if asset is None:
        raise KeyError(asset_id)
    if asset.scope != "library":
        asset.scope = "library"
        session.add(asset)
        session.commit()
        session.refresh(asset)
        log.info("Promoted asset %s to library (source=%s)", asset.id, asset.source)
    return asset


def get_asset(*, asset_id: str, session: Session) -> Asset | None:
    """Fetch an asset by id (no ownership filter)."""
    return session.get(Asset, asset_id)


def _get_owned(*, asset_id: str, session: Session, principal: Principal) -> Asset | None:
    """Fetch an asset by id, scoped to the principal's owner."""
    asset = session.get(Asset, asset_id)
    if asset is None or asset.owner_id != principal.owner_id:
        return None
    return asset


def open_pil(*, asset: Asset, storage: StorageProvider) -> PILImage:
    """Open an asset's stored binary as an RGB PIL image, for inference inputs (DESIGN §5.5)."""
    _ensure_heif()
    path = storage.local_path(asset.storage_key)
    if path is not None and path.exists():
        img = Image.open(path)
    else:
        img = Image.open(io.BytesIO(storage.get_bytes(asset.storage_key)))
    img.load()
    return img if img.mode == "RGB" else img.convert("RGB")


def list_assets(
    *,
    session: Session,
    principal: Principal,
    scope: str | None = None,
    tag: str | None = None,
    q: str | None = None,
    source: str | None = None,
) -> list[Asset]:
    """List assets for the owner, filtered by scope/source/tag and a name/description query."""
    stmt = select(Asset).where(Asset.owner_id == principal.owner_id)
    if scope is not None:
        stmt = stmt.where(Asset.scope == scope)
    if source is not None:
        stmt = stmt.where(Asset.source == source)
    if tag is not None:
        # The tags column uses the base ARRAY type (DESIGN §6 / models.base), whose
        # ``contains`` isn't implemented; use the PostgreSQL ``@>`` containment operator
        # with an explicitly-typed text[] literal.
        stmt = stmt.where(Asset.tags.op("@>")(cast(array([tag]), ARRAY(String))))  # type: ignore[attr-defined]
    stmt = stmt.order_by(Asset.created_at.desc())  # type: ignore[attr-defined]
    assets = list(session.exec(stmt).all())
    if q:
        needle = q.lower()
        assets = [
            a for a in assets if needle in a.name.lower() or needle in a.description.lower()
        ]
    return assets


def update_asset(
    *,
    asset_id: str,
    session: Session,
    principal: Principal,
    name: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    collection: str | None = None,
) -> Asset:
    """Rename / re-tag / move an asset (DESIGN §7). ``None`` fields are left unchanged."""
    asset = _get_owned(asset_id=asset_id, session=session, principal=principal)
    if asset is None:
        raise KeyError(asset_id)
    if name is not None:
        asset.name = name
    if description is not None:
        asset.description = description
    if tags is not None:
        asset.tags = tags
    if collection is not None:
        asset.collection = collection
    session.add(asset)
    session.commit()
    session.refresh(asset)
    return asset


def delete_asset(
    *,
    asset_id: str,
    session: Session,
    principal: Principal,
    storage: StorageProvider,
) -> None:
    """Delete an asset row and its binaries (DESIGN §7).

    Binaries are content-addressed and may be shared by other rows (de-dup); they are
    only removed from storage once no other asset references the same keys.
    """
    asset = _get_owned(asset_id=asset_id, session=session, principal=principal)
    if asset is None:
        raise KeyError(asset_id)
    storage_key = asset.storage_key
    thumb_key = asset.thumb_key
    sha = asset.sha256
    session.delete(asset)
    session.commit()

    others = session.exec(select(Asset).where(Asset.storage_key == storage_key)).first()
    if others is None:
        storage.delete(storage_key)
        if thumb_key:
            storage.delete(thumb_key)
        if sha:
            storage.delete(_sidecar_key(sha))
    log.info("Deleted asset %s", asset_id)
