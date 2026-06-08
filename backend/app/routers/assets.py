"""Assets API router (DESIGN §5.1, §7).

Exposes upload (ephemeral), promote-to-library, list/filter, binary + thumbnail
streaming, update (rename/tag/move), and delete. All work is delegated to
:mod:`app.services.asset_store`; binaries are served through the
:class:`~app.interfaces.storage.StorageProvider` (never absolute paths).
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from app.deps import CurrentPrincipal, DbSession, Storage
from app.interfaces.storage import StorageProvider
from app.schemas.assets import AssetRead, AssetUpdate, UploadResponse
from app.services import asset_store

router = APIRouter(prefix="/api/assets", tags=["assets"])

# Streamed binaries are chunked through the storage provider's read handle.
_STREAM_CHUNK = 64 * 1024
_THUMB_MEDIA_TYPE = "image/webp"


@router.post("/upload", response_model=UploadResponse)
async def upload_assets(
    session: DbSession,
    principal: CurrentPrincipal,
    storage: Storage,
    files: list[UploadFile] = File(...),  # noqa: B008
) -> UploadResponse:
    """Import one or many uploaded files as ephemeral assets (DESIGN §7)."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    created = []
    for upload in files:
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"Empty file: {upload.filename!r}")
        try:
            asset = asset_store.ingest_upload(
                data=data,
                filename=upload.filename or "upload",
                session=session,
                storage=storage,
                principal=principal,
                scope="ephemeral",
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=415, detail=f"Could not read image {upload.filename!r}: {exc}"
            ) from exc
        created.append(AssetRead.from_asset(asset))
    return UploadResponse(assets=created)


@router.post("/{asset_id}/promote", response_model=AssetRead)
def promote_asset(
    asset_id: str,
    session: DbSession,
    principal: CurrentPrincipal,
) -> AssetRead:
    """Promote an ephemeral upload or output to the persistent library (DESIGN §5.1)."""
    try:
        asset = asset_store.promote_to_library(
            asset_id=asset_id, session=session, principal=principal
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Asset not found") from exc
    return AssetRead.from_asset(asset)


@router.get("", response_model=list[AssetRead])
def list_assets(
    session: DbSession,
    principal: CurrentPrincipal,
    scope: str | None = None,
    tag: str | None = None,
    q: str | None = None,
    source: str | None = None,
) -> list[AssetRead]:
    """List assets for the owner, filtered by scope/tag/source and a text query (DESIGN §7)."""
    assets = asset_store.list_assets(
        session=session, principal=principal, scope=scope, tag=tag, q=q, source=source
    )
    return [AssetRead.from_asset(a) for a in assets]


@router.get("/{asset_id}", response_model=AssetRead)
def get_asset_detail(
    asset_id: str,
    session: DbSession,
    principal: CurrentPrincipal,
) -> AssetRead:
    """Fetch a single asset by id (used for send-output-to-input / reuse) (DESIGN §5.5)."""
    asset = asset_store.get_asset(asset_id=asset_id, session=session)
    if asset is None or asset.owner_id != principal.owner_id:
        raise HTTPException(status_code=404, detail="Asset not found")
    return AssetRead.from_asset(asset)


@router.get("/{asset_id}/file")
def get_asset_file(
    asset_id: str,
    session: DbSession,
    storage: Storage,
) -> StreamingResponse:
    """Stream an asset's binary (DESIGN §7)."""
    asset = asset_store.get_asset(asset_id=asset_id, session=session)
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    media_type = f"image/{asset.format.lower()}" if asset.format else "application/octet-stream"
    return _stream(storage, asset.storage_key, media_type)


@router.get("/{asset_id}/thumb")
def get_asset_thumb(
    asset_id: str,
    session: DbSession,
    storage: Storage,
) -> StreamingResponse:
    """Stream an asset's WebP thumbnail (DESIGN §7)."""
    asset = asset_store.get_asset(asset_id=asset_id, session=session)
    if asset is None or not asset.thumb_key:
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return _stream(storage, asset.thumb_key, _THUMB_MEDIA_TYPE)


@router.patch("/{asset_id}", response_model=AssetRead)
def update_asset(
    asset_id: str,
    body: AssetUpdate,
    session: DbSession,
    principal: CurrentPrincipal,
) -> AssetRead:
    """Rename / re-tag / move an asset to a collection (DESIGN §7)."""
    try:
        asset = asset_store.update_asset(
            asset_id=asset_id,
            session=session,
            principal=principal,
            name=body.name,
            description=body.description,
            tags=body.tags,
            collection=body.collection,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Asset not found") from exc
    return AssetRead.from_asset(asset)


@router.delete("/{asset_id}", status_code=204)
def delete_asset(
    asset_id: str,
    session: DbSession,
    principal: CurrentPrincipal,
    storage: Storage,
) -> Response:
    """Delete an asset and its (unshared) binaries (DESIGN §7)."""
    try:
        asset_store.delete_asset(
            asset_id=asset_id, session=session, principal=principal, storage=storage
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Asset not found") from exc
    return Response(status_code=204)


def _stream(storage: StorageProvider, key: str, media_type: str) -> StreamingResponse:
    """Stream a storage key's bytes in chunks via the storage provider."""

    def _iter() -> object:
        with storage.open_read(key) as fh:
            while chunk := fh.read(_STREAM_CHUNK):
                yield chunk

    return StreamingResponse(_iter(), media_type=media_type)
