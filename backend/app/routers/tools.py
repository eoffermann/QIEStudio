"""Image tools API — background removal + upscale/refine (DESIGN §13 #1, #2, §7).

Both tools take an existing asset (by id, JSON body) **or** an uploaded multipart file,
run a light transform, and store the result as a *new ephemeral asset* via
:mod:`app.services.asset_store` so it flows straight back into the run composer (e.g. drop
a chair photo → auto-cut → place it in a pinned room — DESIGN §13 #1). All work is delegated
to the tool services; binaries live behind the :class:`StorageProvider` (never abs paths).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.deps import CurrentPrincipal, DbSession, Storage
from app.schemas.assets import AssetRead
from app.schemas.tools import (
    BackgroundRemovalRequest,
    ToolResult,
    UpscaleRequest,
)
from app.services import asset_store, background_removal, upscale

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tools", tags=["tools"])


def _resolve_input_image(
    *,
    asset_id: str | None,
    file_bytes: bytes | None,
    session: DbSession,
    storage: Storage,
):  # noqa: ANN202 - returns a PIL image; annotating needs the heavy import
    """Open the tool's input image from an asset id or uploaded bytes (DESIGN §13)."""
    from PIL import Image  # local import keeps the module light at import time

    if file_bytes:
        import io

        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    if asset_id:
        asset = asset_store.get_asset(asset_id=asset_id, session=session)
        if asset is None:
            raise HTTPException(status_code=404, detail="Asset not found")
        return asset_store.open_pil(asset=asset, storage=storage)
    raise HTTPException(status_code=400, detail="Provide an asset_id or upload a file")


def _store_result(
    *,
    image,  # noqa: ANN001 - PIL image
    filename: str,
    session: DbSession,
    storage: Storage,
    principal: CurrentPrincipal,
) -> AssetRead:
    """Persist a tool's output image as a new ephemeral asset (DESIGN §13, §5.1)."""
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    asset = asset_store.ingest_upload(
        data=buf.getvalue(),
        filename=filename,
        session=session,
        storage=storage,
        principal=principal,
        scope="ephemeral",
    )
    return AssetRead.from_asset(asset)


@router.post("/background-removal", response_model=ToolResult)
async def background_removal_endpoint(
    session: DbSession,
    principal: CurrentPrincipal,
    storage: Storage,
    asset_id: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
) -> ToolResult:
    """One-click subject cutout → new ephemeral RGBA asset (DESIGN §13 #1).

    Accepts either an ``asset_id`` form field or an uploaded ``file``. The cutout is run
    via rembg (the [inference]-only dep) and the transparent-background result is stored.
    """
    file_bytes = await file.read() if file is not None else None
    image = _resolve_input_image(
        asset_id=asset_id, file_bytes=file_bytes, session=session, storage=storage
    )
    try:
        cutout = background_removal.remove_background(image=image)
    except RuntimeError as exc:
        # rembg is absent in the dev image; surface a clear 503 rather than a 500.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    asset = _store_result(
        image=cutout,
        filename="cutout.png",
        session=session,
        storage=storage,
        principal=principal,
    )
    return ToolResult(asset=asset)


@router.post("/background-removal/json", response_model=ToolResult)
async def background_removal_json(
    body: BackgroundRemovalRequest,
    session: DbSession,
    principal: CurrentPrincipal,
    storage: Storage,
) -> ToolResult:
    """JSON (asset-id) form of background removal (DESIGN §13 #1)."""
    image = _resolve_input_image(
        asset_id=body.asset_id, file_bytes=None, session=session, storage=storage
    )
    try:
        cutout = background_removal.remove_background(image=image)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    asset = _store_result(
        image=cutout,
        filename="cutout.png",
        session=session,
        storage=storage,
        principal=principal,
    )
    return ToolResult(asset=asset)


@router.post("/upscale", response_model=ToolResult)
async def upscale_endpoint(
    session: DbSession,
    principal: CurrentPrincipal,
    storage: Storage,
    asset_id: str | None = Form(default=None),
    scale: float = Form(default=2.0),
    max_long_edge: int | None = Form(default=None),
    file: UploadFile | None = File(default=None),
) -> ToolResult:
    """High-quality Lanczos upscale → new ephemeral asset (DESIGN §13 #2).

    Accepts either an ``asset_id`` form field or an uploaded ``file``, plus ``scale`` and
    an optional ``max_long_edge`` cap.
    """
    file_bytes = await file.read() if file is not None else None
    image = _resolve_input_image(
        asset_id=asset_id, file_bytes=file_bytes, session=session, storage=storage
    )
    try:
        bigger = upscale.upscale_image(image=image, scale=scale, max_long_edge=max_long_edge)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    asset = _store_result(
        image=bigger,
        filename="upscaled.png",
        session=session,
        storage=storage,
        principal=principal,
    )
    return ToolResult(asset=asset)


@router.post("/upscale/json", response_model=ToolResult)
async def upscale_json(
    body: UpscaleRequest,
    session: DbSession,
    principal: CurrentPrincipal,
    storage: Storage,
) -> ToolResult:
    """JSON (asset-id) form of upscale (DESIGN §13 #2)."""
    image = _resolve_input_image(
        asset_id=body.asset_id, file_bytes=None, session=session, storage=storage
    )
    try:
        bigger = upscale.upscale_image(
            image=image, scale=body.scale, max_long_edge=body.max_long_edge
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    asset = _store_result(
        image=bigger,
        filename="upscaled.png",
        session=session,
        storage=storage,
        principal=principal,
    )
    return ToolResult(asset=asset)
