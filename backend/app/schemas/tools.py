"""Request/response schemas for the image tools API (DESIGN §13 #1, #2).

These shape the JSON surface of ``/api/tools`` — the one-click subject cutout
(background removal) and the high-quality upscale/refine pass. Both tools take an
existing asset (by id) or an uploaded file, run a CPU/GPU-light transform, and store
the result as a *new ephemeral asset* so it flows straight back into the run composer
(e.g. drop a chair photo → auto-cut → place it in a pinned room — DESIGN §13 #1).
"""

from __future__ import annotations

from pydantic import BaseModel

from app.schemas.assets import AssetRead


class BackgroundRemovalRequest(BaseModel):
    """Body for ``POST /api/tools/background-removal`` (asset-id form; DESIGN §13 #1).

    Supplying a multipart file uses the file-upload form of the endpoint instead and
    does not use this body.
    """

    asset_id: str


class UpscaleRequest(BaseModel):
    """Body for ``POST /api/tools/upscale`` (asset-id form; DESIGN §13 #2).

    ``scale`` is the linear magnification factor; ``max_long_edge`` optionally caps the
    output's longest edge (to protect downstream VRAM), overriding ``scale`` when hit.
    """

    asset_id: str
    scale: float = 2.0
    max_long_edge: int | None = None


class ToolResult(BaseModel):
    """Result of a tool run — the new ephemeral :class:`AssetRead` it produced."""

    asset: AssetRead
