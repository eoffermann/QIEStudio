"""Catalog / batch export API (DESIGN §13 #5, §5.6, §7).

Streams a batch (or an explicit set of jobs) as a single ``application/zip`` containing
every output image, a contact-sheet grid PNG, and a CSV + JSON of per-image parameters. All
work is delegated to :mod:`app.services.catalog_export`; binaries are read through the
:class:`StorageProvider` (never absolute paths).
"""

from __future__ import annotations

import io

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import col, select

from app.deps import CurrentPrincipal, DbSession, Storage
from app.models.job import Job
from app.schemas.export import ExportSelection
from app.services import catalog_export

router = APIRouter(prefix="/api/export", tags=["export"])


def _zip_response(data: bytes, filename: str) -> StreamingResponse:
    """Wrap zip bytes in a download StreamingResponse (DESIGN §13 #5)."""
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/batch/{batch_id}")
def export_batch(
    batch_id: str,
    session: DbSession,
    principal: CurrentPrincipal,
    storage: Storage,
) -> StreamingResponse:
    """Export a whole batch as a zip (images + contact sheet + CSV/JSON) (DESIGN §13 #5).

    404s when the batch has no jobs owned by the principal, so the export never leaks
    another owner's outputs.
    """
    owned = session.exec(
        select(Job.id).where(
            col(Job.batch_id) == batch_id,
            col(Job.owner_id) == principal.owner_id,
        )
    ).first()
    if owned is None:
        raise HTTPException(status_code=404, detail="Batch not found")
    data = catalog_export.build_export(session=session, storage=storage, batch_id=batch_id)
    return _zip_response(data, f"batch-{batch_id}.zip")


@router.post("/jobs")
def export_jobs(
    body: ExportSelection,
    session: DbSession,
    principal: CurrentPrincipal,
    storage: Storage,
) -> StreamingResponse:
    """Export an explicit set of jobs as a zip (DESIGN §13 #5, §5.6).

    Only the principal's own jobs are included; unknown/foreign ids are silently dropped.
    """
    owned_ids = list(
        session.exec(
            select(Job.id).where(
                col(Job.id).in_(body.job_ids),
                col(Job.owner_id) == principal.owner_id,
            )
        ).all()
    )
    if not owned_ids:
        raise HTTPException(status_code=404, detail="No matching jobs")
    data = catalog_export.build_export(session=session, storage=storage, job_ids=owned_ids)
    return _zip_response(data, "jobs-export.zip")
