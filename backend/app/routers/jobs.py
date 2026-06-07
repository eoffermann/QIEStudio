"""Jobs API — submit/cancel/history (REST) + live progress (WebSocket) (DESIGN §5.5, §7).

Submitting enqueues work on the single-accelerator queue; progress + live latent previews
stream over ``WS /ws/jobs/{id}`` via the :class:`ProgressHub`.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlmodel import select

from app.db import get_engine
from app.deps import CurrentPrincipal, DbSession, Storage
from app.models.job import Job, JobOutput
from app.schemas.jobs import (
    BatchSubmit,
    BatchSubmitResponse,
    JobOutputRead,
    JobRead,
    JobSubmit,
    JobSubmitResponse,
)
from app.services import job_service
from app.services.progress_hub import get_progress_hub

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

_TERMINAL = {"done", "error", "canceled"}


def _iso(dt: object) -> str | None:
    return dt.isoformat() if dt is not None else None  # type: ignore[union-attr]


def _serialize(job: Job, outputs: list[JobOutput]) -> JobRead:
    return JobRead(
        id=job.id,
        batch_id=job.batch_id,
        mode=job.mode,
        status=job.status,
        precision=job.precision,
        device=job.device,
        model_id=job.model_id,
        model_revision=job.model_revision,
        prompt=job.prompt,
        enhanced_prompt=job.enhanced_prompt,
        rewriter_model=job.rewriter_model,
        params=job.params_json or {},
        progress=job.progress,
        progress_step=job.progress_step,
        progress_total=job.progress_total,
        error=job.error,
        created_at=_iso(job.created_at),
        started_at=_iso(job.started_at),
        ended_at=_iso(job.ended_at),
        outputs=[
            JobOutputRead(
                id=o.id,
                position=o.position,
                seed=o.seed,
                file_url=f"/api/jobs/{job.id}/outputs/{o.id}/file",
                thumb_url=f"/api/jobs/{job.id}/outputs/{o.id}/thumb" if o.thumb_key else None,
                metadata=o.metadata_json or {},
            )
            for o in sorted(outputs, key=lambda x: x.position)
        ],
    )


def _outputs_for(job_id: str, session) -> list[JobOutput]:  # noqa: ANN001
    return list(session.exec(select(JobOutput).where(JobOutput.job_id == job_id)).all())


@router.post("", response_model=JobSubmitResponse, status_code=201)
def submit(
    payload: JobSubmit, session: DbSession, principal: CurrentPrincipal
) -> JobSubmitResponse:
    try:
        job = job_service.submit_job(payload, session=session, principal=principal)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JobSubmitResponse(job_id=job.id)


@router.post("/batch", response_model=BatchSubmitResponse, status_code=201)
def submit_batch(
    payload: BatchSubmit, session: DbSession, principal: CurrentPrincipal
) -> BatchSubmitResponse:
    try:
        batch_id, job_ids = job_service.submit_batch(payload, session=session, principal=principal)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BatchSubmitResponse(batch_id=batch_id, job_ids=job_ids)


@router.get("", response_model=list[JobRead])
def history(
    session: DbSession,
    principal: CurrentPrincipal,
    status: str | None = None,
    mode: str | None = None,
    batch_id: str | None = None,
    q: str | None = None,
    limit: int = Query(default=100, le=500),
) -> list[JobRead]:
    stmt = select(Job).where(Job.owner_id == principal.owner_id)
    if status:
        stmt = stmt.where(Job.status == status)
    if mode:
        stmt = stmt.where(Job.mode == mode)
    if batch_id:
        stmt = stmt.where(Job.batch_id == batch_id)
    if q:
        stmt = stmt.where(Job.prompt.contains(q))  # type: ignore[attr-defined]
    stmt = stmt.order_by(Job.created_at.desc()).limit(limit)  # type: ignore[attr-defined]
    jobs = session.exec(stmt).all()
    return [_serialize(j, _outputs_for(j.id, session)) for j in jobs]


@router.get("/{job_id}", response_model=JobRead)
def get_job(job_id: str, session: DbSession, principal: CurrentPrincipal) -> JobRead:
    job = session.get(Job, job_id)
    if job is None or job.owner_id != principal.owner_id:
        raise HTTPException(status_code=404, detail="Job not found")
    return _serialize(job, _outputs_for(job_id, session))


@router.post("/{job_id}/cancel")
def cancel(job_id: str, session: DbSession, principal: CurrentPrincipal) -> dict[str, bool]:
    job = session.get(Job, job_id)
    if job is None or job.owner_id != principal.owner_id:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"canceled": job_service.cancel_job(job_id, session=session)}


def _stream_output(job_id: str, output_id: str, session, storage, *, thumb: bool):  # noqa: ANN001
    job_out = session.get(JobOutput, output_id)
    if job_out is None or job_out.job_id != job_id:
        raise HTTPException(status_code=404, detail="Output not found")
    key = job_out.thumb_key if thumb else job_out.storage_key
    if not key or not storage.exists(key):
        raise HTTPException(status_code=404, detail="Output binary not found")
    media = "image/webp" if thumb else "image/png"
    return StreamingResponse(storage.open_read(key), media_type=media)


@router.get("/{job_id}/outputs/{output_id}/file")
def output_file(job_id: str, output_id: str, session: DbSession, storage: Storage):  # noqa: ANN201
    return _stream_output(job_id, output_id, session, storage, thumb=False)


@router.get("/{job_id}/outputs/{output_id}/thumb")
def output_thumb(job_id: str, output_id: str, session: DbSession, storage: Storage):  # noqa: ANN201
    return _stream_output(job_id, output_id, session, storage, thumb=True)


@router.websocket("/ws/jobs/{job_id}")
async def job_ws(websocket: WebSocket, job_id: str) -> None:
    """Stream progress/status for a job. Sends a snapshot on connect, then live updates.

    Mounted at ``/api/jobs/ws/jobs/{id}`` by the router prefix; ``app.main`` also exposes a
    top-level ``/ws/jobs/{id}`` alias for the spec's path (DESIGN §7).
    """
    await websocket.accept()
    hub = get_progress_hub()
    queue = await hub.subscribe(job_id)
    try:
        # Initial snapshot: persisted job status + the last published progress message.
        from sqlmodel import Session

        with Session(get_engine()) as session:
            job = session.get(Job, job_id)
            if job is not None:
                await websocket.send_json(
                    {"type": "snapshot", "status": job.status, "progress": job.progress,
                     "step": job.progress_step, "total": job.progress_total}
                )
                if job.status in _TERMINAL:
                    return
        last = hub.last(job_id)
        if last is not None:
            await websocket.send_json(last)

        while True:
            msg = await queue.get()
            await websocket.send_json(msg)
            if msg.get("type") == "status" and msg.get("status") in _TERMINAL:
                break
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:  # pragma: no cover
        raise
    finally:
        hub.unsubscribe(job_id, queue)
