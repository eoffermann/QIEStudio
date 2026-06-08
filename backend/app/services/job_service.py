"""Job service — the integrative layer that runs the single-accelerator queue (DESIGN §5.5).

Ties together resolution (§5.4), prompt slots (§5.2), assets (§5.1), LoRAs (§5.3), the
pipeline (§2/§9), and reproducibility metadata (§5.5). Responsibilities:

- **Submit** a run (or a batch/sweep, §5.6): resolve the ordered Edit inputs (pinned + slot
  fills), concrete resolution, model id/precision/device, and LoRA selections *at submit
  time*, persist a ``Job`` (+ ``JobInput`` rows with input hashes) as ``queued``, and enqueue
  it on the in-process :class:`JobQueue` (one inference at a time).
- **Run** a job on the worker thread: load/reuse the pipeline, stream throttled progress +
  live latent previews over the :class:`ProgressHub`, save outputs through the asset store
  with full reproducibility metadata embedded, and persist status transitions so jobs
  survive reloads.
- **Recover** ``queued``/``running`` jobs on startup (resumability, RUN §7) by re-enqueueing
  them (inference isn't checkpointed, so a job interrupted mid-run re-runs from the top).
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any

from sqlmodel import Session, select

from app.config import get_settings
from app.interfaces.auth import Principal
from app.interfaces.queue import QueuedTask, get_job_queue
from app.interfaces.storage import StorageProvider, build_storage_provider
from app.logging_utils import phase
from app.models.base import utcnow
from app.models.job import Job, JobInput, JobOutput
from app.services import asset_store, lora_manager, prompt_store, resolution
from app.services.progress_hub import get_progress_hub

if TYPE_CHECKING:
    from app.schemas.jobs import BatchSubmit, JobSubmit

log = logging.getLogger(__name__)

_WORKER_PRINCIPAL = Principal()  # jobs run as the implicit local owner (v1)


# --- Submit ----------------------------------------------------------------------------


def _model_id_for(mode: str, override: str | None) -> str:
    if override:
        return override
    settings = get_settings()
    return settings.default_edit_model if mode == "edit" else settings.default_generate_model


def _resolve_edit_inputs(submit: JobSubmit, session: Session) -> list[str]:
    """Resolve the ordered Edit input asset ids from a prompt's slots or explicit ids."""
    if submit.prompt_id:
        prompt = prompt_store.get_prompt(prompt_id=submit.prompt_id, session=session)
        if prompt is None:
            raise ValueError(f"Prompt {submit.prompt_id!r} not found")
        images = prompt_store.get_prompt_images(prompt_id=submit.prompt_id, session=session)
        resolved = prompt_store.resolve_inputs(
            prompt=prompt, prompt_images=images, slot_fills=submit.slot_fills, session=session
        )
        return [r.asset_id for r in resolved]
    return list(submit.input_asset_ids)


def _resolve_resolution(
    submit: JobSubmit, first_input_dims: tuple[int, int] | None
) -> tuple[int, int]:
    spec = submit.resolution
    if spec.width and spec.height:
        return int(spec.width), int(spec.height)
    base = spec.base if spec.base is not None else 1024
    orientation = spec.orientation or "landscape"
    return resolution.resolve(
        base=base,
        orientation=orientation,
        aspect=spec.aspect,
        source_dims=first_input_dims,
        max_long_edge=spec.max_long_edge,
    )


def create_job(
    submit: JobSubmit,
    *,
    session: Session,
    principal: Principal,
    batch_id: str | None = None,
) -> Job:
    """Resolve + persist a single queued job (does not enqueue)."""
    mode = submit.mode
    model_id = _model_id_for(mode, submit.model_id)

    # Resolve ordered Edit inputs + their hashes / first-image dims (for match-source).
    input_asset_ids: list[str] = []
    input_hashes: list[str] = []
    first_dims: tuple[int, int] | None = None
    if mode == "edit":
        input_asset_ids = _resolve_edit_inputs(submit, session)
        if not input_asset_ids:
            raise ValueError("Edit mode requires at least one input image")
        for aid in input_asset_ids:
            asset = asset_store.get_asset(asset_id=aid, session=session)
            if asset is None:
                raise ValueError(f"Input asset {aid!r} not found")
            input_hashes.append(asset.sha256)
            if first_dims is None:
                first_dims = (asset.width, asset.height)

    width, height = _resolve_resolution(submit, first_dims)

    # Validate LoRA selections exist.
    lora_sel: list[dict[str, Any]] = []
    for sel in submit.loras:
        lora = lora_manager.get_lora(lora_id=sel.lora_id, session=session, principal=principal)
        if lora is None:
            raise ValueError(f"LoRA {sel.lora_id!r} not found")
        lora_sel.append({"lora_id": sel.lora_id, "weight": float(sel.weight)})

    settings = get_settings()
    preview_n = (
        submit.preview_every_n_steps
        if submit.preview_every_n_steps is not None
        else settings.preview_every_n_steps
    )

    params: dict[str, Any] = {
        "negative_prompt": submit.negative_prompt,
        "width": width,
        "height": height,
        "num_inference_steps": submit.num_inference_steps,
        "true_cfg_scale": submit.true_cfg_scale,
        "guidance_scale": submit.guidance_scale,
        "seed": submit.seed,
        "batch": submit.batch,
        "loras": lora_sel,
        "output_format": submit.output_format,
        "output_quality": submit.output_quality,
        "preview_every_n_steps": preview_n,
        "input_asset_ids": input_asset_ids,
        "offload": {
            "model_cpu": submit.enable_model_cpu_offload,
            "sequential_cpu": submit.enable_sequential_cpu_offload,
            "attention_slicing": submit.enable_attention_slicing,
            "vae_tiling": submit.enable_vae_tiling,
        },
    }

    job = Job(
        batch_id=batch_id,
        owner_id=principal.owner_id,
        workspace_id=principal.workspace_id,
        mode=mode,
        status="queued",
        precision=submit.precision,
        model_id=model_id,
        model_revision=submit.model_revision,
        prompt=submit.prompt,
        enhanced_prompt=submit.enhanced_prompt,
        rewriter_model=submit.rewriter_model,
        params_json=params,
        progress_total=int(submit.num_inference_steps),
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    for pos, (aid, sha) in enumerate(zip(input_asset_ids, input_hashes, strict=False)):
        session.add(JobInput(job_id=job.id, position=pos, asset_id=aid, sha256=sha))
    if input_asset_ids:
        session.commit()

    log.info("Created job %s (%s, %s, %dx%d)", job.id, mode, submit.precision, width, height)
    return job


def submit_job(submit: JobSubmit, *, session: Session, principal: Principal) -> Job:
    """Create + enqueue a single job."""
    job = create_job(submit, session=session, principal=principal)
    _enqueue(job.id)
    return job


def submit_batch(
    batch: BatchSubmit, *, session: Session, principal: Principal
) -> tuple[str, list[str]]:
    """Expand a sweep (§5.6) into child jobs sharing a batch id; enqueue each."""
    from uuid import uuid4


    batch_id = uuid4().hex
    base = batch.base
    sweep = batch.sweep
    variants: list[JobSubmit] = []

    if sweep.kind == "slot":
        if not sweep.slot_name:
            raise ValueError("slot sweep requires slot_name")
        for aid in sweep.slot_asset_ids:
            v = base.model_copy(deep=True)
            v.slot_fills = {**base.slot_fills, sweep.slot_name: [aid]}
            variants.append(v)
    elif sweep.kind == "seed":
        for seed in sweep.seeds:
            v = base.model_copy(deep=True)
            v.seed = seed
            variants.append(v)
    elif sweep.kind == "param":
        steps_grid = sweep.steps_grid or [base.num_inference_steps]
        cfg_grid = sweep.cfg_grid or [base.true_cfg_scale]
        for steps in steps_grid:
            for cfg in cfg_grid:
                v = base.model_copy(deep=True)
                v.num_inference_steps = steps
                v.true_cfg_scale = cfg
                variants.append(v)

    if not variants:
        raise ValueError("Sweep produced no jobs")

    job_ids: list[str] = []
    for v in variants:
        job = create_job(v, session=session, principal=principal, batch_id=batch_id)
        job_ids.append(job.id)
    for jid in job_ids:
        _enqueue(jid)
    log.info("Submitted batch %s with %d job(s)", batch_id, len(job_ids))
    return batch_id, job_ids


def _enqueue(job_id: str) -> None:
    get_job_queue().submit(QueuedTask(job_id=job_id, fn=lambda: run_job(job_id), label="inference"))


# --- Cancel ----------------------------------------------------------------------------


def cancel_job(job_id: str, *, session: Session) -> bool:
    job = session.get(Job, job_id)
    if job is None or job.status in ("done", "error", "canceled"):
        return False
    queue = get_job_queue()
    queue.request_cancel(job_id)
    queue.remove_pending(job_id)  # drop it from the queue if it hasn't started
    if job.status == "queued":  # not yet running — cancel immediately
        job.status = "canceled"
        job.ended_at = utcnow()
        session.add(job)
        session.commit()
        get_progress_hub().publish(job_id, {"type": "status", "status": "canceled"})
    return True


def delete_job(
    job_id: str, *, session: Session, storage: StorageProvider, principal: Principal
) -> bool:
    """Delete a job and its outputs (DESIGN §13 queue management).

    Cancels + dequeues the job first if it is queued/running, then removes its output assets
    (binaries + rows), its JobInput/JobOutput rows, and the job row itself. Input assets
    (user uploads) are left intact — only the JobInput links are removed.
    """
    job = session.get(Job, job_id)
    if job is None or job.owner_id != principal.owner_id:
        return False

    queue = get_job_queue()
    queue.request_cancel(job_id)
    queue.remove_pending(job_id)

    outputs = session.exec(select(JobOutput).where(JobOutput.job_id == job_id)).all()
    output_asset_ids = [o.asset_id for o in outputs if o.asset_id]
    # Delete JobOutput rows first so the asset FK can be removed without a violation.
    for o in outputs:
        session.delete(o)
    for ji in session.exec(select(JobInput).where(JobInput.job_id == job_id)).all():
        session.delete(ji)
    session.flush()
    for aid in output_asset_ids:
        try:
            asset_store.delete_asset(
                asset_id=aid, session=session, principal=principal, storage=storage
            )
        except Exception:  # noqa: BLE001 — best-effort binary cleanup; never block the delete
            log.warning("Could not delete output asset %s for job %s", aid, job_id, exc_info=True)
    session.delete(job)
    session.commit()
    log.info("Deleted job %s (+%d output assets)", job_id, len(output_asset_ids))
    return True


def reorder_pending(ordered_ids: list[str]) -> list[str]:
    """Reorder the pending (not-yet-started) jobs to match ``ordered_ids`` (DESIGN §13).

    Returns the resulting pending order.
    """
    queue = get_job_queue()
    queue.reorder_pending(ordered_ids)
    return queue.pending_ids()


def queue_state() -> dict[str, Any]:
    """Current queue snapshot: the running job id (if any) + pending ids in execution order."""
    queue = get_job_queue()
    return {"running": queue.running_job, "pending": queue.pending_ids()}


# --- Run (worker thread) ---------------------------------------------------------------


def _build_run_request(job: Job, *, session: Session, storage: StorageProvider) -> Any:
    """Construct a pipeline ``RunRequest`` from a persisted job + its inputs."""
    from app.services.device_manager import get_device_info
    from app.services.pipeline_service import RunRequest

    params = job.params_json or {}
    device = get_device_info()

    images: list[Any] = []
    if job.mode == "edit":
        inputs = session.exec(
            select(JobInput).where(JobInput.job_id == job.id).order_by(JobInput.position)
        ).all()
        for ji in inputs:
            asset = asset_store.get_asset(asset_id=ji.asset_id, session=session)
            if asset is None:
                raise ValueError(f"Input asset {ji.asset_id!r} vanished before run")
            images.append(asset_store.open_pil(asset=asset, storage=storage))

    loras: list[dict[str, Any]] = []
    for sel in params.get("loras", []):
        lora = lora_manager.get_lora(
            lora_id=sel["lora_id"], session=session, principal=_WORKER_PRINCIPAL
        )
        if lora is None:
            continue
        local = storage.local_path(lora.storage_key)
        if local is None:
            raise ValueError("LoRA weights not available as a local path for this storage backend")
        loras.append({"path": str(local), "weight": sel["weight"], "adapter_name": sel["lora_id"]})

    prompt_text = job.enhanced_prompt or job.prompt
    return RunRequest(
        mode=job.mode,
        model_id=job.model_id,
        model_revision=job.model_revision,
        precision=job.precision,
        device=device.device_str,
        prompt=prompt_text,
        negative_prompt=params.get("negative_prompt", " "),
        width=params.get("width"),
        height=params.get("height"),
        num_inference_steps=int(params.get("num_inference_steps", 40)),
        true_cfg_scale=float(params.get("true_cfg_scale", 4.0)),
        guidance_scale=float(params.get("guidance_scale", 1.0)),
        seed=params.get("seed"),
        batch=int(params.get("batch", 1)),
        images=images,
        loras=loras,
        preview_every_n_steps=int(params.get("preview_every_n_steps", 5)),
    )


def run_job(job_id: str) -> None:
    """Execute one job on the worker thread (idempotent re-run on restart)."""
    from app.db import get_engine
    from app.services.pipeline_service import (
        CanceledError,
        LoadOptions,
        StepProgress,
        get_pipeline_service,
    )
    from app.services.reproducibility import build_metadata

    hub = get_progress_hub()
    queue = get_job_queue()
    storage = build_storage_provider()

    with Session(get_engine()) as session:
        job = session.get(Job, job_id)
        if job is None:
            log.warning("run_job: job %s not found", job_id)
            return
        if job.status in ("done", "canceled"):
            return
        if queue.is_canceled(job_id):
            job.status = "canceled"
            job.ended_at = utcnow()
            session.add(job)
            session.commit()
            hub.publish(job_id, {"type": "status", "status": "canceled"})
            return

        from app.services.device_manager import get_device_info

        device = get_device_info()
        job.status = "running"
        job.started_at = utcnow()
        job.device = f"{device.device_str} ({device.name}"
        if device.compute_capability:
            job.device += f", SM {device.compute_capability[0]}.{device.compute_capability[1]}"
        job.device += ")"
        session.add(job)
        session.commit()
        hub.publish(job_id, {"type": "status", "status": "running", "device": job.device})

        params = job.params_json or {}
        total = int(params.get("num_inference_steps", 40))
        preview_n = int(params.get("preview_every_n_steps", 5))

        def on_step(p: StepProgress) -> None:
            msg: dict[str, Any] = {
                "type": "progress",
                "step": p.step,
                "total": p.total,
                "progress": round(p.step / p.total, 4) if p.total else 0.0,
            }
            if p.preview_png is not None:
                b64 = base64.b64encode(p.preview_png).decode("ascii")
                msg["preview"] = f"data:image/png;base64,{b64}"
            hub.publish(job_id, msg)
            # Persist progress on the (throttled) preview cadence + final step only.
            if p.step >= p.total or (preview_n > 0 and p.step % preview_n == 0):
                job.progress = msg["progress"]
                job.progress_step = p.step
                session.add(job)
                session.commit()

        try:
            req = _build_run_request(job, session=session, storage=storage)
            offload = params.get("offload", {})
            svc = get_pipeline_service()
            svc.load(
                mode=req.mode,
                model_id=req.model_id,
                model_revision=req.model_revision,
                precision=req.precision,
                device=req.device,
                options=LoadOptions(
                    enable_model_cpu_offload=offload.get("model_cpu", False),
                    enable_sequential_cpu_offload=offload.get("sequential_cpu", False),
                    enable_attention_slicing=offload.get("attention_slicing", False),
                    enable_vae_tiling=offload.get("vae_tiling", False),
                ),
            )
            with phase(log, f"Job {job_id}: inference"):
                images = svc.run(
                    req, on_step=on_step, is_canceled=lambda: queue.is_canceled(job_id)
                )

            input_hashes = [
                ji.sha256
                for ji in session.exec(
                    select(JobInput).where(JobInput.job_id == job.id).order_by(JobInput.position)
                ).all()
            ]
            outputs: list[dict[str, Any]] = []
            for i, image in enumerate(images):
                seed = req.seed
                meta = build_metadata(
                    mode=job.mode,
                    model_id=job.model_id,
                    model_revision=job.model_revision,
                    precision=job.precision,
                    device=job.device,
                    prompt=job.prompt,
                    enhanced_prompt=job.enhanced_prompt,
                    negative_prompt=req.negative_prompt,
                    width=req.width,
                    height=req.height,
                    steps=total,
                    true_cfg_scale=req.true_cfg_scale,
                    guidance_scale=req.guidance_scale,
                    seed=seed,
                    loras=params.get("loras", []),
                    input_hashes=input_hashes,
                    rewriter_model=job.rewriter_model,
                    output_format=params.get("output_format", "png"),
                )
                _asset, jo = asset_store.save_output_image(
                    image=image,
                    job_id=job.id,
                    position=i,
                    seed=seed,
                    metadata=meta,
                    session=session,
                    storage=storage,
                    principal=_WORKER_PRINCIPAL,
                )
                outputs.append({"id": jo.id, "position": i, "seed": seed})

            job.status = "done"
            job.progress = 1.0
            job.progress_step = total
            job.ended_at = utcnow()
            session.add(job)
            session.commit()
            hub.publish(job_id, {"type": "status", "status": "done", "outputs": outputs})
            log.info("Job %s done with %d output(s)", job_id, len(outputs))

        except CanceledError:
            session.rollback()
            job = session.get(Job, job_id)
            if job is not None:
                job.status = "canceled"
                job.ended_at = utcnow()
                session.add(job)
                session.commit()
            hub.publish(job_id, {"type": "status", "status": "canceled"})
            log.info("Job %s canceled", job_id)
        except Exception as exc:  # noqa: BLE001 — record the failure on the job, don't crash worker
            log.exception("Job %s failed", job_id)
            session.rollback()
            job = session.get(Job, job_id)
            if job is not None:
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.ended_at = utcnow()
                session.add(job)
                session.commit()
            hub.publish(job_id, {"type": "status", "status": "error", "error": str(exc)})


def recover_jobs() -> None:
    """Re-enqueue queued/running jobs on startup (resumability, RUN §7)."""
    from app.db import get_engine

    with Session(get_engine()) as session:
        stmt = select(Job).where(Job.status.in_(("queued", "running"))).order_by(Job.created_at)  # type: ignore[attr-defined]
        jobs = session.exec(stmt).all()
        for job in jobs:
            if job.status == "running":
                job.status = "queued"  # cannot resume mid-inference; re-run from the top
                job.progress = 0.0
                job.progress_step = 0
                session.add(job)
        session.commit()
        ids = [j.id for j in jobs]
    for jid in ids:
        _enqueue(jid)
    if ids:
        log.info("Recovered %d queued/running job(s) on startup: %s", len(ids), ids)
