"""Real-GPU smoke test: the self-contained Generate -> Edit loop (RUN.md §5, §6).

Runs inside the CUDA image on the A6000. Drives the **real job flow** (``job_service`` →
``pipeline_service`` → ``asset_store``), so a completed job records the full DESIGN §5.5
reproducibility metadata (mode, model id/revision, precision, device, seed, LoRAs, input
hashes, original + enhanced prompts) and embeds it in the PNG sidecar — exactly the §6
acceptance bar.

Flow:
  1. Probe device + run the model advisor (bf16/fp8/int4) and print the recommendation.
  2. **Generate** a small image from a text prompt (real Qwen-Image), via a persisted job.
  3. Feed that generated image into an **Edit** job (real Qwen-Image-Edit-2511), preserving
     input order — a self-contained end-to-end loop needing no external fixtures.
  4. Copy both outputs (and their metadata JSON) into the output dir (default ``/out`` →
     bind-mounted to the repo ``images/``) for committing, and print the metadata.

Progress is logged before every slow phase with flushed output (project convention).

Usage (in-container):  python -m scripts.smoke_generate_edit [--precision bf16] [--steps 20]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from app.db import get_engine
from app.interfaces.auth import Principal
from app.interfaces.storage import build_storage_provider
from app.logging_utils import configure_logging, phase
from app.models.asset import Asset
from app.models.job import Job, JobOutput
from app.schemas.jobs import JobSubmit, ResolutionSpec
from app.services import job_service
from app.services.device_manager import get_device_info
from app.services.model_advisor import advise
from sqlmodel import Session, select

log = logging.getLogger("smoke")


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _output_asset_for(job_id: str, session: Session) -> Asset | None:
    return session.exec(select(Asset).where(Asset.source_job_id == job_id)).first()


def _export_output(job_id: str, out_dir: Path, label: str, storage) -> dict:  # noqa: ANN001
    """Copy a job's first output PNG + its metadata sidecar into out_dir; return metadata."""
    with Session(get_engine()) as session:
        out = session.exec(select(JobOutput).where(JobOutput.job_id == job_id)).first()
        if out is None:
            raise RuntimeError(f"No output produced for {label} job {job_id}")
        meta = out.metadata_json or {}
        src = storage.local_path(out.storage_key)
        dest = out_dir / f"{label}_{_stamp()}.png"
        if src is not None and Path(src).exists():
            shutil.copyfile(src, dest)
        else:
            dest.write_bytes(storage.get_bytes(out.storage_key))
        (out_dir / f"{dest.stem}.json").write_text(json.dumps(meta, indent=2))
        log.info("Saved %s output -> %s", label, dest.name)
        return {"job_id": job_id, "file": dest.name, "metadata": meta}


def _run_job_blocking(submit: JobSubmit, *, label: str) -> str:
    with Session(get_engine()) as session:
        job = job_service.create_job(submit, session=session, principal=Principal())
        job_id = job.id
    log.info("Running %s job %s ...", label, job_id)
    job_service.run_job(job_id)  # synchronous (we are the worker)
    with Session(get_engine()) as session:
        job = session.get(Job, job_id)
        if job is None or job.status != "done":
            raise RuntimeError(
                f"{label} job {job_id} did not complete: "
                f"status={getattr(job, 'status', '?')} error={getattr(job, 'error', '?')}"
            )
        log.info("%s job done on %s", label, job.device)
    return job_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp8", "int4"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--out", default=os.environ.get("QIE_SMOKE_OUT", "/out"))
    parser.add_argument("--offload", action="store_true",
                        help="enable model CPU offload for VRAM headroom (DESIGN §9.3)")
    args = parser.parse_args()

    configure_logging("INFO")
    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = build_storage_provider()

    # 1) Device + advisor ---------------------------------------------------------------
    dev = get_device_info(refresh=True)
    log.info("Device: backend=%s name=%s SM=%s VRAM=%d/%d MB", dev.backend, dev.name,
             dev.compute_capability, dev.free_vram_mb, dev.total_vram_mb)
    for mode in ("generate", "edit"):
        adv = advise(mode=mode, longer_edge=args.size, batch=1, num_loras=0, device=dev)
        log.info("Advisor[%s @ %d]: recommended=%s", mode, args.size, adv.recommended)
        for opt in adv.options:
            log.info("  - %-5s %-10s peak~%dMB headroom~%dMB %s", opt.precision, opt.status,
                     opt.est_peak_vram_mb, opt.headroom_mb, opt.rationale)

    summary: dict[str, object] = {"precision": args.precision, "steps": args.steps,
                                  "size": args.size, "device": dev.name}

    # 2) Generate -----------------------------------------------------------------------
    with phase(log, f"GENERATE ({args.precision}, {args.steps} steps, {args.size}px)"):
        gen_submit = JobSubmit(
            mode="generate",
            prompt="a single ripe red apple on a plain white studio table, soft lighting, "
                   "high detail product photograph",
            precision=args.precision,
            num_inference_steps=args.steps,
            seed=12345,
            resolution=ResolutionSpec(base=args.size, orientation="square", aspect="1:1"),
            preview_every_n_steps=0,
            enable_model_cpu_offload=args.offload,
        )
        gen_job_id = _run_job_blocking(gen_submit, label="generate")
    summary["generate"] = _export_output(gen_job_id, out_dir, "generate", storage)

    # Locate the generated output asset to feed into Edit.
    with Session(get_engine()) as session:
        gen_asset = _output_asset_for(gen_job_id, session)
        if gen_asset is None:
            raise RuntimeError("Generated output was not registered as an asset")
        gen_asset_id = gen_asset.id
    log.info("Generated asset %s -> feeding into Edit", gen_asset_id)

    # 3) Edit (feeds the generated image back in) --------------------------------------
    with phase(log, f"EDIT ({args.precision}, {args.steps} steps)"):
        edit_submit = JobSubmit(
            mode="edit",
            prompt="place the apple on a rustic wooden cutting board in a cozy kitchen, "
                   "warm morning light",
            precision=args.precision,
            input_asset_ids=[gen_asset_id],
            num_inference_steps=args.steps,
            seed=777,
            resolution=ResolutionSpec(base="match"),
            preview_every_n_steps=0,
            enable_model_cpu_offload=args.offload,
        )
        edit_job_id = _run_job_blocking(edit_submit, label="edit")
    summary["edit"] = _export_output(edit_job_id, out_dir, "edit", storage)

    summary_path = out_dir / f"smoke_summary_{_stamp()}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info("SMOKE OK — Generate->Edit loop complete in %.1fs. Outputs in %s",
             time.time() - t0, out_dir)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
