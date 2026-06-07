"""Real-GPU smoke tests for the remaining RUN §5 paths: a **LoRA-applied** generation, a
**live latent preview** decode, and the **Qwen-VL prompt enhancer** (RUN.md §5).

Run inside the CUDA image on the A6000. Two tasks:

- ``--task lora``  — import a real Qwen-Image LoRA (lightx2v Lightning) via ``lora_manager``,
  run a **bf16** Generate with it applied through the real job flow (``apply_loras`` →
  ``load_lora_weights`` + ``set_adapters``), commit the output + reproducibility metadata
  (which records the LoRA + weight), then capture one **live latent preview** PNG from the
  resident pipeline to validate the throttled VAE decode on the real Qwen VAE (§5.5).
- ``--task enhancer`` — load the **Qwen3-VL** rewriter and run ``enhance()`` in generate mode
  (text) and edit mode (**multimodal** — the VL model sees an input image), writing the
  original + enhanced prompts to the output dir (§5.8).

Outputs go to ``--out`` (default ``/out`` → bind-mounted to the repo ``images/``).
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

from app.config import get_settings
from app.db import get_engine
from app.interfaces.auth import Principal
from app.interfaces.storage import build_storage_provider
from app.logging_utils import configure_logging, phase
from app.models.job import Job, JobOutput
from app.schemas.jobs import JobSubmit, LoraSelection, ResolutionSpec
from app.services import job_service, lora_manager
from sqlmodel import Session, select

log = logging.getLogger("smoke_extras")

_LORA_URL = (
    "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/"
    "Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors"
)


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _export_job_output(job_id: str, out_dir: Path, label: str, storage) -> dict:  # noqa: ANN001
    with Session(get_engine()) as session:
        out = session.exec(select(JobOutput).where(JobOutput.job_id == job_id)).first()
        if out is None:
            raise RuntimeError(f"No output for {label} job {job_id}")
        meta = out.metadata_json or {}
        dest = out_dir / f"{label}_{_stamp()}.png"
        src = storage.local_path(out.storage_key)
        if src is not None and Path(src).exists():
            shutil.copyfile(src, dest)
        else:
            dest.write_bytes(storage.get_bytes(out.storage_key))
        (out_dir / f"{dest.stem}.json").write_text(json.dumps(meta, indent=2))
        log.info("Saved %s -> %s", label, dest.name)
        return {"file": dest.name, "metadata": meta}


def task_lora(out_dir: Path) -> dict:
    """Import a LoRA, run a bf16 Generate with it, then capture a live preview frame."""
    storage = build_storage_provider()
    principal = Principal()

    with Session(get_engine()) as session:
        with phase(log, "Importing Qwen-Image Lightning LoRA via lora_manager"):
            lora = lora_manager.import_from_url(
                url=_LORA_URL, name="Qwen-Image-Lightning-8step-V2",
                session=session, storage=storage, principal=principal,
            )
        lora_id = lora.id
        lora_key = lora.storage_key
        log.info("Imported LoRA %s (sha256=%s, %d bytes)", lora_id, lora.sha256, lora.bytes)

    # bf16 Generate with the LoRA applied (Lightning -> 8 steps, cfg 1.0). model_cpu_offload
    # for headroom (bf16 transformer is ~40 GB).
    submit = JobSubmit(
        mode="generate",
        prompt="a cozy reading nook by a rainy window, soft warm light, detailed",
        precision="bf16",
        num_inference_steps=8,
        true_cfg_scale=1.0,
        seed=42,
        resolution=ResolutionSpec(base=1024, orientation="square", aspect="1:1"),
        loras=[LoraSelection(lora_id=lora_id, weight=1.0)],
        enable_model_cpu_offload=True,
        preview_every_n_steps=0,
    )
    with Session(get_engine()) as session:
        job = job_service.create_job(submit, session=session, principal=principal)
        job_id = job.id
    with phase(log, f"Running bf16+LoRA generate job {job_id}"):
        job_service.run_job(job_id)
    with Session(get_engine()) as session:
        job = session.get(Job, job_id)
        if job is None or job.status != "done":
            raise RuntimeError(f"LoRA job failed: {getattr(job, 'error', '?')}")
    result = _export_job_output(job_id, out_dir, "lora_generate", storage)
    assert result["metadata"].get("loras"), "repro metadata must record the applied LoRA"
    log.info("LoRA recorded in metadata: %s", result["metadata"]["loras"])

    # --- Live latent preview validation (§5.5): reuse the resident bf16+LoRA pipeline and
    # capture the first decoded preview frame to disk. ---
    from app.services.pipeline_service import RunRequest, StepProgress, get_pipeline_service

    local = storage.local_path(lora_key)
    preview_holder: dict[str, bytes] = {}

    def _capture(p: StepProgress) -> None:
        if p.preview_png and "png" not in preview_holder:
            preview_holder["png"] = p.preview_png
            log.info("Captured live preview at step %d/%d (%d bytes)",
                     p.step, p.total, len(p.preview_png))

    req = RunRequest(
        mode="generate", model_id=get_settings().default_generate_model, model_revision=None,
        precision="bf16", device="cuda:0",
        prompt="a cozy reading nook by a rainy window, soft warm light, detailed",
        width=1024, height=1024, num_inference_steps=8, true_cfg_scale=1.0, seed=42,
        loras=[{"path": str(local), "weight": 1.0, "adapter_name": lora_id}],
        preview_every_n_steps=2,
    )
    with phase(log, "Capturing live latent preview (real Qwen VAE decode)"):
        get_pipeline_service().run(req, on_step=_capture)
    if "png" in preview_holder:
        pv = out_dir / f"live_preview_{_stamp()}.png"
        pv.write_bytes(preview_holder["png"])
        result["preview_file"] = pv.name
        log.info("Saved live preview -> %s", pv.name)
    else:
        log.warning("No live preview frame was decoded (preview path degraded to skip)")
        result["preview_file"] = None
    return result


def task_enhancer(out_dir: Path) -> dict:
    """Run the Qwen-VL prompt enhancer: text (generate) + multimodal (edit)."""
    from app.services.rewriter import get_rewriter_service

    svc = get_rewriter_service()
    out: dict[str, object] = {}

    original = "make it look cinematic"
    with phase(log, "Enhancer: generate-mode (text) rewrite via Qwen3-VL"):
        enhanced_gen = svc.enhance(mode="generate", prompt=original)
    log.info("GENERATE original=%r -> enhanced=%r", original, enhanced_gen)
    out["generate"] = {"original": original, "enhanced": enhanced_gen}

    # Multimodal edit-mode enhance: the VL model sees a real input image (the generated apple
    # from the prior smoke, if present).
    img = None
    apples = sorted(out_dir.glob("generate_*.png"))
    if apples:
        from PIL import Image

        img = Image.open(apples[-1]).convert("RGB")
        log.info("Edit-mode enhance will see input image %s", apples[-1].name)
    edit_original = "put it in a sunny kitchen"
    with phase(log, "Enhancer: edit-mode (multimodal) rewrite via Qwen3-VL"):
        enhanced_edit = svc.enhance(
            mode="edit", prompt=edit_original, images=[img] if img is not None else None,
            edit_model_id="Qwen/Qwen-Image-Edit-2511",
        )
    log.info("EDIT original=%r -> enhanced=%r", edit_original, enhanced_edit)
    out["edit"] = {
        "original": edit_original, "enhanced": enhanced_edit,
        "saw_image": img is not None,
    }
    (out_dir / f"enhancer_{_stamp()}.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["lora", "enhancer"])
    parser.add_argument("--out", default=os.environ.get("QIE_SMOKE_OUT", "/out"))
    args = parser.parse_args()

    configure_logging("INFO")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    result = task_lora(out_dir) if args.task == "lora" else task_enhancer(out_dir)

    log.info("smoke_extras[%s] OK in %.1fs", args.task, time.time() - t0)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
