"""Tests for the job service + jobs router (no torch — the real run is a GPU test).

Covers submit-time resolution (resolution math, ordered Edit slot inputs, JobInput rows),
batch/sweep expansion, queued-job cancellation, the ProgressHub, and the WebSocket snapshot
path. The actual inference (``run_job`` → pipeline) is exercised by the CUDA-image GPU tests.
"""

from __future__ import annotations

import pytest
from app.interfaces.auth import Principal
from app.models.asset import Asset
from app.models.job import Job, JobInput
from app.models.prompt import Prompt, PromptImage
from app.schemas.jobs import BatchSubmit, JobSubmit, ResolutionSpec, SweepSpec
from app.services import job_service
from sqlmodel import select

pytestmark = pytest.mark.db


@pytest.fixture
def principal() -> Principal:
    return Principal()


@pytest.fixture(autouse=True)
def _no_enqueue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Don't actually run inference during these tests."""
    monkeypatch.setattr(job_service, "_enqueue", lambda job_id: None)


def _make_asset(session, *, w: int, h: int, sha: str) -> Asset:  # noqa: ANN001
    a = Asset(scope="library", source="upload", storage_key=f"assets/{sha}.png",
              width=w, height=h, format="PNG", sha256=sha)
    session.add(a)
    session.commit()
    session.refresh(a)
    return a


def test_create_generate_job_resolves_resolution(session, principal) -> None:
    submit = JobSubmit(
        mode="generate", prompt="a cat",
        resolution=ResolutionSpec(base=1024, orientation="landscape", aspect="3:2"),
        num_inference_steps=30,
    )
    job = job_service.create_job(submit, session=session, principal=principal)
    assert job.status == "queued"
    assert job.mode == "generate"
    assert job.params_json["width"] == 1024
    assert job.params_json["height"] == 688  # spec example, snapped to ×16
    assert job.progress_total == 30


def test_create_edit_job_orders_slot_inputs(session, principal) -> None:
    bg = _make_asset(session, w=1024, h=768, sha="bg")
    chair = _make_asset(session, w=500, h=500, sha="chair")
    sofa = _make_asset(session, w=500, h=500, sha="sofa")

    prompt = Prompt(name="staging", text="place furniture", mode="edit")
    session.add(prompt)
    session.commit()
    session.refresh(prompt)
    session.add(PromptImage(prompt_id=prompt.id, position=0, role="pinned", asset_id=bg.id))
    session.add(PromptImage(prompt_id=prompt.id, position=1, role="slot", slot_name="furniture",
                            slot_min=1, slot_max=3))
    session.commit()

    submit = JobSubmit(
        mode="edit", prompt="place furniture", prompt_id=prompt.id,
        slot_fills={"furniture": [chair.id, sofa.id]},
        resolution=ResolutionSpec(base="match"),
    )
    job = job_service.create_job(submit, session=session, principal=principal)

    inputs = session.exec(select(JobInput).where(JobInput.job_id == job.id)).all()
    ordered = sorted(inputs, key=lambda r: r.position)
    assert [r.asset_id for r in ordered] == [bg.id, chair.id, sofa.id]
    # "match source" uses the first input's dims, snapped to ×16.
    assert job.params_json["width"] == 1024
    assert job.params_json["height"] == 768


def test_edit_requires_inputs(session, principal) -> None:
    submit = JobSubmit(mode="edit", prompt="x")
    with pytest.raises(ValueError, match="at least one input"):
        job_service.create_job(submit, session=session, principal=principal)


def test_batch_seed_sweep_expands(session, principal) -> None:
    base = JobSubmit(mode="generate", prompt="sweep",
                     resolution=ResolutionSpec(base=512, orientation="square", aspect="1:1"))
    batch = BatchSubmit(base=base, sweep=SweepSpec(kind="seed", seeds=[1, 2, 3]))
    batch_id, job_ids = job_service.submit_batch(batch, session=session, principal=principal)
    assert len(job_ids) == 3
    jobs = session.exec(select(Job).where(Job.batch_id == batch_id)).all()
    assert {j.params_json["seed"] for j in jobs} == {1, 2, 3}


def test_batch_param_sweep_grid(session, principal) -> None:
    base = JobSubmit(mode="generate", prompt="grid",
                     resolution=ResolutionSpec(base=512, orientation="square", aspect="1:1"))
    batch = BatchSubmit(base=base, sweep=SweepSpec(kind="param", steps_grid=[20, 40],
                                                   cfg_grid=[3.0, 4.0]))
    _bid, job_ids = job_service.submit_batch(batch, session=session, principal=principal)
    assert len(job_ids) == 4  # 2 steps × 2 cfg


def test_cancel_queued_job(session, principal) -> None:
    submit = JobSubmit(mode="generate", prompt="x",
                       resolution=ResolutionSpec(base=512, orientation="square", aspect="1:1"))
    job = job_service.create_job(submit, session=session, principal=principal)
    assert job_service.cancel_job(job.id, session=session) is True
    session.refresh(job)
    assert job.status == "canceled"
    # Cancelling a terminal job is a no-op.
    assert job_service.cancel_job(job.id, session=session) is False


def test_progress_hub_last_and_clear() -> None:
    from app.services.progress_hub import ProgressHub

    hub = ProgressHub()
    assert hub.last("j") is None
    hub.publish("j", {"type": "progress", "step": 1})  # no loop set -> just stores last
    assert hub.last("j") == {"type": "progress", "step": 1}
    hub.clear("j")
    assert hub.last("j") is None


def test_ws_snapshot_for_terminal_job(client) -> None:  # noqa: ANN001
    """A WebSocket client connecting to a finished job gets a snapshot and the stream ends."""
    from app.db import get_engine
    from sqlmodel import Session

    with Session(get_engine()) as s:
        job = Job(mode="generate", status="done", progress=1.0, progress_step=10,
                  progress_total=10, model_id="Qwen/Qwen-Image")
        s.add(job)
        s.commit()
        s.refresh(job)
        job_id = job.id

    with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "snapshot"
        assert msg["status"] == "done"


def test_queue_pending_reorder_and_remove() -> None:
    """The in-process queue supports listing, reordering, and removing pending jobs (§13)."""
    from app.interfaces.queue import InProcessJobQueue, QueuedTask

    q = InProcessJobQueue()  # do NOT start the worker, so tasks stay pending
    for jid in ("a", "b", "c"):
        q.submit(QueuedTask(job_id=jid, fn=lambda: None))
    assert q.pending_ids() == ["a", "b", "c"]
    q.reorder_pending(["c", "a", "b"])
    assert q.pending_ids() == ["c", "a", "b"]
    assert q.remove_pending("a") is True
    assert q.pending_ids() == ["c", "b"]
    assert q.remove_pending("zzz") is False


def test_delete_job_removes_row(session, principal) -> None:
    from app.interfaces.storage import build_storage_provider

    submit = JobSubmit(mode="generate", prompt="x",
                       resolution=ResolutionSpec(base=512, orientation="square", aspect="1:1"))
    job = job_service.create_job(submit, session=session, principal=principal)
    storage = build_storage_provider()
    assert job_service.delete_job(
        job.id, session=session, storage=storage, principal=principal
    ) is True
    assert session.get(Job, job.id) is None
    # Deleting a non-existent job is a no-op.
    assert job_service.delete_job(
        "nope", session=session, storage=storage, principal=principal
    ) is False
