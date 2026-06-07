"""Tests for the catalog/batch export service (DESIGN §13 #5, §5.6).

DB-backed + Pillow: seeds Job + JobOutput rows with tiny stored PNGs, builds the export, and
asserts the zip contains the contact-sheet grid, parameters CSV + JSON, and every image.
Uses the ``session`` fixture and skips when no PostgreSQL is reachable.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile

import pytest
from app.interfaces.auth import Principal
from app.interfaces.storage import LocalFsStorage
from app.models.job import Job, JobOutput
from app.services import catalog_export
from PIL import Image

pytestmark = pytest.mark.db

PRINCIPAL = Principal()


@pytest.fixture
def storage(tmp_path) -> LocalFsStorage:  # noqa: ANN001
    return LocalFsStorage(tmp_path)


def _store_png(storage: LocalFsStorage, key: str, color: tuple[int, int, int]) -> None:
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), color).save(buf, format="PNG")
    storage.put_bytes(key, buf.getvalue())


def _seed_batch(
    session: object,
    storage: LocalFsStorage,
    batch_id: str,
    n: int = 3,
) -> list[JobOutput]:
    """Create one batch with ``n`` jobs, each with one stored-PNG output."""
    outputs: list[JobOutput] = []
    for i in range(n):
        job = Job(
            id=f"{batch_id}-job-{i}",
            batch_id=batch_id,
            mode="generate",
            status="done",
            owner_id=PRINCIPAL.owner_id,
            workspace_id=PRINCIPAL.workspace_id,
        )
        session.add(job)
        key = f"assets/{batch_id}-{i}.png"
        _store_png(storage, key, (50 * i % 255, 40, 200))
        out = JobOutput(
            job_id=job.id,
            position=0,
            storage_key=key,
            seed=1000 + i,
            metadata_json={
                "mode": "generate",
                "prompt": f"prompt {i}",
                "precision": "bf16",
                "loras": [{"lora_id": "L1", "weight": 0.8}],  # nested -> JSON-encoded in CSV
            },
        )
        session.add(out)
        outputs.append(out)
    session.commit()
    return outputs


def _open_zip(data: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(data))


def test_export_batch_contains_all_parts(session: object, storage: LocalFsStorage) -> None:
    _seed_batch(session, storage, "B1", n=3)
    data = catalog_export.build_export(session=session, storage=storage, batch_id="B1")

    with _open_zip(data) as zf:
        names = set(zf.namelist())
        assert "contact_sheet.png" in names
        assert "parameters.csv" in names
        assert "parameters.json" in names
        # One image per output.
        images = [n for n in names if n.startswith("images/")]
        assert len(images) == 3

        # Contact sheet is a valid PNG.
        with Image.open(io.BytesIO(zf.read("contact_sheet.png"))) as sheet:
            assert sheet.format == "PNG"
            assert sheet.width > 0 and sheet.height > 0

        # JSON has one entry per output with structured metadata preserved.
        params = json.loads(zf.read("parameters.json"))
        assert len(params) == 3
        assert params[0]["prompt"] == "prompt 0"
        assert isinstance(params[0]["loras"], list)  # structure kept in JSON

        # CSV has a header + one row per output.
        rows = list(csv.DictReader(io.StringIO(zf.read("parameters.csv").decode())))
        assert len(rows) == 3
        assert {"filename", "job_id", "position", "seed"} <= set(rows[0].keys())
        # Nested metadata flattened to a JSON string cell in CSV.
        assert rows[0]["loras"].startswith("[")


def test_export_explicit_job_ids(session: object, storage: LocalFsStorage) -> None:
    outs = _seed_batch(session, storage, "B2", n=3)
    job_ids = [outs[0].job_id, outs[2].job_id]
    data = catalog_export.build_export(session=session, storage=storage, job_ids=job_ids)

    with _open_zip(data) as zf:
        images = [n for n in zf.namelist() if n.startswith("images/")]
        assert len(images) == 2
        params = json.loads(zf.read("parameters.json"))
        assert {p["job_id"] for p in params} == set(job_ids)


def test_export_empty_batch_still_valid_zip(session: object, storage: LocalFsStorage) -> None:
    data = catalog_export.build_export(session=session, storage=storage, batch_id="missing")
    with _open_zip(data) as zf:
        names = set(zf.namelist())
        assert "contact_sheet.png" in names
        assert "parameters.csv" in names
        assert "parameters.json" in names
        assert [n for n in names if n.startswith("images/")] == []
        assert json.loads(zf.read("parameters.json")) == []


def test_export_requires_a_selector(session: object, storage: LocalFsStorage) -> None:
    with pytest.raises(ValueError, match="batch_id or job_ids"):
        catalog_export.build_export(session=session, storage=storage)


def test_export_skips_missing_binaries(session: object, storage: LocalFsStorage) -> None:
    """An output whose binary is gone is dropped from images but kept in the param rows."""
    job = Job(
        id="B3-job", batch_id="B3", mode="generate", status="done",
        owner_id=PRINCIPAL.owner_id, workspace_id=PRINCIPAL.workspace_id,
    )
    session.add(job)
    session.add(
        JobOutput(
            job_id="B3-job", position=0, storage_key="assets/gone.png",
            seed=7, metadata_json={"prompt": "ghost"},
        )
    )
    session.commit()

    data = catalog_export.build_export(session=session, storage=storage, batch_id="B3")
    with _open_zip(data) as zf:
        assert [n for n in zf.namelist() if n.startswith("images/")] == []
        params = json.loads(zf.read("parameters.json"))
        assert len(params) == 1 and params[0]["prompt"] == "ghost"
