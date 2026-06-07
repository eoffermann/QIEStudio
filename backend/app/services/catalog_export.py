"""Catalog / batch export service (DESIGN §13 #5, §5.6).

Exports a batch (or an explicit set of jobs) as a single **zip** — the natural companion
to batch/sweep mode (DESIGN §5.6). The archive contains:

- ``images/`` — every output image (the stored PNG bytes), named ``<job>-<pos>.png``.
- ``contact_sheet.png`` — a thumbnail **grid** composed with Pillow.
- ``parameters.csv`` and ``parameters.json`` — per-image reproducibility parameters drawn
  from each :class:`~app.models.job.JobOutput`'s ``metadata_json``.

All binaries are read through the :class:`~app.interfaces.storage.StorageProvider`; the DB
holds only keys (DESIGN §4.1a). The whole archive is built in memory and returned as bytes,
which the router streams as ``application/zip``.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import zipfile
from typing import TYPE_CHECKING, Any

from PIL import Image
from sqlmodel import col, select

from app.logging_utils import phase
from app.models.job import Job, JobOutput

if TYPE_CHECKING:
    from sqlmodel import Session

    from app.interfaces.storage import StorageProvider

log = logging.getLogger(__name__)

# Contact-sheet layout (DESIGN §13 #5).
SHEET_COLUMNS = 4
SHEET_CELL = 256  # px square cell per thumbnail
SHEET_PAD = 8  # px padding around each cell
SHEET_BG = (24, 24, 27)  # dark surface to match the 2026 UI aesthetic


def _collect_outputs(
    *,
    session: Session,
    batch_id: str | None,
    job_ids: list[str] | None,
) -> list[JobOutput]:
    """Gather the JobOutput rows for a batch id or an explicit job-id list, ordered.

    Outputs are ordered by ``(job_id, position)`` for stable, deterministic archives.
    """
    if batch_id is not None:
        job_id_list = list(
            session.exec(select(Job.id).where(col(Job.batch_id) == batch_id)).all()
        )
    elif job_ids is not None:
        job_id_list = list(job_ids)
    else:  # pragma: no cover - guarded by the public entry point
        raise ValueError("Provide either batch_id or job_ids")

    if not job_id_list:
        return []
    stmt = (
        select(JobOutput)
        .where(col(JobOutput.job_id).in_(job_id_list))
        .order_by(col(JobOutput.job_id), col(JobOutput.position))
    )
    return list(session.exec(stmt).all())


def _flatten_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Flatten a one-level-nested metadata dict to scalar CSV-friendly cells.

    Nested dicts/lists are JSON-encoded so every value is a single CSV cell while the
    full structure is still preserved verbatim in ``parameters.json``.
    """
    flat: dict[str, Any] = {}
    for key, value in meta.items():
        flat[key] = value if isinstance(value, (str, int, float, bool)) or value is None else (
            json.dumps(value, default=str)
        )
    return flat


def _build_contact_sheet(thumbs: list[Image.Image]) -> bytes:
    """Compose a grid contact sheet PNG from thumbnail images (DESIGN §13 #5)."""
    count = max(1, len(thumbs))
    cols = min(SHEET_COLUMNS, count)
    rows = (count + cols - 1) // cols
    cell = SHEET_CELL + 2 * SHEET_PAD
    sheet = Image.new("RGB", (cols * cell, rows * cell), SHEET_BG)

    for index, thumb in enumerate(thumbs):
        cell_img = thumb.convert("RGB") if thumb.mode != "RGB" else thumb.copy()
        cell_img.thumbnail((SHEET_CELL, SHEET_CELL), Image.LANCZOS)
        col_i, row_i = index % cols, index // cols
        # Center each thumbnail within its padded cell.
        x = col_i * cell + SHEET_PAD + (SHEET_CELL - cell_img.width) // 2
        y = row_i * cell + SHEET_PAD + (SHEET_CELL - cell_img.height) // 2
        sheet.paste(cell_img, (x, y))

    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue()


def build_export(
    *,
    session: Session,
    storage: StorageProvider,
    batch_id: str | None = None,
    job_ids: list[str] | None = None,
) -> bytes:
    """Build a zip export for a batch or explicit job ids (DESIGN §13 #5, §5.6).

    The archive bundles every output image, a contact-sheet grid PNG, and a CSV + JSON of
    per-image parameters (from each output's ``metadata_json``). Returns the raw zip bytes.

    Args:
        session: DB session used to resolve jobs/outputs.
        storage: Storage provider used to read the stored output binaries.
        batch_id: Export all outputs of this batch. Mutually exclusive with ``job_ids``.
        job_ids: Export the outputs of exactly these jobs. Mutually exclusive with
            ``batch_id``.

    Returns:
        The zip archive as ``bytes``.

    Raises:
        ValueError: If neither ``batch_id`` nor ``job_ids`` is supplied.
    """
    if batch_id is None and job_ids is None:
        raise ValueError("Provide either batch_id or job_ids")

    label = batch_id if batch_id is not None else f"{len(job_ids or [])} job(s)"
    with phase(log, f"Building catalog export for {label}"):
        outputs = _collect_outputs(session=session, batch_id=batch_id, job_ids=job_ids)

        rows: list[dict[str, Any]] = []
        thumbs: list[Image.Image] = []
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for out in outputs:
                name = f"{out.job_id}-{out.position}.png"
                if out.storage_key and storage.exists(out.storage_key):
                    image_bytes = storage.get_bytes(out.storage_key)
                    zf.writestr(f"images/{name}", image_bytes)
                    thumbs.append(Image.open(io.BytesIO(image_bytes)).copy())

                meta = dict(out.metadata_json or {})
                row = {
                    "filename": name,
                    "job_id": out.job_id,
                    "position": out.position,
                    "seed": out.seed,
                }
                row.update(_flatten_metadata(meta))
                rows.append({"_meta": meta, **row})

            # parameters.json — full, structured metadata per output.
            json_rows = [{k: v for k, v in r.items() if k != "_meta"} | r["_meta"] for r in rows]
            zf.writestr("parameters.json", json.dumps(json_rows, indent=2, default=str))

            # parameters.csv — one row per output; union of all scalar columns.
            csv_rows = [{k: v for k, v in r.items() if k != "_meta"} for r in rows]
            fieldnames: list[str] = []
            for r in csv_rows:
                for k in r:
                    if k not in fieldnames:
                        fieldnames.append(k)
            csv_buf = io.StringIO()
            writer = csv.DictWriter(csv_buf, fieldnames=fieldnames or ["filename"])
            writer.writeheader()
            writer.writerows(csv_rows)
            zf.writestr("parameters.csv", csv_buf.getvalue())

            # contact_sheet.png — thumbnail grid (always present, even when empty).
            zf.writestr("contact_sheet.png", _build_contact_sheet(thumbs))

        log.info("Export built: %d output(s), %d bytes", len(outputs), buf.tell())
        return buf.getvalue()
