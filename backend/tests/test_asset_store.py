"""Asset store tests (DESIGN §5.1, §6).

Covers import + normalization, content de-dup, thumbnail generation, promote-to-
library, output saving with embedded PNG metadata, list/filter, update, and delete.
All DB tests use the ``session`` fixture and skip when no PostgreSQL is reachable.
"""

from __future__ import annotations

import io
import json

import pytest
from app.interfaces.auth import Principal
from app.interfaces.storage import LocalFsStorage
from app.models.asset import Asset
from app.models.job import Job, JobOutput
from app.services import asset_store
from PIL import Image, PngImagePlugin


def _make_job(session, job_id: str, mode: str = "generate") -> Job:  # noqa: ANN001
    """Persist a minimal Job row (JobOutput has an FK to job.id)."""
    job = Job(id=job_id, mode=mode, status="running")
    session.add(job)
    session.commit()
    return job



@pytest.fixture
def storage(tmp_path) -> LocalFsStorage:  # noqa: ANN001
    return LocalFsStorage(tmp_path)


@pytest.fixture
def principal() -> Principal:
    return Principal()


def _png_bytes(color: tuple[int, int, int] = (200, 30, 30), size=(64, 48)) -> bytes:  # noqa: ANN001
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(color=(10, 120, 200), size=(40, 40)) -> bytes:  # noqa: ANN001
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _rgba_png_bytes(size=(32, 32)) -> bytes:  # noqa: ANN001
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_ingest_upload_normalizes_and_thumbnails(session, storage, principal) -> None:  # noqa: ANN001
    asset = asset_store.ingest_upload(
        data=_png_bytes(size=(800, 600)),
        filename="photo.png",
        session=session,
        storage=storage,
        principal=principal,
    )
    assert asset.id
    assert asset.scope == "ephemeral"
    assert asset.source == "upload"
    assert asset.format == "PNG"
    assert (asset.width, asset.height) == (800, 600)
    assert asset.bytes > 0
    assert len(asset.sha256) == 64
    # Working copy + thumbnail both exist in storage.
    assert storage.exists(asset.storage_key)
    assert asset.thumb_key and storage.exists(asset.thumb_key)
    # Thumbnail is a WebP with longest edge <= 384, aspect preserved.
    with Image.open(io.BytesIO(storage.get_bytes(asset.thumb_key))) as thumb:
        assert thumb.format == "WEBP"
        assert max(thumb.size) <= asset_store.THUMB_MAX_EDGE
        assert thumb.width > thumb.height  # landscape preserved


def test_ingest_normalizes_jpeg_and_rgba(session, storage, principal) -> None:  # noqa: ANN001
    jpeg = asset_store.ingest_upload(
        data=_jpeg_bytes(),
        filename="x.jpg",
        session=session,
        storage=storage,
        principal=principal,
    )
    # Normalized working copy is always PNG.
    assert jpeg.format == "PNG"
    with Image.open(io.BytesIO(storage.get_bytes(jpeg.storage_key))) as im:
        assert im.format == "PNG"
        assert im.mode == "RGB"

    rgba = asset_store.ingest_upload(
        data=_rgba_png_bytes(),
        filename="a.png",
        session=session,
        storage=storage,
        principal=principal,
    )
    with Image.open(io.BytesIO(storage.get_bytes(rgba.storage_key))) as im:
        assert im.mode == "RGB"  # alpha flattened


def test_dedup_returns_existing_asset(session, storage, principal) -> None:  # noqa: ANN001
    data = _png_bytes()
    first = asset_store.ingest_upload(
        data=data, filename="a.png", session=session, storage=storage, principal=principal
    )
    second = asset_store.ingest_upload(
        data=data, filename="b.png", session=session, storage=storage, principal=principal
    )
    assert first.id == second.id
    # Only one row stored.
    rows = asset_store.list_assets(session=session, principal=principal)
    assert len([r for r in rows if r.sha256 == first.sha256]) == 1


def test_promote_to_library(session, storage, principal) -> None:  # noqa: ANN001
    asset = asset_store.ingest_upload(
        data=_png_bytes(), filename="a.png", session=session, storage=storage, principal=principal
    )
    assert asset.scope == "ephemeral"
    promoted = asset_store.promote_to_library(
        asset_id=asset.id, session=session, principal=principal
    )
    assert promoted.id == asset.id
    assert promoted.scope == "library"
    assert promoted.source == "upload"  # source metadata preserved


def test_promote_missing_raises(session, principal) -> None:  # noqa: ANN001
    with pytest.raises(KeyError):
        asset_store.promote_to_library(asset_id="nope", session=session, principal=principal)


def test_save_output_embeds_png_metadata(session, storage, principal) -> None:  # noqa: ANN001
    _make_job(session, "job-abc")
    img = Image.new("RGB", (50, 50), (12, 34, 56))
    metadata = {"mode": "generate", "seed": 1234, "prompt": "a cat", "precision": "bf16"}
    asset, output = asset_store.save_output_image(
        image=img,
        job_id="job-abc",
        position=0,
        seed=1234,
        metadata=metadata,
        session=session,
        storage=storage,
        principal=principal,
    )
    assert isinstance(asset, Asset)
    assert isinstance(output, JobOutput)
    assert asset.source == "output"
    assert asset.source_job_id == "job-abc"
    assert asset.scope == "ephemeral"
    assert output.job_id == "job-abc"
    assert output.seed == 1234
    assert output.metadata_json == metadata

    # Metadata embedded in the PNG text chunk.
    raw = storage.get_bytes(asset.storage_key)
    with Image.open(io.BytesIO(raw)) as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        embedded = json.loads(im.text["qie-metadata"])
        assert embedded["seed"] == 1234
        assert embedded["prompt"] == "a cat"

    # JSON sidecar written too.
    sidecar_key = asset_store._sidecar_key(asset.sha256)
    assert storage.exists(sidecar_key)
    assert json.loads(storage.get_bytes(sidecar_key))["mode"] == "generate"


def test_open_pil_returns_rgb(session, storage, principal) -> None:  # noqa: ANN001
    asset = asset_store.ingest_upload(
        data=_png_bytes(), filename="a.png", session=session, storage=storage, principal=principal
    )
    img = asset_store.open_pil(asset=asset, storage=storage)
    assert img.mode == "RGB"
    assert (img.width, img.height) == (asset.width, asset.height)


def test_list_filter_by_scope_source_tag_and_query(session, storage, principal) -> None:  # noqa: ANN001, E501
    up = asset_store.ingest_upload(
        data=_png_bytes(color=(1, 2, 3)),
        filename="alpha.png",
        session=session,
        storage=storage,
        principal=principal,
    )
    _make_job(session, "j1", mode="edit")
    out_img = Image.new("RGB", (20, 20), (9, 9, 9))
    asset_store.save_output_image(
        image=out_img,
        job_id="j1",
        position=0,
        seed=None,
        metadata={"mode": "edit"},
        session=session,
        storage=storage,
        principal=principal,
    )
    asset_store.update_asset(
        asset_id=up.id, session=session, principal=principal, tags=["staging"], name="Alpha Room"
    )

    assert len(asset_store.list_assets(session=session, principal=principal)) == 2
    uploads = asset_store.list_assets(session=session, principal=principal, source="upload")
    assert [a.id for a in uploads] == [up.id]
    outputs = asset_store.list_assets(session=session, principal=principal, source="output")
    assert len(outputs) == 1 and outputs[0].source == "output"
    tagged = asset_store.list_assets(session=session, principal=principal, tag="staging")
    assert [a.id for a in tagged] == [up.id]
    found = asset_store.list_assets(session=session, principal=principal, q="alpha room")
    assert [a.id for a in found] == [up.id]
    none = asset_store.list_assets(session=session, principal=principal, q="zzz-nomatch")
    assert none == []


def test_update_asset_fields(session, storage, principal) -> None:  # noqa: ANN001
    asset = asset_store.ingest_upload(
        data=_png_bytes(), filename="a.png", session=session, storage=storage, principal=principal
    )
    updated = asset_store.update_asset(
        asset_id=asset.id,
        session=session,
        principal=principal,
        name="Renamed",
        description="desc",
        tags=["t1", "t2"],
        collection="my-coll",
    )
    assert updated.name == "Renamed"
    assert updated.description == "desc"
    assert set(updated.tags) == {"t1", "t2"}
    assert updated.collection == "my-coll"


def test_delete_removes_row_and_binaries(session, storage, principal) -> None:  # noqa: ANN001
    asset = asset_store.ingest_upload(
        data=_png_bytes(), filename="a.png", session=session, storage=storage, principal=principal
    )
    storage_key, thumb_key = asset.storage_key, asset.thumb_key
    asset_store.delete_asset(
        asset_id=asset.id, session=session, principal=principal, storage=storage
    )
    assert asset_store.get_asset(asset_id=asset.id, session=session) is None
    assert not storage.exists(storage_key)
    assert not storage.exists(thumb_key)


def test_delete_keeps_shared_binary_for_dedup(session, storage, principal) -> None:  # noqa: ANN001
    """If two rows share content-addressed binaries, deleting one keeps the bytes."""
    data = _png_bytes()
    a = asset_store.ingest_upload(
        data=data, filename="a.png", session=session, storage=storage, principal=principal
    )
    # Manually add a second row pointing at the same keys (simulating cross-scope share).
    twin = Asset(
        owner_id=principal.owner_id,
        workspace_id=principal.workspace_id,
        scope="library",
        source="upload",
        storage_key=a.storage_key,
        thumb_key=a.thumb_key,
        format="PNG",
        width=a.width,
        height=a.height,
        bytes=a.bytes,
        sha256=a.sha256,
    )
    session.add(twin)
    session.commit()

    asset_store.delete_asset(asset_id=a.id, session=session, principal=principal, storage=storage)
    # Bytes remain because the twin still references them.
    assert storage.exists(twin.storage_key)


def test_delete_missing_raises(session, storage, principal) -> None:  # noqa: ANN001
    with pytest.raises(KeyError):
        asset_store.delete_asset(
            asset_id="nope", session=session, principal=principal, storage=storage
        )
