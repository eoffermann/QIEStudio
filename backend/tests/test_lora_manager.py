"""Tests for the LoRA registry + multi-source importer (DESIGN §5.3).

DB-backed (``session`` fixture; skips without PostgreSQL). All network access is injected:
URL/CivitAI downloads use a ``download_fn`` stub, HF uses ``list_files_fn``/``download_fn``
stubs, and the CivitAI metadata call monkeypatches ``civitai._get_json``. No network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.interfaces.auth import Principal
from app.interfaces.storage import LocalFsStorage
from app.models.lora import Lora
from app.services import civitai, integrations, lora_manager

pytestmark = pytest.mark.db


def make_safetensors(payload: dict | None = None) -> bytes:
    """Build a minimal but structurally-valid safetensors blob (8-byte len + JSON header)."""
    header = json.dumps(payload or {"__metadata__": {"k": "v"}}).encode("utf-8")
    return len(header).to_bytes(8, "little") + header + b"\x00\x00"


@pytest.fixture
def principal() -> Principal:
    return Principal()


@pytest.fixture
def storage(tmp_path: Path) -> LocalFsStorage:
    return LocalFsStorage(tmp_path / "data")


# --- is_safetensors -------------------------------------------------------------------


def test_is_safetensors_by_extension() -> None:
    assert lora_manager.is_safetensors(filename="adapter.safetensors")
    assert not lora_manager.is_safetensors(filename="adapter.ckpt", data=b"junkjunkjunk")


def test_is_safetensors_by_header() -> None:
    assert lora_manager.is_safetensors(data=make_safetensors())
    assert not lora_manager.is_safetensors(data=b"not a safetensors file at all")


# --- upload ---------------------------------------------------------------------------


def test_import_from_upload(session, storage, principal) -> None:
    data = make_safetensors()
    lora = lora_manager.import_from_upload(
        data=data, filename="my_lora.safetensors", name="My LoRA",
        session=session, storage=storage, principal=principal,
    )
    assert lora.name == "My LoRA"
    assert lora.source == "upload"
    assert lora.bytes == len(data)
    assert lora.sha256
    assert storage.exists(lora.storage_key)
    assert lora.storage_key.startswith("loras/")


def test_upload_rejects_non_safetensors(session, storage, principal) -> None:
    with pytest.raises(ValueError):
        lora_manager.import_from_upload(
            data=b"definitely not safetensors", filename="evil.bin",
            session=session, storage=storage, principal=principal,
        )


def test_upload_default_name_from_filename(session, storage, principal) -> None:
    lora = lora_manager.import_from_upload(
        data=make_safetensors(), filename="cool-style.safetensors",
        session=session, storage=storage, principal=principal,
    )
    assert lora.name == "cool-style"


# --- url ------------------------------------------------------------------------------


def test_import_from_url(session, storage, principal) -> None:
    blob = make_safetensors()
    calls: list[str] = []

    def fake_download(url: str, token: str | None) -> bytes:
        calls.append(url)
        return blob

    lora = lora_manager.import_from_url(
        url="https://host/path/weights.safetensors?dl=1",
        session=session, storage=storage, principal=principal,
        download_fn=fake_download,
    )
    assert lora.source == "url"
    assert lora.name == "weights"
    assert calls == ["https://host/path/weights.safetensors?dl=1"]


# --- civitai --------------------------------------------------------------------------


def test_import_from_civitai(session, storage, principal, monkeypatch) -> None:
    # Configure a CivitAI key so the importer reads it (encrypted at rest).
    monkeypatch.setattr(integrations, "_get", lambda url, headers, params=None: 200)
    integrations.set_key(provider="civitai", key="civ-key", session=session, principal=principal)

    def fake_get_json(path: str, *, api_key=None, params=None) -> dict:
        assert api_key == "civ-key"  # key flows through from the encrypted store
        return {
            "id": 42,
            "name": "Civit LoRA",
            "allowCommercialUse": ["Image"],
            "modelVersions": [
                {
                    "id": 7,
                    "baseModel": "Qwen-Image-Edit",
                    "trainedWords": ["trig"],
                    "images": [{"url": "https://img/p.jpg"}],
                    "files": [{"name": "x.safetensors", "downloadUrl": "https://dl/x"}],
                }
            ],
        }

    monkeypatch.setattr(civitai, "_get_json", fake_get_json)

    blob = make_safetensors()

    def fake_download(url: str, token: str | None) -> bytes:
        assert url == "https://dl/x"
        assert token == "civ-key"
        return blob

    lora = lora_manager.import_from_civitai(
        url_or_id="https://civitai.green/models/42?modelVersionId=7",
        session=session, storage=storage, principal=principal,
        download_fn=fake_download,
    )
    assert lora.source == "civitai"
    assert lora.name == "Civit LoRA"
    assert lora.trigger_words == ["trig"]
    assert lora.base_compat == "qwen-image-edit"
    assert lora.source_ref == "42@7"


def test_import_from_civitai_no_download_url(session, storage, principal, monkeypatch) -> None:
    monkeypatch.setattr(
        civitai, "_get_json",
        lambda path, **kw: {"id": 1, "name": "n", "modelVersions": [{"id": 2, "files": []}]},
    )
    with pytest.raises(ValueError):
        lora_manager.import_from_civitai(
            url_or_id="1", session=session, storage=storage, principal=principal,
            download_fn=lambda url, token: b"",
        )


# --- hugging face ---------------------------------------------------------------------


def test_import_from_hf_autodetect(session, storage, principal, tmp_path) -> None:
    blob = make_safetensors()
    weight_path = tmp_path / "pytorch_lora_weights.safetensors"
    weight_path.write_bytes(blob)

    def fake_list(repo_id: str, token: str | None) -> list[str]:
        return ["README.md", "config.json", "pytorch_lora_weights.safetensors"]

    def fake_download(*, repo_id, filename, token, cache_dir) -> str:
        assert filename == "pytorch_lora_weights.safetensors"
        return str(weight_path)

    lora = lora_manager.import_from_hf(
        repo_id="someuser/some-lora",
        session=session, storage=storage, principal=principal,
        list_files_fn=fake_list, download_fn=fake_download,
    )
    assert lora.source == "hf"
    assert lora.name == "some-lora"
    assert lora.source_ref == "someuser/some-lora:pytorch_lora_weights.safetensors"


def test_import_from_hf_no_weight_file(session, storage, principal) -> None:
    with pytest.raises(ValueError):
        lora_manager.import_from_hf(
            repo_id="u/r", session=session, storage=storage, principal=principal,
            list_files_fn=lambda repo_id, token: ["README.md"],
            download_fn=lambda **kw: "unused",
        )


def test_pick_hf_weight_file_prefers_lora_name() -> None:
    files = ["other.safetensors", "my_lora.safetensors"]
    assert lora_manager._pick_hf_weight_file(files) == "my_lora.safetensors"
    assert lora_manager._pick_hf_weight_file(["a.safetensors"]) == "a.safetensors"
    assert lora_manager._pick_hf_weight_file(["a.bin"]) is None


# --- registry CRUD --------------------------------------------------------------------


def test_list_update_set_enabled_delete(session, storage, principal) -> None:
    lora = lora_manager.import_from_upload(
        data=make_safetensors(), filename="a.safetensors",
        session=session, storage=storage, principal=principal,
    )
    assert len(lora_manager.list_loras(session=session, principal=principal)) == 1

    updated = lora_manager.update_lora(
        lora_id=lora.id, name="Renamed", trigger_words=["w1", "w2"],
        recommended_weight=0.7, session=session, principal=principal,
    )
    assert updated is not None
    assert updated.name == "Renamed"
    assert updated.trigger_words == ["w1", "w2"]
    assert updated.recommended_weight == 0.7

    toggled = lora_manager.set_enabled(
        lora_id=lora.id, enabled=False, session=session, principal=principal
    )
    assert toggled is not None and toggled.enabled is False

    key = lora.storage_key
    assert lora_manager.delete_lora(
        lora_id=lora.id, session=session, storage=storage, principal=principal
    )
    assert not storage.exists(key)
    assert lora_manager.list_loras(session=session, principal=principal) == []


def test_update_missing_returns_none(session, principal) -> None:
    assert lora_manager.update_lora(
        lora_id="nope", name="x", session=session, principal=principal
    ) is None


def test_delete_shared_blob_kept(session, storage, principal) -> None:
    # Two uploads of identical bytes share one content-addressed key.
    data = make_safetensors()
    a = lora_manager.import_from_upload(
        data=data, filename="a.safetensors", session=session, storage=storage, principal=principal
    )
    b = lora_manager.import_from_upload(
        data=data, filename="b.safetensors", session=session, storage=storage, principal=principal
    )
    assert a.storage_key == b.storage_key
    lora_manager.delete_lora(
        lora_id=a.id, session=session, storage=storage, principal=principal
    )
    # Blob must survive because b still references it.
    assert storage.exists(b.storage_key)


# --- check_compat ---------------------------------------------------------------------


def test_check_compat_clean() -> None:
    lora = Lora(name="L", storage_key="k", base_compat="qwen-image", modes=["generate"])
    assert lora_manager.check_compat(lora=lora, mode="generate", base="qwen-image") == []


def test_check_compat_base_mismatch() -> None:
    lora = Lora(name="L", storage_key="k", base_compat="qwen-image", modes=[])
    warns = lora_manager.check_compat(lora=lora, mode="edit", base="qwen-image-edit")
    assert any("targets qwen-image" in w for w in warns)


def test_check_compat_mode_mismatch() -> None:
    lora = Lora(name="L", storage_key="k", base_compat="qwen-image", modes=["generate"])
    warns = lora_manager.check_compat(lora=lora, mode="edit", base="qwen-image")
    assert any("'edit'" in w for w in warns)


def test_check_compat_disabled() -> None:
    lora = Lora(name="L", storage_key="k", base_compat="qwen-image", modes=[], enabled=False)
    warns = lora_manager.check_compat(lora=lora, mode="generate", base="qwen-image")
    assert any("disabled" in w for w in warns)
