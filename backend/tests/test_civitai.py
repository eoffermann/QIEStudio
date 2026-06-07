"""Tests for CivitAI URL normalization + metadata client (DESIGN §5.3; CLAUDE.md).

``normalize_civitai_url`` is tested exhaustively WITHOUT network across all three colour
domains and URL forms. ``fetch_model_metadata`` / ``search_models`` monkeypatch the single
``_get_json`` network seam.
"""

from __future__ import annotations

import pytest
from app.services import civitai


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # Plain .com forms.
        ("https://civitai.com/models/12345", (12345, None)),
        ("https://civitai.com/models/12345?modelVersionId=67890", (12345, 67890)),
        ("https://civitai.com/models/12345/some-slug", (12345, None)),
        ("https://civitai.com/models/12345/some-slug?modelVersionId=67890", (12345, 67890)),
        # Colour domains resolve to the same canonical ids.
        ("https://civitai.red/models/12345/foo", (12345, None)),
        ("https://civitai.green/models/12345?modelVersionId=67890", (12345, 67890)),
        ("http://www.civitai.red/models/999", (999, None)),
        # Canonical API forms.
        ("https://civitai.com/api/v1/models/12345", (12345, None)),
        ("https://civitai.com/api/v1/model-versions/67890", (None, 67890)),
        # No scheme.
        ("civitai.com/models/12345", (12345, None)),
        ("civitai.green/models/55?modelVersionId=66", (55, 66)),
        # Bare-id shorthands.
        ("12345", (12345, None)),
        ("12345@67890", (12345, 67890)),
        ("12345:67890", (12345, 67890)),
    ],
)
def test_normalize_civitai_url(url: str, expected: tuple[int | None, int | None]) -> None:
    assert civitai.normalize_civitai_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "https://example.com/models/12345",
        "https://civitai.com/articles/123",  # no model/version id present
        "not a url at all",
    ],
)
def test_normalize_civitai_url_rejects(url: str) -> None:
    with pytest.raises(ValueError):
        civitai.normalize_civitai_url(url)


def test_query_param_overrides_path_version() -> None:
    # A modelVersionId query param should win over any path-derived version.
    url = "https://civitai.com/model-versions/111?modelVersionId=222"
    assert civitai.normalize_civitai_url(url) == (None, 222)


def _model_payload() -> dict:
    return {
        "id": 12345,
        "name": "Cool LoRA",
        "allowCommercialUse": ["Image"],
        "allowDerivatives": True,
        "modelVersions": [
            {
                "id": 67890,
                "baseModel": "Qwen-Image",
                "trainedWords": ["coolstyle", "vivid"],
                "images": [{"url": "https://img/1.jpg"}, {"nourl": True}],
                "files": [
                    {"name": "model.ckpt", "downloadUrl": "https://dl/ckpt"},
                    {"name": "lora.safetensors", "downloadUrl": "https://dl/safetensors"},
                ],
            },
            {"id": 11111, "baseModel": "Qwen-Image", "files": []},
        ],
    }


def test_fetch_model_metadata_by_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_json(path: str, *, api_key=None, params=None) -> dict:
        assert path == "/models/12345"
        return _model_payload()

    monkeypatch.setattr(civitai, "_get_json", fake_get_json)
    meta = civitai.fetch_model_metadata(model_id=12345)

    assert meta.model_id == 12345
    assert meta.version_id == 67890  # first version chosen
    assert meta.name == "Cool LoRA"
    assert meta.trigger_words == ["coolstyle", "vivid"]
    assert meta.download_url == "https://dl/safetensors"
    assert meta.preview_image_urls == ["https://img/1.jpg"]
    assert meta.base_model == "Qwen-Image"
    assert "allowCommercialUse" in meta.license


def test_fetch_model_metadata_specific_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        civitai, "_get_json", lambda path, **kw: _model_payload()
    )
    meta = civitai.fetch_model_metadata(model_id=12345, version_id=11111)
    assert meta.version_id == 11111
    assert meta.download_url is None  # that version has no files


def test_fetch_model_metadata_version_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_get_json(path: str, *, api_key=None, params=None) -> dict:
        calls.append(path)
        if path == "/model-versions/67890":
            return {"id": 67890, "modelId": 12345}
        return _model_payload()

    monkeypatch.setattr(civitai, "_get_json", fake_get_json)
    meta = civitai.fetch_model_metadata(version_id=67890)
    assert meta.model_id == 12345
    assert "/model-versions/67890" in calls
    assert "/models/12345" in calls


def test_fetch_requires_an_id() -> None:
    with pytest.raises(ValueError):
        civitai.fetch_model_metadata()


def test_search_models(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_json(path: str, *, api_key=None, params=None) -> dict:
        assert path == "/models"
        assert params["query"] == "anime"
        return {
            "items": [
                {
                    "id": 1,
                    "name": "Anime LoRA",
                    "modelVersions": [{"id": 9, "baseModel": "Qwen-Image", "trainedWords": ["a"]}],
                }
            ]
        }

    monkeypatch.setattr(civitai, "_get_json", fake_get_json)
    results = civitai.search_models(query="anime")
    assert results == [
        {
            "model_id": 1,
            "version_id": 9,
            "name": "Anime LoRA",
            "base_model": "Qwen-Image",
            "trigger_words": ["a"],
        }
    ]
