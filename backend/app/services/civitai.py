"""CivitAI URL normalization + API client (DESIGN §5.3, §5.7; CLAUDE.md constraint).

CivitAI exposes three "front doors" — ``civitai.com``, ``civitai.red`` and
``civitai.green`` — that are *filtered views over one database* sharing the canonical
``civitai.com/api/v1`` API. The importer therefore **strips the domain colour**, extracts
the model id and optional ``modelVersionId``, and resolves everything through the canonical
API. The API key authorizes mature content.

Network access is isolated in :func:`_get_json` (a single ``httpx`` call) so unit tests can
monkeypatch it and never touch the network. :func:`normalize_civitai_url` is pure and is
unit-tested exhaustively across all three domains and URL forms.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

import httpx

from app.logging_utils import phase

log = logging.getLogger(__name__)

# Canonical API host — all colour-domains resolve here.
CIVITAI_API_BASE = "https://civitai.com/api/v1"

# Accepted host suffixes (the three "front doors" over one DB).
_CIVITAI_HOSTS = ("civitai.com", "civitai.red", "civitai.green")

# Matches a leading ``/models/<id>`` path segment (optionally with a trailing slug/path).
_MODELS_PATH_RE = re.compile(r"/models/(\d+)")
# Matches a ``/model-versions/<id>`` (canonical API) or ``/api/v1/models/<id>`` form.
_MODEL_VERSIONS_PATH_RE = re.compile(r"/model-versions/(\d+)")
_API_MODELS_PATH_RE = re.compile(r"/api/v\d+/models/(\d+)")


@dataclass
class CivitaiModelMeta:
    """Normalized metadata resolved from the canonical CivitAI API.

    Only the fields QIE Studio needs for a LoRA import are surfaced; the raw payload is
    kept on :attr:`raw` for callers that want more.
    """

    model_id: int
    version_id: int | None
    name: str
    trigger_words: list[str] = field(default_factory=list)
    recommended_weight: float | None = None
    preview_image_urls: list[str] = field(default_factory=list)
    download_url: str | None = None
    base_model: str = ""
    license: str = ""
    raw: dict = field(default_factory=dict)


def _is_civitai_host(host: str) -> bool:
    host = host.lower()
    return any(host == h or host.endswith("." + h) for h in _CIVITAI_HOSTS)


def normalize_civitai_url(url: str) -> tuple[int | None, int | None]:
    """Extract ``(model_id, version_id)`` from any CivitAI URL/colour-domain or bare id.

    Handles, across ``civitai.com`` / ``.red`` / ``.green``:

    - ``civitai.com/models/12345``
    - ``civitai.com/models/12345?modelVersionId=67890``
    - ``civitai.red/models/12345/some-slug``
    - ``civitai.green/models/12345/some-slug?modelVersionId=67890``
    - ``https://civitai.com/api/v1/models/12345``
    - ``https://civitai.com/api/v1/model-versions/67890`` (version only)
    - a bare model id (``"12345"``)
    - a bare ``"12345@67890"`` (model@version) shorthand

    Returns ``(model_id, version_id)`` where either component may be ``None``. The domain
    colour is irrelevant — all three resolve to the same canonical ids.
    """
    if url is None:
        raise ValueError("URL is required")
    raw = url.strip()
    if not raw:
        raise ValueError("URL is required")

    # --- Bare-id shorthands: "12345" or "12345@67890" or "12345:67890". ---
    bare = re.fullmatch(r"(\d+)(?:[@:](\d+))?", raw)
    if bare:
        model_id = int(bare.group(1))
        version_id = int(bare.group(2)) if bare.group(2) else None
        return model_id, version_id

    # Ensure urlparse sees a scheme so the host lands in ``netloc`` not ``path``.
    parse_target = raw if "//" in raw else f"//{raw}"
    parsed = urlparse(parse_target)
    host = parsed.hostname or ""
    if host and not _is_civitai_host(host):
        raise ValueError(f"Not a recognized CivitAI URL: {url!r}")

    path = parsed.path or ""
    query = parse_qs(parsed.query or "")

    # A ``modelVersionId`` query param overrides any path-derived version.
    version_id: int | None = None
    mv = query.get("modelVersionId") or query.get("modelversionid")
    if mv and mv[0].isdigit():
        version_id = int(mv[0])

    # Version-only canonical API form: ``/model-versions/<id>``.
    vm = _MODEL_VERSIONS_PATH_RE.search(path)
    if vm and version_id is None:
        version_id = int(vm.group(1))

    # Model id from ``/api/v1/models/<id>`` or plain ``/models/<id>``.
    am = _API_MODELS_PATH_RE.search(path)
    mm = am or _MODELS_PATH_RE.search(path)
    model_id = int(mm.group(1)) if mm else None

    if model_id is None and version_id is None:
        raise ValueError(f"Could not extract a CivitAI model/version id from {url!r}")
    return model_id, version_id


def _auth_headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _get_json(path: str, *, api_key: str | None = None, params: dict | None = None) -> dict:
    """Fetch JSON from the canonical CivitAI API (the single network seam to monkeypatch).

    ``path`` is appended to :data:`CIVITAI_API_BASE` (e.g. ``"/models/123"``). Follows
    redirects so colour-domain front doors land on canonical content.
    """
    url = f"{CIVITAI_API_BASE}{path}"
    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        resp = client.get(url, headers=_auth_headers(api_key), params=params)
        resp.raise_for_status()
        return resp.json()


def _pick_version(model_payload: dict, version_id: int | None) -> dict | None:
    """Choose the version dict from a ``/models/{id}`` payload (the requested or first)."""
    versions = model_payload.get("modelVersions") or []
    if not versions:
        return None
    if version_id is not None:
        for v in versions:
            if v.get("id") == version_id:
                return v
    return versions[0]


def _safetensors_download_url(version: dict) -> str | None:
    """Pick the ``.safetensors`` file's download url from a version payload."""
    files = version.get("files") or []
    for f in files:
        name = (f.get("name") or "").lower()
        if name.endswith(".safetensors") and f.get("downloadUrl"):
            return f["downloadUrl"]
    # Fall back to the version-level downloadUrl if present.
    return version.get("downloadUrl")


def _build_meta(model_id: int, model_payload: dict, version: dict | None) -> CivitaiModelMeta:
    """Assemble a :class:`CivitaiModelMeta` from raw model + version payloads."""
    trained_words: list[str] = []
    download_url: str | None = None
    base_model = ""
    images: list[str] = []
    resolved_version_id: int | None = None

    if version is not None:
        resolved_version_id = version.get("id")
        trained_words = [w for w in (version.get("trainedWords") or []) if isinstance(w, str)]
        download_url = _safetensors_download_url(version)
        base_model = version.get("baseModel") or ""
        images = [
            img["url"]
            for img in (version.get("images") or [])
            if isinstance(img, dict) and img.get("url")
        ]

    return CivitaiModelMeta(
        model_id=model_id,
        version_id=resolved_version_id,
        name=model_payload.get("name") or f"civitai-{model_id}",
        trigger_words=trained_words,
        recommended_weight=None,  # CivitAI does not expose a canonical recommended weight
        preview_image_urls=images,
        download_url=download_url,
        base_model=base_model,
        license=_license_string(model_payload),
        raw=model_payload,
    )


def _license_string(model_payload: dict) -> str:
    """Summarize the permission flags CivitAI returns into a short license string."""
    permission_keys = (
        "allowCommercialUse",
        "allowDerivatives",
        "allowNoCredit",
        "allowDifferentLicense",
    )
    flags = [k for k in permission_keys if model_payload.get(k)]
    return ", ".join(flags)


def fetch_model_metadata(
    *,
    model_id: int | None = None,
    version_id: int | None = None,
    api_key: str | None = None,
) -> CivitaiModelMeta:
    """Resolve canonical metadata for a CivitAI model/version.

    Exactly one of ``model_id`` / ``version_id`` is required. When only ``version_id`` is
    given, the version is fetched first to discover its parent ``model_id``. The network
    call lives in :func:`_get_json`, which tests monkeypatch.

    Returns:
        A :class:`CivitaiModelMeta` with name, trigger words, preview urls, the
        ``.safetensors`` download url, the base-model tag and a license summary.
    """
    if model_id is None and version_id is None:
        raise ValueError("model_id or version_id is required")

    with phase(log, f"Fetching CivitAI metadata (model={model_id} version={version_id})"):
        if model_id is None:
            # Version-only: resolve the version to find its parent model.
            version_payload = _get_json(f"/model-versions/{version_id}", api_key=api_key)
            model_id = version_payload.get("modelId")
            if model_id is None:
                raise ValueError(f"CivitAI version {version_id} has no parent modelId")
            model_payload = _get_json(f"/models/{model_id}", api_key=api_key)
            version = _pick_version(model_payload, version_id) or version_payload
        else:
            model_payload = _get_json(f"/models/{model_id}", api_key=api_key)
            version = _pick_version(model_payload, version_id)

    return _build_meta(int(model_id), model_payload, version)


def search_models(*, query: str, api_key: str | None = None, limit: int = 20) -> list[dict]:
    """Search CivitAI models by name (DESIGN §7 ``/api/loras/search``).

    Returns a list of compact result dicts. Network failures or a missing key bubble up to
    the caller, which degrades gracefully to ``[]`` (the router decides).
    """
    with phase(log, f"Searching CivitAI for {query!r}"):
        payload = _get_json(
            "/models",
            api_key=api_key,
            params={"query": query, "limit": limit, "types": "LORA"},
        )
    items = payload.get("items") or []
    results: list[dict] = []
    for item in items:
        versions = item.get("modelVersions") or []
        first = versions[0] if versions else {}
        results.append(
            {
                "model_id": item.get("id"),
                "version_id": first.get("id"),
                "name": item.get("name"),
                "base_model": first.get("baseModel", ""),
                "trigger_words": first.get("trainedWords") or [],
            }
        )
    return results
