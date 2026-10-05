import os
import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from typing import Any

from . import settings

CIVITAI_API_BASE = "https://civitai.com/api/v1"
CIVITAI_URL_RE   = re.compile(
    r"(?:https?://)?(?:[a-zA-Z0-9-]+\.)?civitai\.(?:com|red)/models/(\d+)(?:.*?modelVersionId=(\d+))?",
    re.IGNORECASE,
)
CIVITAI_VERSION_RE = re.compile(
    r"(?:https?://)?(?:[a-zA-Z0-9-]+\.)?civitai\.(?:com|red)/(?:model-versions|api/download/models)/(\d+)",
    re.IGNORECASE,
)
_REQUEST_TIMEOUT = 20


class CivitaiAPIError(Exception):
    pass


def _build_headers(api_key: str | None = None) -> dict[str, str]:
    key = settings.resolve_api_key(api_key)
    headers: dict[str, str] = {"User-Agent": "sd-forge-civitai-helper/2.0"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def with_api_token(url: str, api_key: str | None = None) -> str:
    """Attach a Civitai API token to /api/download URLs.

    Civitai authenticated downloads are more reliable with the token in the
    query string than with only an Authorization header, especially now that
    both civitai.com and civitai.red front doors exist and downloads redirect
    to pre-signed storage URLs. Non-Civitai URLs are left untouched.
    """
    key = settings.resolve_api_key(api_key)
    if not key:
        return url

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host not in {"civitai.com", "civitai.red"} and not (
        host.endswith(".civitai.com") or host.endswith(".civitai.red")
    ):
        return url
    if not parsed.path.startswith("/api/download/"):
        return url

    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if query.get("token"):
        return url
    query["token"] = key
    return urlunparse(parsed._replace(query=urlencode(query)))


def _get(url: str, api_key: str | None = None, **kwargs) -> dict:
    try:
        resp = requests.get(url, headers=_build_headers(api_key),
                            timeout=_REQUEST_TIMEOUT, **kwargs)
    except requests.exceptions.ConnectionError as exc:
        raise CivitaiAPIError(f"Connection failed: {exc}") from exc
    except requests.exceptions.Timeout:
        raise CivitaiAPIError("Request timed out.") from None
    except requests.exceptions.RequestException as exc:
        raise CivitaiAPIError(f"Network error: {exc}") from exc

    if resp.status_code == 401:
        raise CivitaiAPIError("Access denied (401). CivitAI API Key missing or invalid.")
    if resp.status_code == 404:
        raise CivitaiAPIError("Resource not found (404).")
    if resp.status_code == 429:
        raise CivitaiAPIError("Rate limit exceeded (429). Please wait.")
    if not resp.ok:
        raise CivitaiAPIError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        return resp.json()
    except Exception as exc:
        raise CivitaiAPIError("CivitAI returned an invalid (non-JSON) response.") from exc


def parse_model_url(url: str) -> tuple[str | None, str | None]:
    url = url.strip()
    if url.isdigit():
        return url, None
    match = CIVITAI_URL_RE.search(url)
    if match:
        return match.group(1), match.group(2)
    v_match = CIVITAI_VERSION_RE.search(url)
    if v_match:
        return None, v_match.group(1)
    return None, None


def fetch_model_info(model_id: str, api_key: str = "") -> dict:
    return _get(f"{CIVITAI_API_BASE}/models/{model_id}", api_key)


def fetch_version_by_id(version_id: str, api_key: str = "") -> dict:
    return _get(f"{CIVITAI_API_BASE}/model-versions/{version_id}", api_key)


def fetch_version_by_hash(sha256: str, api_key: str = "") -> dict | None:
    try:
        return _get(f"{CIVITAI_API_BASE}/model-versions/by-hash/{sha256}", api_key)
    except CivitaiAPIError as exc:
        if "404" in str(exc):
            return None
        raise


def search_models(query: str, model_type: str | None = None,
                  limit: int = 20, page: int = 1,
                  api_key: str = "", nsfw: bool = False) -> dict:
    params: dict[str, Any] = {
        "limit": min(limit, 100),
        "page":  page,
        "nsfw":  str(nsfw).lower(),
    }
    if query.strip():
        params["query"] = query.strip()
    if model_type and model_type not in ("Tous", "All"):
        params["types"] = model_type
    return _get(f"{CIVITAI_API_BASE}/models", api_key, params=params)


def extract_versions(model_info: dict) -> list[dict]:
    versions = []
    for v in model_info.get("modelVersions", []):
        files = [
            {
                "name":    f.get("name", "unknown"),
                "url":     f.get("downloadUrl", ""),
                "size_kb": f.get("sizeKB", 0),
                "type":    f.get("type", "Model"),
                "format":  f.get("metadata", {}).get("format", ""),
                "sha256":  (f.get("hashes") or {}).get("SHA256", ""),
            }
            for f in v.get("files", [])
            if f.get("downloadUrl")
        ]
        if not files:
            continue
        # Prioritize primary safetensors model file
        files.sort(
            key=lambda f: (
                0 if (f["name"].endswith(".safetensors") and f.get("type") == "Model") else
                1 if f["name"].endswith(".safetensors") else
                2 if f.get("type") == "Model" else
                3
            )
        )
        versions.append({
            "id":            v["id"],
            "name":          v.get("name", ""),
            "label":         f"{v.get('name', '')} — base: {v.get('baseModel', '?')}",
            "base_model":    v.get("baseModel", ""),
            "files":         files,
            "images":        v.get("images", []),
            "trained_words": v.get("trainedWords", []),
        })
    return versions
