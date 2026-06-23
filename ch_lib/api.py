import os
import re
import requests
from typing import Any

CIVITAI_API_BASE = "https://civitai.red/api/v1"
CIVITAI_URL_RE   = re.compile(
    r"(?:https?://)?(?:civitai\.com|civitai\.red)/models/(\d+)(?:.*?modelVersionId=(\d+))?"
)
_REQUEST_TIMEOUT = 20


class CivitaiAPIError(Exception):
    pass


def _build_headers(api_key: str | None = None) -> dict[str, str]:
    key = (api_key or os.environ.get("CIVITAI_API_KEY", "")).strip()
    headers: dict[str, str] = {"User-Agent": "sd-forge-civitai-helper/2.0"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _get(url: str, api_key: str | None = None, **kwargs) -> dict:
    try:
        resp = requests.get(url, headers=_build_headers(api_key),
                            timeout=_REQUEST_TIMEOUT, **kwargs)
    except requests.exceptions.ConnectionError as exc:
        raise CivitaiAPIError(f"Connexion impossible : {exc}") from exc
    except requests.exceptions.Timeout:
        raise CivitaiAPIError("Timeout.") from None

    if resp.status_code == 401:
        raise CivitaiAPIError("Accès refusé (401). Clé API manquante ou invalide.")
    if resp.status_code == 404:
        raise CivitaiAPIError("Ressource introuvable (404).")
    if resp.status_code == 429:
        raise CivitaiAPIError("Trop de requêtes (429). Patientez.")
    if not resp.ok:
        raise CivitaiAPIError(f"HTTP {resp.status_code} : {resp.text[:200]}")
    return resp.json()


def parse_model_url(url: str) -> tuple[str | None, str | None]:
    url = url.strip()
    if url.isdigit():
        return url, None
    match = CIVITAI_URL_RE.search(url)
    if match:
        return match.group(1), match.group(2)
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
    if model_type and model_type != "Tous":
        params["types"] = model_type
    return _get(f"{CIVITAI_API_BASE}/models", api_key, params=params)


def extract_versions(model_info: dict) -> list[dict]:
    versions = []
    for v in model_info.get("modelVersions", []):
        files = [
            {
                "name":    f.get("name", "unknown"),
                "url":     f.get("downloadUrl", "").replace("civitai.com", "civitai.red"),
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
