import re
import time
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from typing import Any

from . import settings, utils

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
_MAX_ATTEMPTS    = 4
_MAX_WAIT        = 60.0
_RETRY_STATUSES  = {429, 500, 502, 503, 504}
_sleep           = time.sleep

PREVIEW_WIDTH = 640
_IMAGE_SIZE_SEGMENT_RE = re.compile(r"^(?:width=\d+|original=true)(?:,.*)?$", re.IGNORECASE)
_VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".gif")


class CivitaiAPIError(Exception):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


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

    The resulting URL carries a secret: pass anything derived from it through
    utils.redact() before logging.
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


def _retry_wait(resp, attempt: int) -> float:
    retry_after = resp.headers.get("Retry-After", "") if resp is not None else ""
    try:
        wait = float(retry_after)
    except (TypeError, ValueError):
        wait = float(2 ** attempt)
    return max(0.0, min(wait, _MAX_WAIT))


def _get(url: str, api_key: str | None = None, **kwargs) -> dict:
    """GET a JSON document, retrying on rate limits and transient failures."""
    last_error = CivitaiAPIError("Request failed.")
    for attempt in range(_MAX_ATTEMPTS):
        resp = None
        try:
            resp = requests.get(url, headers=_build_headers(api_key),
                                timeout=_REQUEST_TIMEOUT, **kwargs)
        except requests.exceptions.Timeout:
            last_error = CivitaiAPIError("Request timed out.")
        except requests.exceptions.ConnectionError as exc:
            last_error = CivitaiAPIError(f"Connection failed: {utils.redact(exc)}")
        except requests.exceptions.RequestException as exc:
            raise CivitaiAPIError(f"Network error: {utils.redact(exc)}") from exc

        if resp is not None:
            status = resp.status_code
            if status == 429:
                last_error = CivitaiAPIError("Rate limit exceeded (429). Please wait.", status)
            elif status in _RETRY_STATUSES:
                last_error = CivitaiAPIError(f"CivitAI server error (HTTP {status}).", status)
            elif status == 401:
                raise CivitaiAPIError("Access denied (401). CivitAI API Key missing or invalid.", status)
            elif status == 404:
                raise CivitaiAPIError("Resource not found (404).", status)
            elif not resp.ok:
                raise CivitaiAPIError(f"HTTP {status}: {utils.redact(resp.text[:200])}", status)
            else:
                try:
                    return resp.json()
                except Exception as exc:
                    raise CivitaiAPIError("CivitAI returned an invalid (non-JSON) response.", status) from exc

        if attempt < _MAX_ATTEMPTS - 1:
            _sleep(_retry_wait(resp, attempt))

    raise last_error


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
        if exc.status == 404:
            return None
        raise


def search_models(query: str, model_type: str | None = None,
                  limit: int = 20, cursor: str = "",
                  api_key: str = "", nsfw: bool = False) -> dict:
    """Search models.

    CivitAI rejects `page` together with `query`, so pagination is
    cursor-based: pass `metadata.nextCursor` of the previous response to get
    the following page.
    """
    params: dict[str, Any] = {
        "limit": min(limit, 100),
        "nsfw":  str(nsfw).lower(),
    }
    if query.strip():
        params["query"] = query.strip()
    if model_type and model_type not in ("Tous", "All"):
        params["types"] = model_type
    if cursor:
        params["cursor"] = cursor
    return _get(f"{CIVITAI_API_BASE}/models", api_key, params=params)


# ── Preview images ───────────────────────────────────────────────────────────

def is_video(image: dict) -> bool:
    if str(image.get("type", "")).lower() == "video":
        return True
    path = urlparse(str(image.get("url", ""))).path.lower()
    return path.endswith(_VIDEO_EXTENSIONS)


def preview_image_urls(*image_lists: list | None) -> list[str]:
    """Still-image URLs from the first list that has any.

    CivitAI showcases often start with a video, which cannot be used as an
    Extra Networks thumbnail.
    """
    for images in image_lists:
        urls = [
            img["url"] for img in (images or [])
            if isinstance(img, dict) and img.get("url") and not is_video(img)
        ]
        if urls:
            return urls
    return []


def sized_image_url(url: str, width: int = PREVIEW_WIDTH) -> str:
    """Ask the CivitAI image CDN for a resized copy instead of the multi-MB original."""
    parsed = urlparse(url)
    if (parsed.hostname or "").lower() != "image.civitai.com":
        return url
    parts = parsed.path.split("/")
    for i, part in enumerate(parts):
        if _IMAGE_SIZE_SEGMENT_RE.match(part):
            parts[i] = f"width={width}"
            return urlunparse(parsed._replace(path="/".join(parts)))
    return url


# ── Versions ─────────────────────────────────────────────────────────────────

def extract_versions(model_info: dict) -> list[dict]:
    versions = []
    for v in model_info.get("modelVersions", []):
        files = [
            {
                "name":    utils.safe_filename(f.get("name"), "unknown"),
                "url":     f.get("downloadUrl", ""),
                "size_kb": f.get("sizeKB", 0),
                "type":    f.get("type", "Model"),
                "format":  (f.get("metadata") or {}).get("format", ""),
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


def latest_version_for(versions: list[dict], local_base_model: str = "") -> dict | None:
    """Newest version sharing the local file's base model.

    A new Flux release is not an update for an SDXL LoRA. Falls back to the
    newest version overall when the base model is unknown or has no match.
    """
    if not versions:
        return None
    base = (local_base_model or "").strip().lower()
    if base:
        same = [v for v in versions if str(v.get("base_model", "")).strip().lower() == base]
        if same:
            return same[0]
    return versions[0]
