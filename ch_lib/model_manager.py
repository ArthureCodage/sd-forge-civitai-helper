import json
import threading
from pathlib import Path

from . import api, settings, utils

_MAX_LOG_LINES = 1000


# ── .civitai.info ────────────────────────────────────────────────────────────

def load_info(model_path: Path) -> dict | None:
    info_path = utils.info_file_path(model_path)
    if not info_path.exists():
        return None
    try:
        data = json.loads(info_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def save_info(model_path: Path, data: dict) -> None:
    utils.info_file_path(model_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_info(version: dict, model_info: dict | None = None, sha256: str = "") -> dict:
    """Merge version and model data into the `.civitai.info` layout.

    `version` is either a raw API version or an entry of
    api.extract_versions(); both end up with the API key names (`modelId`,
    `baseModel`, `trainedWords`) so that the update check and the card
    buttons work whichever way the file was obtained.
    """
    model_info = model_info or {}
    embedded = version.get("model") if isinstance(version.get("model"), dict) else {}
    info = dict(version)

    if not info.get("modelId"):
        model_id = model_info.get("id") or embedded.get("id")
        if model_id:
            info["modelId"] = model_id
    if not info.get("baseModel") and info.get("base_model"):
        info["baseModel"] = info["base_model"]
    if not info.get("trainedWords") and info.get("trained_words"):
        info["trainedWords"] = info["trained_words"]

    info["model"] = {
        "id":          info.get("modelId"),
        "name":        model_info.get("name") or embedded.get("name", ""),
        "type":        model_info.get("type") or embedded.get("type", ""),
        "tags":        model_info.get("tags") or embedded.get("tags", []),
        "description": model_info.get("description") or embedded.get("description", ""),
    }
    if sha256:
        info["sha256"] = sha256
    return info


def info_model_id(info: dict | None):
    if not info:
        return None
    return info.get("modelId") or (info.get("model") or {}).get("id")


def info_trigger_words(info: dict | None) -> list[str] | str:
    if not info:
        return []
    return info.get("trainedWords") or info.get("trained_words") or []


def mark_not_on_civitai(model_path: Path) -> None:
    save_info(model_path, {"not_on_civitai": True})


def is_not_on_civitai(info: dict | None) -> bool:
    return bool(info and info.get("not_on_civitai"))


def save_trigger_words(model_path: Path, trigger_words: list[str] | str, overwrite: bool = True) -> Path | None:
    if not settings.get_auto_txt():
        return None
    if isinstance(trigger_words, list):
        text = ", ".join(w.strip() for w in trigger_words if isinstance(w, str) and w.strip())
    else:
        text = str(trigger_words).strip()
    if not text:
        return None

    txt_path = model_path.with_name(f"{model_path.stem}.txt")
    if txt_path.exists() and not overwrite:
        return txt_path

    try:
        txt_path.write_text(text, encoding="utf-8")
        return txt_path
    except OSError as exc:
        utils.safe_print(f"[CivitAI Helper] Failed to save trigger words txt for {model_path.name}: {exc}")
        return None


# ── Preview ──────────────────────────────────────────────────────────────────

def _fetch_image(url: str, headers: dict) -> tuple[bytes, str] | None:
    """Download one still image; returns (bytes, extension) or None."""
    import requests

    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    if not resp.content:
        return None

    content_type = resp.headers.get("Content-Type", "").lower()
    if content_type.startswith("video/"):
        return None
    if "png" in content_type:
        ext = "png"
    elif "webp" in content_type:
        ext = "webp"
    elif "jpeg" in content_type or "jpg" in content_type:
        ext = "jpg"
    else:
        url_ext = url.rsplit(".", 1)[-1].split("?")[0].lower()
        ext = url_ext if url_ext in ("png", "jpg", "jpeg", "webp") else "jpg"
    return resp.content, ext


def save_preview_images(
    model_path: Path,
    image_urls: list[str] | str,
    api_key: str = "",
    max_count: int = 1,
) -> Path | None:
    if isinstance(image_urls, str):
        urls = [image_urls]
    elif isinstance(image_urls, list):
        urls = [u for u in image_urls if isinstance(u, str) and u]
    else:
        urls = []
    if not urls:
        return None

    wanted = max(1, max_count)
    headers = api._build_headers(api_key)
    first_saved = None
    saved = 0

    for url in urls:
        if saved >= wanted:
            break
        if url.startswith("//"):
            url = "https:" + url

        # Resized copy first (≈100 KB instead of several MB), original as fallback.
        attempts = list(dict.fromkeys([api.sized_image_url(url), url]))
        fetched = None
        for candidate in attempts:
            try:
                fetched = _fetch_image(candidate, headers)
                break
            except Exception as exc:
                if candidate == attempts[-1]:
                    utils.safe_print(
                        f"[CivitAI Helper] Preview download failed for {model_path.name}: {utils.redact(exc)}"
                    )
        if not fetched:
            continue
        content, ext = fetched

        try:
            if saved == 0:
                dest = utils.preview_file_path(model_path, ext)
                primary = model_path.with_name(f"{model_path.stem}.{ext}")
                if primary != dest and not primary.exists():
                    try:
                        primary.write_bytes(content)
                    except OSError:
                        pass
            else:
                dest = model_path.with_name(f"{model_path.stem}.preview.{saved}.{ext}")
            dest.write_bytes(content)
        except OSError as exc:
            utils.safe_print(f"[CivitAI Helper] Could not write preview for {model_path.name}: {exc}")
            continue

        if saved == 0:
            first_saved = dest
        saved += 1

    return first_saved


def save_preview_image(model_path: Path, image_url: str, api_key: str = "") -> Path | None:
    return save_preview_images(model_path, image_url, api_key=api_key, max_count=1)


def save_previews_from_info(model_path: Path, info: dict, api_key: str = "") -> tuple[Path | None, bool]:
    """Save previews listed in `info`. Returns (first saved file, any still image listed)."""
    urls = api.preview_image_urls(info.get("images"))
    if not urls:
        return None, False
    result = save_preview_images(
        model_path, urls, api_key=api_key, max_count=settings.get_max_previews()
    )
    return result, True


def has_preview(model_path: Path) -> bool:
    return any(
        model_path.with_name(f"{model_path.stem}{suf}").exists()
        for suf in utils.PREVIEW_SUFFIXES
    )


# ── Background job state ─────────────────────────────────────────────────────

class _JobState:
    """Shared state of one background job; try_begin() makes starts atomic."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clear()

    def _clear(self) -> None:
        self.running = False
        self.cancel  = False
        self.total   = 0
        self.done    = 0
        self.current = ""
        self.log: list[str] = []

    def reset(self) -> None:
        with self._lock:
            self._clear()

    def try_begin(self) -> bool:
        """Claim the job. False when it is already running."""
        with self._lock:
            if self.running:
                return False
            self._clear()
            self.running = True
            return True

    def append_log(self, msg: str) -> None:
        msg = utils.redact(msg)
        self.log.append(msg)
        if len(self.log) > _MAX_LOG_LINES:
            del self.log[:-_MAX_LOG_LINES]
        utils.safe_print(f"[CivitAI Helper] {msg}")

    @property
    def progress(self) -> float:
        return self.done / self.total if self.total else 0.0

    @property
    def summary(self) -> str:
        return f"{self.done}/{self.total} — {self.current}"


class ScanState(_JobState):
    pass


_scan_state = ScanState()


def get_scan_state() -> ScanState:
    return _scan_state


# ── Scan ─────────────────────────────────────────────────────────────────────

def _identify(model_path: Path, sha256: str, api_key: str) -> dict | None:
    """Look a file up by hash and write its `.civitai.info`. None if unknown to CivitAI."""
    version_info = api.fetch_version_by_hash(sha256, api_key)
    if version_info is None:
        mark_not_on_civitai(model_path)
        return None

    model_id = version_info.get("modelId")
    try:
        model_info = api.fetch_model_info(str(model_id), api_key) if model_id else {}
    except api.CivitaiAPIError:
        model_info = {}

    info = build_info(version_info, model_info, sha256)
    save_info(model_path, info)
    return info


def scan_models(api_key: str = "", skip_existing: bool = True) -> None:
    state = _scan_state
    if not state.try_begin():
        return

    try:
        model_files = utils.iter_model_files()
        state.total = len(model_files)
        state.append_log(f"Scan started: found {state.total} model file(s).")

        for model_path in model_files:
            if state.cancel:
                state.append_log("Scan cancelled.")
                break

            state.current = model_path.name

            if skip_existing and load_info(model_path) is not None:
                state.append_log(f"[SKIP] {model_path.name}")
                state.done += 1
                continue

            state.append_log(f"[HASH] {model_path.name}…")
            try:
                sha256 = utils.sha256_of_file(
                    model_path,
                    progress_callback=lambda d, t, name=model_path.name: setattr(
                        state, "current", f"{name} ({d * 100 // t if t else 100}%)"
                    ),
                )
            except OSError as exc:
                state.append_log(f"[ERR]  Read error: {exc}")
                state.done += 1
                continue

            state.append_log(f"[API]  Querying ({sha256[:12]}…)…")
            try:
                info = _identify(model_path, sha256, api_key)
            except api.CivitaiAPIError as exc:
                state.append_log(f"[ERR]  API error: {exc}")
                state.done += 1
                continue
            except OSError as exc:
                state.append_log(f"[ERR]  Could not write info file: {exc}")
                state.done += 1
                continue

            if info is None:
                state.append_log(f"[NOT FOUND] {model_path.name}")
                state.done += 1
                continue

            state.append_log(f"[OK]   {model_path.name}")

            words = info_trigger_words(info)
            if words:
                save_trigger_words(model_path, words, overwrite=False)

            if not has_preview(model_path):
                result, _ = save_previews_from_info(model_path, info, api_key)
                if result:
                    state.append_log(f"[IMG]  {result.name}")

            state.done += 1

        state.append_log(f"Scan finished: {state.done}/{state.total} models processed.")
    finally:
        state.running = False


def download_missing_previews(api_key: str = "") -> None:
    state = _scan_state
    if not state.try_begin():
        return

    try:
        model_files = utils.iter_model_files()
        missing_models = [m for m in model_files if not has_preview(m)]
        state.total = len(missing_models)
        state.append_log(f"Searching missing thumbnails: {state.total} model(s) without preview.")

        if not missing_models:
            state.append_log("✅ All models already have preview images!")
            return

        for model_path in missing_models:
            if state.cancel:
                state.append_log("Operation cancelled.")
                break

            state.current = model_path.name
            info = load_info(model_path)

            if info is None:
                state.append_log(f"[HASH] {model_path.name}…")
                try:
                    sha256 = utils.sha256_of_file(model_path)
                    info = _identify(model_path, sha256, api_key)
                except Exception as exc:
                    state.append_log(f"[ERR] {model_path.name}: {exc}")
                    state.done += 1
                    continue

            if info is None or is_not_on_civitai(info):
                state.append_log(f"[NOT FOUND] Not found on CivitAI: {model_path.name}")
                state.done += 1
                continue

            words = info_trigger_words(info)
            if words:
                save_trigger_words(model_path, words, overwrite=False)

            result, had_images = save_previews_from_info(model_path, info, api_key)
            if result:
                state.append_log(f"[IMG] ✅ {model_path.name} -> {result.name}")
            elif had_images:
                state.append_log(f"[ERR] Failed downloading preview for {model_path.name}")
            else:
                state.append_log(f"[WARN] No still preview image available for {model_path.name}")

            state.done += 1

        state.append_log(f"Finished: {state.done}/{state.total} previews processed.")
    finally:
        state.running = False


# ── Updates check ─────────────────────────────────────────────────────────────

class UpdateCheckState(_JobState):
    def _clear(self) -> None:
        super()._clear()
        self.results: list[dict] = []


_update_state = UpdateCheckState()


def get_update_state() -> UpdateCheckState:
    return _update_state


def check_for_updates(api_key: str = "") -> None:
    state = _update_state
    if not state.try_begin():
        return

    try:
        model_files = utils.iter_model_files()
        state.total = len(model_files)
        state.append_log(f"Checking updates for {len(model_files)} model(s)…")

        # Several local files often belong to the same CivitAI model:
        # query each model once per run. None records a failed lookup.
        remote_versions: dict[str, list[dict] | None] = {}

        for model_path in model_files:
            if state.cancel:
                state.append_log("Update check cancelled.")
                break
            state.current = model_path.name
            state.done += 1

            info = load_info(model_path)
            if not info or is_not_on_civitai(info):
                continue

            local_version_id = info.get("id")
            model_id         = info_model_id(info)
            if not local_version_id or not model_id:
                continue

            key = str(model_id)
            if key not in remote_versions:
                try:
                    remote_versions[key] = api.extract_versions(api.fetch_model_info(key, api_key))
                except api.CivitaiAPIError as exc:
                    remote_versions[key] = None
                    state.append_log(f"[ERR] {model_path.name}: {exc}")
            versions = remote_versions[key]
            if versions is None:
                continue

            local_base = info.get("baseModel") or info.get("base_model") or ""
            latest = api.latest_version_for(versions, local_base)
            if latest and str(latest["id"]) != str(local_version_id):
                model = info.get("model") or {}
                state.results.append({
                    "model_name":        model.get("name") or model_path.stem,
                    "model_path":        str(model_path),
                    "model_id":          model_id,
                    "local_version_id":  local_version_id,
                    "latest_version_id": latest["id"],
                    "latest_label":      latest["label"],
                    "files":             latest["files"],
                    "model_type":        model.get("type") or "Other",
                })
                state.append_log(f"[UPD] {model_path.name} → {latest['label']}")
            else:
                state.append_log(f"[OK]  {model_path.name} is up to date.")

        state.append_log(f"Update check finished. {len(state.results)} update(s) available.")
    finally:
        state.running = False
