import json
import threading
from pathlib import Path

from . import api, utils


# ── .civitai.info ────────────────────────────────────────────────────────────

def load_info(model_path: Path) -> dict | None:
    info_path = utils.info_file_path(model_path)
    if not info_path.exists():
        return None
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_info(model_path: Path, data: dict) -> None:
    utils.info_file_path(model_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def mark_not_on_civitai(model_path: Path) -> None:
    save_info(model_path, {"not_on_civitai": True})


def is_not_on_civitai(info: dict | None) -> bool:
    return bool(info and info.get("not_on_civitai"))


# ── Preview ──────────────────────────────────────────────────────────────────

def save_preview_image(model_path: Path, image_url: str) -> Path | None:
    import requests
    ext = image_url.rsplit(".", 1)[-1].split("?")[0].lower()
    if ext not in ("png", "jpg", "jpeg", "webp"):
        ext = "jpg"
    dest = utils.preview_file_path(model_path, ext)
    try:
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return dest
    except Exception as exc:
        print(f"[CivitAI Helper] Preview non téléchargée : {exc}")
        return None


def has_preview(model_path: Path) -> bool:
    return any(
        model_path.with_suffix(suf).exists()
        for suf in utils.PREVIEW_SUFFIXES
    )


# ── Scan ─────────────────────────────────────────────────────────────────────

class ScanState:
    def __init__(self) -> None:
        self.running = False
        self.cancel  = False
        self.total   = 0
        self.done    = 0
        self.current = ""
        self.log: list[str] = []

    def reset(self) -> None:
        self.__init__()

    def append_log(self, msg: str) -> None:
        self.log.append(msg)
        print(f"[CivitAI Helper] {msg}")

    @property
    def progress(self) -> float:
        return self.done / self.total if self.total else 0.0

    @property
    def summary(self) -> str:
        return f"{self.done}/{self.total} — {self.current}"


_scan_state = ScanState()


def get_scan_state() -> ScanState:
    return _scan_state


def scan_models(api_key: str = "", skip_existing: bool = True) -> None:
    state = _scan_state
    state.reset()
    state.running = True

    model_files  = utils.iter_model_files()
    state.total  = len(model_files)
    state.append_log(f"Scan démarré : {state.total} fichier(s) trouvé(s).")

    for model_path in model_files:
        if state.cancel:
            state.append_log("Scan annulé.")
            break

        state.current      = model_path.name
        existing_info      = load_info(model_path)

        if skip_existing and existing_info is not None:
            state.append_log(f"[SKIP] {model_path.name}")
            state.done += 1
            continue

        state.append_log(f"[HASH] {model_path.name}…")
        try:
            sha256 = utils.sha256_of_file(
                model_path,
                progress_callback=lambda d, t: setattr(
                    state, "current", f"{model_path.name} ({d * 100 // t}%)"
                ),
            )
        except OSError as exc:
            state.append_log(f"[ERR]  Lecture impossible : {exc}")
            state.done += 1
            continue

        state.append_log(f"[API]  Recherche ({sha256[:12]}…)…")
        try:
            version_info = api.fetch_version_by_hash(sha256, api_key)
        except api.CivitaiAPIError as exc:
            state.append_log(f"[ERR]  API : {exc}")
            state.done += 1
            continue

        if version_info is None:
            state.append_log(f"[N/F]  Introuvable : {model_path.name}")
            mark_not_on_civitai(model_path)
            state.done += 1
            continue

        try:
            model_id   = version_info.get("modelId")
            model_info = api.fetch_model_info(str(model_id), api_key) if model_id else {}
        except api.CivitaiAPIError:
            model_info = {}

        combined = {
            **version_info,
            "model": {
                "name":        model_info.get("name", ""),
                "type":        model_info.get("type", ""),
                "tags":        model_info.get("tags", []),
                "description": model_info.get("description", ""),
            },
            "sha256": sha256,
        }
        save_info(model_path, combined)
        state.append_log(f"[OK]   {model_path.name}")

        if not has_preview(model_path):
            images = version_info.get("images", [])
            if images:
                url = images[0].get("url")
                if url:
                    result = save_preview_image(model_path, url)
                    if result:
                        state.append_log(f"[IMG]  {result.name}")

        state.done += 1

    state.running = False
    state.append_log(f"Scan terminé : {state.done}/{state.total} modèles traités.")


# ── Vérification MAJ ─────────────────────────────────────────────────────────

class UpdateCheckState:
    def __init__(self) -> None:
        self.running = False
        self.results: list[dict] = []
        self.log: list[str]      = []

    def reset(self) -> None:
        self.__init__()

    def append_log(self, msg: str) -> None:
        self.log.append(msg)
        print(f"[CivitAI Helper] {msg}")


_update_state = UpdateCheckState()


def get_update_state() -> UpdateCheckState:
    return _update_state


def check_for_updates(api_key: str = "") -> None:
    state = _update_state
    state.reset()
    state.running = True

    model_files = utils.iter_model_files()
    state.append_log(f"Vérification MAJ : {len(model_files)} modèle(s)…")

    for model_path in model_files:
        info = load_info(model_path)
        if not info or is_not_on_civitai(info):
            continue

        local_version_id = info.get("id")
        model_id         = info.get("modelId")
        if not local_version_id or not model_id:
            continue

        try:
            remote = api.fetch_model_info(str(model_id), api_key)
        except api.CivitaiAPIError as exc:
            state.append_log(f"[ERR] {model_path.name} : {exc}")
            continue

        versions = api.extract_versions(remote)
        latest   = versions[0] if versions else None
        if latest and latest["id"] != local_version_id:
            state.results.append({
                "model_name":        info.get("model", {}).get("name", model_path.stem),
                "model_path":        str(model_path),
                "model_id":          model_id,
                "local_version_id":  local_version_id,
                "latest_version_id": latest["id"],
                "latest_label":      latest["label"],
                "files":             latest["files"],
                "model_type":        info.get("model", {}).get("type", "Other"),
            })
            state.append_log(f"[UPD] {model_path.name} → {latest['label']}")
        else:
            state.append_log(f"[OK]  {model_path.name} à jour.")

    state.running = False
    state.append_log(f"Terminé. {len(state.results)} MAJ disponible(s).")
