import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

from . import api, utils
from .model_manager import save_info, save_preview_image


@dataclass
class DownloadTask:
    url:        str
    dest:       Path
    filename:   str
    sha256_expected: str = ""
    total_bytes:     int = 0
    downloaded_bytes: int = 0
    done:      bool = False
    cancelled: bool = False
    error:     str  = ""

    @property
    def progress(self) -> float:
        return self.downloaded_bytes / self.total_bytes if self.total_bytes else 0.0


class DownloadQueue:
    def __init__(self) -> None:
        self.current:  DownloadTask | None = None
        self.running:  bool       = False
        self._cancel:  bool       = False
        self.log: list[str]       = []
        self._lock = threading.Lock()

    def cancel(self) -> None:
        self._cancel = True

    def append_log(self, msg: str) -> None:
        self.log.append(msg)
        if len(self.log) > 200:
            self.log = self.log[-200:]
        print(f"[CivitAI Helper] {msg}")

    def download(
        self,
        task: DownloadTask,
        api_key: str = "",
        version_data: dict | None = None,
        model_info: dict | None   = None,
        download_preview: bool    = True,
    ) -> None:
        self.running  = True
        self._cancel  = False
        self.current  = task
        self.log      = []

        dest = task.dest
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Resume support
        resume_pos = dest.stat().st_size if dest.exists() else 0
        headers    = utils._build_headers(api_key) if hasattr(utils, "_build_headers") else {}
        headers["User-Agent"] = "sd-forge-civitai-helper/2.0"
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if resume_pos:
            headers["Range"] = f"bytes={resume_pos}-"
            self.append_log(f"Resume depuis {utils.format_size(resume_pos / 1024)}…")

        try:
            resp = requests.get(task.url, headers=headers, stream=True, timeout=30)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            task.error = str(exc)
            self.append_log(f"[ERR] {exc}")
            self.running = False
            return

        total = int(resp.headers.get("Content-Length", 0)) + resume_pos
        task.total_bytes     = total
        task.downloaded_bytes = resume_pos

        mode = "ab" if resume_pos else "wb"
        self.append_log(f"Téléchargement : {task.filename} ({utils.format_size(total / 1024)})")

        try:
            with open(dest, mode) as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if self._cancel:
                        task.cancelled = True
                        self.append_log("Téléchargement annulé.")
                        self.running = False
                        return
                    fh.write(chunk)
                    task.downloaded_bytes += len(chunk)
        except OSError as exc:
            task.error = str(exc)
            self.append_log(f"[ERR] Écriture : {exc}")
            self.running = False
            return

        # Vérification SHA256
        if task.sha256_expected:
            self.append_log("Vérification SHA256…")
            actual = utils.sha256_of_file(dest)
            if actual.lower() != task.sha256_expected.lower():
                task.error = "SHA256 invalide — fichier corrompu."
                self.append_log(f"[ERR] {task.error}")
                dest.unlink(missing_ok=True)
                self.running = False
                return
            self.append_log("SHA256 OK ✓")

        task.done = True
        self.append_log(f"✅ Téléchargé : {task.filename}")

        # Sauvegarde des métadonnées
        if version_data:
            combined = {
                **version_data,
                "model": {
                    "name":        (model_info or {}).get("name", ""),
                    "type":        (model_info or {}).get("type", ""),
                    "tags":        (model_info or {}).get("tags", []),
                    "description": (model_info or {}).get("description", ""),
                },
            }
            save_info(dest, combined)

        # Preview
        if download_preview and version_data:
            images = version_data.get("images", [])
            if images:
                img_url = images[0].get("url")
                if img_url:
                    result = save_preview_image(dest, img_url)
                    if result:
                        self.append_log(f"Preview : {result.name}")

        self.running = False

    def start_async(
        self,
        task: DownloadTask,
        api_key: str = "",
        version_data: dict | None = None,
        model_info: dict | None   = None,
        download_preview: bool    = True,
    ) -> None:
        if self.running:
            return
        thread = threading.Thread(
            target=self.download,
            args=(task, api_key, version_data, model_info, download_preview),
            daemon=True,
        )
        thread.start()


_queue = DownloadQueue()


def get_queue() -> DownloadQueue:
    return _queue


# ── Batch download ────────────────────────────────────────────────────────────

@dataclass
class BatchItem:
    url:          str
    model_name:   str   = ""
    version_name: str   = ""
    filename:     str   = ""
    size_kb:      float = 0.0
    status:       str   = "en attente"   # en attente / téléchargement / terminé / erreur / annulé
    progress:     float = 0.0
    error:        str   = ""
    # ── internal ──
    _dl_url:       str              = field(default="",   repr=False)
    _sha256:       str              = field(default="",   repr=False)
    _dest_dir:     Optional[Path]   = field(default=None, repr=False)
    _version_data: Optional[dict]   = field(default=None, repr=False)
    _model_info:   Optional[dict]   = field(default=None, repr=False)


class BatchQueue:
    def __init__(self) -> None:
        self.items:         list[BatchItem]       = []
        self.running:       bool                  = False
        self._cancel:       bool                  = False
        self.log:           list[str]             = []
        self._current_task: Optional[DownloadTask] = None

    def append_log(self, msg: str) -> None:
        self.log.append(msg)
        if len(self.log) > 500:
            self.log = self.log[-500:]
        print(f"[CivitAI Batch] {msg}")

    def add_item(self, item: BatchItem) -> None:
        self.items.append(item)

    def clear(self) -> None:
        if not self.running:
            self.items = []
            self.log   = []

    def cancel(self) -> None:
        self._cancel = True

    @property
    def summary(self) -> str:
        done   = sum(1 for i in self.items if i.status == "terminé")
        errors = sum(1 for i in self.items if i.status == "erreur")
        total  = len(self.items)
        return f"{done}/{total} terminé(s), {errors} erreur(s)"

    def start(self, api_key: str = "") -> None:
        if self.running:
            return
        if not any(i.status == "en attente" for i in self.items):
            return
        self._cancel = False
        threading.Thread(target=self._run, args=(api_key,), daemon=True).start()

    def _run(self, api_key: str) -> None:
        self.running = True
        for item in self.items:
            if self._cancel:
                if item.status == "en attente":
                    item.status = "annulé"
                continue
            if item.status != "en attente":
                continue

            item.status = "téléchargement"
            self.append_log(f"Début : {item.filename}")

            if not item._dl_url or item._dest_dir is None:
                item.status = "erreur"
                item.error  = "Configuration manquante."
                self.append_log(f"[ERR] {item.filename} : {item.error}")
                continue

            dest_path = item._dest_dir / item.filename
            task = DownloadTask(
                url             = item._dl_url,
                dest            = dest_path,
                filename        = item.filename,
                sha256_expected = item._sha256,
            )
            self._current_task = task

            # Synchronous download — runs inside this background thread
            _dq = DownloadQueue()
            _dq.download(
                task             = task,
                api_key          = api_key,
                version_data     = item._version_data,
                model_info       = item._model_info,
                download_preview = True,
            )

            self._current_task = None

            if task.error:
                item.status = "erreur"
                item.error  = task.error
                self.append_log(f"[ERR] {item.filename} : {task.error}")
            elif task.cancelled:
                item.status = "annulé"
                self.append_log(f"[ANN] {item.filename}")
            else:
                item.status   = "terminé"
                item.progress = 1.0
                self.append_log(f"✅ {item.filename}")

        self.running = False
        self.append_log(f"Lot terminé : {self.summary}")


_batch_queue = BatchQueue()


def get_batch_queue() -> BatchQueue:
    return _batch_queue
