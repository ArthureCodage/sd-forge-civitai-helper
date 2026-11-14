import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

from . import api, card_index, settings, utils
from .model_manager import build_info, info_trigger_words, save_info, save_preview_images, save_trigger_words

MAX_ATTEMPTS = 3
_sleep = time.sleep

# Outcomes of one transfer attempt
_OK, _RETRY, _STOP = "ok", "retry", "stop"


@dataclass
class DownloadTask:
    url:              str
    dest:             Path
    filename:         str
    sha256_expected:  str = ""
    total_bytes:      int = 0
    downloaded_bytes: int = 0
    done:             bool = False
    cancelled:        bool = False
    error:            str  = ""

    @property
    def progress(self) -> float:
        return self.downloaded_bytes / self.total_bytes if self.total_bytes else 0.0


def _int_header(resp, name: str) -> int:
    try:
        return int(resp.headers.get(name, 0) or 0)
    except (TypeError, ValueError):
        return 0


class DownloadQueue:
    def __init__(self) -> None:
        self.current:  DownloadTask | None = None
        self.running:  bool                = False
        self._cancel:  bool                = False
        self.log:      list[str]           = []
        self._lock                         = threading.Lock()

    def cancel(self) -> None:
        self._cancel = True

    def append_log(self, msg: str) -> None:
        msg = utils.redact(msg)
        self.log.append(msg)
        if len(self.log) > 200:
            del self.log[:-200]
        utils.safe_print(f"[CivitAI Helper] {msg}")

    def _fail(self, task: DownloadTask, message: object, prefix: str = "") -> None:
        task.error = utils.redact(message)
        self.append_log(f"[ERR] {prefix}{task.error}")

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

        try:
            dest = task.dest
            dest.parent.mkdir(parents=True, exist_ok=True)
            part_dest = dest.with_name(f"{dest.name}.part")

            # Check if destination file already exists and is complete
            if dest.exists() and task.sha256_expected:
                self.append_log(f"File already exists: {task.filename}. Checking SHA256...")
                try:
                    existing_hash = utils.sha256_of_file(dest)
                    if existing_hash.lower() == task.sha256_expected.lower():
                        task.done = True
                        task.total_bytes = dest.stat().st_size
                        task.downloaded_bytes = dest.stat().st_size
                        self.append_log(f"✅ File already present and SHA256 matches: {task.filename}")
                        self._save_metadata(dest, version_data, model_info, download_preview, api_key)
                        return
                except OSError:
                    pass

            outcome = _STOP
            for attempt in range(1, MAX_ATTEMPTS + 1):
                outcome = self._transfer(task, part_dest, api_key)
                if outcome != _RETRY or attempt == MAX_ATTEMPTS or self._cancel:
                    break
                wait = 2 ** attempt
                self.append_log(f"Retrying in {wait}s (attempt {attempt + 1}/{MAX_ATTEMPTS})…")
                _sleep(wait)
                task.error = ""

            if outcome != _OK or self._cancel or task.cancelled:
                return

            # Verification SHA256 on the downloaded .part file
            if task.sha256_expected:
                self.append_log("Verifying SHA256…")
                actual = utils.sha256_of_file(part_dest)
                if actual.lower() != task.sha256_expected.lower():
                    self._fail(task, "SHA256 mismatch — downloaded file is corrupted.")
                    part_dest.unlink(missing_ok=True)
                    return
                self.append_log("SHA256 OK ✓")

            # Atomically rename .part file to final destination
            part_dest.replace(dest)
            task.done = True
            card_index.invalidate()
            self.append_log(f"✅ Downloaded: {task.filename}")

            # Save metadata and preview images
            self._save_metadata(dest, version_data, model_info, download_preview, api_key)

        except OSError as exc:
            self._fail(task, exc, "File error: ")
        finally:
            self.running = False

    def _transfer(self, task: DownloadTask, part_dest: Path, api_key: str) -> str:
        """One attempt at fetching the file into `part_dest` (resuming if possible)."""
        resume_pos = part_dest.stat().st_size if part_dest.exists() else 0
        headers = api._build_headers(api_key)
        download_url = api.with_api_token(task.url, api_key)

        use_resume = resume_pos > 0
        if use_resume:
            headers["Range"] = f"bytes={resume_pos}-"
            self.append_log(f"Resuming download from {utils.format_size(resume_pos / 1024)}…")

        try:
            resp = requests.get(download_url, headers=headers, stream=True, timeout=30)

            # Handle 416 Range Not Satisfiable (file changed on server or range out of bounds)
            if resp.status_code == 416 and use_resume:
                resp.close()
                self.append_log("Range invalid or file updated on server, restarting from beginning…")
                part_dest.unlink(missing_ok=True)
                resume_pos = 0
                headers.pop("Range", None)
                task.downloaded_bytes = 0
                resp = requests.get(download_url, headers=headers, stream=True, timeout=30)
        except requests.exceptions.RequestException as exc:
            self._fail(task, exc, "Network error: ")
            return _RETRY

        try:
            status = resp.status_code
            if status in (401, 403):
                self._fail(task, f"Access denied (HTTP {status}). CivitAI API key missing or invalid, "
                                 "or this model requires a logged-in account.")
                return _STOP
            if status == 429 or status >= 500:
                self._fail(task, f"CivitAI is busy (HTTP {status}).")
                return _RETRY
            if status >= 400:
                self._fail(task, f"Download refused (HTTP {status}).")
                return _STOP

            # A login or error page must never be saved as a model file.
            content_type = str(resp.headers.get("Content-Type", "")).lower()
            if content_type.startswith(("text/html", "application/json")):
                self._fail(task, "CivitAI returned a web page instead of the file. "
                                 "Check your API key: the model probably requires login.")
                return _STOP

            # Proper HTTP 206 vs 200 handling to avoid corrupting file
            if status == 206:
                content_range = resp.headers.get("Content-Range", "")
                try:
                    total = int(content_range.rsplit("/", 1)[-1])
                except ValueError:
                    total = _int_header(resp, "Content-Length") + resume_pos
                mode = "ab"
            else:
                resume_pos = 0
                total = _int_header(resp, "Content-Length")
                mode = "wb"

            task.total_bytes      = total
            task.downloaded_bytes = resume_pos

            self.append_log(f"Downloading: {task.filename} ({utils.format_size(total / 1024)})")

            try:
                with open(part_dest, mode) as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunk for high throughput
                        if self._cancel:
                            task.cancelled = True
                            self.append_log("Download cancelled.")
                            return _STOP
                        if chunk:
                            fh.write(chunk)
                            task.downloaded_bytes += len(chunk)
            # requests' network errors derive from OSError: test them first.
            except requests.exceptions.RequestException as exc:
                self._fail(task, exc, "Download interrupted: ")
                return _RETRY
            except OSError as exc:
                self._fail(task, exc, "Could not write file: ")
                return _STOP
            except Exception as exc:
                self._fail(task, exc, "Download interrupted: ")
                return _RETRY

            if total and task.downloaded_bytes < total:
                self._fail(task, f"Connection closed early ({task.downloaded_bytes}/{total} bytes).")
                return _RETRY
            return _OK
        finally:
            resp.close()

    def _save_metadata(
        self,
        dest: Path,
        version_data: dict | None,
        model_info: dict | None,
        download_preview: bool,
        api_key: str,
    ) -> None:
        # The model file is already safely on disk: a metadata problem is
        # reported but must not turn the download into a failure.
        try:
            if version_data:
                info = build_info(version_data, model_info)
                save_info(dest, info)
                if settings.get_auto_txt():
                    words = info_trigger_words(info)
                    if words:
                        save_trigger_words(dest, words)

            if download_preview:
                urls = api.preview_image_urls(
                    (version_data or {}).get("images"), (model_info or {}).get("images")
                )
                if urls:
                    max_cnt = settings.get_max_previews()
                    result = save_preview_images(dest, urls, api_key=api_key, max_count=max_cnt)
                    if result:
                        self.append_log(f"Preview saved: {result.name}")
        except Exception as exc:
            self.append_log(f"[WARN] Could not save metadata/preview: {exc}")

    def start_async(
        self,
        task: DownloadTask,
        api_key: str = "",
        version_data: dict | None = None,
        model_info: dict | None   = None,
        download_preview: bool    = True,
    ) -> bool:
        """Start the download in the background. False if one is already running."""
        with self._lock:
            if self.running:
                return False
            self.running = True
        thread = threading.Thread(
            target=self.download,
            args=(task, api_key, version_data, model_info, download_preview),
            daemon=True,
        )
        thread.start()
        return True


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
    status:       str   = "pending"      # pending / downloading / completed / error / cancelled
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
        self.items:         list[BatchItem]           = []
        self.running:       bool                      = False
        self._cancel:       bool                      = False
        self.log:           list[str]                 = []
        self._current_task: Optional[DownloadTask]    = None
        self._active_queue: Optional[DownloadQueue]   = None
        self._lock                                    = threading.Lock()

    def append_log(self, msg: str) -> None:
        msg = utils.redact(msg)
        self.log.append(msg)
        if len(self.log) > 500:
            del self.log[:-500]
        utils.safe_print(f"[CivitAI Batch] {msg}")

    def add_item(self, item: BatchItem) -> None:
        self.items.append(item)

    def clear(self) -> None:
        if not self.running:
            self.items = []
            self.log   = []

    def cancel(self) -> None:
        self._cancel = True
        if self._active_queue:
            self._active_queue.cancel()

    @property
    def summary(self) -> str:
        done   = sum(1 for i in self.items if i.status == "completed")
        errors = sum(1 for i in self.items if i.status == "error")
        total  = len(self.items)
        return f"{done}/{total} completed, {errors} error(s)"

    def start(self, api_key: str = "") -> bool:
        """Start processing pending items. False if already running or nothing to do."""
        with self._lock:
            if self.running:
                return False
            if not any(i.status == "pending" for i in self.items):
                return False
            self.running = True
            self._cancel = False
        threading.Thread(target=self._run, args=(api_key,), daemon=True).start()
        return True

    def _run(self, api_key: str) -> None:
        self.running = True
        try:
            for item in list(self.items):
                if self._cancel:
                    if item.status == "pending":
                        item.status = "cancelled"
                    continue
                if item.status != "pending":
                    continue

                item.status = "downloading"
                self.append_log(f"Starting: {item.filename}")

                if not item._dl_url or item._dest_dir is None:
                    item.status = "error"
                    item.error  = "Configuration missing."
                    self.append_log(f"[ERR] {item.filename} : {item.error}")
                    continue

                filename  = utils.safe_filename(item.filename)
                dest_path = item._dest_dir / filename
                task = DownloadTask(
                    url             = item._dl_url,
                    dest            = dest_path,
                    filename        = filename,
                    sha256_expected = item._sha256,
                )
                self._current_task = task

                # Synchronous download inside this background worker thread
                _dq = DownloadQueue()
                self._active_queue = _dq
                _dq.download(
                    task             = task,
                    api_key          = api_key,
                    version_data     = item._version_data,
                    model_info       = item._model_info,
                    download_preview = True,
                )

                self._active_queue = None
                self._current_task = None

                if task.error:
                    item.status = "error"
                    item.error  = task.error
                    self.append_log(f"[ERR] {item.filename} : {task.error}")
                elif task.cancelled or self._cancel:
                    item.status = "cancelled"
                    self.append_log(f"[CANCELLED] {item.filename}")
                else:
                    item.status   = "completed"
                    item.progress = 1.0
                    self.append_log(f"✅ {item.filename}")

        finally:
            self.running = False
            self._active_queue = None
            self._current_task = None
            self.append_log(f"Batch finished: {self.summary}")


_batch_queue = BatchQueue()


def get_batch_queue() -> BatchQueue:
    return _batch_queue
