"""In-memory index of local model files, used to answer card lookups.

Extra Networks shows hundreds of cards and each one asks which model file it
belongs to. Walking every model folder per card is far too slow, so the walk
is done once and reused for a short while.
"""
import threading
import time
from pathlib import Path

from . import utils

_TTL_SECONDS = 30.0

_lock = threading.Lock()
_root: Path | None = None
_built_at: float = 0.0
_by_name: dict[str, list[Path]] = {}
_by_stem: dict[str, list[Path]] = {}


def invalidate() -> None:
    """Forget the index (call after a model file was added)."""
    global _built_at
    with _lock:
        _built_at = 0.0


def _ensure_fresh() -> None:
    global _root, _built_at, _by_name, _by_stem
    root = utils.get_sd_root()
    now = time.monotonic()
    if _built_at and root == _root and now - _built_at < _TTL_SECONDS:
        return
    by_name: dict[str, list[Path]] = {}
    by_stem: dict[str, list[Path]] = {}
    for path in utils.iter_model_files(root):
        by_name.setdefault(path.name.lower(), []).append(path)
        by_stem.setdefault(path.stem.lower(), []).append(path)
    _root, _by_name, _by_stem, _built_at = root, by_name, by_stem, now


def find_model(name: str) -> Path | None:
    """Resolve a card name ("my_lora", "sub/dir/model.safetensors") to a file.

    Only exact file-name or stem matches count: "abc" must never resolve to
    "abc_v2", and "flux.1-dev" must not be cut down to "flux".
    """
    norm = (name or "").strip().replace("\\", "/").lower().strip("/")
    if not norm:
        return None
    last = norm.rsplit("/", 1)[-1]

    with _lock:
        _ensure_fresh()
        candidates = list(_by_name.get(last) or _by_stem.get(last) or [])

    if not candidates:
        return None
    if "/" in norm and len(candidates) > 1:
        for path in candidates:
            posix = path.as_posix().lower()
            if posix.endswith("/" + norm) or posix.rsplit(".", 1)[0].endswith("/" + norm):
                return path
    return candidates[0]
