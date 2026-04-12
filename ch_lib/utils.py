import os
import hashlib
from pathlib import Path

CIVITAI_INFO_SUFFIX = ".civitai.info"
PREVIEW_SUFFIXES    = (".preview.png", ".preview.jpg", ".preview.jpeg", ".preview.webp")

MODEL_TYPE_DIRS: dict[str, str] = {
    "Checkpoint":        "models/Stable-diffusion",
    "LORA":              "models/Lora",
    "LoCon":             "models/Lora",
    "DoRA":              "models/Lora",
    "TextualInversion":  "embeddings",
    "VAE":               "models/VAE",
    "ControlNet":        "models/ControlNet",
    "Upscaler":          "models/ESRGAN",
    "Hypernetwork":      "models/hypernetworks",
    "Other":             "models/Other",
}

MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}


def get_sd_root() -> Path:
    try:
        from modules import shared
        data_dir = getattr(shared.cmd_opts, "data_dir", None)
        if data_dir:
            return Path(data_dir)
    except Exception:
        pass
    return Path(__file__).resolve().parents[3]


def resolve_model_dir(model_type: str, custom_dir: str | None = None) -> Path:
    root = get_sd_root()
    if custom_dir and custom_dir.strip():
        p = Path(custom_dir.strip())
        destination = p if p.is_absolute() else root / p
    else:
        relative = MODEL_TYPE_DIRS.get(model_type, "models/Other")
        destination = root / relative
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def sha256_of_file(path: Path, progress_callback=None) -> str:
    h     = hashlib.sha256()
    total = path.stat().st_size
    done  = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
            done += len(chunk)
            if progress_callback:
                progress_callback(done, total)
    return h.hexdigest()


def format_size(size_kb: float) -> str:
    if size_kb >= 1_000_000:
        return f"{size_kb / 1_000_000:.2f} GB"
    if size_kb >= 1_000:
        return f"{size_kb / 1_000:.2f} MB"
    return f"{size_kb:.0f} KB"


def iter_model_files(root: Path | None = None) -> list[Path]:
    if root is None:
        root = get_sd_root()
    results: list[Path] = []
    for model_dir in MODEL_TYPE_DIRS.values():
        scan_dir = root / model_dir
        if not scan_dir.exists():
            continue
        for f in scan_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in MODEL_EXTENSIONS:
                if not any(part.startswith(".") for part in f.parts):
                    results.append(f)
    return results


def info_file_path(model_path: Path) -> Path:
    return model_path.with_suffix("").with_suffix(CIVITAI_INFO_SUFFIX)


def preview_file_path(model_path: Path, ext: str = "png") -> Path:
    return model_path.with_suffix(f".preview.{ext}")
