import os
import sys
import hashlib
from pathlib import Path


def safe_print(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        try:
            enc = sys.stdout.encoding or "utf-8"
            print(msg.encode(enc, errors="replace").decode(enc))
        except Exception:
            print(msg.encode("ascii", errors="replace").decode("ascii"))

CIVITAI_INFO_SUFFIX = ".civitai.info"
PREVIEW_SUFFIXES    = (
    ".preview.png", ".preview.jpg", ".preview.jpeg", ".preview.webp",
    ".png", ".jpg", ".jpeg", ".webp"
)

MODEL_TYPE_DIRS: dict[str, str] = {
    "Checkpoint":        "models/Stable-diffusion",
    "LORA":              "models/Lora",
    "LoCon":             "models/Lora",
    "DoRA":              "models/Lora",
    "TextualInversion":  "embeddings",
    "VAE":               "models/VAE",
    "ControlNet":        "models/ControlNet",
    "Controlnet":        "models/ControlNet",
    "Upscaler":          "models/ESRGAN",
    "Hypernetwork":      "models/hypernetworks",
    "MotionModule":      "models/AnimateDiff",
    "Poses":             "models/Poses",
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
        norm_type = (model_type or "Other").strip().lower()
        custom_target = None
        try:
            from modules import shared
            if hasattr(shared, "cmd_opts"):
                if norm_type == "checkpoint" and getattr(shared.cmd_opts, "ckpt_dir", None):
                    custom_target = Path(shared.cmd_opts.ckpt_dir)
                elif norm_type in ("lora", "locon", "dora") and getattr(shared.cmd_opts, "lora_dir", None):
                    custom_target = Path(shared.cmd_opts.lora_dir)
        except Exception:
            pass

        if custom_target:
            destination = custom_target
        else:
            dirs_map = {k.lower(): v for k, v in MODEL_TYPE_DIRS.items()}
            relative = dirs_map.get(norm_type, "models/Other")
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
    seen: set[Path] = set()

    # Deduplicate directory names to avoid scanning the same directory multiple times
    unique_dirs = list(dict.fromkeys(MODEL_TYPE_DIRS.values()))

    # Check custom command-line flags from WebUI / Forge
    custom_dirs: list[Path] = []
    try:
        from modules import shared
        if hasattr(shared, "cmd_opts"):
            ckpt_dir = getattr(shared.cmd_opts, "ckpt_dir", None)
            if ckpt_dir:
                custom_dirs.append(Path(ckpt_dir))
            lora_dir = getattr(shared.cmd_opts, "lora_dir", None)
            if lora_dir:
                custom_dirs.append(Path(lora_dir))
    except Exception:
        pass

    dirs_to_scan = [root / d for d in unique_dirs] + custom_dirs

    for scan_dir in dirs_to_scan:
        if not scan_dir.exists():
            continue
        try:
            for f in scan_dir.rglob("*"):
                if f.is_file() and f.suffix.lower() in MODEL_EXTENSIONS:
                    resolved = f.resolve()
                    if resolved not in seen and not any(part.startswith(".") for part in f.parts):
                        seen.add(resolved)
                        results.append(f)
        except OSError:
            continue
    return results


def info_file_path(model_path: Path) -> Path:
    target = model_path.with_name(f"{model_path.stem}{CIVITAI_INFO_SUFFIX}")
    if not target.exists():
        # Fallback to legacy double-suffix path if it was previously created
        legacy = model_path.with_suffix("").with_suffix(CIVITAI_INFO_SUFFIX)
        if legacy.exists():
            return legacy
    return target


def preview_file_path(model_path: Path, ext: str = "png") -> Path:
    return model_path.with_name(f"{model_path.stem}.preview.{ext}")
