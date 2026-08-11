import os
from typing import Any

API_KEY_OPTION = "sd_forge_civitai_helper_api_key"
MAX_PREVIEWS_OPTION = "sd_forge_civitai_helper_max_previews"
AUTO_TXT_OPTION = "sd_forge_civitai_helper_auto_txt"
ENV_API_KEY = "CIVITAI_API_KEY"


def _get_shared() -> Any | None:
    try:
        from modules import shared
        return shared
    except Exception:
        return None


def get_saved_api_key() -> str:
    shared = _get_shared()
    if shared is None:
        return ""
    return str(getattr(shared.opts, API_KEY_OPTION, "") or "").strip()


def get_max_previews() -> int:
    shared = _get_shared()
    if shared is None:
        return 1
    val = getattr(shared.opts, MAX_PREVIEWS_OPTION, 1)
    try:
        return max(1, min(5, int(val)))
    except (ValueError, TypeError):
        return 1


def get_auto_txt() -> bool:
    shared = _get_shared()
    if shared is None:
        return True
    return bool(getattr(shared.opts, AUTO_TXT_OPTION, True))


def resolve_api_key(api_key: str | None = None) -> str:
    """Use explicit UI key first, then WebUI setting, then CIVITAI_API_KEY env."""
    explicit = (api_key or "").strip()
    if explicit:
        return explicit
    saved = get_saved_api_key()
    if saved:
        return saved
    return os.environ.get(ENV_API_KEY, "").strip()


def register_options() -> None:
    shared = _get_shared()
    if shared is None:
        return
    try:
        section = ("sd_forge_civitai_helper", "CivitAI Helper")
        shared.opts.add_option(
            API_KEY_OPTION,
            shared.OptionInfo(
                "",
                "CivitAI API Key (used when the extension field is empty; stored in WebUI config.json)",
                section=section,
            ),
        )
        shared.opts.add_option(
            MAX_PREVIEWS_OPTION,
            shared.OptionInfo(
                1,
                "Number of preview images to download per model (1-5)",
                shared.NumberSlider,
                {"minimum": 1, "maximum": 5, "step": 1},
                section=section,
            ),
        )
        shared.opts.add_option(
            AUTO_TXT_OPTION,
            shared.OptionInfo(
                True,
                "Automatically save trigger words in a .txt file alongside the model",
                section=section,
            ),
        )
    except Exception as exc:
        print(f"[CivitAI Helper] Could not register settings: {exc}")
