import sys
import threading
from pathlib import Path

# Allow importing ch_lib from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import gradio as gr
except ImportError:
    gr = None

try:
    from modules import script_callbacks
except ImportError:
    script_callbacks = None

from ch_lib import api, utils, model_manager, downloader, settings
from ch_lib.downloader import DownloadTask, BatchItem

# ── Batch analysis phase state ─────────────────────────────────────────
_resolve: dict = {"running": False, "total": 0, "done": 0, "log": []}

# ── Card info cache for extra networks ─────────────────────────────────
_card_cache: dict[str, dict] = {}


def get_card_info_data(filename: str) -> dict:
    if not filename:
        return {"civitai_url": "", "trigger_words": ""}

    clean_name = filename.strip()
    if clean_name in _card_cache:
        return _card_cache[clean_name]

    model_files = utils.iter_model_files()
    matched_path = None
    clean_stem = Path(clean_name).stem.lower()

    for p in model_files:
        if p.stem.lower() == clean_stem or p.name.lower() == clean_name.lower():
            matched_path = p
            break
        if clean_name.replace("\\", "/").lower() in str(p).replace("\\", "/").lower():
            matched_path = p
            break

    if not matched_path:
        res = {"civitai_url": "", "trigger_words": ""}
        _card_cache[clean_name] = res
        return res

    info = model_manager.load_info(matched_path)
    civitai_url = ""
    trigger_words = ""

    if info:
        model_id = info.get("modelId") or (info.get("model") or {}).get("id")
        version_id = info.get("id")
        if model_id and version_id:
            civitai_url = f"https://civitai.com/models/{model_id}?modelVersionId={version_id}"
        elif model_id:
            civitai_url = f"https://civitai.com/models/{model_id}"

        words = info.get("trainedWords") or info.get("trained_words") or []
        if isinstance(words, list) and words:
            trigger_words = ", ".join(w for w in words if isinstance(w, str))
        elif isinstance(words, str):
            trigger_words = words

    txt_path = matched_path.with_name(f"{matched_path.stem}.txt")
    if txt_path.exists():
        try:
            txt_content = txt_path.read_text(encoding="utf-8").strip()
            if txt_content:
                trigger_words = txt_content
        except Exception:
            pass

    res = {"civitai_url": civitai_url, "trigger_words": trigger_words}
    _card_cache[clean_name] = res
    return res


# ── Callbacks Download ───────────────────────────────────────────────────────

def cb_fetch_model(url_or_id: str, api_key: str):
    api_key = settings.resolve_api_key(api_key)
    model_id, version_id = api.parse_model_url(url_or_id)

    if not model_id and not version_id:
        return (
            "❌ Invalid URL or ID.", gr.update(choices=[]), gr.update(choices=[]),
            "", "", [], "Other", None,
        )

    try:
        if not model_id and version_id:
            # Direct version URL or download URL was supplied
            v_data = api.fetch_version_by_id(version_id, api_key)
            model_id = str(v_data.get("modelId", ""))
            if not model_id:
                return (
                    "❌ Model not found for version ID.", gr.update(choices=[]), gr.update(choices=[]),
                    "", "", [], "Other", None,
                )

        model_info = api.fetch_model_info(model_id, api_key)
    except api.CivitaiAPIError as exc:
        return (
            f"❌ {exc}", gr.update(choices=[]), gr.update(choices=[]),
            "", "", [], "Other", None,
        )

    versions = api.extract_versions(model_info)
    if not versions:
        return (
            "❌ No downloadable versions found.", gr.update(choices=[]),
            gr.update(choices=[]), "", "", [], "Other", None,
        )

    # Pre-select version from URL if specified
    default_version = versions[0]
    if version_id:
        match = next((v for v in versions if str(v["id"]) == str(version_id)), None)
        if match:
            default_version = match

    version_labels = [v["label"] for v in versions]
    file_names     = [f["name"] for f in default_version["files"]]
    trigger_words  = ", ".join(default_version.get("trained_words", []))
    model_type     = model_info.get("type", "Other")

    cover_url = None
    images = default_version.get("images", []) or model_info.get("images", [])
    for img in images:
        if isinstance(img, dict) and img.get("url"):
            cover_url = img.get("url")
            break

    base_model = default_version.get("base_model") or "?"
    stats = model_info.get("stats", {})
    downloads = stats.get("downloadCount", 0)
    rating = round(stats.get("rating", 0), 1)

    summary = (
        f"### 📦 **{model_info.get('name', '?')}**\n"
        f"🏷️ **Type:** `{model_type}` | 🧩 **Base:** `{base_model}` | "
        f"⬇️ **Downloads:** `{downloads:,}` | ⭐ **Rating:** `{rating}`"
    )

    return (
        f"✅ Model found: {model_info.get('name', '?')}",
        gr.update(choices=version_labels, value=default_version["label"]),
        gr.update(choices=file_names, value=file_names[0] if file_names else None),
        summary,
        trigger_words,
        versions,
        model_type,
        cover_url,
    )


def cb_version_change(version_label: str, versions_cache: list):
    version = next((v for v in versions_cache if v["label"] == version_label), None)
    if not version:
        return gr.update(choices=[]), "", None
    file_names    = [f["name"] for f in version["files"]]
    trigger_words = ", ".join(version.get("trained_words", []))
    cover_url = None
    images = version.get("images", [])
    for img in images:
        if isinstance(img, dict) and img.get("url"):
            cover_url = img.get("url")
            break
    return gr.update(choices=file_names, value=file_names[0] if file_names else None), trigger_words, cover_url


def cb_start_download(
    url_or_id, version_label, file_name,
    api_key, custom_dir, download_preview,
    versions_cache, model_type,
):
    api_key = settings.resolve_api_key(api_key)
    if not versions_cache:
        return "❌ Fetch model info first."
    if not version_label or not file_name:
        return "❌ Select a version and file."

    q = downloader.get_queue()
    if q.running:
        return "⚠️ A download is already in progress."

    version = next((v for v in versions_cache if v["label"] == version_label), None)
    if not version:
        return "❌ Version not found."

    file_info = next((f for f in version["files"] if f["name"] == file_name), None)
    if not file_info:
        return "❌ File not found."

    model_id, _ = api.parse_model_url(url_or_id)
    try:
        model_info = api.fetch_model_info(model_id, api_key) if model_id else {}
    except api.CivitaiAPIError:
        model_info = {}

    dest_dir  = utils.resolve_model_dir(model_type, custom_dir)
    dest_path = dest_dir / file_info["name"]

    task = DownloadTask(
        url             = file_info["url"],
        dest            = dest_path,
        filename        = file_info["name"],
        sha256_expected = file_info.get("sha256", ""),
    )

    q.start_async(
        task             = task,
        api_key          = api_key,
        version_data     = version,
        model_info       = model_info,
        download_preview = download_preview,
    )
    return f"⏳ Starting download: {file_info['name']}…"


def cb_cancel_download():
    q = downloader.get_queue()
    if q.running:
        q.cancel()
        return "🛑 Cancelling..."
    return "ℹ️ No active download."


def cb_poll_download():
    q    = downloader.get_queue()
    task = q.current
    log  = "\n".join(q.log[-40:])

    if task:
        if task.error:
            status = f"❌ Error: {task.error}"
            progress = 0.0
        elif task.cancelled:
            status   = "⚠️ Cancelled"
            progress = 0.0
        elif task.done:
            status   = f"✅ Completed: {task.filename}"
            progress = 1.0
        else:
            pct      = task.progress * 100
            done_mb  = utils.format_size(task.downloaded_bytes / 1024)
            total_mb = utils.format_size(task.total_bytes / 1024)
            status   = f"⏳ {task.filename} — {done_mb} / {total_mb} ({pct:.1f}%)"
            progress = task.progress
    else:
        status   = "💤 Waiting"
        progress = 0.0

    return gr.update(value=status), gr.update(value=log), gr.update(value=progress)


# ── Callbacks Scan ───────────────────────────────────────────────────────────

def cb_start_scan(api_key: str, skip_existing: bool):
    api_key = settings.resolve_api_key(api_key)
    state = model_manager.get_scan_state()
    if state.running:
        return "⚠️ Scan already in progress."
    threading.Thread(
        target=model_manager.scan_models,
        args=(api_key, skip_existing),
        daemon=True,
    ).start()
    return "⏳ Scan started..."


def cb_start_missing_previews(api_key: str):
    api_key = settings.resolve_api_key(api_key)
    state = model_manager.get_scan_state()
    if state.running:
        return "⚠️ Operation already in progress."
    threading.Thread(
        target=model_manager.download_missing_previews,
        args=(api_key,),
        daemon=True,
    ).start()
    return "⏳ Downloading missing previews..."


def cb_cancel_scan():
    state = model_manager.get_scan_state()
    if state.running:
        state.cancel = True
        return "🛑 Cancelling..."
    return "ℹ️ No active scan."


def cb_poll_scan():
    state    = model_manager.get_scan_state()
    log      = "\n".join(state.log[-40:])
    progress = state.progress

    if state.running:
        status = f"⏳ {state.summary}"
    elif state.log:
        status = f"✅ {state.log[-1]}"
    else:
        status = "💤"

    return gr.update(value=status), gr.update(value=log), gr.update(value=progress)


# ── Callbacks Updates ────────────────────────────────────────────────────────

def cb_check_updates(api_key: str):
    api_key = settings.resolve_api_key(api_key)
    state = model_manager.get_update_state()
    if state.running:
        return "⚠️ Check already in progress.", []
    threading.Thread(
        target=model_manager.check_for_updates,
        args=(api_key,),
        daemon=True,
    ).start()
    return "⏳ Checking for updates...", []


def cb_poll_updates():
    state = model_manager.get_update_state()
    log   = "\n".join(state.log[-30:])

    if state.running:
        return gr.update(value="⏳ In progress..."), gr.update(value=log), gr.update()

    if not state.results:
        msg = "✅ All your models are up to date." if state.log else "💤"
        return gr.update(value=msg), gr.update(value=log), gr.update(value=[])

    rows = [
        [r["model_name"], r["local_version_id"],
         r["latest_label"], r["model_type"], r["model_path"]]
        for r in state.results
    ]
    return (
        gr.update(value=f"🆕 {len(rows)} update(s) available."),
        gr.update(value=log),
        gr.update(value=rows),
    )


def cb_download_update(selected_rows, api_key: str):
    api_key = settings.resolve_api_key(api_key)
    state = model_manager.get_update_state()
    if not state.results or not selected_rows:
        return "❌ No rows selected."

    messages = []
    items_to_download = []

    for row_idx in selected_rows:
        try:
            result = state.results[int(row_idx)]
        except (IndexError, ValueError):
            continue

        try:
            version_info = api.fetch_version_by_id(str(result["latest_version_id"]), api_key)
        except api.CivitaiAPIError as exc:
            messages.append(f"❌ {result['model_name']} : {exc}")
            continue

        files = [
            {
                "name":    f.get("name", ""),
                "url":     f.get("downloadUrl", ""),
                "size_kb": f.get("sizeKB", 0),
                "sha256":  (f.get("hashes") or {}).get("SHA256", ""),
            }
            for f in version_info.get("files", [])
            if f.get("downloadUrl")
        ]
        if not files:
            messages.append(f"⚠️ {result['model_name']}: no files found.")
            continue

        primary = next((f for f in files if f["name"].endswith(".safetensors")), files[0])
        existing_model_path = Path(result.get("model_path", ""))
        dest_dir = existing_model_path.parent if existing_model_path.exists() else utils.resolve_model_dir(result["model_type"])

        items_to_download.append((result, primary, dest_dir, version_info))

    if not items_to_download:
        return "\n".join(messages) if messages else "❌ No valid updates to download."

    if len(items_to_download) == 1:
        result, primary, dest_dir, version_info = items_to_download[0]
        q = downloader.get_queue()
        if q.running:
            return "⚠️ A download is already in progress."
        task = DownloadTask(
            url             = primary["url"],
            dest            = dest_dir / primary["name"],
            filename        = primary["name"],
            sha256_expected = primary.get("sha256", ""),
        )
        q.start_async(
            task             = task,
            api_key          = api_key,
            version_data     = version_info,
            download_preview = True,
        )
        messages.append(f"⏳ Downloading update for {result['model_name']}...")
    else:
        bq = downloader.get_batch_queue()
        if bq.running:
            return "⚠️ A batch download is already in progress."
        for result, primary, dest_dir, version_info in items_to_download:
            item = BatchItem(
                url           = f"https://civitai.com/models/{result.get('model_id', '')}",
                model_name    = result["model_name"],
                version_name  = result["latest_label"],
                filename      = primary["name"],
                size_kb       = primary.get("size_kb", 0),
                status        = "pending",
                _dl_url       = primary["url"],
                _sha256       = primary.get("sha256", ""),
                _dest_dir     = dest_dir,
                _version_data = version_info,
                _model_info   = None,
            )
            bq.add_item(item)
        bq.start(api_key)
        messages.append(f"⏳ Queued {len(items_to_download)} updates in batch...")

    return "\n".join(messages)


# ── Callbacks Search ─────────────────────────────────────────────────────────

def cb_search(query, model_type, page, api_key, nsfw):
    api_key = settings.resolve_api_key(api_key)
    try:
        data = api.search_models(
            query=query,
            model_type=model_type if model_type not in ("Tous", "All") else None,
            limit=20, page=int(page), api_key=api_key, nsfw=nsfw,
        )
    except api.CivitaiAPIError as exc:
        return gr.update(value=[]), f"❌ {exc}"

    items = data.get("items", [])
    if not items:
        return gr.update(value=[]), "No results found."

    rows = [
        [
            m.get("name", ""),
            m.get("type", ""),
            m.get("stats", {}).get("downloadCount", 0),
            round(m.get("stats", {}).get("rating", 0), 2),
            f"https://civitai.com/models/{m['id']}",
        ]
        for m in items
    ]
    total = data.get("metadata", {}).get("totalItems", "?")
    return gr.update(value=rows), f"🔍 {len(rows)} result(s) (total: {total})"


# ── Callbacks Batch ──────────────────────────────────────────────────────────

def cb_batch_analyze(urls_text: str, api_key: str, custom_dir: str):
    api_key = settings.resolve_api_key(api_key)
    urls = [u.strip() for u in urls_text.strip().splitlines() if u.strip()]
    if not urls:
        return "❌ No URLs provided."

    bq = downloader.get_batch_queue()
    if bq.running:
        return "⚠️ A batch is already downloading."
    if _resolve["running"]:
        return "⚠️ Analysis already in progress."

    bq.clear()
    _resolve["running"] = True
    _resolve["total"]   = len(urls)
    _resolve["done"]    = 0
    _resolve["log"]     = []

    def _run():
        try:
            for url in urls:
                _resolve["log"].append(f"Analyzing: {url[:70]}...")
                model_id, version_id = api.parse_model_url(url)

                if not model_id and not version_id:
                    _resolve["log"].append("  ❌ Invalid URL.")
                    _resolve["done"] += 1
                    continue

                try:
                    if not model_id and version_id:
                        v_info = api.fetch_version_by_id(version_id, api_key)
                        model_id = str(v_info.get("modelId", ""))

                    model_info = api.fetch_model_info(model_id, api_key)
                except api.CivitaiAPIError as exc:
                    _resolve["log"].append(f"  ❌ API: {exc}")
                    _resolve["done"] += 1
                    continue

                versions = api.extract_versions(model_info)
                if not versions:
                    _resolve["log"].append(f"  ❌ No versions: {model_info.get('name','?')}")
                    _resolve["done"] += 1
                    continue

                version = versions[0]
                if version_id:
                    match = next((v for v in versions if str(v["id"]) == str(version_id)), None)
                    if match:
                        version = match

                files = version["files"]
                best = (
                    next((f for f in files if f["name"].endswith(".safetensors") and f.get("type", "") == "Model"), None)
                    or next((f for f in files if f["name"].endswith(".safetensors")), None)
                    or files[0]
                )

                model_type = model_info.get("type", "Other")
                dest_dir   = utils.resolve_model_dir(model_type, custom_dir)

                item = BatchItem(
                    url          = url,
                    model_name   = model_info.get("name", "?"),
                    version_name = version.get("name", ""),
                    filename     = best["name"],
                    size_kb      = best.get("size_kb", 0),
                    status       = "pending",
                    _dl_url      = best["url"],
                    _sha256      = best.get("sha256", ""),
                    _dest_dir    = dest_dir,
                    _version_data = version,
                    _model_info  = model_info,
                )
                bq.add_item(item)
                size_str = utils.format_size(best.get("size_kb", 0)) if best.get("size_kb") else "?"
                _resolve["log"].append(f"  ✅ {model_info.get('name','?')} → {best['name']} ({size_str})")
                _resolve["done"] += 1
        finally:
            _resolve["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return f"⏳ Analyzing {len(urls)} URL(s)..."


def cb_batch_start(api_key: str):
    api_key = settings.resolve_api_key(api_key)
    bq = downloader.get_batch_queue()
    if bq.running:
        return "⚠️ Already in progress."
    if _resolve["running"]:
        return "⚠️ Wait for analysis to finish."
    if not bq.items:
        return "❌ No models. Analyze URLs first."
    pending = sum(1 for i in bq.items if i.status == "pending")
    if not pending:
        return "ℹ️ No pending items."
    bq.start(api_key)
    return f"⏳ Batch started — {pending} download(s) in queue..."


def cb_batch_cancel():
    bq = downloader.get_batch_queue()
    if bq.running:
        bq.cancel()
        return "🛑 Cancelling..."
    return "ℹ️ No active batch."


def cb_batch_clear():
    bq = downloader.get_batch_queue()
    if bq.running:
        return "⚠️ Cannot clear while downloading."
    bq.clear()
    _resolve["log"] = []
    return "🗑️ List cleared."


def cb_poll_batch():
    bq = downloader.get_batch_queue()

    if bq._current_task:
        for item in bq.items:
            if item.status == "downloading":
                item.progress = bq._current_task.progress
                break

    rows = []
    for item in bq.items:
        pct      = f"{item.progress * 100:.0f}%" if item.progress > 0 else "—"
        size_str = utils.format_size(item.size_kb) if item.size_kb else "?"
        rows.append([item.model_name, item.filename, size_str, item.status, pct])

    log_lines = (_resolve["log"] + bq.log)[-40:]
    log       = "\n".join(log_lines)

    if bq.running:
        done   = sum(1 for i in bq.items if i.status == "completed")
        status = f"⏳ {done}/{len(bq.items)} completed..."
    elif _resolve["running"]:
        status = f"🔍 Analyzing {_resolve['done']}/{_resolve['total']}..."
    elif bq.items:
        done   = sum(1 for i in bq.items if i.status == "completed")
        errors = sum(1 for i in bq.items if i.status == "error")
        if done + errors == len(bq.items) and bq.log:
            status = f"✅ Batch completed: {done} OK, {errors} error(s)"
        else:
            pending = sum(1 for i in bq.items if i.status == "pending")
            status  = f"📋 {len(bq.items)} model(s) — {pending} pending"
    else:
        status = "💤 List empty"

    return gr.update(value=rows), gr.update(value=log), gr.update(value=status)


# ── UI Construction ──────────────────────────────────────────────────────────

def _load_css() -> str:
    css_path = Path(__file__).resolve().parents[1] / "style.css"
    if css_path.exists():
        return css_path.read_text(encoding="utf-8")
    return ""


def build_ui():
    with gr.Blocks(elem_id="civitai_helper_root", css=_load_css()) as ui:
        gr.Markdown("# 🐘 CivitAI Helper")

        with gr.Row():
            api_key_input = gr.Textbox(
                label="CivitAI API Key",
                placeholder="Optional - required for restricted/NSFW models",
                value=settings.get_saved_api_key(),
                type="password", scale=3,
            )

        with gr.Tab("⬇️ Download"):
            _tab_download(api_key_input)

        with gr.Tab("📦 Batch"):
            _tab_batch(api_key_input)

        with gr.Tab("🔍 Search"):
            _tab_search(api_key_input)

        with gr.Tab("🔄 Scan & Update"):
            _tab_scan(api_key_input)

    return ui


def _tab_download(api_key_input):
    gr.Markdown("### Download a model from a CivitAI URL or ID")

    with gr.Row():
        with gr.Column(scale=3):
            url_input     = gr.Textbox(label="URL or ID", placeholder="https://civitai.com/models/12345")
            fetch_btn     = gr.Button("🔍 Fetch Info", variant="primary")
            model_status  = gr.Markdown("")
            model_summary = gr.Markdown("")
        with gr.Column(scale=2):
            preview_img   = gr.Image(label="Cover Preview", interactive=False, height=240)

    with gr.Row():
        version_dd = gr.Dropdown(label="Version", choices=[], interactive=True, scale=2)
        file_dd    = gr.Dropdown(label="File",    choices=[], interactive=True, scale=2)

    trigger_words_box = gr.Textbox(label="Trigger words", interactive=False)

    with gr.Accordion("⚙️ Options", open=False):
        custom_dir_input = gr.Textbox(
            label="Destination folder",
            placeholder="Leave empty for auto-detection by model type",
        )
        preview_checkbox = gr.Checkbox(label="Download preview images", value=True)

    with gr.Row():
        dl_btn      = gr.Button("⬇️ Download", variant="primary")
        cancel_btn  = gr.Button("🛑 Cancel",   variant="stop")
        refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary")

    dl_result = gr.Markdown("")

    with gr.Accordion("📋 Progress", open=True):
        dl_progress = gr.Slider(minimum=0, maximum=1, value=0,
                                label="Progress", interactive=False)
        dl_status   = gr.Markdown("💤 Waiting")
        dl_log      = gr.Textbox(label="Logs", lines=8, interactive=False, max_lines=15)

    versions_cache = gr.State([])
    model_type     = gr.State("Other")

    fetch_btn.click(
        fn=cb_fetch_model,
        inputs=[url_input, api_key_input],
        outputs=[model_status, version_dd, file_dd,
                 model_summary, trigger_words_box, versions_cache, model_type, preview_img],
    )
    version_dd.change(
        fn=cb_version_change,
        inputs=[version_dd, versions_cache],
        outputs=[file_dd, trigger_words_box, preview_img],
    )
    dl_btn.click(
        fn=cb_start_download,
        inputs=[url_input, version_dd, file_dd, api_key_input,
                custom_dir_input, preview_checkbox, versions_cache, model_type],
        outputs=[dl_result],
    )
    cancel_btn.click(fn=cb_cancel_download, outputs=[dl_result])
    refresh_btn.click(fn=cb_poll_download, outputs=[dl_status, dl_log, dl_progress])

    if hasattr(gr, "Timer"):
        poll_timer = gr.Timer(value=2.5)
        poll_timer.tick(fn=cb_poll_download, outputs=[dl_status, dl_log, dl_progress])


def _tab_batch(api_key_input):
    gr.Markdown("### Download multiple models at once")
    gr.Markdown(
        "Paste one CivitAI URL per line. "
        "The extension automatically detects model versions and selects "
        "the best available `.safetensors` file."
    )

    urls_input = gr.Textbox(
        label="CivitAI URLs (one per line)",
        placeholder="https://civitai.com/models/12345\nhttps://civitai.com/models/67890",
        lines=6,
    )

    with gr.Accordion("⚙️ Options", open=False):
        custom_dir_batch = gr.Textbox(
            label="Destination folder",
            placeholder="Leave empty for auto-detection by model type",
        )

    with gr.Row():
        analyze_btn = gr.Button("🔍 Analyze URLs", variant="primary")
        start_btn   = gr.Button("⬇️ Download All",  variant="primary")
        cancel_btn  = gr.Button("🛑 Cancel",        variant="stop")
        clear_btn   = gr.Button("🗑️ Clear List")
        refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary")

    batch_status = gr.Markdown("💤 Empty list")

    batch_table = gr.Dataframe(
        headers=["Model", "File", "Size", "Status", "Progress"],
        datatype=["str", "str", "str", "str", "str"],
        interactive=False,
        wrap=True,
        label="Download Queue",
    )

    batch_log = gr.Textbox(label="Logs", lines=8, interactive=False, max_lines=15)

    analyze_btn.click(
        fn=cb_batch_analyze,
        inputs=[urls_input, api_key_input, custom_dir_batch],
        outputs=[batch_status],
    )
    start_btn.click(
        fn=cb_batch_start,
        inputs=[api_key_input],
        outputs=[batch_status],
    )
    cancel_btn.click(fn=cb_batch_cancel, outputs=[batch_status])
    clear_btn.click(fn=cb_batch_clear,   outputs=[batch_status])
    refresh_btn.click(fn=cb_poll_batch,  outputs=[batch_table, batch_log, batch_status])

    if hasattr(gr, "Timer"):
        batch_timer = gr.Timer(value=2.5)
        batch_timer.tick(fn=cb_poll_batch, outputs=[batch_table, batch_log, batch_status])


def _tab_search(api_key_input):
    gr.Markdown("### Search for models on CivitAI")

    with gr.Row():
        search_input = gr.Textbox(label="Keywords", placeholder="e.g., realistic portrait lora...", scale=3)
        type_filter  = gr.Dropdown(
            label="Type",
            choices=["All", "Checkpoint", "LORA", "TextualInversion", "VAE", "ControlNet", "Upscaler"],
            value="All", scale=1,
        )
        nsfw_toggle  = gr.Checkbox(label="NSFW", value=False, scale=1)
        page_input   = gr.Number(label="Page", value=1, minimum=1, precision=0, scale=1)

    search_btn    = gr.Button("🔍 Search", variant="primary")
    search_status = gr.Markdown("")
    search_results = gr.Dataframe(
        headers=["Name", "Type", "Downloads", "Rating", "URL"],
        datatype=["str", "str", "number", "number", "str"],
        interactive=False, wrap=True,
    )

    search_btn.click(
        fn=cb_search,
        inputs=[search_input, type_filter, page_input, api_key_input, nsfw_toggle],
        outputs=[search_results, search_status],
    )


def _tab_scan(api_key_input):
    gr.Markdown("#### 📂 Scan local models")
    gr.Markdown(
        "Calculates SHA256 of each local model, queries CivitAI "
        "and generates `.civitai.info`, trigger words `.txt`, and preview images."
    )

    with gr.Row():
        skip_existing_cb    = gr.Checkbox(label="Skip already scanned models", value=True)
        scan_btn            = gr.Button("🔍 Start Scan", variant="primary")
        missing_preview_btn = gr.Button("📸 Download Missing Previews", variant="secondary")
        scan_cancel_btn     = gr.Button("🛑 Stop",        variant="stop")
        refresh_scan_btn    = gr.Button("🔄 Refresh Status", variant="secondary")

    scan_status   = gr.Markdown("💤")
    scan_progress = gr.Slider(minimum=0, maximum=1, value=0,
                              label="Progress", interactive=False)
    scan_log      = gr.Textbox(label="Logs", lines=8, interactive=False, max_lines=20)

    gr.Markdown("---")
    gr.Markdown("#### 🆕 Check for updates")

    update_btn       = gr.Button("🔄 Check Updates", variant="primary")
    update_status    = gr.Markdown("💤")
    update_log       = gr.Textbox(label="Update Logs", lines=5, interactive=False)
    update_table     = gr.Dataframe(
        headers=["Model", "Local Version", "New Version", "Type", "Path"],
        datatype=["str", "str", "str", "str", "str"],
        interactive=True, label="Available updates",
    )
    dl_update_btn    = gr.Button("⬇️ Download Selected Updates", variant="primary")
    dl_update_result = gr.Markdown("")

    selected_rows = gr.State([])

    scan_btn.click(
        fn=cb_start_scan,
        inputs=[api_key_input, skip_existing_cb],
        outputs=[scan_status],
    )
    missing_preview_btn.click(
        fn=cb_start_missing_previews,
        inputs=[api_key_input],
        outputs=[scan_status],
    )
    scan_cancel_btn.click(fn=cb_cancel_scan, outputs=[scan_status])
    refresh_scan_btn.click(fn=cb_poll_scan,  outputs=[scan_status, scan_log, scan_progress])

    if hasattr(gr, "Timer"):
        scan_timer = gr.Timer(value=2.0)
        scan_timer.tick(fn=cb_poll_scan, outputs=[scan_status, scan_log, scan_progress])

    update_btn.click(
        fn=cb_check_updates,
        inputs=[api_key_input],
        outputs=[update_status, update_table],
    )

    if hasattr(gr, "Timer"):
        update_timer = gr.Timer(value=3.0)
        update_timer.tick(fn=cb_poll_updates, outputs=[update_status, update_log, update_table])

    def on_table_select(evt: gr.SelectData):
        return [evt.index[0]]

    update_table.select(fn=on_table_select, outputs=[selected_rows])
    dl_update_btn.click(
        fn=cb_download_update,
        inputs=[selected_rows, api_key_input],
        outputs=[dl_update_result],
    )


# ── Registration Callbacks ───────────────────────────────────────────────────

def on_app_started(demo=None, app=None):
    if app is None:
        return
    try:
        from fastapi.responses import JSONResponse

        @app.get("/civitai_helper/card_info")
        async def api_card_info(filename: str = ""):
            return JSONResponse(get_card_info_data(filename))
    except Exception as exc:
        utils.safe_print(f"[CivitAI Helper] Failed to register API endpoint: {exc}")


def on_ui_settings():
    settings.register_options()


def on_ui_tabs():
    return [(build_ui(), "🐘 CivitAI Helper", "sd_forge_civitai_helper")]


if script_callbacks:
    script_callbacks.on_app_started(on_app_started)
    script_callbacks.on_ui_settings(on_ui_settings)
    script_callbacks.on_ui_tabs(on_ui_tabs)
