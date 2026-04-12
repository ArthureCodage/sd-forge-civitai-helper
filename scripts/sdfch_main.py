import sys
import threading
from pathlib import Path

# Permet l'import de ch_lib depuis scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
from modules import script_callbacks

from ch_lib import api, utils, model_manager, downloader
from ch_lib.downloader import DownloadTask, BatchItem

# ── Batch analysis phase state ─────────────────────────────────────────
_resolve: dict = {"running": False, "total": 0, "done": 0, "log": []}


# ── Callbacks Download ───────────────────────────────────────────────────────

def cb_fetch_model(url_or_id: str, api_key: str):
    model_id, version_id = api.parse_model_url(url_or_id)
    if not model_id:
        return (
            "❌ Invalid URL or ID.", gr.update(choices=[]), gr.update(choices=[]),
            "", "", [], "Other",
        )
    try:
        model_info = api.fetch_model_info(model_id, api_key)
    except api.CivitaiAPIError as exc:
        return (
            f"❌ {exc}", gr.update(choices=[]), gr.update(choices=[]),
            "", "", [], "Other",
        )

    versions   = api.extract_versions(model_info)
    if not versions:
        return (
            "❌ No downloadable versions found.", gr.update(choices=[]),
            gr.update(choices=[]), "", "", [], "Other",
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

    summary = (
        f"**{model_info.get('name', '?')}** — "
        f"Type: `{model_type}` — "
        f"{len(versions)} version(s) available"
    )

    return (
        f"✅ Model found: {model_info.get('name', '?')}",
        gr.update(choices=version_labels, value=default_version["label"]),
        gr.update(choices=file_names, value=file_names[0] if file_names else None),
        summary,
        trigger_words,
        versions,
        model_type,
    )


def cb_version_change(version_label: str, versions_cache: list):
    version = next((v for v in versions_cache if v["label"] == version_label), None)
    if not version:
        return gr.update(choices=[]), ""
    file_names    = [f["name"] for f in version["files"]]
    trigger_words = ", ".join(version.get("trained_words", []))
    return gr.update(choices=file_names, value=file_names[0] if file_names else None), trigger_words


def cb_start_download(
    url_or_id, version_label, file_name,
    api_key, custom_dir, download_preview,
    versions_cache, model_type,
):
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
    state = model_manager.get_scan_state()
    if state.running:
        return "⚠️ Scan already in progress."
    threading.Thread(
        target=model_manager.scan_models,
        args=(api_key, skip_existing),
        daemon=True,
    ).start()
    return "⏳ Scan started..."


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


# ── Callbacks MAJ ────────────────────────────────────────────────────────────

def cb_check_updates(api_key: str):
    state = model_manager.get_update_state()
    if state.running:
        return "⚠️ Check already in progress.", []
    threading.Thread(
        target=model_manager.check_for_updates,
        args=(api_key,),
        daemon=True,
    ).start()
    return "⏳ Checking...", []


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
    state = model_manager.get_update_state()
    if not state.results or not selected_rows:
        return "❌ No rows selected."

    messages = []
    for row_idx in selected_rows:
        result = state.results[int(row_idx)]
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
            messages.append(f"⚠️ {result['model_name']}: no files.")
            continue

        primary   = next((f for f in files if f["name"].endswith(".safetensors")), files[0])
        dest_dir  = utils.resolve_model_dir(result["model_type"])
        dest_path = dest_dir / primary["name"]
        task      = DownloadTask(
            url=primary["url"], dest=dest_path,
            filename=primary["name"], sha256_expected=primary.get("sha256", ""),
        )
        q = downloader.get_queue()
        if not q.running:
            q.start_async(task=task, api_key=api_key,
                          version_data=version_info, download_preview=True)
            messages.append(f"⏳ {result['model_name']} en cours…")
        else:
            messages.append(f"⚠️ {result['model_name']}: queue occupied.")

    return "\n".join(messages)


# ── Callbacks Recherche ──────────────────────────────────────────────────────

def cb_search(query, model_type, page, api_key, nsfw):
    try:
        data = api.search_models(
            query=query,
            model_type=model_type if model_type != "Tous" else None,
            limit=20, page=int(page), api_key=api_key, nsfw=nsfw,
        )
    except api.CivitaiAPIError as exc:
        return gr.update(value=[]), f"❌ {exc}"

    items = data.get("items", [])
    if not items:
        return gr.update(value=[]), "No results."

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
    return gr.update(value=rows), f"🔍 {len(rows)} results (total: {total})"


# ── Callbacks Batch ──────────────────────────────────────────────────────────

def cb_batch_analyze(urls_text: str, api_key: str, custom_dir: str):
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
        for url in urls:
            _resolve["log"].append(f"Analyzing: {url[:70]}...")
            model_id, version_id = api.parse_model_url(url)
            if not model_id:
                _resolve["log"].append(f"  ❌ Invalid URL.")
                _resolve["done"] += 1
                continue

            try:
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

            # Select version
            version = versions[0]
            if version_id:
                match = next((v for v in versions if str(v["id"]) == str(version_id)), None)
                if match:
                    version = match

            # Select best file: Model-type safetensors first
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

        _resolve["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return f"⏳ Analyzing {len(urls)} URL(s)..."


def cb_batch_start(api_key: str):
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

    # Update current item progress
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


# ── Construction UI ──────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(elem_id="civitai_helper_root", css=_load_css()) as ui:
        gr.Markdown("# 🐘 CivitAI Helper")

        with gr.Row():
            api_key_input = gr.Textbox(
                label="CivitAI API Key",
                placeholder="Optional - required for restricted/NSFW models",
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


def _load_css() -> str:
    css_path = Path(__file__).resolve().parents[1] / "style.css"
    if css_path.exists():
        return css_path.read_text(encoding="utf-8")
    return ""


def _tab_download(api_key_input):
    gr.Markdown("### Download a model from a CivitAI URL or ID")

    url_input    = gr.Textbox(label="URL or ID", placeholder="https://civitai.com/models/12345")
    fetch_btn    = gr.Button("🔍 Fetch Info", variant="primary")
    model_status = gr.Markdown("")
    model_summary = gr.Markdown("")

    with gr.Row():
        version_dd = gr.Dropdown(label="Version", choices=[], interactive=True, scale=2)
        file_dd    = gr.Dropdown(label="File",  choices=[], interactive=True, scale=2)

    trigger_words_box = gr.Textbox(label="Trigger words", interactive=False)

    with gr.Accordion("⚙️ Options", open=False):
        custom_dir_input = gr.Textbox(
            label="Destination folder",
            placeholder="Leave empty for auto-detection",
        )
        preview_checkbox = gr.Checkbox(label="Download preview", value=True)

    with gr.Row():
        dl_btn     = gr.Button("⬇️ Download", variant="primary")
        cancel_btn = gr.Button("🛑 Cancel",      variant="stop")

    dl_result = gr.Markdown("")

    with gr.Accordion("📋 Progress", open=True):
        dl_progress = gr.Slider(minimum=0, maximum=1, value=0,
                                label="Progress", interactive=False)
        dl_status   = gr.Markdown("💤 Waiting")
        dl_log      = gr.Textbox(label="Logs", lines=8, interactive=False, max_lines=15)

    versions_cache = gr.State([])
    model_type     = gr.State("Other")
    poll_timer     = gr.Timer(value=2.5)

    fetch_btn.click(
        fn=cb_fetch_model,
        inputs=[url_input, api_key_input],
        outputs=[model_status, version_dd, file_dd,
                 model_summary, trigger_words_box, versions_cache, model_type],
    )
    version_dd.change(
        fn=cb_version_change,
        inputs=[version_dd, versions_cache],
        outputs=[file_dd, trigger_words_box],
    )
    dl_btn.click(
        fn=cb_start_download,
        inputs=[url_input, version_dd, file_dd, api_key_input,
                custom_dir_input, preview_checkbox, versions_cache, model_type],
        outputs=[dl_result],
    )
    cancel_btn.click(fn=cb_cancel_download, outputs=[dl_result])
    poll_timer.tick(fn=cb_poll_download, outputs=[dl_status, dl_log, dl_progress])


def _tab_batch(api_key_input):
    gr.Markdown("### Download multiple models at once")
    gr.Markdown(
        "Paste one CivitAI URL per line. "
        "The extension automatically selects the latest version "
        "and best available `.safetensors` file."
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
        start_btn   = gr.Button("⬇️ Download tout",  variant="primary")
        cancel_btn  = gr.Button("🛑 Cancel",           variant="stop")
        clear_btn   = gr.Button("🗑️ Clear List")

    batch_status = gr.Markdown("💤 Empty list")

    batch_table = gr.Dataframe(
        headers=["Model", "File", "Size", "Status", "Progress"],
        datatype=["str", "str", "str", "str", "str"],
        interactive=False,
        wrap=True,
        label="Download Queue",
    )

    batch_log = gr.Textbox(label="Logs", lines=8, interactive=False, max_lines=15)

    batch_timer = gr.Timer(value=2.5)

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
    batch_timer.tick(fn=cb_poll_batch,   outputs=[batch_table, batch_log, batch_status])


def _tab_search(api_key_input):
    gr.Markdown("### Search for models on CivitAI")

    with gr.Row():
        search_input = gr.Textbox(label="Keywords", placeholder="e.g., realistic portrait lora...", scale=3)
        type_filter  = gr.Dropdown(
            label="Type",
            choices=["Tous", "Checkpoint", "LORA", "TextualInversion", "VAE", "ControlNet"],
            value="Tous", scale=1,
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
        "Calculates SHA256 of each model, queries CivitAI "
        "and generates `.civitai.info` + preview for each found model."
    )

    with gr.Row():
        skip_existing_cb = gr.Checkbox(label="Skip already scanned models", value=True)
        scan_btn         = gr.Button("🔍 Start Scan", variant="primary")
        scan_cancel_btn  = gr.Button("🛑 Stop",        variant="stop")

    scan_status   = gr.Markdown("💤")
    scan_progress = gr.Slider(minimum=0, maximum=1, value=0,
                              label="Progress", interactive=False)
    scan_log      = gr.Textbox(label="Logs", lines=8, interactive=False, max_lines=20)

    gr.Markdown("---")
    gr.Markdown("#### 🆕 Check for updates")

    update_btn    = gr.Button("🔄 Check Updates", variant="primary")
    update_status = gr.Markdown("💤")
    update_log    = gr.Textbox(label="Update Logs", lines=5, interactive=False)
    update_table  = gr.Dataframe(
        headers=["Model", "Local Version", "New Version", "Type", "Path"],
        datatype=["str", "str", "str", "str", "str"],
        interactive=True, label="Available updates",
    )
    dl_update_btn    = gr.Button("⬇️ Download Selected", variant="primary")
    dl_update_result = gr.Markdown("")

    selected_rows = gr.State([])
    scan_timer    = gr.Timer(value=2.0)
    update_timer  = gr.Timer(value=3.0)

    scan_btn.click(
        fn=cb_start_scan,
        inputs=[api_key_input, skip_existing_cb],
        outputs=[scan_status],
    )
    scan_cancel_btn.click(fn=cb_cancel_scan, outputs=[scan_status])
    scan_timer.tick(fn=cb_poll_scan, outputs=[scan_status, scan_log, scan_progress])

    update_btn.click(
        fn=cb_check_updates,
        inputs=[api_key_input],
        outputs=[update_status, update_table],
    )
    update_timer.tick(fn=cb_poll_updates, outputs=[update_status, update_log, update_table])

    def on_table_select(evt: gr.SelectData):
        return [evt.index[0]]

    update_table.select(fn=on_table_select, outputs=[selected_rows])
    dl_update_btn.click(
        fn=cb_download_update,
        inputs=[selected_rows, api_key_input],
        outputs=[dl_update_result],
    )


# ── Enregistrement ───────────────────────────────────────────────────────────

def on_ui_tabs():
    return [(build_ui(), "🐘 CivitAI Helper", "sd_forge_civitai_helper")]


script_callbacks.on_ui_tabs(on_ui_tabs)
