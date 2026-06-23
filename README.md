# 🐘 SD Forge — CivitAI Helper

A **Stable Diffusion Forge** extension for downloading, scanning, and updating your models directly from CivitAI.

> [!NOTE]
> **Domain Update**: This extension has been updated to use **civitai.red** as the main API base and download domain. This resolves connection/DNS blocking issues and ensures both SFW and NSFW content are accessible out of the box.

## Installation

```bash
cd <SD_Forge_root>/extensions/
git clone <repo_url> sd-forge-civitai-helper
# Restart Forge — install.py runs automatically
```

Or via the UI: **Extensions → Install from URL**.

## Features

| Tab | Functionality |
|---|---|
| ⬇️ Download | Download via URL/ID, version/file selection, resume, SHA256 verification, preview |
| 🔍 Search | Keyword search, type filtering, pagination |
| 📦 Batch | Batch download multiple models, auto-selects latest version & best `.safetensors` file |
| 🔄 Scan & Update | Local SHA256 scanning, `.civitai.info` generation, detects newer versions |

## API Key

Optional for public models.
Required for restricted/NSFW content.

Get your key at: https://civitai.red/user/account

You can also set it as an environment variable:
```bash
export CIVITAI_API_KEY=your_key
```

## Generated Files

For each scanned model, the extension creates:
- `model_name.civitai.info` — JSON metadata
- `model_name.preview.jpg` — preview image
