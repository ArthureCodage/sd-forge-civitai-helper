# 🐘 SD Forge — CivitAI Helper

<div align="center">

[![CivitAI API](https://img.shields.io/badge/CivitAI-API_v1-3b82f6?style=for-the-badge&logo=civitai)](https://civitai.com)
[![SD Forge Neo](https://img.shields.io/badge/SD_Forge_Neo-Compatible-f97316?style=for-the-badge&logo=python)](https://github.com/Haoming02/sd-webui-forge-classic)
[![A1111 WebUI](https://img.shields.io/badge/A1111_WebUI-Supported-8b5cf6?style=for-the-badge)](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
[![License MIT](https://img.shields.io/badge/License-MIT-10b981?style=for-the-badge)](LICENSE)

**The ultimate model, LoRA, and preview manager for Stable Diffusion Forge Neo & AUTOMATIC1111 WebUI.**

[Features](#-features) • [Installation](#-installation) • [Usage Guide](#-usage-guide) • [API Key Setup](#-api-key-setup) • [FAQ & Troubleshooting](#-faq--troubleshooting)

</div>

---

## 🌟 Overview

**SD Forge CivitAI Helper** is a powerful extension designed to streamline downloading, organizing, and updating Stable Diffusion models (Checkpoints, LoRAs, ControlNets, VAEs, Embeddings) directly from **CivitAI**.

Whether you're running **Flux.1**, **SDXL**, **Pony Diffusion**, **Illustrious**, or **SD 1.5**, CivitAI Helper automatically fetches official metadata (`.civitai.info`), model thumbnails (`.preview.png` / `.png`), and trigger words (`.txt`) for an effortless Extra Networks card view.

---

## ✨ Features

### 📸 Instant Thumbnail & Preview Downloader
- Automatically downloads official CivitAI preview images for every model or LoRA.
- Saves thumbnails in both `model.preview.png` and `model.png` formats for maximum compatibility with **SD Forge Neo** and **A1111 WebUI** extra networks views.
- Support for downloading **1 to 5 preview images** per model.

### 📝 Auto Trigger Words Generator (`.txt`)
- Automatically extracts trained trigger words from CivitAI metadata and saves them in a `.txt` file alongside your `.safetensors` file.
- SD Forge Neo reads these trigger word files natively when clicking LoRA cards.

### 📦 Batch Model Downloader
- Paste multiple CivitAI URLs (one per line) for unattended batch downloading.
- Automatically selects the latest model version and the optimal `.safetensors` file format.
- Progress tracking with resume support and SHA256 integrity verification.

### 📸 Fast "Missing Previews" Scanner
- Have models on disk without thumbnails? The new **"Download Missing Previews"** button scans your local model folders and fetches missing thumbnails in seconds without requiring heavy SHA256 re-hashing!

### 🔍 In-App CivitAI Model Search
- Search CivitAI models directly inside your WebUI/Forge interface by keyword, model type, rating, or base model (SDXL, Flux, Pony, SD 1.5).
- Live cover image previews and safety rating tags (`SFW`, `Soft`, `Mature`, `X`).

---

## 🛠️ Installation

### Method 1: Via SD Forge / WebUI Interface (Recommended)
1. Open **SD Forge Neo** or **AUTOMATIC1111 WebUI**.
2. Go to the **Extensions** tab ➔ **Install from URL**.
3. Paste the repository URL:
   ```text
   https://github.com/ArthureCodage/sd-forge-civitai-helper.git
   ```
4. Click **Install**, then click **Apply and restart UI**.

### Method 2: Via Git Clone
```bash
# Navigate to your extensions folder
cd sd-forge-neo/extensions/

# Clone the repository
git clone https://github.com/ArthureCodage/sd-forge-civitai-helper.git

# Restart SD Forge Neo
```

---

## 🚀 Usage Guide

### 1. Downloading a Model (`⬇️ Download` Tab)
1. Paste any CivitAI URL or Model ID (e.g. `https://civitai.com/models/12345`).
2. Click **🔍 Fetch Info** to load the model's cover thumbnail, base model badge, and version options.
3. Select your desired version and `.safetensors` file.
4. Click **⬇️ Download**. The model, preview image, metadata, and trigger words `.txt` file will be placed in the appropriate directory (e.g., `models/Lora/` or `models/Stable-diffusion/`).

### 2. Downloading Missing Thumbnails (`🔄 Scan & Update` Tab)
1. Open the **🔄 Scan & Update** tab.
2. Click **📸 Download Missing Previews**.
3. The extension checks your local models, finds those missing preview thumbnails, and downloads them directly from CivitAI.

### 3. Batch Downloading (`📦 Batch` Tab)
1. Paste multiple CivitAI model URLs into the text box (one per line).
2. Click **🔍 Analyze URLs**, then click **⬇️ Download All**.

---

## 🔑 API Key Setup

An API key is **optional** for public SFW models, but **required** for downloading restricted or age-restricted (NSFW) models.

1. Generate an API Key in your [CivitAI Account Settings](https://civitai.com/user/account).
2. Enter your API Key in the **CivitAI API Key** field at the top of the extension UI.
3. *Optional:* Save your API Key globally in **Settings ➔ CivitAI Helper** so it persists across sessions.

---

## ❓ FAQ & Troubleshooting

| Issue | Solution |
|---|---|
| **Thumbnails not showing up on cards?** | Click the **Refresh** button on the Extra Networks panel in SD Forge Neo, or run **📸 Download Missing Previews**. |
| **Download fails with HTTP 401?** | Ensure your CivitAI API Key is entered correctly. Restricted models require a valid API key. |
| **Where are LoRAs saved?** | By default, models are automatically routed: LoRAs ➔ `models/Lora/`, Checkpoints ➔ `models/Stable-diffusion/`, VAEs ➔ `models/VAE/`. |
| **How to download multiple sample images?** | Adjust the preview image limit slider in **Settings ➔ CivitAI Helper**. |

---

## 🏷️ Keywords & Tags

`stable-diffusion-forge` • `sd-forge-neo` • `civitai-helper` • `lora-preview-downloader` • `flux-lora-downloader` • `sdxl-model-manager` • `automatic1111-extension` • `civitai-api-v1`

---

## 📄 License

Distributed under the [MIT License](LICENSE). Open-source and free to modify.
