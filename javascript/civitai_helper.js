"use strict";

/**
 * CivitAI Helper — Frontend
 * - Extra Networks Card Buttons (Open CivitAI, Insert Trigger Words)
 * - Auto-detect active prompt (txt2img vs img2img)
 * - Client-side caching and debounced card enrichment
 * - Auto-paste CivitAI URLs from clipboard
 */

(function () {
  const CIVITAI_URL_RE = /(?:https?:\/\/)?(?:[a-zA-Z0-9-]+\.)?civitai\.(?:com|red)\/(?:models\/\d+|model-versions\/\d+|api\/download\/models\/\d+)/i;
  const cardCache = new Map();
  let scanTimeout = null;

  // ── Prompt Detection ───────────────────────────────────────────────────────

  function getActivePromptTextarea() {
    const img2imgTab = document.getElementById("tab_img2img");
    const isImg2ImgActive = img2imgTab && (
      img2imgTab.style.display !== "none" &&
      window.getComputedStyle(img2imgTab).display !== "none"
    );
    if (isImg2ImgActive) {
      const ta = document.querySelector("#img2img_prompt textarea");
      if (ta) return ta;
    }
    return document.querySelector("#txt2img_prompt textarea");
  }

  // ── Extra Networks Card Buttons ───────────────────────────────────────────

  function createCardButton(emoji, title, onClick) {
    const btn = document.createElement("button");
    btn.className   = "ch-card-btn";
    btn.textContent = emoji;
    btn.title       = title;
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      e.preventDefault();
      onClick(btn);
    });
    return btn;
  }

  function injectCardButtons(card) {
    if (card.querySelector(".ch-card-actions")) return;

    const container = document.createElement("div");
    container.className = "ch-card-actions";

    // 🌐 Open on CivitAI
    container.appendChild(createCardButton("🌐", "Open on CivitAI", (btn) => {
      const url = card.dataset.chCivitaiUrl;
      if (url) {
        window.open(url, "_blank", "noopener,noreferrer");
      } else {
        btn.title = "No CivitAI URL found";
      }
    }));

    // 💡 Add trigger words to prompt
    container.appendChild(createCardButton("💡", "Insert trigger words into prompt", (btn) => {
      const words = card.dataset.chTriggerWords;
      if (!words) {
        btn.title = "No trigger words available";
        return;
      }
      const ta = getActivePromptTextarea();
      if (ta) {
        const current = ta.value.trim();
        ta.value = current ? `${current}, ${words}` : words;
        ta.dispatchEvent(new Event("input", { bubbles: true }));
        ta.dispatchEvent(new Event("change", { bubbles: true }));

        const original = btn.textContent;
        btn.textContent = "✓";
        setTimeout(() => { btn.textContent = original; }, 1200);
      }
    }));

    card.appendChild(container);
  }

  async function enrichCard(card) {
    const filename = card.dataset.name;
    if (!filename) {
      injectCardButtons(card);
      return;
    }

    if (cardCache.has(filename)) {
      const info = cardCache.get(filename);
      if (info.civitai_url)   card.dataset.chCivitaiUrl   = info.civitai_url;
      if (info.trigger_words) card.dataset.chTriggerWords = info.trigger_words;
      injectCardButtons(card);
      return;
    }

    try {
      const resp = await fetch(
        `/civitai_helper/card_info?filename=${encodeURIComponent(filename)}`
      );
      if (resp.ok) {
        const info = await resp.json();
        cardCache.set(filename, info);
        if (info.civitai_url)   card.dataset.chCivitaiUrl   = info.civitai_url;
        if (info.trigger_words) card.dataset.chTriggerWords = info.trigger_words;
      }
    } catch (_) {
      // Optional endpoint fallback
    }

    injectCardButtons(card);
  }

  function scanCards() {
    clearTimeout(scanTimeout);
    scanTimeout = setTimeout(() => {
      document.querySelectorAll(".card:not(.ch-enriched)").forEach((card) => {
        card.classList.add("ch-enriched");
        enrichCard(card);
      });
    }, 150);
  }

  // ── Auto-paste URL ────────────────────────────────────────────────────────

  async function tryAutoPaste(input) {
    if (!navigator.clipboard?.readText) return;
    try {
      const text = await navigator.clipboard.readText();
      if (CIVITAI_URL_RE.test(text.trim()) && !input.value.trim()) {
        input.value = text.trim();
        input.dispatchEvent(new Event("input", { bubbles: true }));
        input.dispatchEvent(new Event("change", { bubbles: true }));
      }
    } catch (_) {
      // Clipboard permission denied
    }
  }

  function initDownloadTab() {
    const root = document.querySelector("#civitai_helper_root");
    if (!root) return;
    const inputs = root.querySelectorAll("input[type='text'], textarea");
    inputs.forEach((inp) => {
      inp.addEventListener("focus", () => tryAutoPaste(inp));
    });
  }

  // ── Init ──────────────────────────────────────────────────────────────────

  function init() {
    initDownloadTab();
    scanCards();

    const observer = new MutationObserver(() => scanCards());
    observer.observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
