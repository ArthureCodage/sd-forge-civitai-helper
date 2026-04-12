"use strict";

/**
 * CivitAI Helper — Frontend
 * - Boutons sur les cartes Extra Networks
 * - Auto-paste URL CivitAI depuis le presse-papier
 */

(function () {
  const CIVITAI_URL_RE = /(?:https?:\/\/)?civitai\.com\/models\/\d+/;

  // ── Boutons cartes Extra Networks ─────────────────────────────────────────

  function createCardButton(emoji, title, onClick) {
    const btn = document.createElement("button");
    btn.className   = "ch-card-btn";
    btn.textContent = emoji;
    btn.title       = title;
    btn.addEventListener("click", (e) => { e.stopPropagation(); e.preventDefault(); onClick(); });
    return btn;
  }

  function injectCardButtons(card) {
    if (card.querySelector(".ch-card-actions")) return;

    const container = document.createElement("div");
    container.className = "ch-card-actions";

    // 🌐 Ouvrir sur CivitAI
    container.appendChild(createCardButton("🌐", "Ouvrir sur CivitAI", () => {
      const url = card.dataset.chCivitaiUrl;
      if (url) window.open(url, "_blank", "noopener,noreferrer");
    }));

    // 💡 Copier trigger words dans le prompt
    container.appendChild(createCardButton("💡", "Ajouter les trigger words au prompt", () => {
      const words = card.dataset.chTriggerWords;
      if (!words) return;
      const ta = document.querySelector(
        "#txt2img_prompt textarea, #img2img_prompt textarea"
      );
      if (ta) {
        ta.value += (ta.value.trim() ? ", " : "") + words;
        ta.dispatchEvent(new Event("input", { bubbles: true }));
      }
    }));

    card.appendChild(container);
  }

  async function enrichCard(card) {
    const filename = card.dataset.name;
    if (!filename) { injectCardButtons(card); return; }

    try {
      const resp = await fetch(
        `/civitai_helper/card_info?filename=${encodeURIComponent(filename)}`
      );
      if (resp.ok) {
        const info = await resp.json();
        if (info.civitai_url)   card.dataset.chCivitaiUrl   = info.civitai_url;
        if (info.trigger_words) card.dataset.chTriggerWords = info.trigger_words;
      }
    } catch (_) { /* endpoint optionnel */ }

    injectCardButtons(card);
  }

  function scanCards() {
    document.querySelectorAll(".card:not(.ch-enriched)").forEach((card) => {
      card.classList.add("ch-enriched");
      enrichCard(card);
    });
  }

  // ── Auto-paste URL ────────────────────────────────────────────────────────

  async function tryAutoPaste(input) {
    if (!navigator.clipboard?.readText) return;
    try {
      const text = await navigator.clipboard.readText();
      if (CIVITAI_URL_RE.test(text.trim()) && !input.value.trim()) {
        input.value = text.trim();
        input.dispatchEvent(new Event("input", { bubbles: true }));
      }
    } catch (_) { /* permission refusée */ }
  }

  function initDownloadTab() {
    const root = document.querySelector("#civitai_helper_root");
    if (!root) return;
    const ta = root.querySelector("textarea");
    if (ta) ta.addEventListener("focus", () => tryAutoPaste(ta));
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
