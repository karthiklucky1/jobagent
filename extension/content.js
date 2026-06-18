// HirePath Extension — Content Script
// Fills job application forms in the user's browser tab.

// ── Helpers ──────────────────────────────────────────────────────────────────

function fillInput(el, value) {
  if (!el || value === undefined || value === null || value === "") return;
  const proto = el.tagName === "TEXTAREA"
    ? window.HTMLTextAreaElement.prototype
    : window.HTMLInputElement.prototype;
  const nativeSetter = Object.getOwnPropertyDescriptor(proto, "value");
  if (nativeSetter && nativeSetter.set) {
    nativeSetter.set.call(el, value);
  } else {
    el.value = value;
  }
  ["input", "change", "blur"].forEach(ev =>
    el.dispatchEvent(new Event(ev, { bubbles: true }))
  );
}

function selectOption(el, value) {
  if (!el || !value) return;
  const lower = String(value).toLowerCase();
  for (const opt of el.options) {
    if (opt.text.toLowerCase().includes(lower) || opt.value.toLowerCase().includes(lower)) {
      el.value = opt.value;
      el.dispatchEvent(new Event("change", { bubbles: true }));
      return true;
    }
  }
  return false;
}

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

function waitFor(selector, timeout = 8000, root = document) {
  return new Promise((resolve) => {
    const el = root.querySelector(selector);
    if (el) { resolve(el); return; }
    const obs = new MutationObserver(() => {
      const found = root.querySelector(selector);
      if (found) { obs.disconnect(); resolve(found); }
    });
    obs.observe(root.documentElement || root, { childList: true, subtree: true });
    setTimeout(() => { obs.disconnect(); resolve(null); }, timeout);
  });
}

function labelText(el) {
  const label = el.labels?.[0]?.textContent ||
    document.querySelector(`label[for="${el.id}"]`)?.textContent ||
    el.getAttribute("aria-label") ||
    el.getAttribute("placeholder") ||
    el.getAttribute("name") ||
    el.id || "";
  return label.toLowerCase().trim();
}

// ── Greenhouse ────────────────────────────────────────────────────────────────

async function fillGreenhouse(pack) {
  // If on the job description page (no form visible), click Apply
  let root = document;
  const hasForm = document.querySelector(
    "#first_name, input[name='job_application[first_name]'], #resume_upload_or_paste, #resume, form#application_form"
  );
  if (!hasForm) {
    const applyBtns = [
      ...document.querySelectorAll("a[href*='/apply'], button"),
    ].filter(el => /apply/i.test(el.textContent) && el.offsetParent);
    if (applyBtns[0]) {
      applyBtns[0].click();
      await delay(3000);
    }
  }

  // Check for Greenhouse embed iframe (custom career pages)
  for (const fr of document.querySelectorAll("iframe")) {
    try {
      if (fr.src?.includes("greenhouse.io/embed")) {
        root = fr.contentDocument;
        break;
      }
    } catch (e) {}
  }

  // ── Fixed fields (standard Greenhouse IDs) ──
  const fixed = {
    "#first_name": pack.first_name,
    "#last_name": pack.last_name,
    "#email": pack.email,
    "#phone": pack.phone,
    "input[name='job_application[first_name]']": pack.first_name,
    "input[name='job_application[last_name]']": pack.last_name,
    "input[name='job_application[email]']": pack.email,
    "input[name='job_application[phone]']": pack.phone,
    "#job_application_first_name": pack.first_name,
    "#job_application_last_name": pack.last_name,
    "#job_application_email": pack.email,
    "#job_application_phone": pack.phone,
  };
  for (const [sel, val] of Object.entries(fixed)) {
    const el = root.querySelector(sel);
    if (el) fillInput(el, val);
  }

  await delay(300);

  // ── Cover letter (textarea) ──
  for (const sel of [
    "#cover_letter",
    "textarea[name='job_application[cover_letter]']",
    "textarea[id*='cover']",
    "textarea[name*='cover']",
  ]) {
    const el = root.querySelector(sel);
    if (el) { fillInput(el, pack.cover_letter || ""); break; }
  }

  // ── LinkedIn / GitHub / Portfolio / Website ──
  for (const [kw, val] of [
    ["linkedin", pack.linkedin_url],
    ["github", pack.github_url],
    ["portfolio", pack.portfolio_url],
    ["website", pack.portfolio_url],
    ["twitter", ""],
  ]) {
    if (!val) continue;
    const el = root.querySelector(
      `input[id*="${kw}" i], input[placeholder*="${kw}" i], input[name*="${kw}" i]`
    );
    if (el) fillInput(el, val);
  }

  // ── Custom questions via label matching ──
  // Greenhouse custom questions render as <label> + <input|textarea|select>
  // We scan every visible field and match by its label text.
  const allInputs = root.querySelectorAll(
    "input:not([type='hidden']):not([type='file']):not([type='submit']):not([type='checkbox']):not([type='radio']), textarea, select"
  );
  for (const inp of allInputs) {
    const lbl = labelText(inp);
    if (!lbl) continue;

    // Skip already-filled fixed fields
    if (inp.value && inp.value.trim()) continue;

    if (/first.?name|given.?name/i.test(lbl)) fillInput(inp, pack.first_name);
    else if (/last.?name|family.?name|surname/i.test(lbl)) fillInput(inp, pack.last_name);
    else if (/\bemail\b/i.test(lbl)) fillInput(inp, pack.email);
    else if (/phone|mobile|telephone/i.test(lbl)) fillInput(inp, pack.phone);
    else if (/city|location|where.*based|where.*live/i.test(lbl)) fillInput(inp, pack.location || "");
    else if (/linkedin/i.test(lbl)) fillInput(inp, pack.linkedin_url || "");
    else if (/github/i.test(lbl)) fillInput(inp, pack.github_url || "");
    else if (/portfolio|personal.*site|website|personal.*url/i.test(lbl)) fillInput(inp, pack.portfolio_url || "");
    else if (/cover.?letter/i.test(lbl) && inp.tagName === "TEXTAREA") fillInput(inp, pack.cover_letter || "");
    else if (/years?.*(of\s+)?experience|how.*long.*experience/i.test(lbl)) fillInput(inp, String(pack.years_experience || ""));
    else if (/current.*title|job.*title|position/i.test(lbl)) fillInput(inp, pack.current_title || "");
    else if (/salary|compensation|expected.*pay/i.test(lbl)) fillInput(inp, String(pack.salary_min || ""));
    else if (/pronouns/i.test(lbl)) fillInput(inp, "");
    else if (inp.tagName === "SELECT") {
      if (/gender/i.test(lbl)) selectOption(inp, "decline");
      else if (/race|ethnic/i.test(lbl)) selectOption(inp, "decline");
      else if (/veteran/i.test(lbl)) selectOption(inp, "decline");
      else if (/disability/i.test(lbl)) selectOption(inp, "decline");
      else if (/sponsor|visa|authoriz/i.test(lbl)) {
        if (pack.work_authorization) selectOption(inp, pack.work_authorization);
      }
      else if (/country/i.test(lbl)) selectOption(inp, "United States");
    }
  }

  // ── EEOC selects (by ID pattern, catches any missed above) ──
  for (const el of root.querySelectorAll(
    "select[id*='gender'], select[id*='race'], select[id*='ethnicity'], select[id*='veteran'], select[id*='disability']"
  )) {
    if (!el.value || el.value === "") selectOption(el, "decline");
  }

  // ── Work authorization radio/select ──
  if (pack.work_authorization) {
    const authSel = root.querySelector("select[id*='authoriz'], select[name*='authoriz']");
    if (authSel && !authSel.value) selectOption(authSel, pack.work_authorization);
  }

  // Fill essay questions from AI-generated answers in pack
  if (pack.ai_answers) {
    await fillEssayQuestions(root, pack);
  }

  return true;
}

// ── Lever ─────────────────────────────────────────────────────────────────────

async function fillLever(pack) {
  // Navigate to /apply if on description page
  if (!window.location.pathname.includes("/apply")) {
    const applyLink = document.querySelector("a[href*='/apply']");
    if (applyLink) {
      applyLink.click();
      await delay(2500);
    } else {
      const url = window.location.href.replace(/\/?$/, "/apply");
      window.location.href = url;
      return true; // page reloads, content script re-runs
    }
  }

  const map = {
    "input[name='name']": `${pack.first_name} ${pack.last_name}`.trim(),
    "input[name='email']": pack.email,
    "input[name='phone']": pack.phone,
    "input[name='org']": pack.current_title || "",
    "input[name='urls[LinkedIn]']": pack.linkedin_url || "",
    "input[name='urls[GitHub]']": pack.github_url || "",
    "input[name='urls[Portfolio]']": pack.portfolio_url || "",
    "input[name='urls[Other]']": pack.portfolio_url || "",
  };

  for (const [sel, val] of Object.entries(map)) {
    const el = document.querySelector(sel);
    if (el && val) fillInput(el, val);
  }

  const cl = document.querySelector("textarea[name='comments'], textarea[id*='cover']");
  if (cl) fillInput(cl, pack.cover_letter || "");

  if (pack.ai_answers) {
    await fillEssayQuestions(document, pack);
  }

  return true;
}

// ── Ashby ─────────────────────────────────────────────────────────────────────

async function fillAshby(pack) {
  const inputs = document.querySelectorAll("input, textarea");
  for (const inp of inputs) {
    const lbl = labelText(inp);
    if (/first.*(name)?/i.test(lbl)) fillInput(inp, pack.first_name);
    else if (/last.*(name)?/i.test(lbl)) fillInput(inp, pack.last_name);
    else if (/email/i.test(lbl)) fillInput(inp, pack.email);
    else if (/phone|mobile/i.test(lbl)) fillInput(inp, pack.phone);
    else if (/linkedin/i.test(lbl)) fillInput(inp, pack.linkedin_url || "");
    else if (/github/i.test(lbl)) fillInput(inp, pack.github_url || "");
    else if (/portfolio|website/i.test(lbl)) fillInput(inp, pack.portfolio_url || "");
    else if (/cover.*letter/i.test(lbl) && inp.tagName === "TEXTAREA") fillInput(inp, pack.cover_letter || "");
  }

  if (pack.ai_answers) {
    await fillEssayQuestions(document, pack);
  }

  return true;
}

// ── LinkedIn Easy Apply ────────────────────────────────────────────────────────

async function fillLinkedIn(pack) {
  // Wait for job detail to render
  await delay(1500);

  // Click Easy Apply if modal not open yet
  if (!document.querySelector(".jobs-easy-apply-modal, [data-test-modal], .jobs-easy-apply-content")) {
    const applyBtn = document.querySelector(
      "button.jobs-apply-button, .jobs-s-apply button, [data-control-name='jobdetail_topcard_inapply'], button[aria-label*='Easy Apply']"
    );
    if (applyBtn) {
      applyBtn.click();
      await delay(2500);
    } else {
      // No Easy Apply — this is an external application, nothing to fill here
      showBanner("⚠️ This job uses an external application. Open the external form link to use HirePath autofill.");
      return false;
    }
  }

  const modalRoot = document.querySelector(".jobs-easy-apply-modal, [data-test-modal], .jobs-easy-apply-content") || document;
  const inputs = modalRoot.querySelectorAll("input, textarea, select");

  for (const inp of inputs) {
    const lbl = labelText(inp);
    if (/first.*name|given.*name/i.test(lbl)) fillInput(inp, pack.first_name);
    else if (/last.*name|family.*name|surname/i.test(lbl)) fillInput(inp, pack.last_name);
    else if (/\bemail\b/i.test(lbl)) fillInput(inp, pack.email);
    else if (/phone|mobile/i.test(lbl)) fillInput(inp, pack.phone);
    else if (/city|location/i.test(lbl)) fillInput(inp, pack.location || "");
    else if (/linkedin/i.test(lbl)) fillInput(inp, pack.linkedin_url || "");
    else if (/cover.*letter|additional.*info/i.test(lbl) && inp.tagName === "TEXTAREA") fillInput(inp, pack.cover_letter || "");
    else if (/year.*experience|how many year/i.test(lbl)) fillInput(inp, String(pack.years_experience || ""));
    else if (/salary|compensation/i.test(lbl)) fillInput(inp, String(pack.salary_min || ""));
    else if (inp.tagName === "SELECT" && /gender|race|ethnicity|veteran|disability/i.test(lbl)) selectOption(inp, "decline");
    else if (/sponsor/i.test(lbl) && pack.requires_sponsorship === false) {
      const noRadio = inp.closest("fieldset")?.querySelector("input[value*='No' i], input[value*='no' i]");
      if (noRadio) noRadio.click();
    }
  }
  await delay(300);
  return true;
}

// ── Indeed ────────────────────────────────────────────────────────────────────

async function fillIndeed(pack) {
  const inputs = document.querySelectorAll("input, textarea, select");
  for (const inp of inputs) {
    const lbl = labelText(inp);
    if (/first.*name/i.test(lbl)) fillInput(inp, pack.first_name);
    else if (/last.*name/i.test(lbl)) fillInput(inp, pack.last_name);
    else if (/\bemail\b/i.test(lbl)) fillInput(inp, pack.email);
    else if (/phone|mobile/i.test(lbl)) fillInput(inp, pack.phone);
    else if (/city|location/i.test(lbl)) fillInput(inp, pack.location || "");
    else if (/cover.*letter/i.test(lbl) && inp.tagName === "TEXTAREA") fillInput(inp, pack.cover_letter || "");
    else if (/year.*experience/i.test(lbl)) fillInput(inp, String(pack.years_experience || ""));
    else if (inp.tagName === "SELECT" && /gender|race|ethnicity|veteran|disability/i.test(lbl)) selectOption(inp, "decline");
  }
  return true;
}

// ── Workday ───────────────────────────────────────────────────────────────────

async function fillWorkday(pack) {
  await delay(2000); // Workday renders slowly

  // data-automation-id based
  const autos = document.querySelectorAll("[data-automation-id]");
  for (const el of autos) {
    const aid = el.getAttribute("data-automation-id") || "";
    if (/firstName|legalFirstName/i.test(aid)) fillInput(el, pack.first_name);
    else if (/lastName|legalLastName/i.test(aid)) fillInput(el, pack.last_name);
    else if (/email/i.test(aid)) fillInput(el, pack.email);
    else if (/phone/i.test(aid)) fillInput(el, pack.phone);
    else if (/address|city|location/i.test(aid)) fillInput(el, pack.location || "");
    else if (/coverLetter|coverletterText/i.test(aid)) fillInput(el, pack.cover_letter || "");
    else if (/linkedIn/i.test(aid)) fillInput(el, pack.linkedin_url || "");
  }

  // aria-label fallback
  for (const inp of document.querySelectorAll("input[aria-label], textarea[aria-label]")) {
    const lbl = (inp.getAttribute("aria-label") || "").toLowerCase();
    if (/first/i.test(lbl)) fillInput(inp, pack.first_name);
    else if (/last/i.test(lbl)) fillInput(inp, pack.last_name);
    else if (/email/i.test(lbl)) fillInput(inp, pack.email);
    else if (/phone/i.test(lbl)) fillInput(inp, pack.phone);
  }
  return true;
}

// ── Smartrecruiters ───────────────────────────────────────────────────────────

async function fillSmartrecruiters(pack) {
  const map = {
    "#first-name": pack.first_name,
    "#last-name": pack.last_name,
    "#email": pack.email,
    "#phone": pack.phone,
    "input[name='firstName']": pack.first_name,
    "input[name='lastName']": pack.last_name,
    "input[name='email']": pack.email,
    "input[name='phoneNumber']": pack.phone,
  };
  for (const [sel, val] of Object.entries(map)) {
    const el = document.querySelector(sel);
    if (el && val) fillInput(el, val);
  }
  return true;
}

// ── Generic fallback ──────────────────────────────────────────────────────────

async function fillGeneric(pack) {
  const inputs = document.querySelectorAll("input:not([type='hidden']):not([type='submit']):not([type='button']), textarea, select");
  for (const inp of inputs) {
    const lbl = labelText(inp);
    if (/first.*name/i.test(lbl)) fillInput(inp, pack.first_name);
    else if (/last.*name/i.test(lbl)) fillInput(inp, pack.last_name);
    else if (/^(full.?)?name$/i.test(lbl)) fillInput(inp, `${pack.first_name} ${pack.last_name}`);
    else if (/\bemail\b/i.test(lbl)) fillInput(inp, pack.email);
    else if (/phone|mobile|tel/i.test(lbl)) fillInput(inp, pack.phone);
    else if (/city|location/i.test(lbl)) fillInput(inp, pack.location || "");
    else if (/linkedin/i.test(lbl)) fillInput(inp, pack.linkedin_url || "");
    else if (/github/i.test(lbl)) fillInput(inp, pack.github_url || "");
    else if (/portfolio|personal.*site|website/i.test(lbl)) fillInput(inp, pack.portfolio_url || "");
    else if (/cover.*letter/i.test(lbl) && inp.tagName === "TEXTAREA") fillInput(inp, pack.cover_letter || "");
    else if (/year.*experience/i.test(lbl)) fillInput(inp, String(pack.years_experience || ""));
    else if (inp.tagName === "SELECT" && /gender|race|ethnicity|veteran|disability/i.test(lbl)) selectOption(inp, "decline");
  }

  if (pack && pack.ai_answers) await fillEssayQuestions(document, pack);
  return true;
}

// ── Essay answer filling ───────────────────────────────────────────────────────
// Strategy: cached answers (free) first, then on-demand AI per question (~$0.002
// first time only). Never pre-generates — only calls AI for questions that actually
// appear on THIS form.

async function fillEssayQuestions(root, pack) {
  const textareas = Array.from(root.querySelectorAll("textarea")).filter(ta => {
    if (ta.value && ta.value.trim()) return false; // already filled
    const q = labelText(ta);
    return q && q.length >= 8; // skip tiny/unlabeled textareas
  });

  if (!textareas.length) return;

  for (const ta of textareas) {
    const q = labelText(ta);

    // 1. Check pre-cached answers from fill-pack (free, no API call)
    let answer = null;
    if (pack.ai_answers) {
      answer = pack.ai_answers[q];
      if (!answer) {
        for (const [key, val] of Object.entries(pack.ai_answers)) {
          if (keywordOverlap(q, key) > 0.4) { answer = val; break; }
        }
      }
    }

    // 2. If not cached, call /api/answer-question on-demand (~$0.002, cached after)
    if (!answer && pack.hirepath_url && pack.auth_token && pack.app_id) {
      try {
        const res = await fetch(`${pack.hirepath_url}/api/answer-question`, {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${pack.auth_token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ question: q, app_id: pack.app_id }),
        });
        if (res.ok) {
          const data = await res.json();
          answer = data.answer || null;
          if (answer) console.log(`[HirePath] AI answered "${q.slice(0, 60)}…" (cached=${data.cached})`);
        }
      } catch (e) {
        console.warn("[HirePath] answer-question failed:", e.message);
      }
    }

    if (answer) fillInput(ta, answer);

    // 3. Always observe — if user edits, save their answer to memory
    observeAnswer(ta, pack);
  }
}

function keywordOverlap(a, b) {
  const stopWords = new Set(['do','you','want','to','the','a','an','is','are','your','this','that','at','for','in','of','and','or','why','what','how','tell','us','about']);
  const wordsA = a.toLowerCase().split(/\W+/).filter(w => w.length > 2 && !stopWords.has(w));
  const wordsB = b.toLowerCase().split(/\W+/).filter(w => w.length > 2 && !stopWords.has(w));
  if (!wordsA.length || !wordsB.length) return 0;
  const setA = new Set(wordsA);
  const matches = wordsB.filter(w => setA.has(w)).length;
  return matches / Math.max(wordsA.length, wordsB.length);
}

function observeAnswer(ta, pack) {
  if (!pack.hirepath_url || !pack.auth_token) return;
  const origValue = ta.value;
  ta.addEventListener('blur', function handler() {
    const newVal = ta.value.trim();
    if (!newVal || newVal === origValue.trim()) return;
    const q = labelText(ta);
    if (!q) return;
    ta.removeEventListener('blur', handler);
    fetch(`${pack.hirepath_url}/api/save-answer`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${pack.auth_token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ question: q, answer: newVal, app_id: pack.app_id })
    }).catch(() => {});
    console.log('[HirePath] Saved answer for:', q);
  });
}

// ── Router ────────────────────────────────────────────────────────────────────

async function fillForm(fillPack) {
  const host = window.location.hostname;
  let filled = false;
  try {
    if (host.includes("greenhouse.io")) filled = await fillGreenhouse(fillPack);
    else if (host.includes("lever.co")) filled = await fillLever(fillPack);
    else if (host.includes("ashbyhq.com")) filled = await fillAshby(fillPack);
    else if (host.includes("linkedin.com")) filled = await fillLinkedIn(fillPack);
    else if (host.includes("indeed.com")) filled = await fillIndeed(fillPack);
    else if (host.includes("myworkdayjobs.com") || host.includes("workday.com")) filled = await fillWorkday(fillPack);
    else if (host.includes("smartrecruiters.com")) filled = await fillSmartrecruiters(fillPack);
    else filled = await fillGeneric(fillPack);

    showBanner(
      filled
        ? "✅ HirePath filled your form! Review every field, then click Submit."
        : "⚠️ HirePath: some fields may need manual entry on this site."
    );
  } catch (e) {
    console.error("[HirePath]", e);
    showBanner("❌ HirePath error: " + e.message + " — try filling manually.");
  }
}

// ── Banner ────────────────────────────────────────────────────────────────────

function showBanner(msg) {
  const existing = document.getElementById("hirepath-ext-banner");
  if (existing) existing.remove();

  const banner = document.createElement("div");
  banner.id = "hirepath-ext-banner";
  banner.style.cssText = [
    "position:fixed", "top:0", "left:0", "right:0", "z-index:2147483647",
    "padding:12px 20px", "display:flex", "align-items:center", "gap:12px",
    "background:linear-gradient(135deg,#4f46e5,#7c3aed)",
    "color:white", "font:bold 13px/1.4 system-ui,sans-serif",
    "box-shadow:0 4px 24px rgba(0,0,0,0.35)",
  ].join(";");

  const text = document.createElement("span");
  text.style.flex = "1";
  text.textContent = msg;

  const close = document.createElement("button");
  close.textContent = "✕";
  close.style.cssText = "background:none;border:none;color:white;cursor:pointer;font-size:18px;line-height:1;padding:0 4px;opacity:0.8;";
  close.addEventListener("click", () => banner.remove());

  banner.appendChild(text);
  banner.appendChild(close);
  document.body.prepend(banner);
  document.body.style.paddingTop = "46px";
  setTimeout(() => { banner.remove(); document.body.style.paddingTop = ""; }, 12000);
}

// ── Extension context guard ───────────────────────────────────────────────────
// If the extension is reloaded/updated while this tab is open, the content
// script becomes orphaned and all chrome.* calls throw "Extension context
// invalidated". Wrap every chrome API call with this helper.
function chromeCall(fn) {
  try { return fn(); } catch (e) {
    if (e?.message?.includes("Extension context invalidated")) {
      console.warn("[HirePath] Extension was reloaded — please refresh this tab.");
    } else { console.error("[HirePath]", e); }
  }
}

// ── Message listener (from background.js) ────────────────────────────────────

chromeCall(() => chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "DO_FILL") {
    console.log("[HirePath] DO_FILL received, starting fill for:", msg.fillPack?.job_title);
    fillForm(msg.fillPack).then(() => sendResponse({ ok: true }));
    return true;
  }
  if (msg.type === "PING") {
    sendResponse({ ok: true });
  }
}));

// ── Bridge: dashboard postMessage → background.js → new tab ──────────────────
window.addEventListener("message", (e) => {
  if (e.data?.type === "HIREPATH_LOAD_PACK" && e.data?.pack) {
    console.log("[HirePath] Received HIREPATH_LOAD_PACK from dashboard, forwarding to background");
    // Send ACK only AFTER background confirms it received the pack.
    // If context is invalidated or background doesn't respond, no ACK is sent
    // and the dashboard falls back to window.open() automatically.
    try {
      chrome.runtime.sendMessage({ type: "OPEN_AND_FILL", payload: e.data.pack }, (res) => {
        if (chrome.runtime.lastError) {
          console.warn("[HirePath] Background error:", chrome.runtime.lastError.message);
          // No ACK → dashboard will window.open() as fallback
        } else {
          console.log("[HirePath] Background opened tab, sending ACK to dashboard");
          window.postMessage({ type: "HIREPATH_EXT_ACK" }, "*");
        }
      });
    } catch (err) {
      // Extension context invalidated (extension was reloaded while this tab was open).
      // No ACK sent → dashboard falls back to window.open().
      console.warn("[HirePath] Extension context invalidated — refresh this tab to restore autofill");
    }
  }
});

// ── Auto-fill on page load if background already set the flag ─────────────────
chromeCall(() => chrome.storage.local.get(["hirepath_fill_pack", "hirepath_auto_fill"], (data) => {
  if (data.hirepath_auto_fill && data.hirepath_fill_pack) {
    console.log("[HirePath] Auto-fill flag found in storage, filling after delay");
    chrome.storage.local.set({ hirepath_auto_fill: false });
    setTimeout(() => fillForm(data.hirepath_fill_pack), 2500);
  } else {
    console.log("[HirePath] Content script loaded on", window.location.hostname, "— no pending auto-fill");
  }
}));
