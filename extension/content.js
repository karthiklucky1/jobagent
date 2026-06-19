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

// ── Avature ─────────────────────────────────────────────────────────────────
// Used by Accenture, plus many large enterprises (Siemens, KPMG, etc.).
// Avature renders fields inside .avature-field / data-field wrappers and uses
// custom dropdown widgets (not native <select>).

// Detect Avature even on custom domains (Accenture, etc.) via page markers.
function isAvaturePage() {
  return !!(
    document.querySelector("[class*='avature'], [id*='avature'], [data-avature], form[action*='avature']") ||
    /avature/i.test(document.documentElement.innerHTML.slice(0, 5000))
  );
}

async function fillAvature(pack) {
  await delay(1500); // Avature hydrates fields after load

  // 1. Standard inputs — Avature uses id/name like "fieldFirstName", "txtEmail",
  //    or data-field attributes. Cover both plus label matching.
  const inputs = document.querySelectorAll(
    "input:not([type='hidden']):not([type='submit']):not([type='button']):not([type='checkbox']):not([type='radio']), textarea"
  );
  for (const inp of inputs) {
    if (inp.value && inp.value.trim()) continue;
    const ctx = (
      labelText(inp) + " " +
      (inp.getAttribute("name") || "") + " " +
      (inp.id || "") + " " +
      (inp.getAttribute("data-field") || "")
    ).toLowerCase();

    if (/first.?name|given.?name|forename/.test(ctx)) fillInput(inp, pack.first_name);
    else if (/last.?name|family.?name|surname/.test(ctx)) fillInput(inp, pack.last_name);
    else if (/full.?name|^name$|candidate.?name/.test(ctx)) fillInput(inp, `${pack.first_name} ${pack.last_name}`);
    else if (/\bemail\b|e-?mail/.test(ctx)) fillInput(inp, pack.email);
    else if (/phone|mobile|telephone|contact.?number/.test(ctx)) fillInput(inp, pack.phone);
    else if (/city|town|location|address|where.*based/.test(ctx)) fillInput(inp, pack.location || "");
    else if (/linkedin/.test(ctx)) fillInput(inp, pack.linkedin_url || "");
    else if (/github/.test(ctx)) fillInput(inp, pack.github_url || "");
    else if (/portfolio|personal.*site|website|personal.*url/.test(ctx)) fillInput(inp, pack.portfolio_url || "");
    else if (/cover.?letter/.test(ctx) && inp.tagName === "TEXTAREA") fillInput(inp, pack.cover_letter || "");
    else if (/years?.*experience|experience.*years?/.test(ctx)) fillInput(inp, String(pack.years_experience || ""));
    else if (/current.*title|job.*title|current.*position/.test(ctx)) fillInput(inp, pack.current_title || "");
    else if (/salary|compensation|expected.*pay/.test(ctx)) fillInput(inp, String(pack.salary_min || ""));
  }

  // 2. Native selects (EEOC / country / authorization)
  for (const sel of document.querySelectorAll("select")) {
    if (sel.value && sel.value !== "") continue;
    const ctx = (labelText(sel) + " " + (sel.getAttribute("name") || "") + " " + (sel.id || "")).toLowerCase();
    if (/gender|race|ethnic|veteran|disability/.test(ctx)) selectOption(sel, "decline");
    else if (/country/.test(ctx)) selectOption(sel, "United States");
    else if (/sponsor|visa|authoriz/.test(ctx) && pack.work_authorization) selectOption(sel, pack.work_authorization);
  }

  // 3. Avature custom dropdowns (div-based, role=combobox/listbox)
  for (const combo of document.querySelectorAll("[role='combobox'], .avature-select, .dropdown-trigger")) {
    const ctx = (labelText(combo) + " " + (combo.getAttribute("aria-label") || "")).toLowerCase();
    let want = null;
    if (/gender|race|ethnic|veteran|disability/.test(ctx)) want = "decline";
    else if (/country/.test(ctx)) want = "United States";
    if (!want) continue;
    try {
      combo.click();
      await delay(300);
      const opts = document.querySelectorAll("[role='option'], li[role='option'], .dropdown-option");
      for (const o of opts) {
        if (o.textContent.toLowerCase().includes(want === "decline" ? "decline" : want.toLowerCase())) {
          o.click();
          break;
        }
      }
      await delay(150);
    } catch (e) { /* leave for user */ }
  }

  // 4. Essay questions
  await fillEssayQuestions(document, pack);
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

// ── API helper ──────────────────────────────────────────────────────────────────
// Routes through the background service worker to bypass CORS (content-script
// fetches run in the page origin and get blocked).
function apiFetch(url, method, token, body) {
  return new Promise((resolve) => {
    try {
      chrome.runtime.sendMessage(
        { type: "API_FETCH", payload: { url, method, token, body } },
        (res) => {
          if (chrome.runtime.lastError) {
            resolve({ ok: false, error: chrome.runtime.lastError.message });
          } else {
            resolve(res || { ok: false, error: "no response" });
          }
        }
      );
    } catch (e) {
      resolve({ ok: false, error: e.message });
    }
  });
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
      const res = await apiFetch(
        `${pack.hirepath_url}/api/answer-question`,
        "POST",
        pack.auth_token,
        { question: q, app_id: pack.app_id }
      );
      if (res.ok && res.data) {
        answer = res.data.answer || null;
        if (answer) console.log(`[HirePath] AI answered "${q.slice(0, 60)}…" (cached=${res.data.cached})`);
      } else {
        console.warn("[HirePath] answer-question failed:", res.error || res.status);
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
    apiFetch(`${pack.hirepath_url}/api/save-answer`, 'POST', pack.auth_token,
      { question: q, answer: newVal, app_id: pack.app_id });
    console.log('[HirePath] Saved answer for:', q);
  });
}

// ── Resume file attachment ──────────────────────────────────────────────────────
// Fetches the tailored resume .docx (base64) via the background worker and sets
// it on any empty file inputs using a synthetic DataTransfer.
async function attachResume(root, pack) {
  if (!pack.hirepath_url || !pack.auth_token || !pack.app_id) return;
  const fileInputs = Array.from(root.querySelectorAll('input[type="file"]')).filter(fi => {
    if (fi.files && fi.files.length) return false; // already has a file
    const ctx = (labelText(fi) + " " + (fi.name || "") + " " + (fi.id || "")).toLowerCase();
    // Only target resume/CV inputs; skip cover-letter/other doc uploads
    return /resume|cv|résumé/.test(ctx) || /resume/.test(ctx);
  });
  if (!fileInputs.length) return;

  const res = await apiFetch(
    `${pack.hirepath_url}/api/fill-pack/${pack.app_id}/resume`,
    "GET", pack.auth_token, null
  );
  if (!res.ok || !res.data || !res.data.base64) {
    console.warn("[HirePath] resume fetch failed:", res.error || res.status);
    return;
  }

  const { filename, mime, base64 } = res.data;
  let file;
  try {
    const bin = atob(base64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    file = new File([bytes], filename, { type: mime });
  } catch (e) {
    console.warn("[HirePath] resume decode failed:", e.message);
    return;
  }

  for (const fi of fileInputs) {
    try {
      const dt = new DataTransfer();
      dt.items.add(file);
      fi.files = dt.files;
      fi.dispatchEvent(new Event("input", { bubbles: true }));
      fi.dispatchEvent(new Event("change", { bubbles: true }));
      console.log("[HirePath] Attached tailored resume:", filename);
    } catch (e) {
      console.warn("[HirePath] Could not set file input:", e.message);
    }
  }
}

// ── Router ────────────────────────────────────────────────────────────────────
// Copilot mode: fill what we can, highlight gaps in yellow, wait for user
// to click Next, then repeat on the next page. User interaction at every step.

let _copilotActive = false;
let _copilotPack = null;

async function fillForm(fillPack) {
  _copilotPack = fillPack;
  _copilotActive = true;
  _lastUrl = location.href;

  // Persist the session so copilot survives page navigations (e.g. clicking
  // Apply sends you to a new URL / domain). Resumed by the load handler below.
  chromeCall(() => chrome.storage.local.set({
    hirepath_copilot_pack: fillPack,
    hirepath_copilot_ts: Date.now(),
  }));

  // Decide: are we on a real application FORM, or just a job description page?
  // A description page (e.g. accenture.com/.../jobdetails) has search boxes and
  // cookie inputs but no application fields — so we must find & click Apply first.
  if (!hasApplicationForm()) {
    findAndClickApply(fillPack);
    return;
  }
  await runCopilotStep();
}

// True only if the page looks like an actual application form (not a JD page).
function hasApplicationForm() {
  // Strong signals: email input, resume file input, or name fields
  if (document.querySelector("input[type='email'], input[type='file']")) return true;
  const named = document.querySelectorAll(
    "input[name*='first' i], input[name*='last' i], input[name*='name' i]," +
    "input[id*='first' i], input[id*='last' i], textarea"
  );
  if (named.length >= 2) return true;
  // Fallback: many visible text inputs usually means a form
  const visibleText = Array.from(
    document.querySelectorAll("input[type='text'], input:not([type])")
  ).filter(el => el.offsetParent !== null);
  return visibleText.length >= 3;
}

// ── Step runner ───────────────────────────────────────────────────────────────

async function runCopilotStep() {
  if (!_copilotActive || !_copilotPack) return;
  const pack = _copilotPack;

  // Detect login wall or CAPTCHA — pause and guide user
  if (isLoginWall()) {
    showOverlay('🔐 Please log in to continue.<br><small>I\'ll auto-resume once you\'re in.</small>', [], false);
    watchForFormAppearance();
    return;
  }
  if (isCaptcha()) {
    showOverlay('🤖 Please solve the CAPTCHA.<br><small>I\'ll continue automatically after.</small>', [], false);
    watchForFormAppearance();
    return;
  }

  // Run the fill
  const result = await fillCurrentPage(pack);
  showStepOverlay(result, pack);

  // Attach resume (don't block the overlay showing)
  attachResume(document, pack).catch(() => {});

  // Watch for user advancing to next page
  watchForPageAdvance(pack);
}

// ── Fill current page ─────────────────────────────────────────────────────────

async function fillCurrentPage(pack) {
  const host = window.location.hostname;
  let platformFilled = false;
  try {
    if (host.includes('greenhouse.io'))       platformFilled = await fillGreenhouse(pack);
    else if (host.includes('lever.co'))       platformFilled = await fillLever(pack);
    else if (host.includes('ashbyhq.com'))    platformFilled = await fillAshby(pack);
    else if (host.includes('linkedin.com'))   platformFilled = await fillLinkedIn(pack);
    else if (host.includes('indeed.com'))     platformFilled = await fillIndeed(pack);
    else if (host.includes('myworkdayjobs.com') || host.includes('workday.com'))
                                              platformFilled = await fillWorkday(pack);
    else if (host.includes('smartrecruiters.com')) platformFilled = await fillSmartrecruiters(pack);
    else if (host.includes('avature.net') || isAvaturePage()) platformFilled = await fillAvature(pack);
    else                                      platformFilled = await fillGeneric(pack);
  } catch (e) {
    console.warn('[HirePath] platform fill error:', e.message);
  }

  // Fill essay questions via AI
  await fillEssayQuestions(document, pack);

  // Audit all form fields — classify as filled / unfilled / unknown
  const allInputs = Array.from(document.querySelectorAll(
    'input:not([type="hidden"]):not([type="submit"]):not([type="button"]):not([type="image"]),' +
    'textarea, select'
  )).filter(el => el.offsetParent !== null); // visible only

  let filled = 0, needUser = 0, skipped = 0;
  const needUserEls = [];

  for (const el of allInputs) {
    if (el.type === 'file') { skipped++; continue; }
    if (el.type === 'checkbox' || el.type === 'radio') { skipped++; continue; }
    const val = el.value ? el.value.trim() : '';
    if (val) {
      filled++;
      highlightField(el, 'green');
    } else {
      needUser++;
      needUserEls.push(el);
      highlightField(el, 'yellow');
    }
  }

  return { filled, needUser, needUserEls, platformFilled };
}

// ── Field highlighting ────────────────────────────────────────────────────────

function highlightField(el, color) {
  el.style.transition = 'box-shadow 0.3s ease, border-color 0.3s ease';
  if (color === 'green') {
    el.style.boxShadow = '0 0 0 2px rgba(16,185,129,0.5)';
    el.style.borderColor = 'rgba(16,185,129,0.7)';
  } else {
    el.style.boxShadow = '0 0 0 2px rgba(245,158,11,0.6)';
    el.style.borderColor = 'rgba(245,158,11,0.8)';
    // Add pulsing attention ring
    el.dataset.hirepath = 'needs-fill';
    el.addEventListener('input', () => {
      if (el.value.trim()) highlightField(el, 'green');
    }, { once: true });
  }
}

function clearHighlights() {
  document.querySelectorAll('[data-hirepath]').forEach(el => {
    el.style.boxShadow = '';
    el.style.borderColor = '';
    delete el.dataset.hirepath;
  });
}

// ── Step overlay ──────────────────────────────────────────────────────────────

function showStepOverlay(result, pack) {
  const { filled, needUser } = result;
  const stepText = detectStepText();

  let statusHtml = '';
  if (filled > 0) statusHtml += `<span style="color:#10b981;font-weight:700">✅ ${filled} filled</span>`;
  if (needUser > 0) statusHtml += `${filled > 0 ? ' &nbsp;·&nbsp; ' : ''}<span style="color:#f59e0b;font-weight:700">⚠️ ${needUser} need you</span>`;
  if (filled === 0 && needUser === 0) statusHtml = '<span style="color:#94a3b8">No fields found on this page</span>';

  const stepLabel = stepText ? `<div style="color:#94a3b8;font-size:11px;margin-bottom:6px">${stepText}</div>` : '';

  const instructions = needUser > 0
    ? `<div style="font-size:11px;color:#cbd5e1;margin-top:6px">Fill the <span style="color:#f59e0b;font-weight:600">yellow fields</span> above, then click Next.</div>`
    : `<div style="font-size:11px;color:#cbd5e1;margin-top:6px">All fields filled! Review and click Next.</div>`;

  showOverlay(`${stepLabel}${statusHtml}${instructions}`, [], true);
}

// ── Pause overlay (login / captcha) ──────────────────────────────────────────

function showOverlay(html, _unused, dismissable) {
  removeOverlay();
  const div = document.createElement('div');
  div.id = 'hp-copilot-overlay';
  div.innerHTML = `
    <div style="display:flex;align-items:center;gap:10px">
      <span style="font-size:20px;flex-shrink:0">⚡</span>
      <div style="flex:1;line-height:1.5">${html}</div>
      ${dismissable ? `<button onclick="document.getElementById('hp-copilot-overlay').remove()" style="background:none;border:none;color:#64748b;cursor:pointer;font-size:16px;padding:0 4px;line-height:1" title="Dismiss">✕</button>` : ''}
    </div>`;
  Object.assign(div.style, {
    position: 'fixed', bottom: '20px', right: '20px', zIndex: '2147483647',
    background: 'linear-gradient(135deg,rgba(15,23,42,0.97),rgba(30,41,59,0.97))',
    border: '1px solid rgba(99,102,241,0.4)', borderRadius: '16px',
    padding: '14px 16px', maxWidth: '320px', minWidth: '220px',
    boxShadow: '0 8px 32px rgba(0,0,0,0.5), 0 0 0 1px rgba(99,102,241,0.1)',
    fontFamily: 'system-ui,sans-serif', fontSize: '13px', color: '#e2e8f0',
    backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)',
    transition: 'opacity 0.3s ease',
  });
  document.body.appendChild(div);
  // Auto-hide after 30s if dismissable
  if (dismissable) setTimeout(() => div?.remove(), 30000);
}

function removeOverlay() {
  document.getElementById('hp-copilot-overlay')?.remove();
}

// ── Detect step text (e.g. "Step 2 of 4") ────────────────────────────────────

function detectStepText() {
  // Look for common step indicators
  const patterns = [
    /step\s+\d+\s+of\s+\d+/i,
    /\d+\s*\/\s*\d+/,
    /page\s+\d+\s+of\s+\d+/i,
    /\d+\s+of\s+\d+\s+steps/i,
  ];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    const t = node.textContent.trim();
    for (const p of patterns) {
      if (p.test(t)) return t.slice(0, 40);
    }
  }
  return null;
}

// ── Login / CAPTCHA detection ─────────────────────────────────────────────────

function isLoginWall() {
  const body = document.body.innerText.toLowerCase();
  const hasLoginForm = !!document.querySelector('input[type="password"]');
  const hasLoginText = /sign in|log in|login|create an? account|register to apply/.test(body);
  const noRealForm = !document.querySelector('input[type="text"], input[type="email"], textarea');
  return hasLoginForm || (hasLoginText && noRealForm);
}

function isCaptcha() {
  return !!(
    document.querySelector('iframe[src*="recaptcha"], iframe[src*="hcaptcha"], .cf-turnstile, #challenge-stage') ||
    document.body.innerText.toLowerCase().includes('prove you are human')
  );
}

// ── Watch for page advance (user clicked Next) ────────────────────────────────

let _advanceObserver = null;
let _lastUrl = location.href;

function watchForPageAdvance(pack) {
  if (_advanceObserver) { _advanceObserver.disconnect(); _advanceObserver = null; }

  // Strategy 1: URL change (standard navigations)
  const urlPoll = setInterval(() => {
    if (!_copilotActive) { clearInterval(urlPoll); return; }
    if (location.href !== _lastUrl) {
      _lastUrl = location.href;
      clearInterval(urlPoll);
      if (_advanceObserver) { _advanceObserver.disconnect(); _advanceObserver = null; }
      clearHighlights();
      removeOverlay();
      setTimeout(() => runCopilotStep(), 1500); // wait for new page content
    }
  }, 400);

  // Strategy 2: Major DOM change on same URL (SPA / Workday / React)
  let mutationTimer = null;
  const snapshot = document.querySelectorAll('input, textarea, select').length;
  _advanceObserver = new MutationObserver(() => {
    const now = document.querySelectorAll('input, textarea, select').length;
    if (now !== snapshot && Math.abs(now - snapshot) >= 2) {
      clearTimeout(mutationTimer);
      mutationTimer = setTimeout(() => {
        if (!_copilotActive) return;
        if (location.href === _lastUrl) { // only if URL didn't change (handled above)
          clearHighlights();
          removeOverlay();
          runCopilotStep();
        }
      }, 800);
    }
  });
  _advanceObserver.observe(document.body, { childList: true, subtree: true });
}

// ── Watch for form appearance (after login / captcha) ─────────────────────────

function watchForFormAppearance() {
  const check = setInterval(() => {
    if (!_copilotActive) { clearInterval(check); return; }
    if (!isLoginWall() && !isCaptcha()) {
      clearInterval(check);
      removeOverlay();
      setTimeout(() => runCopilotStep(), 1000);
    }
  }, 1500);
}

// ── Apply button finder ───────────────────────────────────────────────────────
// Used when the extension is on a job description page, not a form yet.

function findAndClickApply(pack) {
  const clickables = Array.from(
    document.querySelectorAll("a, button, [role='button'], input[type='button'], input[type='submit']")
  ).filter(el => el.offsetParent !== null);

  const labelOf = (el) => ((el.textContent || '') + ' ' + (el.getAttribute('aria-label') || '')).trim();

  // Workday "Start Your Application" chooser: prefer "Apply Manually" so we
  // control the fill (avoid Workday's own resume parser / LinkedIn flows).
  const chooser = clickables.find(el => /apply\s*manually/i.test(labelOf(el)));
  if (chooser) {
    chooser.click();
    console.log('[HirePath] Clicked "Apply Manually" (Workday chooser)');
    showOverlay('🖱️ Starting application…<br><small style="color:#94a3b8">If a sign-in appears, log in — I\'ll resume.</small>', [], false);
    watchForFormAppearance();
    return true;
  }

  // Match buttons/links whose visible text, aria-label, or href signals "apply".
  // Lenient (contains "apply") but guarded so we don't match legal text like
  // "By applying you agree…".
  const isApply = (el) => {
    const href = (el.getAttribute('href') || '');
    const label = labelOf(el);
    // Avoid LinkedIn/social apply shortcuts — we want the native form
    if (/with\s+linkedin|with\s+indeed/i.test(label)) return false;
    if (label.length > 0 && label.length <= 30 && /\bapply\b/i.test(label)) return true;
    if (/\/apply\b|applynow|apply-now/i.test(href)) return true;
    return false;
  };
  const candidates = clickables.filter(isApply);

  if (candidates.length > 0) {
    candidates[0].click();
    console.log('[HirePath] Clicked Apply button:', labelOf(candidates[0]));
    showOverlay('🖱️ Clicked <strong>Apply</strong>. Waiting for the form to load…', [], false);
    watchForFormAppearance();
    return true;
  }
  // No apply button found — guide the user
  showOverlay(
    '👆 I couldn\'t find the <strong>Apply</strong> button automatically.<br>' +
    '<small style="color:#94a3b8">Please click it yourself — I\'ll take over once the form loads.</small>',
    [], true
  );
  watchForFormAppearance();
  return false;
}

// ── Banner (legacy, used only if copilot not active) ─────────────────────────

function showBanner(msg) {
  const existing = document.getElementById('hirepath-ext-banner');
  if (existing) existing.remove();
  const banner = document.createElement('div');
  banner.id = 'hirepath-ext-banner';
  banner.style.cssText = [
    'position:fixed','top:0','left:0','right:0','z-index:2147483647',
    'padding:12px 20px','display:flex','align-items:center','gap:12px',
    'background:linear-gradient(135deg,#4f46e5,#7c3aed)',
    'color:white','font-family:system-ui,sans-serif','font-size:13px','font-weight:500',
    'box-shadow:0 2px 20px rgba(79,70,229,0.4)',
  ].join(';');
  banner.innerHTML = `<span style="font-size:18px">🚀</span><span style="flex:1">${msg}</span>` +
    `<button onclick="this.parentElement.remove()" style="background:rgba(255,255,255,0.2);border:none;color:white;cursor:pointer;padding:4px 10px;border-radius:8px;font-size:12px">✕</button>`;
  document.body.prepend(banner);
  setTimeout(() => banner?.remove(), 12000);
}

// ── Extension context guard ───────────────────────────────────────────────────
// If the extension is reloaded while a tab is open, the content script becomes
// orphaned and chrome.* calls throw "Extension context invalidated".
function chromeCall(fn) {
  try { return fn(); } catch (e) {
    if (e?.message?.includes('Extension context invalidated')) {
      console.warn('[HirePath] Extension was reloaded — please refresh this tab.');
    } else { console.error('[HirePath]', e); }
  }
}

// ── Message listener ──────────────────────────────────────────────────────────

chromeCall(() => chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === 'DO_FILL') {
    console.log('[HirePath] DO_FILL received, starting copilot for:', msg.fillPack?.job_title);
    fillForm(msg.fillPack).then(() => sendResponse({ ok: true }));
    return true;
  }
}));

// ── Bridge: dashboard postMessage → background.js → new tab ──────────────────
window.addEventListener('message', (e) => {
  if (e.data?.type === 'HIREPATH_LOAD_PACK' && e.data?.pack) {
    console.log('[HirePath] Received HIREPATH_LOAD_PACK from dashboard, forwarding to background');
    try {
      chrome.runtime.sendMessage({ type: 'OPEN_AND_FILL', payload: e.data.pack }, (res) => {
        if (chrome.runtime.lastError) {
          console.warn('[HirePath] Background error:', chrome.runtime.lastError.message);
        } else {
          console.log('[HirePath] Background opened tab, sending ACK to dashboard');
          window.postMessage({ type: 'HIREPATH_EXT_ACK' }, '*');
        }
      });
    } catch (err) {
      console.warn('[HirePath] Extension context invalidated — refresh this tab to restore autofill');
    }
  }
});

// ── Auto-fill on page load ────────────────────────────────────────────────────
chromeCall(() => chrome.storage.local.get(
  ['hirepath_fill_pack', 'hirepath_auto_fill', 'hirepath_copilot_pack', 'hirepath_copilot_ts'],
  (data) => {
    const host = window.location.hostname;
    // Never auto-run on the HirePath dashboard itself
    const onDashboard = /hirepath\.dev$/i.test(host) || host === 'localhost' || host === '127.0.0.1';

    if (data.hirepath_auto_fill && data.hirepath_fill_pack) {
      console.log('[HirePath] Auto-fill flag found, starting copilot after delay');
      chrome.storage.local.set({ hirepath_auto_fill: false });
      setTimeout(() => fillForm(data.hirepath_fill_pack), 2500);
      return;
    }

    // Resume an in-progress copilot session across navigations (e.g. after
    // clicking Apply, even across domains: accenture.com → myworkdayjobs.com).
    // 30-min window absorbs slow logins / account creation on Workday etc.
    const SESSION_MS = 30 * 60 * 1000;
    const fresh = data.hirepath_copilot_ts && (Date.now() - data.hirepath_copilot_ts) < SESSION_MS;
    if (!onDashboard && data.hirepath_copilot_pack && fresh) {
      if (isActionablePage()) {
        console.log('[HirePath] Resuming copilot session after navigation');
        setTimeout(() => fillForm(data.hirepath_copilot_pack), 2000);
      } else {
        console.log('[HirePath] Copilot session active but page not actionable yet — waiting');
        setTimeout(() => {
          if (_copilotActive) return;
          if (isActionablePage()) fillForm(data.hirepath_copilot_pack);
        }, 3000);
      }
      return;
    }

    console.log('[HirePath] Content script loaded on', host, '— no pending autofill');

    // Reliable manual trigger: on any job/application page, show a floating
    // "Fill with HirePath" button. Works regardless of session timing or
    // navigation — uses the last pack you loaded from the dashboard.
    if (!onDashboard) {
      const pack = data.hirepath_copilot_pack || data.hirepath_fill_pack || null;
      setTimeout(() => {
        if (_copilotActive) return;
        if (isActionablePage()) injectFillButton(pack);
      }, 1500);
    }
  }
));

// ── Floating manual "Fill" button ─────────────────────────────────────────────

function injectFillButton(pack) {
  if (document.getElementById('hp-fill-btn')) return;
  const btn = document.createElement('button');
  btn.id = 'hp-fill-btn';
  btn.innerHTML = '⚡ Fill with HirePath';
  Object.assign(btn.style, {
    position: 'fixed', bottom: '20px', left: '20px', zIndex: '2147483647',
    padding: '12px 18px', borderRadius: '14px', border: 'none', cursor: 'pointer',
    background: 'linear-gradient(135deg,#4f46e5,#7c3aed)', color: '#fff',
    fontFamily: 'system-ui,sans-serif', fontSize: '13px', fontWeight: '700',
    boxShadow: '0 8px 24px rgba(79,70,229,0.45)',
  });
  btn.addEventListener('click', () => {
    btn.remove();
    if (pack) {
      fillForm(pack);
    } else {
      showOverlay(
        '⚠️ No job loaded yet.<br><small style="color:#c4b5fd;font-weight:400">Go to your <b>HirePath dashboard</b>, find the job, and click <b>Auto Fill</b>. HirePath will open the application here and fill it automatically.</small>',
        [], true
      );
    }
  });
  document.body.appendChild(btn);
}

// ── Page classification helpers ───────────────────────────────────────────────

function isKnownATS() {
  const h = window.location.hostname;
  return /greenhouse\.io|lever\.co|ashbyhq\.com|myworkdayjobs\.com|workday\.com|smartrecruiters\.com|avature\.net|icims\.com|taleo\.net|successfactors|brassring|jobvite\.com|workable\.com|bamboohr\.com/i.test(h)
    || isAvaturePage();
}

function isActionablePage() {
  return isKnownATS() || hasApplicationForm() || hasApplyButton();
}

function hasApplyButton() {
  const isApply = (el) => {
    const label = ((el.textContent || '') + ' ' + (el.getAttribute('aria-label') || '')).trim();
    return label.length > 0 && label.length <= 30 && /\bapply\b/i.test(label) && el.offsetParent !== null;
  };
  return Array.from(document.querySelectorAll("a, button, [role='button']")).some(isApply);
}
