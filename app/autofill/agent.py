"""Autofill agent.

Approach: per-platform handlers. Greenhouse and Lever pages have predictable
DOM structures so we hand-write resilient selectors. For Workday and one-off
career pages, fall back to a generic field-finder + Claude-assisted mapping.

The agent NEVER clicks submit. It opens the page, fills what it can, and
returns a list of PendingQuestion records for whatever it couldn't determine.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import List
from urllib.parse import urlparse

from playwright.async_api import Page, async_playwright
from sqlmodel import select

from app.config import settings
from app.db.init_db import get_session
from app.db.models import Application, ApplicationStatus, Job, PendingQuestion, AnswerMemory
from app.matching.pipeline import _load_resume
from app.qa_store.resolver import QAResolver

log = logging.getLogger(__name__)


class CaptchaDetectedError(Exception):
    """Custom exception raised when CAPTCHA is detected in headless mode."""
    pass


# Comprehensive stealth init script — patches all fingerprint vectors that
# reCAPTCHA Enterprise and Lever bot detection check.
_STEALTH_INIT_JS = """
(() => {
  // 1. Hide webdriver flag
  Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

  // 2. Restore plugins (empty in headless)
  Object.defineProperty(navigator, 'plugins', {
    get: () => [1, 2, 3, 4, 5],
  });

  // 3. Restore mimeTypes
  Object.defineProperty(navigator, 'mimeTypes', {
    get: () => [1, 2, 3],
  });

  // 4. Add chrome runtime object (absent in headless)
  if (!window.chrome) {
    window.chrome = { runtime: {}, loadTimes: () => {}, csi: () => {}, app: {} };
  }

  // 5. Spoof permissions query (Notification permission)
  const origQuery = window.navigator.permissions && window.navigator.permissions.query.bind(window.navigator.permissions);
  if (origQuery) {
    window.navigator.permissions.query = (parameters) =>
      parameters.name === 'notifications'
        ? Promise.resolve({ state: Notification.permission })
        : origQuery(parameters);
  }

  // 6. Restore languages
  Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });

  // 7. Hardware concurrency
  Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });

  // 8. deviceMemory
  Object.defineProperty(navigator, 'deviceMemory', { get: () => 8 });

  // 9. Platform
  Object.defineProperty(navigator, 'platform', { get: () => 'MacIntel' });

  // 10. Remove "HeadlessChrome" from UA products list
  Object.defineProperty(navigator, 'appVersion', {
    get: () => navigator.appVersion.replace('Headless', ''),
  });
})();
"""

# Track active headful browser page contexts: application_id -> Page
_active_previews: dict[int, Page] = {}
_main_loop: asyncio.AbstractEventLoop | None = None

# Initialize canonical QA Resolver
qa_resolver = QAResolver()

# Memory state for CAPTCHA/Review confirmations
_pending_events: dict[str, asyncio.Event] = {}
_pending_event_loops: dict[str, asyncio.AbstractEventLoop] = {}
_event_data: dict[str, str] = {}

async def _send_telegram_photo(caption: str, photo_path: str, reply_markup: dict | None = None) -> bool:
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        log.warning("Telegram credentials missing, cannot send photo.")
        return False
    try:
        import httpx
        import json
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendPhoto"
        async with httpx.AsyncClient() as client:
            with open(photo_path, "rb") as f:
                files = {"photo": f}
                data = {
                    "chat_id": settings.telegram_chat_id,
                    "caption": caption,
                    "parse_mode": "Markdown",
                }
                if reply_markup:
                    data["reply_markup"] = json.dumps(reply_markup)
                resp = await client.post(url, data=data, files=files, timeout=20)
                return resp.status_code == 200
    except Exception as e:
        log.warning("Failed to send Telegram photo: %s", e)
        return False

async def _detect_captcha(page: Page) -> bool:
    captcha_selectors = [
        "iframe[src*='recaptcha']",
        "iframe[src*='hcaptcha']",
        "iframe[src*='turnstile']",
        ".g-recaptcha",
        "#cf-turnstile-iframe",
        "iframe[title*='recaptcha']",
        "iframe[title*='hcaptcha']",
        "iframe[title*='turnstile']"
    ]
    for sel in captcha_selectors:
        try:
            el = await page.query_selector(sel)
            if el and await el.is_visible():
                return True
        except Exception:
            continue
    try:
        # Check visible text of the body instead of raw HTML content to avoid matching hidden inputs, scripts, or styles
        body_text = await page.locator("body").inner_text()
        body_text_lower = body_text.lower()
        if (
            "verify you are human" in body_text_lower
            or "please solve the challenge" in body_text_lower
            or "solve the captcha" in body_text_lower
        ):
            return True
    except Exception:
        pass
    return False

async def _handle_captcha(page: Page, application_id: int, job: Job) -> bool:
    """Detect CAPTCHA, take screenshot, notify via Telegram, and raise CaptchaDetectedError to abort headless run."""
    if not await _detect_captcha(page):
        return False
        
    log.warning("CAPTCHA detected on page for %s", job.company)
    screenshot_path = f"./data/captcha_{application_id}.png"
    import os
    os.makedirs("./data", exist_ok=True)
    try:
        await page.screenshot(path=screenshot_path)
    except Exception as e:
        log.warning("Failed to take CAPTCHA screenshot: %s", e)
        screenshot_path = None
        
    caption = (
        f"🔒 *CAPTCHA detected* for *{job.company}* — *{job.title}*.\n\n"
        f"Headless autofill was blocked by a CAPTCHA. I've moved this application to *Awaiting Review*.\n\n"
        f"Please click **Open & Submit in Browser** on your dashboard or use the button below to solve the CAPTCHA and complete the form."
    )
    
    reply_markup = {
        "inline_keyboard": [
            [
                {"text": "🔓 Solve CAPTCHA / Open Browser", "callback_data": f"captcha:solve:{application_id}"}
            ]
        ]
    }
    
    if screenshot_path:
        await _send_telegram_photo(caption, screenshot_path, reply_markup)
    else:
        try:
            import httpx
            import json
            await httpx.post(
                f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
                json={
                    "chat_id": settings.telegram_chat_id,
                    "text": caption,
                    "parse_mode": "Markdown",
                    "reply_markup": reply_markup,
                },
                timeout=10
            )
        except Exception as e:
            log.warning("Failed to send CAPTCHA text message: %s", e)
            
    raise CaptchaDetectedError(f"CAPTCHA detected on page for {job.company}")

async def _handle_pre_submit_review(page: Page, application_id: int, job: Job, verify_note: str = "") -> str:
    """Take full page screenshot of the filled form, send to Telegram, and wait for human approval."""
    screenshot_path = f"./data/pre_submit_{application_id}.png"
    import os
    os.makedirs("./data", exist_ok=True)
    try:
        await page.screenshot(path=screenshot_path, full_page=True)
    except Exception as e:
        log.warning("Failed to take pre-submit screenshot: %s", e)
        screenshot_path = None
        
    event_id = f"review_{application_id}"
    event = asyncio.Event()
    _pending_events[event_id] = event
    _pending_event_loops[event_id] = asyncio.get_event_loop()
    
    caption = (
        f"📋 *Ready to submit* to *{job.company}* for *{job.title}*.\n"
        f"Please review the filled application screenshot. Select Approve to submit, or Reject to abort."
    )
    if verify_note:
        caption += f"\n\n🔎 {verify_note}"
    
    reply_markup = {
        "inline_keyboard": [
            [
                {"text": "Approve ✅", "callback_data": f"review:approve:{application_id}"},
                {"text": "Reject ❌", "callback_data": f"review:reject:{application_id}"}
            ]
        ]
    }
    
    if screenshot_path:
        await _send_telegram_photo(caption, screenshot_path, reply_markup)
    else:
        try:
            import httpx
            await httpx.post(
                f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
                json={
                    "chat_id": settings.telegram_chat_id,
                    "text": caption,
                    "parse_mode": "Markdown",
                    "reply_markup": reply_markup
                },
                timeout=10
            )
        except Exception as e:
            log.warning("Failed to send review text message: %s", e)
            
    try:
        log.info("Waiting for user pre-submit review...")
        await asyncio.wait_for(event.wait(), timeout=600.0)
        res = _event_data.get(event_id, "timeout")
        return res
    except asyncio.TimeoutError:
        log.warning("Pre-submit review timeout.")
        return "timeout"
    finally:
        _pending_events.pop(event_id, None)
        _pending_event_loops.pop(event_id, None)
        _event_data.pop(event_id, None)
        if screenshot_path and os.path.exists(screenshot_path):
            try:
                os.remove(screenshot_path)
            except Exception:
                pass

async def _click_submit(page: Page) -> bool:
    submit_selectors = [
        "#submit_app", 
        "#btn-submit", 
        "button[type='submit']", 
        "input[type='submit']",
        "button:has-text('Submit')",
        "button:has-text('submit')",
        "button:has-text('Submit Application')",
        "button:has-text('submit application')",
        "input[value='Submit']",
        "input[value='submit']",
        "input[value='Submit Application']",
        "input[value='submit application']",
        "[role='button']:has-text('Submit')",
        "[role='button']:has-text('submit')",
        "button[id*='submit']",
        "button[class*='submit']"
    ]
    for sel in submit_selectors:
        try:
            el = await page.query_selector(sel)
            if el and await el.is_visible():
                await el.click()
                return True
        except Exception:
            continue
    return False

async def _fill_humanlike(el, val: str) -> None:
    """Focus, clear, type with jitter, fire React synthetic events, then blur."""
    import random
    val_str = str(val)
    try:
        await el.focus()
        await asyncio.sleep(random.uniform(0.1, 0.2))
        await el.fill("")
        await el.type(val_str, delay=random.randint(15, 45))
        await asyncio.sleep(random.uniform(0.15, 0.30))
    except Exception as e:
        log.debug("Fallback to standard fill: %s", e)
        try:
            await el.fill(val_str)
        except Exception:
            pass

    # Fire React synthetic events — required so React-controlled inputs (Lever, etc.)
    # update their internal state. Without this, the form submits as empty.
    try:
        await el.evaluate("""(el) => {
            const proto = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')
                       || Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value');
            if (proto && proto.set) proto.set.call(el, el.value);
            el.dispatchEvent(new Event('input',  { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
        }""")
        await el.blur()
        await asyncio.sleep(random.uniform(0.08, 0.18))
    except Exception:
        pass

def set_main_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _main_loop
    _main_loop = loop
    log.info("Main event loop registered in autofill agent")

def get_main_loop() -> asyncio.AbstractEventLoop | None:
    return _main_loop



@dataclass
class UnknownField:
    label: str
    selector: str
    field_type: str
    options: List[str] | None = None


# --- generic personal-info map ---

def _personal_fields() -> dict:
    data = qa_resolver.data
    identity = data.get("identity", {})
    return {
        "first_name": identity.get("first_name", settings.applicant_first_name),
        "last_name": identity.get("last_name", settings.applicant_last_name),
        "email": identity.get("email", settings.applicant_email),
        "phone": identity.get("phone", settings.applicant_phone),
        "location": identity.get("location", settings.applicant_location),
        "github": identity.get("github", settings.applicant_github),
        "linkedin": identity.get("linkedin", settings.applicant_linkedin),
    }


# ── Post-fill verification ────────────────────────────────────────────────────
# After a handler fills the form, confirm the critical text fields actually hold
# the expected values in the DOM. Silent fill failures (wrong field, React state
# not committed, value cleared on blur) are the most common autofill bug — this
# catches them before we ask the human to approve a half-empty form.

_VERIFY_SELECTORS: dict[str, list[str]] = {
    "first_name": [
        "input[name='first_name']", "input[name='firstName']",
        "input[autocomplete='given-name']", "input[id*='first_name']",
        "input[id*='firstName']",
    ],
    "last_name": [
        "input[name='last_name']", "input[name='lastName']",
        "input[autocomplete='family-name']", "input[id*='last_name']",
        "input[id*='lastName']",
    ],
    "email": [
        "input[type='email']", "input[name='email']",
        "input[autocomplete='email']", "input[id*='email']",
    ],
    "phone": [
        "input[type='tel']", "input[name='phone']",
        "input[autocomplete='tel']", "input[id*='phone']",
    ],
}


@dataclass
class FieldVerification:
    field: str
    expected: str
    actual: str
    ok: bool


@dataclass
class VerificationReport:
    checks: List["FieldVerification"]

    @property
    def mismatches(self) -> List["FieldVerification"]:
        return [c for c in self.checks if not c.ok]

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def summary(self) -> str:
        if not self.checks:
            return "no verifiable fields found"
        ok = sum(1 for c in self.checks if c.ok)
        parts = []
        for c in self.checks:
            mark = "✓" if c.ok else "✗"
            parts.append(f"{mark} {c.field}")
        return f"{ok}/{len(self.checks)} fields verified — " + ", ".join(parts)


def _values_match(field: str, expected: str, actual: str) -> bool:
    """Field-appropriate comparison of expected vs actual DOM value."""
    if not expected:
        return True  # nothing was expected, so nothing to verify
    exp = expected.strip().lower()
    act = (actual or "").strip().lower()
    if not act:
        return False
    if field == "email":
        return exp == act
    if field == "phone":
        import re as _re
        exp_d = _re.sub(r"\D", "", exp)
        act_d = _re.sub(r"\D", "", act)
        # phone fields may add/drop country code — match on the last 10 digits
        return exp_d[-10:] == act_d[-10:] and len(act_d) >= 10
    # names and free text: expected should appear within the actual value
    return exp in act


async def _read_field_value(fill_target, selectors: list[str]) -> str | None:
    """Return the value of the first visible, matching input — or None if absent."""
    for sel in selectors:
        try:
            el = await fill_target.query_selector(sel)
            if el and await el.is_visible():
                val = await el.input_value()
                return val
        except Exception:
            continue
    return None


async def _verify_filled_fields(fill_target, expected: dict, retry_fill: bool = True) -> VerificationReport:
    """Verify critical text fields hold the expected values; optionally re-fill mismatches once.

    Only fields that (a) have an expected value and (b) have a locatable input on the
    page are checked — we never penalise a form for not having a phone field, etc.
    """
    checks: List[FieldVerification] = []

    for field_name, selectors in _VERIFY_SELECTORS.items():
        expected_val = str(expected.get(field_name, "") or "")
        if not expected_val:
            continue
        actual = await _read_field_value(fill_target, selectors)
        if actual is None:
            continue  # field not present on this form — not a failure

        ok = _values_match(field_name, expected_val, actual)

        # One re-fill attempt on a mismatch, then re-read
        if not ok and retry_fill:
            for sel in selectors:
                try:
                    el = await fill_target.query_selector(sel)
                    if el and await el.is_visible():
                        log.info("Verification: re-filling '%s' (was %r, expected %r)",
                                 field_name, actual, expected_val)
                        await _fill_humanlike(el, expected_val)
                        actual = await el.input_value()
                        break
                except Exception:
                    continue
            ok = _values_match(field_name, expected_val, actual)

        checks.append(FieldVerification(
            field=field_name, expected=expected_val, actual=actual or "", ok=ok,
        ))

    report = VerificationReport(checks=checks)
    if report.all_ok:
        log.info("Field verification PASSED: %s", report.summary())
    else:
        log.warning("Field verification found mismatches: %s", report.summary())
        for m in report.mismatches:
            log.warning("  ✗ %s: expected %r, got %r", m.field, m.expected, m.actual)
    return report


US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"
}

def _get_state_from_location(location: str | None) -> str | None:
    if not location:
        return None
    import re
    match = re.search(r'\b([A-Z]{2})\b', location)
    if match:
        state_code = match.group(1)
        if state_code in US_STATES:
            return US_STATES[state_code]
    for name in US_STATES.values():
        if name.lower() in location.lower():
            return name
    return None

_EEOC_DEFAULTS: dict[str, str] = {
    # State / location
    "what u.s state do you currently reside in": "Ohio",
    "what state do you currently reside in":     "Ohio",
    "state":                                     "Ohio",
    "current state":                             "Ohio",
    # EEOC demographic — decline to self-identify (safest, avoids bias)
    "gender":                                    "Decline to self-identify",
    "gender identity":                           "Decline to self-identify",
    "race":                                      "Decline to self-identify",
    "race/ethnicity":                            "Decline to self-identify",
    "ethnicity":                                 "Decline to self-identify",
    "veteran status":                            "I am not a protected veteran",
    "veteran":                                   "I am not a protected veteran",
    "disability status":                         "No, I do not have a disability, or history/record of having a disability",
    "disability":                                "No, I do not have a disability, or history/record of having a disability",
    # Referral source
    "how did you first hear about this opportunity": "LinkedIn",
    "how did you hear about us":                 "LinkedIn",
    "how did you hear about this role":          "LinkedIn",
    "referral source":                           "LinkedIn",
    "how did you find this job":                 "LinkedIn",
}


def _check_memory(label: str, job: Job | None = None) -> str | None:
    # 0. EEOC / state defaults — check before QA resolver so they always match
    norm_label = label.lower().strip().rstrip("*").strip()
    for key, default_val in _EEOC_DEFAULTS.items():
        if key in norm_label or norm_label == key:
            return default_val

    # 1. Resolve with canonical QAResolver
    ans, conf = qa_resolver.resolve(label, job)
    if conf >= 0.7:
        return ans

    # 2. Check DB memory
    norm = label.lower().strip()
    
    # Generic blocklist for memory retrieval to avoid cross-company contamination
    generic_blocklist = [
        "additional information", "anything else", "cover letter", "comments", 
        "additional context", "tell us more", "why are you interested", "statement of interest",
        "anything else you'd like us to know"
    ]
    if any(blocked in norm for blocked in generic_blocklist):
        return None

    with get_session() as session:
        mem = session.exec(select(AnswerMemory).where(AnswerMemory.label_normalized == norm)).first()
        if mem:
            from datetime import datetime
            mem.use_count += 1
            mem.last_used_at = datetime.utcnow()
            session.add(mem)
            session.commit()
            return mem.answer
    return None


def _get_system_question_answerer_prompt() -> str:
    data = qa_resolver.data
    identity = data.get("identity", {})
    edu = data.get("education", {})
    exp = data.get("experience", {})
    bg = data.get("background", {})
    
    first_name = identity.get("first_name", "Karthik")
    last_name = identity.get("last_name", "Amruthaluri")
    github = identity.get("github", "github.com/karthiklucky1").replace("https://", "").replace("http://", "")
    email = identity.get("email", "karthikamruthaluri2002@gmail.com")
    
    uni = edu.get("university", "University of Cincinnati")
    degree = edu.get("degree", "Master of Engineering")
    grad_year = edu.get("graduation_year", 2026)
    grad_date = edu.get("graduation_date", "April 30, 2026")
    grad_status = edu.get("graduation_status", "Graduated")
    
    exp_summary = bg.get("experience_summary", "")
    flagship_project = bg.get("flagship_project", "")
    tech_stack = bg.get("tech_stack", "")
    yoe = exp.get("total_yoe", 3)
    
    from datetime import datetime
    current_date = datetime.utcnow().strftime("%B %d, %Y")
    
    prompt = f"""You write short, professional, and honest answers to job application screening questions.

Candidate Context:
- Current Date: {current_date}
- {first_name} {last_name}
- Github: {github} | Email: {email}
- Education: {degree}, {uni} ({grad_status} on {grad_date}).
- Experience: {exp_summary} ({yoe}+ years of experience).
- Flagship Project: {flagship_project}
- Tech Stack: {tech_stack}

Rules:
- Be honest. Do not invent or fabricate experience.
- If the candidate lacks direct experience with a specific tool/skill, use "truthful transfer": explain that they haven't used it directly but have adjacent experience (e.g., "I haven't used Ray directly, but I have built distributed async inference with FastAPI and AWS ECS").
- Keep the answer concise and natural, between 50 and 120 words.
- Write in first person ("I").
- Do not add any placeholder, explanations, or metadata. Return the exact response text to be entered into the form."""
    return prompt


def _answer_question_with_llm(label: str, job: Job, resume_text: str) -> str:
    log.info("Generating LLM answer for screening question: '%s'", label)
    prompt = f"""<resume>
{resume_text[:6000]}
</resume>

<job>
Title: {job.title}
Company: {job.company}
Description: {job.description[:4000]}
</job>

Screening Question: "{label}"

Write a professional response answering this question based on the resume and job context."""
    
    system_prompt = _get_system_question_answerer_prompt()
    
    # 1. Try Anthropic
    if settings.anthropic_api_key:
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=settings.anthropic_api_key)
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                system=[{"type": "text", "text": system_prompt}],
                messages=[{"role": "user", "content": prompt}],
            )
            ans = resp.content[0].text.strip()
            log.info("Generated LLM response (Anthropic): '%s...'", ans[:80])
            return ans
        except Exception as e:
            log.warning("LLM question answering (Anthropic) failed for '%s': %s", label, e)

    # 2. Try OpenAI fallback
    if settings.openai_api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)
            resp = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=300,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            ans = resp.choices[0].message.content.strip()
            log.info("Generated LLM response (OpenAI): '%s...'", ans[:80])
            return ans
        except Exception as e:
            log.warning("LLM question answering (OpenAI) failed for '%s': %s", label, e)
            
    return ""


# ---------- cover letter helper ----------

async def _fill_cover_letter_area(page: Page, cover_text: str) -> bool:
    """Detect and fill any cover letter textarea on the page."""
    if not cover_text:
        return False
    cover_keywords = ["cover letter", "cover_letter", "coverletter", "motivation letter", "why do you want", "why this company"]
    textareas = await page.query_selector_all("textarea")
    for ta in textareas:
        try:
            label = await ta.evaluate("""(e) => {
                if (e.id) {
                    const lbl = document.querySelector(`label[for='${e.id}']`);
                    if (lbl) return lbl.innerText.toLowerCase();
                }
                const parent = e.closest('[class*="field"], [class*="question"], .form-group');
                const lbl = parent?.querySelector('label, .label, legend');
                return lbl?.innerText?.toLowerCase() || e.name?.toLowerCase() || e.id?.toLowerCase() || '';
            }""")
            if any(kw in label for kw in cover_keywords):
                val = await ta.input_value()
                if not val:
                    await ta.fill(cover_text)
                    log.info("Filled cover letter text area (label: '%s')", label)
                    return True
        except Exception:
            continue
    return False


def _safe_id_sel(id_str: str) -> str:
    """Return a valid CSS selector for an element ID.
    IDs that start with a digit are illegal in CSS (#29783) — use attribute selector instead."""
    if id_str and id_str[0].isdigit():
        return f"[id='{id_str}']"
    return f"#{id_str}"


# ---------- React Select helper ----------

async def _fill_react_select(page: Page, field_id: str, answer: str) -> bool:
    """Fill a React Select combobox by clicking the control and selecting the matching option.

    React Select renders a text input with role=combobox inside a .select__control div.
    We click the control to open the menu, then click the option whose text matches `answer`.
    Returns True if an option was successfully clicked.
    """
    try:
        # Bug fix: .select__control is an ANCESTOR of the input, not a sibling descendant.
        # Use :has() to find the wrapping control, then fall back to clicking the input itself.
        id_sel = _safe_id_sel(field_id)
        control = await page.query_selector(f".select__control:has({id_sel})")
        if not control:
            inp = await page.query_selector(id_sel)
            if not inp:
                return False
            await inp.click()
        else:
            await control.click()

        await page.wait_for_timeout(700)

        # Bug fix: scope to the currently open menu so we don't match options from
        # already-closed menus elsewhere on the page.
        menu = await page.query_selector(".select__menu, [class*='menu--is-open']")
        if menu:
            options = await menu.query_selector_all("[class*='option']")
        else:
            # Fallback: React Select generates predictable option IDs
            options = await page.query_selector_all(f"[id^='react-select-{field_id}-option']")
            if not options:
                options = await page.query_selector_all("[role='option']")

        low_answer = answer.lower().strip()
        for opt in options:
            txt = (await opt.inner_text()).strip()
            if txt.lower() == low_answer or txt.lower().startswith(low_answer):
                await opt.click()
                log.info("React Select '%s' → exact match '%s'", field_id, txt)
                await page.wait_for_timeout(300)
                return True

        # Partial match fallback
        for opt in options:
            txt = (await opt.inner_text()).strip()
            if low_answer in txt.lower():
                await opt.click()
                log.info("React Select '%s' → partial match '%s'", field_id, txt)
                await page.wait_for_timeout(300)
                return True

        log.warning("React Select '%s': no option matched '%s'. Available: %s",
                    field_id, answer, [((await o.inner_text()).strip()) for o in options[:8]])
        await page.keyboard.press("Escape")
        return False
    except Exception as e:
        log.warning("React Select fill failed for '%s': %s", field_id, e)
        return False


async def _fill_intl_tel_input(page: Page) -> bool:
    """Evade international phone code dropdown blocks (e.g. intl-tel-input library).
    
    If the phone input uses a flag/country selection list (like on Pinterest),
    clicks the dropdown trigger, filters by 'United States', and selects the option.
    """
    try:
        trigger = await page.query_selector(".iti__selected-country, .iti__selected-flag, .iti__flag-container, .iti__arrow")
        if not trigger:
            return False
        
        # Check if United States +1 is already selected to avoid unnecessary clicking
        title = await trigger.get_attribute("title") or ""
        text = await trigger.inner_text() or ""
        if "United States" in title or "United States" in text or "+1" in title or "+1" in text:
            return True
            
        log.info("intl-tel-input dial code dropdown detected. Opening...")
        await trigger.click()
        await page.wait_for_timeout(500)
        
        # If there is a search filter inside the dropdown, type into it
        search_inp = await page.query_selector(".iti__search-input, #iti-0__search-input")
        if search_inp and await search_inp.is_visible():
            await search_inp.fill("United States")
            await page.wait_for_timeout(400)
            
        # Select United States option
        us_option = await page.query_selector("li[id$='-us'], li[id*='-us'], .iti__country:has-text('United States')")
        if us_option:
            await us_option.click()
            log.info("Selected United States (+1) from international phone dropdown.")
            await page.wait_for_timeout(300)
            return True
        else:
            await page.keyboard.press("Escape")
            return False
    except Exception as e:
        log.debug("Skip international phone dropdown handler: %s", e)
        return False


# ---------- Frame proxy (used when GH form is in an embed iframe) ----------

class _FrameProxy:
    """Wraps a Playwright Frame so DOM queries go to the iframe
    while wait_for_timeout / keyboard / other Page-only APIs delegate to Page."""
    def __init__(self, frame, page: Page):
        self._frame = frame
        self._page  = page

    def __getattr__(self, name: str):
        # Page-only helpers
        if name in ("wait_for_timeout", "wait_for_selector", "wait_for_load_state",
                    "keyboard", "mouse", "screenshot", "reload", "close"):
            return getattr(self._page, name)
        return getattr(self._frame, name)


# ---------- Greenhouse handler ----------

async def _click_apply_and_load_form(page: Page) -> bool:
    """Click the Apply / Apply Now button on custom career pages that embed Greenhouse.

    Companies like Pinterest host jobs on their own domain (pinterestcareers.com)
    with a Greenhouse iframe or redirect triggered by an Apply button.
    After clicking we wait for the form fields to appear, then scroll to them.
    Returns True if an Apply button was found and clicked.
    """
    apply_selectors = [
        # Greenhouse data attributes
        "a[data-mapped='true']",
        "a[data-gh-apply]",
        # Generic text matches
        "a:has-text('Apply for this Job')",
        "a:has-text('apply for this job')",
        "a:has-text('Apply Now')",
        "a:has-text('apply now')",
        "a:has-text('Apply')",
        "a:has-text('apply')",
        "button:has-text('Apply for this Job')",
        "button:has-text('apply for this job')",
        "button:has-text('Apply Now')",
        "button:has-text('apply now')",
        "button:has-text('Apply')",
        "button:has-text('apply')",
        # Input elements
        "input[value='Apply']",
        "input[value='apply']",
        "input[value='Apply Now']",
        "input[value='apply now']",
        # Roles and partial attributes
        "[role='button']:has-text('Apply')",
        "[role='button']:has-text('apply')",
        "button[id*='apply']",
        "button[class*='apply']",
        "a[id*='apply']",
        "a[class*='apply']",
        # Common class names
        ".apply-button",
        ".apply-btn",
        "#apply-button",
        "#apply-btn",
        "a[href*='/apply']",
        "a[href*='apply']",
    ]
    for sel in apply_selectors:
        try:
            el = await page.query_selector(sel)
            if el and await el.is_visible():
                await el.click()
                log.info("Clicked Apply button via selector: %s", sel)
                # Wait for form fields to appear after click / navigation
                try:
                    await page.wait_for_selector(
                        "#first_name, input[name='job_application[first_name]'], #resume",
                        timeout=8000,
                    )
                except Exception:
                    await page.wait_for_timeout(3000)
                # Scroll the form into view
                try:
                    form_el = await page.query_selector(
                        "#first_name, input[name='job_application[first_name]'], form"
                    )
                    if form_el:
                        await form_el.scroll_into_view_if_needed()
                except Exception:
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.3)")
                return True
        except Exception as e:
            log.debug("Apply button attempt failed for %s: %s", sel, e)
    return False


async def _fill_greenhouse(page: Page, resume_docx: str, cover_text: str, job: Job, resume_text: str) -> List[UnknownField]:
    """Handles both boards.greenhouse.io and job-boards.greenhouse.io formats.
    New format uses id-based fields with no name attr and aria-required instead of required.
    """
    import os
    pf = _personal_fields()

    # --- Standard personal fields (both old and new Greenhouse) ---
    field_map = {
        "#first_name": pf["first_name"],
        "#last_name": pf["last_name"],
        "#email": pf["email"],
        "#phone": pf["phone"],
        "#country": "United States",
        "input[name='job_application[first_name]']": pf["first_name"],
        "input[name='job_application[last_name]']": pf["last_name"],
        "input[name='job_application[email]']": pf["email"],
        "input[name='job_application[phone]']": pf["phone"],
    }
    # Select phone country code flag if present
    await _fill_intl_tel_input(page)

    for sel, val in field_map.items():
        if not val:
            continue
        try:
            el = await page.query_selector(sel)
            if el:
                val_str = str(val)
                role = await el.get_attribute("role") or ""
                class_attr = await el.get_attribute("class") or ""
                if role == "combobox" or "select" in class_attr:
                    field_id = sel.replace("#", "") if sel.startswith("#") else ""
                    if field_id:
                        await _fill_react_select(page, field_id, val_str)
                    else:
                        await _fill_humanlike(el, val_str)
                else:
                    curr_val = await el.input_value()
                    if curr_val != val_str:
                        await _fill_humanlike(el, val_str)
                        log.info("Filled %s", sel)
        except Exception as e:
            log.debug("GH fill skipped for %s: %s", sel, e)

    # --- Resume upload (new GH: id='resume', old GH: name contains 'resume') ---
    resume_abs = os.path.abspath(resume_docx) if resume_docx else ""
    uploaded = False
    if resume_abs:
        for file_sel in ["#resume", "input[type='file'][id='resume']", "input[type='file'][name*='resume']", "input[type='file']"]:
            try:
                el = await page.query_selector(file_sel)
                if el:
                    await el.set_input_files(resume_abs)
                    log.info("Resume uploaded via selector: %s", file_sel)
                    uploaded = True
                    break
            except Exception as e:
                log.debug("Resume upload attempt failed for %s: %s", file_sel, e)
    if not uploaded:
        log.warning("Could not upload resume — no matching file input found")

    # --- LinkedIn / GitHub quick-fill for known question IDs ---
    quick_fill = {
        "#question_13303964008": pf["linkedin"],  # LinkedIn Profile
        "#question_13303967008": pf["github"],    # GitHub URL
    }
    for sel, val in quick_fill.items():
        if not val:
            continue
        try:
            el = await page.query_selector(sel)
            if el:
                await _fill_humanlike(el, str(val))
        except Exception:
            pass

    # --- EEO React Select dropdowns (not required, so scanner misses them) ---
    # Some fields (e.g. race) only appear after a prior answer (hispanic_ethnicity=No).
    # Multi-pass: after each fill, wait for DOM to settle, then re-check remaining fields.
    # Any EEO field present on the page but not fillable is added to unknown for bot report.
    data = qa_resolver.data
    eeo = data.get("eeo", {})
    eeo_fields = {
        "gender":             eeo.get("gender", "Male"),
        "hispanic_ethnicity": "No" if not eeo.get("hispanic_latino", False) else "Yes",
        "race":               eeo.get("race", "Asian"),
        "veteran_status":     eeo.get("veteran_status", "I am not a protected veteran"),
        "disability_status":  eeo.get("disability_status", "No, I do not have a disability, or history/record of having a disability"),
    }
    eeo_filled: set = set()

    for _pass in range(4):  # up to 4 passes to catch dynamically revealed fields
        new_fills_this_pass = 0
        for field_id, answer in eeo_fields.items():
            if field_id in eeo_filled:
                continue
            el = await page.query_selector(f"#{field_id}")
            if not el:
                continue
            role = await el.get_attribute("role") or ""
            filled = False
            if role == "combobox":
                filled = await _fill_react_select(page, field_id, answer)
                log.info("EEO field #%s → '%s' (%s)", field_id, answer, "✅" if filled else "❌")
            else:
                try:
                    await el.select_option(label=answer)
                    filled = True
                except Exception:
                    try:
                        await el.fill(answer)
                        filled = True
                    except Exception:
                        pass
                if filled:
                    log.info("EEO field #%s → '%s' ✅", field_id, answer)
            if filled:
                eeo_filled.add(field_id)
                new_fills_this_pass += 1
                await page.wait_for_timeout(700)  # let any newly revealed fields render

                # Selecting "No" for hispanic_ethnicity reveals the race field dynamically.
                # It gets a generated ID (not "#race"), so hunt for it immediately by label.
                if field_id == "hispanic_ethnicity":
                    await page.wait_for_timeout(1200)  # extra wait for race field to fully render
                    race_cbs = await page.query_selector_all("input[role='combobox']")
                    for rcb in race_cbs:
                        rcb_id = await rcb.get_attribute("id") or ""
                        if rcb_id in eeo_filled:
                            continue
                        rcb_label = await rcb.evaluate("""(e) => {
                            if (e.id) {
                                const lbl = document.querySelector(`label[for='${e.id}']`);
                                if (lbl) return lbl.innerText.replace(/[*\\n]+/g, ' ').trim();
                            }
                            const parent = e.closest('[class*="field"], [class*="question"], li');
                            const lbl = parent?.querySelector('label, .label, legend');
                            return lbl?.innerText.replace(/[*\\n]+/g, ' ').trim() || '';
                        }""")
                        if any(kw in rcb_label.lower() for kw in ["race", "ethnicity", "racial"]):
                            race_filled = await _fill_react_select(page, rcb_id, "Asian")
                            if race_filled:
                                eeo_filled.add(rcb_id)
                                log.info("Dynamic race field #%s ('%s') → Asian ✅", rcb_id, rcb_label)
                            else:
                                log.warning("Dynamic race field #%s ('%s') → failed ❌", rcb_id, rcb_label)
                            break

        if new_fills_this_pass == 0:
            break  # nothing new filled this pass, stop early

    # --- Cover letter ---
    await _fill_cover_letter_area(page, cover_text)

    # --- Scan all visible required custom fields ---
    unknown: List[UnknownField] = []

    # EEO fields that were present on the page but couldn't be filled → report to bot
    for field_id, answer in eeo_fields.items():
        if field_id in eeo_filled:
            continue
        el = await page.query_selector(f"#{field_id}")
        if not el:
            continue
        label_text = await el.evaluate("""(e) => {
            if (e.id) {
                const lbl = document.querySelector(`label[for='${e.id}']`);
                if (lbl) return lbl.innerText.replace(/[*\\n]+/g, ' ').trim();
            }
            const parent = e.closest('[class*="field"], [class*="question"], li');
            const lbl = parent?.querySelector('label, .label, legend');
            return lbl?.innerText.replace(/[*\\n]+/g, ' ').trim() || e.id;
        }""") or field_id
        log.warning("EEO field #%s ('%s') present but not filled — adding to missing", field_id, label_text)
        unknown.append(UnknownField(label=label_text, selector=f"#{field_id}", field_type="select"))

    # --- Label-based combobox sweep ---
    # Catches dynamic fields (e.g. race appearing after hispanic=No) that have generated
    # IDs not matching eeo_fields, plus new required dropdowns like "AI Policy".
    # For every role=combobox on the page: fill if label is known, report if required & unknown.
    all_comboboxes = await page.query_selector_all("input[role='combobox']")
    for cb in all_comboboxes:
        try:
            cb_id = await cb.get_attribute("id") or ""
            cb_class = await cb.get_attribute("class") or ""
            if cb_id in eeo_filled or "iti" in cb_id or "iti" in cb_class:
                continue
            # Skip if the React Select control already shows a selected value
            already_selected = await cb.evaluate("""(e) => {
                const ctrl = e.closest('.select__control');
                if (ctrl) {
                    const val = ctrl.querySelector('.select__single-value');
                    return !!(val && val.innerText.trim());
                }
                return false;
            }""")
            if already_selected:
                if cb_id:
                    eeo_filled.add(cb_id)
                continue
            # Get label text for this combobox
            label_text = await cb.evaluate("""(e) => {
                if (e.id) {
                    const lbl = document.querySelector(`label[for='${e.id}']`);
                    if (lbl) return lbl.innerText.replace(/[*\\n]+/g, ' ').trim();
                }
                const parent = e.closest('[class*="field"], [class*="question"], li');
                const lbl = parent?.querySelector('label, .label, legend');
                return lbl?.innerText.replace(/[*\\n]+/g, ' ').trim() || '';
            }""")
            if not label_text:
                continue
            is_required = ('*' in label_text) or (
                await cb.evaluate("(e) => e.getAttribute('aria-required') === 'true'")
            )
            clean_label = label_text.strip().rstrip('*').strip()
            low_label = clean_label.lower()
            known_val = _check_memory(clean_label, job)
            if known_val and cb_id:
                filled = await _fill_react_select(page, cb_id, known_val)
                if filled:
                    eeo_filled.add(cb_id)
                    log.info("Combobox sweep filled '%s' (#%s) → '%s'", clean_label, cb_id, known_val)
                    await page.wait_for_timeout(700)
                    continue
                # Fill failed — if required, fall through to report
            if is_required:
                sel = _safe_id_sel(cb_id) if cb_id else "input[role='combobox']"
                if not any(u.selector == sel for u in unknown):
                    log.warning("Unknown required combobox: '%s' (#%s)", clean_label, cb_id)
                    unknown.append(UnknownField(label=clean_label, selector=sel, field_type="select"))
        except Exception as exc:
            log.debug("Combobox sweep error: %s", exc)
            continue

    # New Greenhouse uses aria-required=true on the input itself
    all_fields = await page.query_selector_all("input[aria-required='true'], textarea[aria-required='true'], select[aria-required='true'], input[required], textarea[required], select[required]")

    # Also grab any field with a question_ id that has no value
    question_fields = await page.query_selector_all("input[id^='question_'], textarea[id^='question_']")
    seen_ids = set()
    combined = all_fields + [f for f in question_fields if f not in all_fields]

    for el in combined:
        try:
            el_id = await el.get_attribute("id") or ""
            el_class = await el.get_attribute("class") or ""
            if el_id in seen_ids or "iti" in el_id or "iti" in el_class:
                continue
            seen_ids.add(el_id)

            # Bug fix: React Select combobox inputs always return "" from input_value()
            # even when filled, so skip EEO fields we already successfully filled above.
            if el_id in eeo_filled:
                continue

            # Skip file inputs and hidden
            el_type = await el.get_attribute("type") or "text"
            if el_type in ("file", "hidden", "submit"):
                continue

            value = await el.input_value()
            if value:
                continue

            # Get label text
            label = await el.evaluate("""(e) => {
                if (e.id) {
                    const lbl = document.querySelector(`label[for='${e.id}']`);
                    if (lbl) return lbl.innerText.replace(/[*\\n]+/g, ' ').trim();
                }
                const parent = e.closest('[class*="field"], [class*="question"], li');
                const lbl = parent?.querySelector('label, .label, legend');
                return lbl?.innerText.replace(/[*\\n]+/g, ' ').trim() || e.placeholder || e.id || '(unlabeled)';
            }""")
            label = label.strip().rstrip("*").strip()
            tag = await el.evaluate("(e) => e.tagName.toLowerCase()")

            # Check against known quick-fills first
            low_label = label.lower()
            known_val = _check_memory(label, job)
            el_role = await el.get_attribute("role") or ""
            _was_llm = False
            if not known_val and tag in ["input", "textarea"] and el_role != "combobox" and el_type not in ["checkbox", "radio"]:
                known_val = _answer_question_with_llm(label, job, resume_text)
                if known_val:
                    _was_llm = True
            if known_val:
                log.info("Answer memory hit for '%s': %s", label, known_val[:80])
                filled = False
                try:
                    if el_role == "combobox":
                        filled = await _fill_react_select(page, el_id, known_val)
                    elif tag == "select":
                        await el.select_option(label=known_val)
                        filled = True
                    else:
                        await _fill_humanlike(el, known_val)
                        filled = True
                except Exception as e:
                    log.warning("Failed to auto-fill '%s': %s", label, e)
                if filled:
                    if _was_llm:
                        try:
                            with get_session() as msession:
                                existing = msession.exec(select(AnswerMemory).where(AnswerMemory.label_normalized == low_label)).first()
                                if not existing:
                                    msession.add(AnswerMemory(label_normalized=low_label, label_original=label, answer=known_val))
                                    msession.commit()
                        except Exception:
                            pass
                    continue

            log.info("Unknown required field: '%s' (%s) role=%s", label, el_id, el_role)
            sel = _safe_id_sel(el_id) if el_id else f"*[placeholder='{await el.get_attribute('placeholder')}']"
            unknown.append(UnknownField(label=label, selector=sel, field_type=tag))
        except Exception as exc:
            log.debug("Error scanning field: %s", exc)
            continue

    # Post-fill standard field verification
    for sel, val in field_map.items():
        try:
            el = await page.query_selector(sel)
            if el:
                current_val = await el.input_value()
                if not current_val:
                    await _fill_humanlike(el, str(val))
                    await page.wait_for_timeout(300)
                    if not await el.input_value():
                        label_name = sel.replace("#", "").replace("input[name='job_application[", "").replace("]']", "").replace("_", " ").title()
                        if not any(u.selector == sel for u in unknown):
                            unknown.append(UnknownField(label=f"Required: {label_name}", selector=sel, field_type="text"))
        except Exception:
            pass

    return unknown



# ---------- Lever radio button helper ----------

# Known work-auth questions and correct answers for Karthik's OPT status
_LEVER_RADIO_MAP = [
    # (keywords_that_must_all_appear_in_label_lower, answer_value_or_label)
    (["eligible", "work", "country"],          "Yes"),   # eligible to work in the US
    (["visa sponsorship", "require"],          "Yes"),   # requires future H-1B sponsorship
    (["visa sponsorship", "future"],           "Yes"),
    (["sponsorship", "continue working"],      "Yes"),
    (["authorized to work"],                   "Yes"),
    (["legally authorized"],                   "Yes"),
    (["currently eligible"],                   "Yes"),
    (["sponsorship", "now or in the future"],  "Yes"),
]


async def _fill_lever_radio_buttons(page: Page, job: Job) -> None:
    """Answer all required radio button questions on a Lever apply form.

    Lever puts work-auth questions in cards with name like cards[UUID][fieldN].
    For each group of radios sharing a name, we read the nearest question text
    and pick the correct Yes/No answer from _LEVER_RADIO_MAP.
    Falls back to LLM if no keyword rule matches.
    """
    try:
        # Group radio buttons by their name attribute
        all_radios = await page.query_selector_all("input[type='radio']")
        seen_names: set[str] = set()
        for radio in all_radios:
            name = await radio.get_attribute("name") or ""
            if not name or name in seen_names:
                continue
            seen_names.add(name)

            # Get all radios in this group
            group = await page.query_selector_all(f"input[type='radio'][name='{name}']")
            if not group:
                continue

            # Check if already answered
            already_checked = False
            for r in group:
                if await r.is_checked():
                    already_checked = True
                    break
            if already_checked:
                continue

            # Get question label from nearest container
            label_text = await group[0].evaluate("""(e) => {
                const card = e.closest('.application-question') ||
                             e.closest('[class*="form-field"]') ||
                             e.closest('[class*="card"]') ||
                             e.closest('li') ||
                             e.parentElement?.parentElement;
                if (!card) return '';
                const lbl = card.querySelector('label:not([for]),legend,h4,h3,p,[class*="label"]');
                return (lbl || card).innerText.replace(/[\\n\\r]+/g, ' ').trim().substring(0, 300);
            }""")
            low = label_text.lower()

            # Match against known rules
            answer = None
            for keywords, ans in _LEVER_RADIO_MAP:
                if all(kw in low for kw in keywords):
                    answer = ans
                    break

            # Fallback: check memory / LLM
            if not answer:
                answer = _check_memory(label_text, job)
            if not answer:
                # Default Yes/No questions to Yes unless sponsorship-related
                if "yes" in low and "no" in low:
                    answer = "Yes"

            if not answer:
                log.info("Lever radio: no answer found for %r — skipping", label_text[:80])
                continue

            # Find and click the matching radio option
            clicked = False
            for r in group:
                # Get label for this specific radio
                r_id = await r.get_attribute("id") or ""
                r_val = await r.get_attribute("value") or ""
                label_el = None
                if r_id:
                    label_el = await page.query_selector(f"label[for='{r_id}']")
                if not label_el:
                    label_el = await r.evaluate_handle("(e) => e.closest('label') || e.nextElementSibling")
                    label_el = label_el.as_element()
                opt_text = ""
                if label_el:
                    opt_text = (await label_el.inner_text()).strip()
                if not opt_text:
                    opt_text = r_val

                if opt_text.lower() == answer.lower() or r_val.lower() == answer.lower():
                    # Use JS click — hCaptcha iframe overlays the form and
                    # blocks Playwright's pointer-based click()
                    await r.evaluate("(e) => e.click()")
                    await r.evaluate("(e) => e.dispatchEvent(new Event('change', {bubbles:true}))")
                    await page.wait_for_timeout(300)
                    clicked = True
                    log.info("Lever radio: '%s' → '%s'", label_text[:60], opt_text)
                    break

            if not clicked:
                log.warning("Lever radio: could not click '%s' for answer '%s'", label_text[:60], answer)
    except Exception as e:
        log.warning("Lever radio button fill error: %s", e)


# ---------- Lever handler ----------

async def _fill_lever(page: Page, resume_docx: str, cover_text: str, job: Job, resume_text: str) -> List[UnknownField]:
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(page.url)
    if not parsed.path.endswith("/apply") and not parsed.path.endswith("/apply/"):
        path = parsed.path.rstrip("/") + "/apply"
        apply_url = urlunparse((parsed.scheme, parsed.netloc, path, parsed.params, parsed.query, parsed.fragment))
        log.info("Redirecting from description to Lever apply form: %s", apply_url)
        await page.goto(apply_url, wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)

    pf = _personal_fields()
    selectors = {
        "input[name='name']":           pf["first_name"] + " " + pf["last_name"],
        "input[name='email']":          pf["email"],
        "input[name='phone']":          pf["phone"],
        "input[name='org']":            "University of Cincinnati",
        "input[name='urls[LinkedIn]']": pf["linkedin"],
        "input[name='urls[GitHub]']":   pf["github"],
    }
    for sel, val in selectors.items():
        if not val or not str(val).strip():
            continue
        try:
            el = await page.query_selector(sel)
            if el:
                curr_val = await el.input_value()
                if curr_val != val:
                    await _fill_humanlike(el, val)
        except Exception as e:
            log.debug("Lever fill skipped for %s: %s", sel, e)

    # ── Location autocomplete ──────────────────────────────────────────────────
    # Lever uses Google Places autocomplete; must type + select to populate
    # the hidden `selectedLocation` field that the backend validates.
    try:
        loc_input = await page.query_selector("input[name='location']")
        if loc_input:
            loc_val = pf.get("location") or "Cincinnati, OH"
            await loc_input.focus()
            await loc_input.type(loc_val.split(",")[0].strip(), delay=60)
            await page.wait_for_timeout(1500)
            # Pick first autocomplete suggestion
            suggestion = await page.query_selector(
                ".pac-item, [class*='suggestion'], [role='option'], "
                "ul[class*='autocomplete'] li"
            )
            if suggestion:
                await suggestion.click()
                log.info("Lever: location autocomplete selected")
            else:
                # Fallback: press Enter to accept the typed value
                await loc_input.press("ArrowDown")
                await page.wait_for_timeout(400)
                await loc_input.press("Enter")
                log.info("Lever: location entered via keyboard")
            await page.wait_for_timeout(500)
    except Exception as e:
        log.warning("Lever location fill failed: %s", e)

    # ── Timezone (hidden field) ────────────────────────────────────────────────
    try:
        tz_el = await page.query_selector("#applicant-timezone")
        if tz_el:
            tz_val = await tz_el.input_value()
            if not tz_val:
                await tz_el.evaluate("(e) => { e.value = 'America/New_York'; e.dispatchEvent(new Event('change', {bubbles:true})); }")
    except Exception:
        pass

    # ── Resume upload ──────────────────────────────────────────────────────────
    try:
        file_input = await page.query_selector("input[type='file'][name='resume']")
        if file_input and resume_docx:
            await file_input.set_input_files(os.path.abspath(resume_docx))
    except Exception as e:
        log.warning("Resume upload failed: %s", e)

    await _fill_cover_letter_area(page, cover_text)

    # ── Required radio buttons (work auth, sponsorship, custom questions) ──────
    await _fill_lever_radio_buttons(page, job)

    unknown: List[UnknownField] = []
    custom_fields = await page.query_selector_all(".application-question input, .application-question textarea, .application-question select")
    for el in custom_fields:
        try:
            value = await el.input_value()
            if value:
                continue
            required = await el.evaluate("(e) => e.required || e.getAttribute('aria-required') === 'true'")
            if not required:
                continue
            label = await el.evaluate("""(e) => {
                const parent = e.closest('.application-question');
                const lbl = parent?.querySelector('.application-label');
                return lbl?.innerText.trim() || e.name || '(unlabeled)';
            }""")
            tag = await el.evaluate("(e) => e.tagName.toLowerCase()")
            
            cached_ans = _check_memory(label, job)
            el_type = await el.get_attribute("type") or "text"
            _was_llm = False
            if not cached_ans and tag in ["input", "textarea"] and el_type not in ["checkbox", "radio"]:
                cached_ans = _answer_question_with_llm(label, job, resume_text)
                if cached_ans:
                    _was_llm = True
            if cached_ans:
                log.info("Answer memory hit for '%s': %s", label, cached_ans)
                filled = False
                try:
                    if tag == "select":
                        await el.select_option(label=cached_ans)
                    else:
                        await _fill_humanlike(el, cached_ans)
                    filled = True
                except Exception as e:
                    log.warning("Failed to auto-fill cached answer for %s: %s", label, e)
                if filled:
                    if _was_llm:
                        try:
                            with get_session() as msession:
                                norm = label.lower().strip()
                                existing = msession.exec(select(AnswerMemory).where(AnswerMemory.label_normalized == norm)).first()
                                if not existing:
                                    msession.add(AnswerMemory(label_normalized=norm, label_original=label, answer=cached_ans))
                                    msession.commit()
                        except Exception:
                            pass
                    continue
                    
            unknown.append(UnknownField(label=label, selector=f"*[name='{await el.get_attribute('name')}']", field_type=tag))
        except Exception:
            continue

    # Post-fill standard field verification
    for sel, val in selectors.items():
        try:
            el = await page.query_selector(sel)
            if el:
                current_val = await el.input_value()
                if not current_val:
                    await el.fill(val)
                    await page.wait_for_timeout(300)
                    if not await el.input_value():
                        label_name = sel.replace("input[name='", "").replace("']", "").replace("urls[", "").replace("]", "").replace("_", " ").title()
                        if not any(u.selector == sel for u in unknown):
                            unknown.append(UnknownField(label=f"Required: {label_name}", selector=sel, field_type="text"))
        except Exception:
            pass

    return unknown


# ---------- Ashby handler ----------

async def _fill_ashby(page: Page, resume_docx: str, cover_text: str, job: Job, resume_text: str) -> List[UnknownField]:
    # 1. Switch to Application tab if present
    app_tab = await page.query_selector("text=Application")
    if app_tab:
        log.info("Ashby: Found 'Application' tab, clicking it to reveal form...")
        await app_tab.click()
        await page.wait_for_timeout(1500)
    pf = _personal_fields()
    name_val = pf["first_name"] + " " + pf["last_name"]
    email_val = pf["email"]
    phone_val = pf["phone"]
    name_el = await page.query_selector("input[name='_systemfield_name'], input[name='name']")
    if name_el:
        await _fill_humanlike(name_el, name_val)

    # Fill Email
    email_el = await page.query_selector("input[name='_systemfield_email'], input[name='email'], input[type='email']")
    if email_el:
        await _fill_humanlike(email_el, email_val)

    # Fill Phone
    phone_el = await page.query_selector("input[type='tel'], input[name*='phone']")
    if phone_el:
        await _fill_intl_tel_input(page)
        await _fill_humanlike(phone_el, phone_val)

    # Fill LinkedIn
    linkedin_val = pf.get("linkedin", "")
    linkedin_el = await page.query_selector("input[name='_systemfield_linkedin'], input[name='linkedin'], input[name*='linkedin']")
    if linkedin_el and linkedin_val:
        await _fill_humanlike(linkedin_el, linkedin_val)

    # Fill GitHub
    github_val = pf.get("github", "")
    github_el = await page.query_selector("input[name='_systemfield_github'], input[name='github'], input[name*='github']")
    if github_el and github_val:
        await _fill_humanlike(github_el, github_val)

    # Upload Resume
    try:
        file_input = await page.query_selector("input[id='_systemfield_resume'], input[type='file']")
        if file_input and resume_docx:
            await file_input.set_input_files(os.path.abspath(resume_docx))
    except Exception as e:
        log.warning("Resume upload failed: %s", e)

    await _fill_cover_letter_area(page, cover_text)

    unknown: List[UnknownField] = []
    containers_count = len(await page.query_selector_all("[class*='_fieldEntry_']"))
    log.info("Ashby: Found %d field entry containers", containers_count)

    for idx in range(containers_count):
        # Re-query on each loop step to avoid React DOM detaching/stale element errors
        containers = await page.query_selector_all("[class*='_fieldEntry_']")
        if idx >= len(containers):
            break
        entry = containers[idx]

        try:
            # Extract question label
            question_text = await entry.evaluate("""(e) => {
                const label_el = e.querySelector('label, legend, span[class*="_label_"], div[class*="_label_"]');
                if (label_el) return label_el.innerText.trim();
                return e.innerText.split('\\n')[0].strip();
            }""")
            
            clean_question = question_text.replace('*', '').replace('\xa0', ' ').strip()
            
            # Check if required
            is_required = await entry.evaluate("""(e) => {
                const text = e.innerText || '';
                if (text.includes('*')) return true;
                const input = e.querySelector('input, textarea, select');
                if (input && (input.required || input.getAttribute('aria-required') === 'true')) return true;
                const req_indicator = e.querySelector('[class*="required"], [class*="Required"], ._asterisk_');
                if (req_indicator) return true;
                return false;
            }""")

            # Find inputs and buttons inside container
            inputs = await entry.query_selector_all("input, textarea, select")
            buttons = await entry.query_selector_all("button")

            # Determine type of question and if already answered
            is_answered = False
            field_type = "text"
            
            yes_no_buttons = []
            for btn in buttons:
                btn_txt = (await btn.inner_text()).strip()
                if btn_txt in ["Yes", "No"]:
                    yes_no_buttons.append(btn)

            input_types = []
            input_roles = []
            input_aria_popups = []
            input_checked_states = []
            input_values = []
            
            for inp in inputs:
                input_types.append(await inp.get_attribute("type") or "")
                input_roles.append(await inp.evaluate("e => e.role === 'combobox'") or "")
                input_aria_popups.append(await inp.get_attribute("aria-haspopup") or "")
                
                tag = await inp.evaluate("e => e.tagName.toLowerCase()")
                el_type = await inp.get_attribute("type") or ""
                if tag == "input" and el_type in ["checkbox", "radio"]:
                    input_checked_states.append(await inp.evaluate("e => e.checked"))
                else:
                    input_checked_states.append(False)
                    
                input_values.append(await inp.input_value() if tag in ["input", "textarea"] else "")
                
            button_classes = []
            for btn in buttons:
                button_classes.append(await btn.get_attribute("class") or "")

            # Classify field entry
            has_text_input = False
            for inp in inputs:
                tag = await inp.evaluate("e => e.tagName.toLowerCase()")
                el_type = await inp.get_attribute("type") or ""
                if el_type in ["text", "email", "tel", "url", "number"] or tag == "textarea":
                    has_text_input = True
                    break

            if has_text_input:
                field_type = "text"
                for inp_idx, val in enumerate(input_values):
                    el_type = input_types[inp_idx] if inp_idx < len(input_types) else ""
                    if el_type in ["text", "email", "tel", "url", "number"] or await inputs[inp_idx].evaluate("e => e.tagName.toLowerCase() === 'textarea'"):
                        if val:
                            is_answered = True
                            break
            elif yes_no_buttons:
                field_type = "yes_no"
                for btn in yes_no_buttons:
                    cls = await btn.get_attribute("class") or ""
                    if "_active_" in cls:
                        is_answered = True
                        break
            elif "radio" in input_types:
                field_type = "radio"
                if any(input_checked_states):
                    is_answered = True
            elif "checkbox" in input_types:
                field_type = "checkbox"
                if any(input_checked_states):
                    is_answered = True
            elif any(input_roles) or any(p == "listbox" for p in input_aria_popups) or any("toggleButton" in c for c in button_classes):
                field_type = "combobox"
                if any(val for val in input_values):
                    is_answered = True
            else:
                if "file" in input_types:
                    field_type = "file"
                    if "resume" in clean_question.lower():
                        is_answered = False
                else:
                    if any(val for val in input_values):
                        is_answered = True

            # Skip standard fields that we already filled globally
            name_attr = await inputs[0].get_attribute("name") or "" if inputs else ""
            el_id = await inputs[0].get_attribute("id") or "" if inputs else ""
            el_type = await inputs[0].get_attribute("type") or "" if inputs else ""
            if (name_attr in ["name", "email", "phone", "resume", "_systemfield_name", "_systemfield_email", "_systemfield_resume"] or
                el_id in ["_systemfield_name", "_systemfield_email", "_systemfield_resume"] or
                el_type == "tel" or
                "resume" in clean_question.lower() or
                "cover letter" in clean_question.lower()):
                continue

            if is_answered and field_type != "file":
                continue

            # Try to get answer from memory/deterministic rules
            ans = _check_memory(clean_question, job)
            _ashby_was_llm = False

            # Text/textarea custom LLM fallback
            if not ans and field_type == "text" and "resume" not in clean_question.lower():
                ans = _answer_question_with_llm(clean_question, job, resume_text)
                if ans:
                    _ashby_was_llm = True

            if ans:
                log.info("Ashby: Filling '%s' -> '%s'", clean_question, ans)
                
                # Re-query elements inside container to avoid DOM detachment after LLM delay
                inputs = await entry.query_selector_all("input, textarea, select")
                buttons = await entry.query_selector_all("button")
                yes_no_buttons = []
                for btn in buttons:
                    btn_txt = (await btn.inner_text()).strip()
                    if btn_txt in ["Yes", "No"]:
                        yes_no_buttons.append(btn)

                if field_type == "yes_no":
                    for btn in yes_no_buttons:
                        btn_txt = (await btn.inner_text()).strip()
                        if btn_txt.lower() == ans.lower():
                            try:
                                await btn.click()
                                await page.wait_for_timeout(300)
                            except Exception:
                                pass
                            break
                elif field_type in ["radio", "checkbox"]:
                    labels = await entry.query_selector_all("label")
                    question_label_el = await entry.query_selector("label[class*='question-title'], legend, .ashby-application-form-question-title")
                    clicked = False
                    for lbl in labels:
                        if question_label_el and await lbl.evaluate("(lbl, qlbl) => lbl === qlbl", question_label_el):
                            continue
                        lbl_txt = (await lbl.inner_text()).strip()
                        if ans.lower() in lbl_txt.lower() or lbl_txt.lower() in ans.lower():
                            try:
                                await lbl.click()
                                await page.wait_for_timeout(300)
                            except Exception:
                                pass
                            clicked = True
                            if field_type == "radio":
                                break
                    if not clicked:
                        opts = await entry.query_selector_all("[class*='_option_']")
                        for opt in opts:
                            opt_txt = (await opt.inner_text()).strip()
                            if ans.lower() in opt_txt.lower() or opt_txt.lower() in ans.lower():
                                try:
                                    await opt.click()
                                    await page.wait_for_timeout(300)
                                except Exception:
                                    pass
                                break
                elif field_type == "combobox":
                    inp = await entry.query_selector("input")
                    if inp:
                        await inp.focus()
                        await page.keyboard.press("Meta+A")
                        await page.keyboard.press("Backspace")
                        await _fill_humanlike(inp, ans)
                        await page.wait_for_timeout(1500)
                        
                        options = await page.query_selector_all("[role='listbox'] [role='option']")
                        clicked = False
                        for opt in options:
                            opt_txt = (await opt.inner_text()).strip()
                            if ans.lower() in opt_txt.lower() or opt_txt.lower() in ans.lower():
                                try:
                                    await opt.click()
                                    await page.wait_for_timeout(300)
                                except Exception:
                                    pass
                                clicked = True
                                break
                        if not clicked and options:
                            try:
                                await options[0].click()
                                await page.wait_for_timeout(300)
                            except Exception:
                                pass
                elif field_type == "file":
                    if resume_docx and "resume" in clean_question.lower():
                        file_input = await entry.query_selector("input[type='file']")
                        if file_input:
                            await file_input.set_input_files(os.path.abspath(resume_docx))
                else:
                    for inp in inputs:
                        el_type = await inp.get_attribute("type") or ""
                        if el_type in ["text", "email", "tel", "url", "number"] or await inp.evaluate("e => e.tagName.toLowerCase() === 'textarea'"):
                            await _fill_humanlike(inp, ans)
                            # Save to memory only after the fill succeeds
                            if _ashby_was_llm:
                                try:
                                    with get_session() as msession:
                                        norm = clean_question.lower().strip()
                                        existing = msession.exec(select(AnswerMemory).where(AnswerMemory.label_normalized == norm)).first()
                                        if not existing:
                                            msession.add(AnswerMemory(label_normalized=norm, label_original=clean_question, answer=ans))
                                            msession.commit()
                                except Exception:
                                    pass
                            # Auto-accept communicationConsent radio — query by name, not type
                            no_consent = await entry.query_selector("input[name='communicationConsent'][value='notGiven']")
                            if no_consent:
                                parent_lbl = await no_consent.evaluate_handle("e => e.closest('label')")
                                if parent_lbl:
                                    elem = parent_lbl.as_element()
                                    if elem:
                                        await elem.click()
                                        await page.wait_for_timeout(300)
                            break
            else:
                if is_required:
                    log.warning("Ashby: Required field unanswered: '%s'", clean_question)
                    inp_id = await inputs[0].get_attribute("id") if inputs else ""
                    name_attr = await inputs[0].get_attribute("name") if inputs else ""
                    if inp_id:
                        sel = f"#{inp_id}"
                    elif name_attr:
                        sel = f"*[name='{name_attr}']"
                    else:
                        field_path = await entry.get_attribute("data-field-path") or ""
                        sel = f"div[data-field-path='{field_path}']" if field_path else "[class*='_fieldEntry_']"
                        
                    unknown.append(UnknownField(label=clean_question, selector=sel, field_type=field_type))
        except Exception as e:
            log.exception("Error filling Ashby field container: %s", e)
            continue


    # Post-fill standard field verification
    selectors = {
        "input[name='_systemfield_name']": name_val,
        "input[name='_systemfield_email']": email_val,
        "input[type='tel']": phone_val,
        "input[name='_systemfield_linkedin']": linkedin_val,
        "input[name='_systemfield_github']": github_val,
    }
    for sel, val in selectors.items():
        try:
            el = await page.query_selector(sel)
            if el:
                current_val = await el.input_value()
                if not current_val:
                    if sel == "input[type='tel']":
                        await _fill_intl_tel_input(page)
                    await _fill_humanlike(el, val)
                    await page.wait_for_timeout(300)
                    if not await el.input_value():
                        label_name = sel.replace("input[name='", "").replace("']", "").replace("_", " ").title()
                        if not any(u.selector == sel for u in unknown):
                            unknown.append(UnknownField(label=f"Required: {label_name}", selector=sel, field_type="text"))
        except Exception:
            pass
    # Resilient fallback for LinkedIn and GitHub inputs in Ashby if standard selectors missed
    try:
        all_inputs = await page.query_selector_all("input")
        for inp in all_inputs:
            inp_type = await inp.get_attribute("type") or ""
            if inp_type not in ["text", "url", ""]:
                continue
            parent = await inp.evaluate_handle("e => e.closest('[class*=\"_fieldEntry_\"]') || e.parentElement")
            if parent:
                parent_text = (await parent.as_element().inner_text()).lower()
                if "linkedin" in parent_text and linkedin_val:
                    curr_val = await inp.input_value()
                    if not curr_val:
                        log.info("Ashby post-fill: Found empty LinkedIn input via parent text, filling...")
                        await _fill_humanlike(inp, linkedin_val)
                        await page.wait_for_timeout(300)
                elif "github" in parent_text and github_val:
                    curr_val = await inp.input_value()
                    if not curr_val:
                        log.info("Ashby post-fill: Found empty GitHub input via parent text, filling...")
                        await _fill_humanlike(inp, github_val)
                        await page.wait_for_timeout(300)
    except Exception as e:
        log.warning("Ashby post-fill fallback check failed: %s", e)

    return unknown


# ---------- dispatcher ----------

async def _autofill_one(application_id: int) -> List[UnknownField]:
    with get_session() as session:
        app = session.get(Application, application_id)
        if not app:
            raise ValueError(f"Application {application_id} not found")
        job = session.get(Job, app.job_id)
        apply_url = app.apply_url or job.url
        resume_path = app.tailored_resume_path
        cover_path = app.cover_letter_path

    cover_text = ""
    if cover_path:
        from pathlib import Path
        cover_text = Path(cover_path).read_text(encoding="utf-8")
        if "---COVER---" in cover_text:
            cover_text = cover_text.split("---COVER---", 1)[-1].strip()

    resume_text = _load_resume()
    from urllib.parse import parse_qs
    parsed_url   = urlparse(apply_url)
    host         = parsed_url.netloc
    query_params = parse_qs(parsed_url.query)

    # Detect Greenhouse by hostname OR by gh_jid / gh_src query params
    # (companies like Pinterest embed GH on their own careers domain)
    is_greenhouse = (
        "greenhouse" in host
        or "boards.greenhouse" in host
        or "gh_jid" in query_params
        or "gh_src" in query_params
    )

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=settings.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--disable-infobars",
                "--window-position=0,0",
                "--ignore-certificate-errors",
            ],
        )
        browser_ver = browser.version
        context = await browser.new_context(
            user_agent=f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{browser_ver} Safari/537.36",
            viewport={"width": 1280, "height": 800},
            locale="en-US",
        )
        await context.add_init_script(_STEALTH_INIT_JS)
        page = await context.new_page()
        await page.goto(apply_url, wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)

        # Check for initial CAPTCHA
        await _handle_captcha(page, application_id, job)

        if is_greenhouse:
            is_custom_domain = "greenhouse" not in host and "boards.greenhouse" not in host
            fill_target = page  # default: fill on main page

            if is_custom_domain:
                # Custom career pages embed Greenhouse via iframe — click Apply to trigger it
                gh_frame = None
                for f in page.frames:
                    if "greenhouse.io/embed" in f.url or "greenhouse.io/embed" in f.url:
                        gh_frame = f
                        break

                if not gh_frame:
                    clicked = await _click_apply_and_load_form(page)
                    log.info("Custom GH domain (%s): Apply clicked=%s, waiting for embed iframe", host, clicked)
                    try:
                        await page.wait_for_function(
                            "() => Array.from(window.frames).some(f => { try { return f.location.href.includes('greenhouse.io/embed'); } catch(e) { return false; } })",
                            timeout=8000,
                        )
                    except Exception:
                        pass
                    for f in page.frames:
                        if "greenhouse.io/embed" in f.url:
                            gh_frame = f
                            break

                if gh_frame:
                    log.info("Found Greenhouse embed iframe: %s", gh_frame.url[:80])
                    await gh_frame.wait_for_load_state("domcontentloaded")
                    await page.wait_for_timeout(1500)
                    fill_target = _FrameProxy(gh_frame, page)
                else:
                    log.warning("No Greenhouse embed iframe found on %s — falling back to main page", host)

            else:
                # Standard job-boards.greenhouse.io / boards.greenhouse.io:
                # URL points to the job description page. Check if form is already present;
                # if not (form fields absent), click the Apply button to navigate to the form.
                form_present = await page.query_selector("#first_name, input[name='job_application[first_name]'], #resume_upload_or_paste")
                if not form_present:
                    log.info("GH standard domain: form fields not visible on %s — clicking Apply button", apply_url)
                    clicked = await _click_apply_and_load_form(page)
                    if clicked:
                        await page.wait_for_timeout(2500)
                        # If we navigated to a new URL, wait for DOM
                        await page.wait_for_load_state("domcontentloaded")
                    else:
                        # Try appending /apply to the URL path
                        from urllib.parse import urlparse as _up2, urlunparse as _uu2
                        _p = _up2(apply_url)
                        if not _p.path.endswith("/apply"):
                            form_url = _uu2((_p.scheme, _p.netloc, _p.path.rstrip("/") + "/apply", "", "", ""))
                            log.info("GH: navigating to form URL: %s", form_url)
                            await page.goto(form_url, wait_until="domcontentloaded")
                            await page.wait_for_timeout(2000)

            unknown = await _fill_greenhouse(fill_target, resume_path, cover_text, job, resume_text)
            verify_target = fill_target
        elif "lever.co" in host:
            unknown = await _fill_lever(page, resume_path, cover_text, job, resume_text)
            verify_target = page
        elif "ashbyhq.com" in host:
            unknown = await _fill_ashby(page, resume_path, cover_text, job, resume_text)
            verify_target = page
        else:
            log.warning("No handler for %s yet — falling through", host)
            unknown = []
            verify_target = page

        log.info("Autofill complete. %d unknown fields.", len(unknown))

        # Check for post-filling CAPTCHA
        await _handle_captcha(page, application_id, job)

        # ── Post-fill verification — confirm critical fields actually stuck ──
        # Re-fills any mismatch once; surfaces the result on the review prompt.
        verify_report = None
        try:
            verify_report = await _verify_filled_fields(verify_target, _personal_fields())
        except Exception as e:
            log.warning("Field verification step failed (continuing): %s", e)

        # If fully filled, run pre-submit screenshot review
        if not unknown:
            from datetime import datetime
            with get_session() as session:
                app_db = session.get(Application, application_id)
                if app_db:
                    app_db.status = ApplicationStatus.READY_TO_SUBMIT
                    app_db.updated_at = datetime.utcnow()
                    session.add(app_db)
                    session.commit()

            verify_note = ""
            if verify_report is not None:
                if verify_report.all_ok:
                    verify_note = f"Field check: {verify_report.summary()}"
                else:
                    miss = ", ".join(m.field for m in verify_report.mismatches)
                    verify_note = f"⚠️ Field check FAILED — could not confirm: {miss}. {verify_report.summary()}"
            review_res = await _handle_pre_submit_review(page, application_id, job, verify_note=verify_note)
            if review_res == "approve":
                log.info("Submission approved. Clicking submit...")
                clicked = await _click_submit(page)
                if clicked:
                    await page.wait_for_timeout(5000)
                    is_submitted = False
                    success_keywords = ["/thanks", "/thank", "success", "confirmation", "submitted"]
                    if any(kw in page.url.lower() for kw in success_keywords):
                        is_submitted = True
                    else:
                        try:
                            body_text = await page.locator("body").inner_text()
                            body_text_lower = body_text.lower()
                            success_texts = [
                                "thank you for applying",
                                "successfully submitted",
                                "application submitted",
                                "received your application",
                                "thanks for applying",
                                "we have received your application"
                            ]
                            if any(st in body_text_lower for st in success_texts):
                                is_submitted = True
                        except Exception:
                            pass
                            
                    if is_submitted:
                        from datetime import datetime
                        from app.analytics.funnel import FunnelTracker
                        with get_session() as session:
                            app_db = session.get(Application, application_id)
                            if app_db:
                                app_db.status = ApplicationStatus.SUBMITTED
                                app_db.submitted_at = datetime.utcnow()
                                app_db.updated_at = datetime.utcnow()
                                session.add(app_db)
                                session.commit()
                        FunnelTracker.record(job.id, "applied", True, metadata={"method": "headless_approved"})
                        try:
                            import httpx
                            msg = f"🚀 *Application successfully submitted* to *{job.company}* for *{job.title}*!"
                            httpx.post(
                                f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
                                json={"chat_id": settings.telegram_chat_id, "text": msg, "parse_mode": "Markdown"},
                                timeout=10,
                            )
                        except Exception as e:
                            log.warning("Failed to send submission success notification: %s", e)
                    else:
                        log.warning("Submit button clicked, but success page/text not detected.")
                else:
                    log.error("Could not find submit button to click.")
            elif review_res == "reject":
                with get_session() as session:
                    app_db = session.get(Application, application_id)
                    if app_db:
                        from datetime import datetime
                        app_db.status = ApplicationStatus.SKIPPED
                        app_db.updated_at = datetime.utcnow()
                        session.add(app_db)
                        session.commit()
                try:
                    import httpx
                    msg = f"❌ *Application aborted* for *{job.company}* — *{job.title}* (marked as skipped)."
                    httpx.post(
                        f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
                        json={"chat_id": settings.telegram_chat_id, "text": msg, "parse_mode": "Markdown"},
                        timeout=10,
                    )
                except Exception as e:
                    log.warning("Failed to send abort notification: %s", e)

        return unknown


async def _preview_one(application_id: int) -> None:
    """Re-fill the form in headful mode and keep the browser open until user closes it."""
    if application_id in _active_previews:
        existing_page = _active_previews[application_id]
        try:
            if not existing_page.is_closed():
                log.info("Bringing existing browser window to front for app %d", application_id)
                await existing_page.bring_to_front()
                return
        except Exception as e:
            log.debug("Failed to bring existing page to front: %s", e)
            _active_previews.pop(application_id, None)

    with get_session() as session:
        app = session.get(Application, application_id)
        if not app:
            raise ValueError(f"Application {application_id} not found")
        job = session.get(Job, app.job_id)
        apply_url = app.apply_url or job.url
        resume_path = app.tailored_resume_path
        cover_path = app.cover_letter_path

    cover_text = ""
    if cover_path:
        from pathlib import Path
        cover_text = Path(cover_path).read_text(encoding="utf-8")
        if "---COVER---" in cover_text:
            cover_text = cover_text.split("---COVER---", 1)[-1].strip()

    resume_text = _load_resume()
    from urllib.parse import parse_qs as _parse_qs
    _parsed   = urlparse(apply_url)
    host      = _parsed.netloc
    _qparams  = _parse_qs(_parsed.query)
    is_greenhouse = (
        "greenhouse" in host
        or "boards.greenhouse" in host
        or "gh_jid" in _qparams
        or "gh_src" in _qparams
    )
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=False,
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
            )
            browser_ver = browser.version
            context = await browser.new_context(
                user_agent=f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{browser_ver} Safari/537.36",
                viewport={"width": 1280, "height": 800},
                locale="en-US",
            )
            await context.add_init_script(_STEALTH_INIT_JS)
            page = await context.new_page()
            _active_previews[application_id] = page

            try:
                await page.goto(apply_url, wait_until="domcontentloaded")
                await page.wait_for_timeout(2000)

                if is_greenhouse:
                    is_custom_domain = "greenhouse" not in host and "boards.greenhouse" not in host
                    fill_target = page
                    if is_custom_domain:
                        gh_frame = None
                        for f in page.frames:
                            if "greenhouse.io/embed" in f.url:
                                gh_frame = f
                                break
                        if not gh_frame:
                            await _click_apply_and_load_form(page)
                            try:
                                await page.wait_for_function(
                                    "() => Array.from(window.frames).some(f => { try { return f.location.href.includes('greenhouse.io/embed'); } catch(e) { return false; } })",
                                    timeout=8000,
                                )
                            except Exception:
                                await page.wait_for_timeout(3000)
                            for f in page.frames:
                                if "greenhouse.io/embed" in f.url:
                                    gh_frame = f
                                    break
                        if gh_frame:
                            await gh_frame.wait_for_load_state("domcontentloaded")
                            fill_target = _FrameProxy(gh_frame, page)
                    await _fill_greenhouse(fill_target, resume_path, cover_text, job, resume_text)
                elif "lever.co" in host:
                    await _fill_lever(page, resume_path, cover_text, job, resume_text)
                elif "ashbyhq.com" in host:
                    await _fill_ashby(page, resume_path, cover_text, job, resume_text)
                else:
                    log.warning("No handler for %s yet — falling through", host)
            except Exception as e:
                log.exception("Error during form filling in preview mode: %s", e)

            log.info("Form filled and open in browser. Close the browser window when done.")
            try:
                is_submitted = False
                for _ in range(3600):
                    try:
                        if page.is_closed():
                            break

                        # --- Live Auto-Save ---
                        try:
                            js_extract_script = """() => {
                                const data = [];
                                const inputs = document.querySelectorAll("input, textarea, select");
                                for (const el of inputs) {
                                    const type = el.getAttribute("type");
                                    if (type === "file" || type === "hidden" || type === "submit") continue;
                                    
                                    let val = "";
                                    let label = "";
                                    const tagName = el.tagName.toLowerCase();
                                    
                                    if (type === "radio" || type === "checkbox") {
                                        if (!el.checked) continue;
                                        
                                        let optionLabel = "";
                                        if (el.id) {
                                            const lbl = document.querySelector(`label[for='${el.id}']`);
                                            if (lbl) optionLabel = lbl.innerText.trim();
                                        }
                                        if (!optionLabel) {
                                            const parentLabel = el.closest('label');
                                            if (parentLabel) optionLabel = parentLabel.innerText.trim();
                                        }
                                        if (!optionLabel) {
                                            optionLabel = el.value ? el.value.trim() : "";
                                        }
                                        if (!optionLabel || optionLabel.toLowerCase() === "on") {
                                            optionLabel = "Yes";
                                        }
                                        val = optionLabel;
                                        
                                        const parent = el.closest('[class*="field"], [class*="question"], [class*="_fieldEntry_"], fieldset, li, .application-question, .form-question');
                                        const lbl = parent?.querySelector('label, .label, legend, .application-label, span[class*="_label_"], div[class*="_label_"]');
                                        label = lbl ? lbl.innerText : "";
                                    } else {
                                        if (tagName === "select") {
                                            const opt = el.options[el.selectedIndex];
                                            val = opt ? opt.text.trim() : "";
                                        } else {
                                            val = el.value ? el.value.trim() : "";
                                        }
                                        
                                        if (el.id) {
                                            const lbl = document.querySelector(`label[for='${el.id}']`);
                                            if (lbl) label = lbl.innerText;
                                        }
                                        if (!label) {
                                            const parent = el.closest('[class*="field"], [class*="question"], [class*="_fieldEntry_"], fieldset, li, .application-question, .form-question');
                                            const lbl = parent?.querySelector('label, .label, legend, .application-label, span[class*="_label_"], div[class*="_label_"]');
                                            label = lbl ? lbl.innerText : "";
                                        }
                                        if (!label) {
                                            label = el.placeholder || el.name || el.id || "";
                                        }
                                    }
                                    if (!val) continue;
                                    
                                    label = label.replace(/[\\*\\n\\r]+/g, ' ').replace(/[:\\s]+$/, '').trim();
                                    if (label && val) {
                                        data.push({ label: label, value: val });
                                    }
                                }
                                
                                const comboboxes = document.querySelectorAll("input[role='combobox']");
                                for (const cb of comboboxes) {
                                    const ctrl = cb.closest('.select__control');
                                    if (ctrl) {
                                        const singleValEl = ctrl.querySelector('.select__single-value');
                                        const val = singleValEl ? singleValEl.innerText.trim() : "";
                                        if (!val) continue;
                                        
                                        let label = "";
                                        if (cb.id) {
                                            const lbl = document.querySelector(`label[for='${cb.id}']`);
                                            if (lbl) label = lbl.innerText;
                                        }
                                        if (!label) {
                                            const parent = cb.closest('[class*="field"], [class*="question"], li, .application-question, .form-question');
                                            const lbl = parent?.querySelector('label, .label, legend, .application-label');
                                            label = lbl ? lbl.innerText : "";
                                        }
                                        label = label.replace(/[\\*\\n\\r]+/g, ' ').replace(/[:\\s]+$/, '').trim();
                                        if (label && val) {
                                            data.push({ label: label, value: val });
                                        }
                                    }
                                }
                                return data;
                            }"""
                            fields = await page.evaluate(js_extract_script)
                            if fields:
                                from datetime import datetime
                                with get_session() as session:
                                    for f in fields:
                                        label_orig = f["label"]
                                        label_norm = label_orig.lower().strip()
                                        val = f["value"]
                                        if len(label_norm) < 3 or len(val) < 1:
                                            continue
                                        existing = session.exec(
                                            select(AnswerMemory).where(AnswerMemory.label_normalized == label_norm)
                                        ).first()
                                        if existing:
                                            if existing.answer != val:
                                                existing.answer = val
                                                existing.last_used_at = datetime.utcnow()
                                                session.add(existing)
                                        else:
                                            session.add(
                                                AnswerMemory(
                                                    label_normalized=label_norm,
                                                    label_original=label_orig,
                                                    answer=val,
                                                    last_used_at=datetime.utcnow(),
                                                    use_count=1
                                                )
                                            )
                                    session.commit()
                        except Exception as e:
                            log.debug("Auto-save failed: %s", e)

                        current_url = page.url.lower()
                        success_keywords = ["/thanks", "/thank", "success", "confirmation", "submitted"]
                        url_matched = any(kw in current_url for kw in success_keywords)
                        
                        text_matched = False
                        if not url_matched:
                            try:
                                body_text = await page.locator("body").inner_text()
                                body_low = body_text.lower()
                                success_phrases = [
                                    "thank you for applying",
                                    "thanks for applying",
                                    "thank you for your interest",
                                    "thanks for your interest",
                                    "application received",
                                    "application has been received",
                                    "successfully submitted",
                                    "application submitted",
                                    "application sent",
                                    "we have received your application",
                                    "we've received your application"
                                ]
                                if any(phrase in body_low for phrase in success_phrases):
                                    text_matched = True
                                    log.info("Submission detected via page success text match")
                            except Exception:
                                pass
                                
                        if url_matched or text_matched:
                            is_submitted = True
                            log.info("Submission detected! URL: %s", page.url)
                            break
                    except Exception as loop_err:
                        log.debug("Preview loop iteration warning (possibly transient): %s", loop_err)
                    await asyncio.sleep(1)

                if is_submitted:
                    from datetime import datetime
                    with get_session() as session:
                        app_db = session.get(Application, application_id)
                        if app_db:
                            app_db.status = ApplicationStatus.SUBMITTED
                            app_db.submitted_at = datetime.utcnow()
                            app_db.updated_at = datetime.utcnow()
                            session.add(app_db)
                            session.commit()
                            log.info("Application %d status updated to SUBMITTED", application_id)

                if not page.is_closed():
                    await page.wait_for_event("close", timeout=0)
            except Exception as e:
                log.warning("Submission detection or close wait failed: %s", e)
    finally:
        _active_previews.pop(application_id, None)


def autofill(application_id: int, bypass_delay: bool = False) -> List[UnknownField]:
    """Sync wrapper — fills form, saves pending questions, updates status, and notifies via Telegram."""
    from datetime import datetime, time
    import time as time_module
    import random
    import httpx

    # 1. Daily limit check
    with get_session() as session:
        app = session.get(Application, application_id)
        if not app:
            raise ValueError(f"Application {application_id} not found")
        job = session.get(Job, app.job_id)
        job_company = job.company
        job_title = job.title
        
        # Count today's submitted applications
        today_start = datetime.combine(datetime.utcnow().date(), time.min)
        today_count = len(session.exec(
            select(Application)
            .where(Application.status == ApplicationStatus.SUBMITTED)
            .where(Application.submitted_at >= today_start)
        ).all())
        
        if today_count >= settings.daily_apply_limit:
            msg = (
                f"⚠️ *Daily Apply Limit Reached* ({today_count}/{settings.daily_apply_limit})\n"
                f"Skipping autofill for *{job_company}* — _{job_title}_ to pace submissions."
            )
            log.warning(msg)
            try:
                httpx.post(
                    f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
                    json={"chat_id": settings.telegram_chat_id, "text": msg, "parse_mode": "Markdown"},
                    timeout=10,
                )
            except Exception as e:
                log.warning("Telegram notification failed: %s", e)
            return []

    # 1b. Manual-track jobs: don't autofill — just prep materials and notify
    with get_session() as session:
        app = session.get(Application, application_id)
        if app and app.apply_track == "manual":
            job = session.get(Job, app.job_id)
            apply_url = app.apply_url or job.url
            job_title = job.title
            job_company = job.company
            job_source = job.source.value if job.source else "unknown"

    if app and app.apply_track == "manual":
        # Ensure materials are tailored first
        with get_session() as session:
            app = session.get(Application, application_id)
            is_shortlisted = app.status == ApplicationStatus.SHORTLISTED
            has_no_resume = not app.tailored_resume_path
        if is_shortlisted or has_no_resume:
            from app.tailoring.tailor import tailor_for_application
            try:
                tailor_for_application(application_id)
            except Exception as e:
                log.exception("Auto-tailoring failed for manual app %d: %s", application_id, e)

        with get_session() as session:
            app = session.get(Application, application_id)
            if app and app.status not in [ApplicationStatus.ERROR, ApplicationStatus.SKIPPED]:
                from datetime import datetime
                app.status = ApplicationStatus.AWAITING_USER
                app.updated_at = datetime.utcnow()
                session.add(app)
                session.commit()

        try:
            import httpx
            msg = (
                f"📌 *Manual Apply Ready* [{job_source.upper()}]\n"
                f"*{job_company}* — _{job_title}_\n\n"
                f"✅ Tailored resume + cover letter prepared.\n\n"
                f"👉 [Open & Apply]({apply_url})\n\n"
                f"Send /skip to dismiss or /manual to see all pending manual jobs."
            )
            httpx.post(
                f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
                json={
                    "chat_id": settings.telegram_chat_id,
                    "text": msg,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": False,
                },
                timeout=10,
            )
        except Exception as e:
            log.warning("Manual-track Telegram notification failed: %s", e)

        log.info("Manual-track app %d: materials ready, user notified.", application_id)
        return []

    # 2. Pacing delay
    if not bypass_delay:
        delay = random.uniform(settings.submission_jitter_min, settings.submission_jitter_max)
        if delay > 0:
            msg = f"⏳ *Submission Pacing:* Waiting {delay:.0f} seconds before autofilling form for *{job_company}*..."
            log.info(msg)
            try:
                httpx.post(
                    f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
                    json={"chat_id": settings.telegram_chat_id, "text": msg, "parse_mode": "Markdown"},
                    timeout=10,
                )
            except Exception as e:
                log.warning("Telegram notification failed: %s", e)
            time_module.sleep(delay)
    else:
        log.info("Pacing delay bypassed for manual/interactive trigger.")

    # Auto-tailor if not done yet
    with get_session() as session:
        app = session.get(Application, application_id)
        if not app:
            raise ValueError(f"Application {application_id} not found")
        is_shortlisted = app.status == ApplicationStatus.SHORTLISTED
        has_no_resume = not app.tailored_resume_path

    if is_shortlisted or has_no_resume:
        from app.tailoring.tailor import tailor_for_application
        log.info("Application %d is not tailored yet. Tailoring now...", application_id)
        try:
            tailor_for_application(application_id)
        except Exception as e:
            log.exception("Auto-tailoring failed for application %d: %s", application_id, e)

    # Clear old unanswered pending questions for this app so we get fresh ones
    with get_session() as session:
        old_pqs = session.exec(
            select(PendingQuestion).where(PendingQuestion.application_id == application_id)
        ).all()
        for pq in old_pqs:
            session.delete(pq)
        session.commit()

    try:
        unknown = asyncio.run(_autofill_one(application_id))
    except CaptchaDetectedError:
        with get_session() as session:
            app = session.get(Application, application_id)
            if app:
                from datetime import datetime
                app.status = ApplicationStatus.AWAITING_USER
                app.updated_at = datetime.utcnow()
                session.add(app)
                session.commit()
        return []

    # Re-check status in case it was submitted/skipped during headless review
    with get_session() as session:
        app = session.get(Application, application_id)
        current_status = app.status if app else None

    # If submitted/skipped, we don't save new questions or notify/preview
    if current_status in [ApplicationStatus.SUBMITTED, ApplicationStatus.SKIPPED]:
        return []

    with get_session() as session:
        for uf in unknown:
            session.add(
                PendingQuestion(
                    application_id=application_id,
                    field_label=uf.label,
                    field_selector=uf.selector,
                    field_type=uf.field_type,
                )
            )
        app = session.get(Application, application_id)
        job = session.get(Job, app.job_id)
        from datetime import datetime
        app.status = (
            ApplicationStatus.AWAITING_USER if unknown else ApplicationStatus.READY_TO_SUBMIT
        )
        app.updated_at = datetime.utcnow()
        session.add(app)
        session.commit()
        job_title = job.title
        job_company = job.company

    # Proactively ping Telegram with the first question
    try:
        import httpx
        if unknown:
            first = unknown[0]
            msg = (
                f"🤖 *JobAgent* — Form filled for *{job_company}*\n"
                f"📋 Role: _{job_title}_\n\n"
                f"Found *{len(unknown)} question(s)* I couldn't answer automatically.\n\n"
                f"*Question 1 of {len(unknown)}:*\n{first.label}\n\n"
                f"Reply with your answer, or send /next to see this again."
            )
        else:
            msg = (
                f"✅ *{job_company}* — Form fully filled!\n"
                f"_{job_title}_\n\nNo custom questions needed. *Launching browser for final verification...*"
            )
        httpx.post(
            f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
            json={"chat_id": settings.telegram_chat_id, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
        log.info("Telegram notified: %d pending questions for app %d", len(unknown), application_id)
        
        # If fully filled, automatically launch preview so the user can "check full form"
        if not unknown:
            log.info("Launching automatic preview for app %d", application_id)
            preview(application_id)
            
    except Exception as e:
        log.warning("Telegram notification or auto-preview failed: %s", e)

    return unknown



def preview(application_id: int) -> None:
    """Re-open the filled form in a visible browser window so the user can review and submit."""
    loop = get_main_loop()
    if loop and loop.is_running():
        log.info("Scheduling preview for app %d on the main event loop", application_id)
        asyncio.run_coroutine_threadsafe(_preview_one(application_id), loop)
    else:
        log.info("Main event loop not running. Running preview synchronously on current thread")
        asyncio.run(_preview_one(application_id))


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python -m app.autofill.agent <application_id>")
        sys.exit(1)
    autofill(int(sys.argv[1]))
