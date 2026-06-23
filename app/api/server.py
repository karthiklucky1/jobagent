"""Small FastAPI surface for status + manual triggers.

Endpoints:
  GET  /health           — liveness
  GET  /stats            — counts by status
  POST /run/discovery    — kick off discovery
  POST /run/matching     — kick off matching
  POST /run/tailor       — tailor all SHORTLISTED
  POST /run/autofill/{id} — autofill one application
"""
from __future__ import annotations

import logging
from typing import Optional, Optional as _Opt

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile

log = logging.getLogger(__name__)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlmodel import select
from sqlalchemy import func, desc

from app.autofill.agent import autofill, preview
from app.db.init_db import get_session
from app.db.models import (
    Application, ApplicationStatus, Job,
    UserSubscription, UserUsage, PlanTier, PLAN_LIMITS,
)
from app.discovery.pipeline import run_discovery
from app.matching.pipeline import run_matching
from app.tailoring.tailor import tailor_all_shortlisted

app = FastAPI(title="JobAgent")

# ── Rate limiting ─────────────────────────────────────────────────────────────
# slowapi wraps the same limits as Flask-Limiter; uses the client IP as the key.
# Sensitive / expensive endpoints get tighter per-minute caps.

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    _limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])
    app.state.limiter = _limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    _RATE_LIMIT_AVAILABLE = True
except ImportError:
    _RATE_LIMIT_AVAILABLE = False
    log.warning("slowapi not installed — rate limiting disabled. Run: pip install slowapi")

def _rate_limit(limit: str):
    """Decorator factory that applies a slowapi rate limit if available, else no-op."""
    def decorator(fn):
        if _RATE_LIMIT_AVAILABLE:
            return _limiter.limit(limit)(fn)
        return fn
    return decorator


# Serve static files (PWA manifest, icons, service worker)
from fastapi.staticfiles import StaticFiles
import os as _os
_static_dir = _os.path.join(_os.path.dirname(__file__), "..", "static")
if _os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ── Supabase session-refresh middleware ─────────────────────────────────────
# When SUPABASE_URL is configured, every response gets a refreshed access
# token in the X-Supabase-Token header so the frontend can keep the session
# alive without the user having to re-login every hour.

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

class SupabaseSessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        from app.config import settings
        if not settings.use_supabase:
            return response
        # Only attempt refresh if a bearer token is present
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return response
        token = auth_header.split(" ", 1)[1]
        try:
            from app.db.supabase_client import anon_client
            sb = anon_client()
            result = sb.auth.refresh_session(token)
            if result and result.session:
                response.headers["X-Supabase-Token"] = result.session.access_token
        except Exception:
            pass
        return response

app.add_middleware(SupabaseSessionMiddleware)


# ── Security audit logging ────────────────────────────────────────────────────
# Logs every 401/403/429 response with IP + path so failed-auth patterns are
# visible in production log streams (Railway / Supabase logs).

import json as _json

_sec_log = logging.getLogger("security")

class SecurityAuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        if response.status_code in (401, 403, 429):
            ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "unknown").split(",")[0].strip()
            event = {
                "event": "auth_failure" if response.status_code in (401, 403) else "rate_limited",
                "status": response.status_code,
                "method": request.method,
                "path": request.url.path,
                "ip": ip,
                "ua": request.headers.get("user-agent", "")[:120],
            }
            _sec_log.warning(_json.dumps(event))
        return response

app.add_middleware(SecurityAuditMiddleware)


# ── Auth helpers ─────────────────────────────────────────────────────────────

def _get_user_id(request: Request) -> str | None:
    """Extract user_id from the Supabase JWT. Returns None if not authenticated.

    Token sources, in order:
    1. Authorization: Bearer header — used by all fetch()/extension API calls.
    2. sb_token cookie — needed for full-page navigations (e.g. GET /dashboard),
       since the browser does NOT attach the Authorization header to those, only
       to fetch(). Without this, server-rendered pages can't identify the user.
    """
    from app.config import settings
    if not settings.use_supabase:
        return "local"   # single-user mode — no auth
    auth = request.headers.get("Authorization", "")
    token = auth.split(" ", 1)[1] if auth.startswith("Bearer ") else None
    if not token:
        token = request.cookies.get("sb_token")
    if not token:
        return None
    from app.db.supabase_client import get_user_id_from_token
    return get_user_id_from_token(token)


def _require_user(request: Request) -> str:
    """Like _get_user_id but raises 401 if not authenticated."""
    uid = _get_user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return uid


def _user_has_resume(uid: str | None) -> bool:
    """True if the user has uploaded (or synthesized) a resume.

    Supabase mode → check the ``resumes/{uid}/`` storage folder.
    Local mode    → check for any ``./data/resume_master.*`` file.
    """
    from app.config import settings
    if settings.use_supabase and uid and uid != "local":
        try:
            from app.db.supabase_client import service_client
            sb = service_client()
            files = sb.storage.from_("resume").list(uid)
            return any((f.get("name") or "").startswith("resume.") for f in (files or []))
        except Exception:
            return False
    import glob
    return bool(glob.glob("./data/resume_master.*"))


def _require_owned_application(request: Request, application_id: int):
    """Load an Application, enforcing that it belongs to the requesting user.

    Raises 401 if unauthenticated, 404 if missing or owned by someone else
    (404 not 403 — don't leak that the ID exists). In SQLite single-user mode
    (uid == "local") ownership is not enforced.
    """
    uid = _require_user(request)
    with get_session() as session:
        application = session.get(Application, application_id)
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")
        if uid != "local" and application.user_id != uid:
            raise HTTPException(status_code=404, detail="Application not found")
    return uid


@app.on_event("startup")
async def startup_event():
    import asyncio
    from app.autofill.agent import set_main_loop
    set_main_loop(asyncio.get_running_loop())
    # Create all DB tables at runtime (after env vars are injected by Railway)
    from app.db.init_db import init_db
    init_db()
    # Start background scheduler — runs discovery + matching every 6 hours
    asyncio.create_task(_scheduler())


async def _scheduler():
    """Run discovery → matching every 6 hours for each user who has a resume."""
    import asyncio
    import logging
    _log = logging.getLogger("scheduler")
    INTERVAL = 6 * 60 * 60
    await asyncio.sleep(120)  # let server fully boot first
    while True:
        try:
            from app.db.models import UserProfile
            with get_session() as session:
                users = session.exec(select(UserProfile)).all()
            user_ids = [u.user_id for u in users if u.user_id and _user_has_resume(u.user_id)]
            if not user_ids:
                _log.info("Scheduler: no users with resumes, skipping run")
            for uid in user_ids:
                try:
                    _log.info("Scheduler: running discovery + matching for user %s", uid)
                    await asyncio.to_thread(_discover_then_match, uid)
                except Exception as e:
                    _log.exception("Scheduler error for user %s: %s", uid, e)
        except Exception as e:
            _log.exception("Scheduler outer error: %s", e)
        await asyncio.sleep(INTERVAL)


try:
    templates = Jinja2Templates(directory="app/templates")
except Exception:
    import os
    templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "..", "templates"))


def _fromjson_filter(value):
    """Parse a JSON string into a Python object for templates; tolerant of
    None / already-parsed lists / malformed strings (returns [] on failure)."""
    if value is None or value == "":
        return []
    if isinstance(value, (list, dict)):
        return value
    try:
        import json as _json
        return _json.loads(value)
    except Exception:
        return []


def _cleantext_filter(value):
    """Turn scraped HTML / entity-laden text into clean, readable plain text:
    block tags → newlines, <li> → bullets, strip remaining tags, unescape
    entities, normalize whitespace. Job descriptions often arrive as raw HTML."""
    if not value:
        return ""
    import re as _re, html as _html
    text = str(value)
    # Preserve structure: turn list / line-break / block-close tags into newlines
    text = _re.sub(r'(?i)<\s*li[^>]*>', '\n• ', text)
    text = _re.sub(r'(?i)<\s*(br|/p|/div|/li|/ul|/h[1-6]|/tr)\s*/?>', '\n', text)
    # Drop everything else that looks like a tag
    text = _re.sub(r'<[^>]+>', '', text)
    # Decode entities (run twice to catch double-escaped &amp;nbsp;)
    text = _html.unescape(_html.unescape(text)).replace('\xa0', ' ')
    # Tidy each line; common scraper bullets (~, -, *, •) → "• "
    out, blank = [], 0
    for ln in text.splitlines():
        ln = _re.sub(r'[ \t]+', ' ', ln).strip()
        ln = _re.sub(r'^[~\-\*•]\s+', '• ', ln)
        if ln == '':
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append(ln)
    return '\n'.join(out).strip()


templates.env.filters["fromjson"] = _fromjson_filter
templates.env.filters["cleantext"] = _cleantext_filter


def _sponsorship_of(job):
    """Jinja global: legal sponsorship assessment for a job (cap-exempt aware)."""
    try:
        from app.intelligence.sponsorship import assess
        return assess(company=getattr(job, "company", "") or "",
                      description=getattr(job, "description", "") or "",
                      url=getattr(job, "url", "") or "")
    except Exception:
        return None


templates.env.globals["sponsorship_of"] = _sponsorship_of


def _urgency_of(job):
    """Jinja global: timing/urgency assessment for a job (fresh / hard-to-fill)."""
    try:
        from app.intelligence.urgency import assess
        return assess(job)
    except Exception:
        return None


templates.env.globals["urgency_of"] = _urgency_of


# ── Public / marketing pages ─────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request=request, name="landing.html", context={})


@app.get("/pricing", response_class=HTMLResponse)
def pricing_page(request: Request):
    return templates.TemplateResponse(request=request, name="pricing.html", context={})


@app.get("/privacy", response_class=HTMLResponse)
def privacy_page(request: Request):
    return templates.TemplateResponse(request=request, name="privacy.html", context={})


@app.get("/terms", response_class=HTMLResponse)
def terms_page(request: Request):
    return templates.TemplateResponse(request=request, name="terms.html", context={})


# ── Auth pages ───────────────────────────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    from app.config import settings
    return templates.TemplateResponse(request=request, name="auth.html", context={
        "supabase_url": settings.supabase_url,
        "supabase_anon_key": settings.supabase_anon_key,
    })


@app.get("/auth/callback", response_class=HTMLResponse)
def auth_callback(request: Request):
    from app.config import settings
    return templates.TemplateResponse(request=request, name="auth_callback.html", context={
        "supabase_url": settings.supabase_url,
        "supabase_anon_key": settings.supabase_anon_key,
    })


@app.get("/auth/reset", response_class=HTMLResponse)
def auth_reset_page(request: Request):
    """Password-reset landing page. Supabase sends the recovery link here with
    a token in the URL hash; this page lets the user set a new password."""
    from app.config import settings
    return templates.TemplateResponse(request=request, name="auth_reset.html", context={
        "supabase_url": settings.supabase_url,
        "supabase_anon_key": settings.supabase_anon_key,
    })


@app.post("/auth/logout")
def logout():
    return {"success": True}


# ── Resume upload ─────────────────────────────────────────────────────────────

@app.post("/api/resume/upload")
async def upload_resume(request: Request):
    """Upload resume file. Stores in Supabase Storage (production) or local disk (dev)."""
    from fastapi import UploadFile, File
    uid = _require_user(request)
    form = await request.form()
    file: UploadFile = form.get("file")
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    content = await file.read()
    filename = file.filename
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "pdf"
    if ext not in ("pdf", "docx", "md", "txt"):
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, MD, TXT allowed")

    # Clear the cached experience/education extraction when a new resume is uploaded
    from app.db.models import AnswerMemory
    from app.db.init_db import get_session
    from sqlmodel import delete as sql_delete
    with get_session() as session:
        session.exec(
            sql_delete(AnswerMemory).where(
                AnswerMemory.label_normalized == "__resume_extracted_experience_education",
                AnswerMemory.user_id == (uid if uid != "local" else None)
            )
        )
        session.commit()

    from app.config import settings
    if settings.use_supabase:
        try:
            from app.db.supabase_client import service_client
            sb = service_client()
            path = f"{uid}/resume.{ext}"
            sb.storage.from_("resume").upload(path, content, {"upsert": "true", "content-type": file.content_type})
            public_url = sb.storage.from_("resume").get_public_url(path)
            return {"success": True, "url": public_url, "path": path}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Storage upload failed: {e}")
    else:
        # Local dev — save to data/
        import os
        os.makedirs("./data", exist_ok=True)
        local_path = f"./data/resume_master.{ext}"
        with open(local_path, "wb") as f:
            f.write(content)
        return {"success": True, "path": local_path}


@app.post("/api/resume/extract-profile")
@_rate_limit("5/minute")
async def extract_profile_from_resume(request: Request) -> dict:
    """Parse the user's uploaded resume and auto-fill their profile fields using Claude."""
    import re as _re
    uid = _require_user(request)

    # Load resume text
    try:
        from app.matching.pipeline import _load_resume
        resume_text = _load_resume(user_id=uid)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not load resume: {exc}")

    if not resume_text or len(resume_text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Resume appears empty")

    # Ask Claude to extract structured info
    import anthropic as _anthropic
    from app.config import settings as _settings
    client = _anthropic.Anthropic(api_key=_settings.anthropic_api_key)

    prompt = f"""Extract the following fields from this resume. Return ONLY a JSON object with these exact keys (use null for missing fields):
first_name, last_name, email, phone, location, current_title, years_experience (integer),
linkedin_url, github_url, portfolio_url, degree, university, graduation_year (integer),
key_skills (comma-separated string), professional_summary (2-3 sentence summary of their background),
suggested_target_roles (an array of 4-6 specific job titles this candidate is genuinely
well-qualified for and should apply to right now, ordered best-fit first. Use real,
searchable titles like "Senior Backend Engineer" or "Data Scientist" — base them on the
candidate's actual experience, seniority and skills, not generic guesses).

Resume:
{resume_text[:6000]}

Return only valid JSON, no markdown, no explanation."""

    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = msg.content[0].text.strip()

    # Strip markdown fences if present
    raw = _re.sub(r"^```(?:json)?\s*", "", raw)
    raw = _re.sub(r"\s*```$", "", raw)

    import json as _json
    try:
        extracted = _json.loads(raw)
    except Exception:
        raise HTTPException(status_code=500, detail="Could not parse extraction response")

    # Save to profile — reuse same single-session logic. Map the sentinel
    # "local" user to None so single-user (SQLite) and SaaS (Supabase) modes
    # read & write the SAME profile row the dashboard later loads.
    from app.db.models import UserProfile
    import datetime as _datetime
    user_id_arg = uid if uid != "local" else None
    with get_session() as session:
        q = select(UserProfile)
        if user_id_arg:
            q = q.where(UserProfile.user_id == user_id_arg)
        else:
            q = q.where(UserProfile.user_id == None)  # noqa: E711 (SQL IS NULL)
        db_profile = session.exec(q).first()
        if not db_profile:
            db_profile = UserProfile(user_id=user_id_arg)
            session.add(db_profile)
            session.flush()

        field_map = [
            "first_name", "last_name", "email", "phone", "location", "current_title",
            "years_experience", "linkedin_url", "github_url", "portfolio_url",
            "degree", "university", "graduation_year", "key_skills", "professional_summary"
        ]
        for field in field_map:
            val = extracted.get(field)
            if val is not None and val != "":
                setattr(db_profile, field, val)
        db_profile.updated_at = _datetime.datetime.utcnow()
        session.add(db_profile)
        session.commit()

    # Auto-seed target_roles from the extracted data — only when the user
    # hasn't already set any roles (so we never overwrite deliberate choices).
    user_id_arg = uid if uid != "local" else None
    seeded_roles: list[str] = []
    with get_session() as session:
        q = select(UserProfile)
        if user_id_arg:
            q = q.where(UserProfile.user_id == user_id_arg)
        p = session.exec(q).first()
        if p and not (p.target_roles or "").strip():
            seen_r: set[str] = set()
            def _add_role(r: str):
                r = (r or "").strip()
                if r and r.lower() not in seen_r and len(seeded_roles) < 6:
                    seen_r.add(r.lower())
                    seeded_roles.append(r)
            # Prefer the roles Claude chose for this candidate.
            ai_roles = extracted.get("suggested_target_roles") or []
            if isinstance(ai_roles, str):
                ai_roles = [r for r in ai_roles.split(",")]
            if isinstance(ai_roles, list):
                for r in ai_roles:
                    _add_role(r if isinstance(r, str) else str(r))
            # Fallback heuristic if Claude returned nothing usable: current_title
            # first, then skill tokens that read like roles.
            if not seeded_roles:
                _add_role(extracted.get("current_title") or "")
                _role_tokens = ("engineer", "scientist", "developer", "manager",
                                "analyst", "designer", "architect", "lead")
                for s in (extracted.get("key_skills") or "").split(","):
                    s = s.strip()
                    if s and any(tok in s.lower() for tok in _role_tokens):
                        _add_role(s)
            if seeded_roles:
                p.target_roles = ", ".join(seeded_roles)
                p.updated_at = _datetime.datetime.utcnow()
                session.add(p)
                session.commit()

    return {"success": True, "extracted": {k: extracted.get(k) for k in field_map},
            "seeded_roles": seeded_roles}


@app.get("/api/resume/status")
def resume_status(request: Request) -> dict:
    """Whether the current user has a resume on file. Drives the Discover gate."""
    uid = _get_user_id(request)
    return {"has_resume": _user_has_resume(uid)}


@app.get("/api/resume/analysis")
def resume_analysis(request: Request) -> dict:
    """General ATS-readiness analysis of the user's résumé — no specific job needed.

    Scores deterministic parse-ability signals (contact info, sections, quantified
    impact, action verbs, length) plus how well the résumé reflects the user's
    saved target roles. Fully local: no LLM / network calls, so it's instant + free.
    """
    import re as _re
    uid = _get_user_id(request)
    from app.config import settings
    if settings.use_supabase and not uid:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not _user_has_resume(uid):
        return {"has_resume": False}
    try:
        from app.matching.pipeline import _load_resume
        text = _load_resume(user_id=uid)
    except Exception as e:
        return {"has_resume": True, "error": f"Could not read résumé: {e}"}
    if not text or len(text.strip()) < 30:
        return {"has_resume": True, "error": "Résumé appears empty or unreadable."}

    low = text.lower()
    words = _re.findall(r"\b\w+\b", text)
    wc = len(words)
    findings: list[dict] = []

    def _add(label, ok, detail):
        findings.append({"label": label, "ok": bool(ok), "detail": detail})

    # 1. Contact info — ATS parsers key on a findable email + phone.
    has_email = bool(_re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text))
    has_phone = bool(_re.search(r"(\+?\d[\d\s().-]{7,}\d)", text))
    _add("Contact details", has_email and has_phone,
         "Email & phone found" if has_email and has_phone else "Add a clear email and phone number")

    # 2. Standard sections
    sections = {
        "experience": any(s in low for s in ("experience", "employment", "work history")),
        "education": "education" in low,
        "skills": any(s in low for s in ("skills", "technologies", "technical")),
    }
    missing_sec = [k for k, v in sections.items() if not v]
    _add("Standard sections", not missing_sec,
         "Experience, education & skills present" if not missing_sec
         else f"Missing section(s): {', '.join(missing_sec)}")

    # 3. Quantified achievements — numbers/%/$ signal impact.
    metric_hits = len(_re.findall(r"\b\d+%|\$\s?\d|\b\d{2,}\b", text))
    _add("Quantified impact", metric_hits >= 3,
         f"{metric_hits} measurable results" if metric_hits >= 3
         else "Add metrics (%, $, counts) to your bullets")

    # 4. Action verbs
    action_verbs = ("built", "led", "designed", "developed", "launched", "created",
                    "improved", "reduced", "increased", "managed", "shipped", "drove",
                    "owned", "architected", "delivered", "implemented", "optimized")
    verb_hits = sum(low.count(v) for v in action_verbs)
    _add("Action verbs", verb_hits >= 5,
         f"{verb_hits} strong action verbs" if verb_hits >= 5
         else "Start bullets with action verbs (Built, Led, Shipped…)")

    # 5. Length
    good_len = 350 <= wc <= 1100
    _add("Length", good_len,
         f"{wc} words — good" if good_len
         else (f"{wc} words — too short, add detail" if wc < 350 else f"{wc} words — consider trimming"))

    # 6. Target-role alignment — ties roles + résumé together.
    roles = _get_target_roles(uid)
    if not roles:
        _add("Target-role match", False, "Add target roles to check résumé alignment")
    else:
        role_terms = set()
        for r in roles:
            for tok in _re.findall(r"\b\w+\b", r.lower()):
                if len(tok) > 2:
                    role_terms.add(tok)
        covered = [t for t in role_terms if t in low]
        role_ok = bool(role_terms) and (len(covered) / len(role_terms) >= 0.5)
        _add("Target-role match", role_ok,
             f"Résumé reflects your target roles ({len(covered)}/{len(role_terms)} terms)" if role_ok
             else "Résumé weakly matches your target roles — weave in relevant keywords")

    score = round(sum(1 for f in findings if f["ok"]) / len(findings) * 100)
    grade = "Strong" if score >= 80 else "Good" if score >= 60 else "Needs work"
    return {"has_resume": True, "score": score, "grade": grade,
            "word_count": wc, "findings": findings}


@app.get("/api/discovery/last-run")
def discovery_last_run(request: Request) -> dict:
    """Return the most recent discovery run's per-source summary for this user."""
    import json
    from app.db.models import DiscoveryRun
    uid = _get_user_id(request)
    with get_session() as session:
        q = select(DiscoveryRun).order_by(desc(DiscoveryRun.id))
        if uid and uid != "local":
            q = q.where(DiscoveryRun.user_id == uid)
        run = session.exec(q).first()
        if not run:
            return {"run": None, "can_run": True, "gate_detail": ""}
        try:
            counts = json.loads(run.source_counts or "{}")
        except Exception:
            counts = {}
        allowed, detail = _discovery_gate(uid)
        return {"can_run": allowed, "gate_detail": detail, "run": {
            "id": run.id,
            "status": run.status,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "finished_at": run.finished_at.isoformat() if run.finished_at else None,
            "total_fetched": run.total_fetched,
            "total_inserted": run.total_inserted,
            "total_shortlisted": run.total_shortlisted,
            "error": run.error,
            "sources": counts,
        }}


@app.get("/api/resume/view")
def view_resume(request: Request) -> dict:
    """Return a temporary signed URL to view/download the user's uploaded resume."""
    from app.config import settings
    uid = _require_user(request)
    if settings.use_supabase and uid and uid != "local":
        try:
            from app.db.supabase_client import service_client
            sb = service_client()
            files = sb.storage.from_("resume").list(uid)
            names = [f.get("name", "") for f in (files or []) if f.get("name", "").startswith("resume.")]
            if names:
                signed = sb.storage.from_("resume").create_signed_url(f"{uid}/{names[0]}", 3600)
                url = (signed or {}).get("signedURL") or (signed or {}).get("signedUrl")
                return {"url": url, "filename": names[0]}
        except Exception as e:
            log.warning("Could not sign resume URL: %s", e)
        return {"url": None}
    # Local mode — serve the file directly
    import glob
    matches = glob.glob("./data/resume_master.*")
    if matches:
        return {"url": "/api/resume/file", "filename": matches[0].split("/")[-1]}
    return {"url": None}


@app.get("/api/resume/file")
def resume_file(request: Request):
    """Serve the locally-stored resume (local/dev mode only)."""
    from fastapi.responses import FileResponse
    import glob
    matches = glob.glob("./data/resume_master.*")
    if not matches:
        raise HTTPException(status_code=404, detail="No resume on file")
    return FileResponse(matches[0])


@app.post("/api/resume/synthesize")
def synthesize_resume(request: Request) -> dict:
    """Build a minimal markdown resume from the user's profile fields.

    Used when the user fills their details in manually instead of uploading a
    file — gives the matcher something to work with so discovery isn't blocked.
    """
    uid = _require_user(request)
    from app.autofill.answer_pack import _get_or_create_profile
    profile = _get_or_create_profile(user_id=uid if uid != "local" else None)
    name = f"{profile.first_name} {profile.last_name}".strip() or "Candidate"
    md = (
        f"# {name}\n\n"
        f"**Email:** {profile.email}\n"
        f"**Location:** {profile.location}\n"
        f"**Current Title:** {profile.current_title}\n\n"
        f"## Summary\n{profile.professional_summary or ''}\n\n"
        f"## Skills\n{profile.key_skills or ''}\n\n"
        f"## Experience\n{profile.current_title or ''}\n"
    )
    from app.config import settings
    if settings.use_supabase and uid != "local":
        try:
            from app.db.supabase_client import service_client
            sb = service_client()
            sb.storage.from_("resume").upload(
                f"{uid}/resume.md",
                md.encode("utf-8"),
                {"upsert": "true", "content-type": "text/markdown"},
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not save resume: {e}")
    else:
        import os
        os.makedirs("./data", exist_ok=True)
        with open("./data/resume_master.md", "w", encoding="utf-8") as f:
            f.write(md)
    return {"success": True}


@app.post("/api/jobs/clear")
def clear_jobs(request: Request) -> dict:
    """Delete all jobs + applications belonging to the current user.

    Used to reset a pool that was populated before the resume gate existed.
    In SQLite single-user mode (uid == "local") this clears the whole DB.
    """
    uid = _require_user(request)
    _scoped = uid and uid != "local"
    deleted_apps = 0
    deleted_jobs = 0
    with get_session() as session:
        aq = select(Application)
        jq = select(Job)
        if _scoped:
            aq = aq.where(Application.user_id == uid)
            jq = jq.where(Job.user_id == uid)
        for a in session.exec(aq).all():
            session.delete(a)
            deleted_apps += 1
        for j in session.exec(jq).all():
            session.delete(j)
            deleted_jobs += 1
        session.commit()
    return {"success": True, "deleted_jobs": deleted_jobs, "deleted_applications": deleted_apps}


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/debug/tenancy")
def debug_tenancy(request: Request) -> dict:
    """Quick check that multi-tenant mode is live and the backend sees the caller
    as a real user (not the shared 'local' tenant). Auth required.

    Healthy production response looks like:
        { "use_supabase": true, "your_uid": "<uuid>", "is_local": false, ... }
    If use_supabase is false or your_uid is "local", DATABASE_URL/SUPABASE_URL
    aren't set on the backend and all users share one tenant.
    """
    from app.config import settings
    auth = request.headers.get("Authorization", "")
    token = auth.split(" ", 1)[1] if auth.startswith("Bearer ") else None
    uid = _get_user_id(request)
    # If a token was sent but didn't resolve to a user, surface WHY — this is the
    # usual "authenticated:false while logged in" cause (bad/missing service key,
    # or a token from a different Supabase project than the backend keys).
    verify_error = None
    if token and uid is None and settings.use_supabase:
        try:
            from app.db.supabase_client import service_client
            service_client().auth.get_user(token)
        except Exception as e:
            verify_error = f"{type(e).__name__}: {str(e)[:180]}"
    return {
        "use_supabase": settings.use_supabase,
        "database_url_set": bool(settings.database_url),
        "supabase_url_set": bool(settings.supabase_url),
        "service_role_key_set": bool(settings.supabase_service_role_key),
        "anon_key_set": bool(settings.supabase_anon_key),
        "token_present": bool(token),
        "authenticated": uid is not None,
        "your_uid": uid,
        "is_local": uid == "local",
        "verify_error": verify_error,
    }


@app.get("/stats")
def stats(request: Request) -> dict:
    uid = _get_user_id(request)
    with get_session() as session:
        q = select(Job).where(Job.is_closed == False)
        if uid and uid != "local":
            q = q.where(Job.user_id == uid)
        total_jobs = len(session.exec(q).all())
        by_status = {}
        for st in ApplicationStatus:
            # Exclude orphan applications (Job row deleted) by joining Job.
            aq = (
                select(func.count(Application.id))
                .join(Job, Application.job_id == Job.id)
                .where(Application.status == st)
            )
            if uid and uid != "local":
                aq = aq.where(Application.user_id == uid)
            count = session.exec(aq).first() or 0
            if count:
                by_status[st.value] = count
    return {"total_jobs": total_jobs, "applications": by_status}


@app.get("/shortlist")
def shortlist(request: Request):
    """Top-scored jobs not yet processed — scoped to the authenticated user."""
    uid = _require_user(request)
    with get_session() as session:
        q = select(Job).where(Job.rerank_score >= 70)
        if uid != "local":
            q = q.where(Job.user_id == uid)
        jobs = session.exec(q.order_by(Job.rerank_score.desc())).all()
    return [
        {
            "id": j.id,
            "company": j.company,
            "title": j.title,
            "location": j.location,
            "url": j.url,
            "similarity": j.similarity_score,
            "rerank": j.rerank_score,
            "hire_probability": j.hire_probability_score,
            "blended": j.blended_score,
            "reason": j.rerank_reasoning,
        }
        for j in jobs
    ]


@app.get("/api/stats")
def api_stats(request: Request) -> dict:
    uid = _get_user_id(request)
    _uid_filter = (uid and uid != "local")
    with get_session() as session:
        # Total jobs in db
        jq = select(func.count(Job.id))
        if _uid_filter:
            jq = jq.where(Job.user_id == uid)
        total_jobs = session.exec(jq).first() or 0

        # Unique companies in Job db
        cq = select(func.count(func.distinct(Job.company)))
        if _uid_filter:
            cq = cq.where(Job.user_id == uid)
        total_companies = session.exec(cq).first() or 0

        # Funnel metrics
        def _jcount(extra=None):
            q = select(func.count(Job.id))
            if _uid_filter:
                q = q.where(Job.user_id == uid)
            if extra is not None:
                q = q.where(extra)
            return session.exec(q).first() or 0

        cross_encoder_passed = _jcount(Job.similarity_score.is_not(None))
        reranker_scored = _jcount(Job.rerank_score.is_not(None))

        # Application counts by status — JOIN Job so orphan applications (whose
        # Job row was deleted) are excluded; otherwise counts here disagree with
        # the dashboard kanban, which inner-joins Job and never shows orphans.
        app_counts = {}
        for status in ApplicationStatus:
            aq = (
                select(func.count(Application.id))
                .join(Job, Application.job_id == Job.id)
                .where(Application.status == status)
            )
            if _uid_filter:
                aq = aq.where(Application.user_id == uid)
            count = session.exec(aq).first() or 0
            app_counts[status.value] = count

        shortlisted = app_counts[ApplicationStatus.SHORTLISTED.value] + app_counts[ApplicationStatus.TAILORED.value]

        # Score distribution
        band_85_100 = _jcount(Job.rerank_score >= 85)
        band_60_84 = _jcount((Job.rerank_score >= 60) & (Job.rerank_score < 85))
        band_40_59 = _jcount((Job.rerank_score >= 40) & (Job.rerank_score < 60))
        band_0_39 = _jcount((Job.rerank_score >= 0) & (Job.rerank_score < 40))
        unranked = _jcount(Job.rerank_score.is_(None))

        # Top companies
        top_q = select(Job.company, func.count(Job.id)).group_by(Job.company).order_by(desc(func.count(Job.id))).limit(10)
        if _uid_filter:
            top_q = top_q.where(Job.user_id == uid)
        top_companies_res = session.exec(top_q).all()
        top_companies = [{"company": company, "count": count} for company, count in top_companies_res]

        # Per-source job counts (for source breakdown bar in UI)
        from app.db.models import JobSource as JS
        source_counts: dict[str, int] = {}
        for src in JS:
            sq = select(func.count(Job.id)).where(Job.source == src)
            if _uid_filter:
                sq = sq.where(Job.user_id == uid)
            cnt = session.exec(sq).first() or 0
            if cnt:
                source_counts[src.value] = cnt

        # Company registry stats (global — not per-user)
        from app.db.models import CompanyRegistry
        total_boards = session.exec(select(func.count(CompanyRegistry.id))).first() or 0
        active_boards = session.exec(select(func.count(CompanyRegistry.id)).where(CompanyRegistry.is_active == True)).first() or 0
        total_validated_jobs = session.exec(select(func.sum(CompanyRegistry.job_count))).first() or 0

    return {
        "total_jobs": total_jobs,
        "total_companies": total_companies,
        "funnel": {
            "total_pool": total_jobs,
            "cross_encoder_passed": cross_encoder_passed,
            "reranker_scored": reranker_scored,
            "shortlisted": shortlisted,
        },
        "applications": app_counts,
        "scores": {
            "band_85_100": band_85_100,
            "band_60_84": band_60_84,
            "band_40_59": band_40_59,
            "band_0_39": band_0_39,
            "unranked": unranked,
        },
        "sources": source_counts,
        "top_companies": top_companies,
        "registry": {
            "total_boards": total_boards,
            "active_boards": active_boards,
            "total_validated_jobs": total_validated_jobs,
        }
    }


_AUTOFILL_SOURCES = {"greenhouse", "lever", "ashby", "workday", "smartrecruiters"}

@app.get("/api/jobs")
def api_jobs(
    request: Request,
    page: int = 1,
    limit: int = 50,
    search: str = None,
    company: str = None,
    status: str = None,
    min_score: int = None,
    max_score: int = None,
    remote: str = None,
    track: str = None,   # "autofill" | "manual"
) -> dict:
    uid = _get_user_id(request)
    _uid_filter = uid and uid != "local"
    offset = (page - 1) * limit

    with get_session() as session:
        # Base query — exclude closed/purged jobs from the Jobs table.
        query = select(Job, Application).outerjoin(Application, Application.job_id == Job.id).where(Job.is_closed == False)
        if _uid_filter:
            query = query.where(Job.user_id == uid)

        # Apply filters
        if search:
            search_pattern = f"%{search}%"
            query = query.where(Job.title.like(search_pattern) | Job.company.like(search_pattern) | Job.location.like(search_pattern))

        if company:
            query = query.where(Job.company == company)

        if track == "autofill":
            query = query.where(Job.source.in_(list(_AUTOFILL_SOURCES)))
        elif track == "manual":
            query = query.where(Job.source.not_in(list(_AUTOFILL_SOURCES)))
            
        if status:
            if status == "unprocessed":
                query = query.where(Application.id.is_(None))
            else:
                query = query.where(Application.status == status)
                
        if min_score is not None:
            query = query.where(Job.rerank_score >= min_score)
            
        if max_score is not None:
            query = query.where(Job.rerank_score <= max_score)
            
        if remote is not None:
            is_remote = remote.lower() == "true"
            query = query.where(Job.remote == is_remote)
            
        # Get total count (for pagination) — also exclude closed jobs.
        count_query = select(func.count(Job.id)).where(Job.is_closed == False)
        if _uid_filter:
            count_query = count_query.where(Job.user_id == uid)
        if search:
            search_pattern = f"%{search}%"
            count_query = count_query.where(Job.title.like(search_pattern) | Job.company.like(search_pattern) | Job.location.like(search_pattern))
        if company:
            count_query = count_query.where(Job.company == company)
        if status:
            if status == "unprocessed":
                count_query = count_query.select_from(Job).outerjoin(Application, Application.job_id == Job.id).where(Application.id.is_(None))
            else:
                count_query = count_query.select_from(Job).join(Application, Application.job_id == Job.id).where(Application.status == status)
        if min_score is not None:
            count_query = count_query.where(Job.rerank_score >= min_score)
        if max_score is not None:
            count_query = count_query.where(Job.rerank_score <= max_score)
        if remote is not None:
            is_remote = remote.lower() == "true"
            count_query = count_query.where(Job.remote == is_remote)
            
        total = session.exec(count_query).first() or 0
        
        # Apply pagination and sorting
        query = query.order_by(desc(Job.blended_score), desc(Job.rerank_score), desc(Job.similarity_score), desc(Job.id)).offset(offset).limit(limit)
        
        results = session.exec(query).all()
        
        jobs_list = []
        for job, app in results:
            jobs_list.append({
                "id": job.id,
                "source": job.source.value if job.source else "manual",
                "company": job.company,
                "title": job.title,
                "location": job.location,
                "remote": job.remote,
                "url": job.url,
                "similarity": job.similarity_score,
                "rerank": job.rerank_score,
                "hire_probability": job.hire_probability_score,
                "blended": job.blended_score,
                "reason": job.rerank_reasoning,
                "application": {
                    "id": app.id,
                    "status": app.status.value,
                    "apply_track": app.apply_track,
                    "created_at": app.created_at.isoformat() if app.created_at else None,
                    "updated_at": app.updated_at.isoformat() if app.updated_at else None,
                } if app else None
            })
            
        import math
        pages = math.ceil(total / limit) if total else 0
        
        return {
            "jobs": jobs_list,
            "total": total,
            "page": page,
            "pages": pages,
            "limit": limit
        }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    """Kanban board UI for tracking application progress."""
    from app.config import settings
    uid = _get_user_id(request)
    _uid_filter = uid and uid != "local"
    # SSR auth: whether THIS page navigation was authenticated (via sb_token
    # cookie). If not, fail closed and render no pipeline data — never leak other
    # tenants' applications. The client auth-guard sets the cookie and reloads.
    ssr_authed = bool(uid) or not settings.use_supabase
    if settings.use_supabase and not uid:
        results = []
    else:
        with get_session() as session:
            q = select(Application, Job).join(Job).order_by(Application.updated_at.desc())
            if _uid_filter:
                q = q.where(Application.user_id == uid)
            results = session.exec(q).all()
        
    shortlisted = []
    bot_filled = []    # autofill-track: form filled, pending review
    manual_queue = []  # manual-track: materials ready, waiting for human to apply
    submitted = []
    skipped = []
    rejected = []      # heard back: no — collected separately

    _AUTOFILL_REVIEW_STATUSES = {
        ApplicationStatus.AUTOFILLED,
        ApplicationStatus.AWAITING_USER,
        ApplicationStatus.READY_TO_SUBMIT,
    }

    for app_model, job_model in results:
        if app_model.status in [ApplicationStatus.SHORTLISTED, ApplicationStatus.TAILORED]:
            shortlisted.append((app_model, job_model))
        elif app_model.status in _AUTOFILL_REVIEW_STATUSES:
            if app_model.apply_track == "manual":
                manual_queue.append((app_model, job_model))
            else:
                bot_filled.append((app_model, job_model))
        elif app_model.status in [ApplicationStatus.SUBMITTED, ApplicationStatus.INTERVIEWING]:
            submitted.append((app_model, job_model))
        elif app_model.status == ApplicationStatus.REJECTED:
            rejected.append((app_model, job_model))
        elif app_model.status == ApplicationStatus.SKIPPED:
            skipped.append((app_model, job_model))

    from datetime import datetime as _dt

    # Legal work-authorization framing for this user (drives the visa-fit panel
    # and the sponsorship-aware ranking boost below).
    visa_framing = None
    try:
        from app.intelligence.work_auth import assess_profile
        from app.autofill.answer_pack import _get_or_create_profile
        _prof = _get_or_create_profile(user_id=uid if uid and uid != "local" else None)
        visa_framing = assess_profile(_prof)
    except Exception as _e:
        log.debug("visa framing unavailable: %s", _e)

    # For users who need sponsorship, float no-lottery (cap-exempt) and known
    # sponsors to the top — those are the jobs that can actually hire them.
    _boost_sponsorship = bool(visa_framing and getattr(visa_framing, "needs_future_sponsorship", False))

    def _priority(job) -> float:
        """Rank by blended score (fit + hiring intent), plus an urgency/timing
        tiebreak, plus a sponsorship-aware boost for visa users."""
        base = job.blended_score if job.blended_score is not None else (job.rerank_score or 0)
        # Urgency is a strong tiebreak (fresh / hard-to-fill float up) but stays
        # secondary to fit: up to ~+14 on a 0-100 scale.
        try:
            urg = _urgency_of(job)
            if urg:
                base += urg.score * 0.15
        except Exception:
            pass
        if _boost_sponsorship:
            try:
                spons = _sponsorship_of(job)
                if spons and spons.cap_exempt:
                    base += 1000          # no-lottery → absolute top
                elif spons and spons.tone == "good":
                    base += 200           # established sponsor → boosted
                elif spons and spons.explicitly_refuses:
                    base -= 500           # explicitly won't sponsor → sink
            except Exception:
                pass
        return base

    # Highest-priority roles first; recency breaks ties so fresh postings float up.
    shortlisted.sort(
        key=lambda x: (_priority(x[1]), x[1].posted_at or x[1].discovered_at or _dt.min),
        reverse=True,
    )
    manual_queue.sort(key=lambda x: _priority(x[1]), reverse=True)

    # Job-based, not company-based: show at most 2 roles per company in the
    # shortlist so a single company can't dominate the view. Keeps the
    # highest-priority 2 (list is already sorted best-first).
    # NOTE: jobs with an empty/unknown company must NOT be collapsed together —
    # otherwise dozens of distinct aggregator postings (RemoteOK/HN often have a
    # blank company field) all share key "" and get capped to 2 globally, which
    # is why the shortlist count showed 19 but only 2 rendered.
    def _cap_per_company(items, cap: int = 2):
        seen: dict[str, int] = {}
        capped = []
        for app_model, job_model in items:
            company = (job_model.company or "").strip().lower()
            # Distinct key per row when company is unknown → never grouped.
            key = company if company else f"__unknown__:{job_model.id}"
            if seen.get(key, 0) >= cap:
                continue
            seen[key] = seen.get(key, 0) + 1
            capped.append((app_model, job_model))
        return capped

    shortlisted = _cap_per_company(shortlisted)
    manual_queue = _cap_per_company(manual_queue)

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "shortlisted": shortlisted,
            "bot_filled": bot_filled,
            "manual_queue": manual_queue,
            "submitted": submitted,
            "skipped": skipped,
            "rejected": rejected,
            "ssr_authed": ssr_authed,
            "visa_framing": visa_framing,
            "supabase_url": settings.supabase_url,
            "supabase_anon_key": settings.supabase_anon_key,
        }
    )


@app.get("/application/{application_id}/details")
def application_details(application_id: int, request: Request) -> dict:
    """Return tailored resume + cover letter text for modal preview."""
    from pathlib import Path
    _require_owned_application(request, application_id)
    with get_session() as session:
        application = session.get(Application, application_id)
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")
        job = session.get(Job, application.job_id)

    resume_text = ""
    cover_text = ""

    if application.tailored_resume_path:
        try:
            from docx import Document
            doc = Document(application.tailored_resume_path)
            resume_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            resume_text = f"(Could not read resume: {e})"

    if application.cover_letter_path:
        try:
            cover_text = Path(application.cover_letter_path).read_text(encoding="utf-8")
        except Exception as e:
            cover_text = f"(Could not read cover letter: {e})"

    return {
        "id": application_id,
        "company": job.company,
        "title": job.title,
        "apply_url": application.apply_url or job.url,
        "status": application.status.value,
        "source": job.source.value,
        "resume": resume_text,
        "cover_letter": cover_text,
    }


@app.get("/api/fill-pack/{application_id}")
def get_fill_pack(application_id: int, request: Request) -> dict:
    """Returns all data the browser extension needs to fill a job application form."""
    import io, zipfile as _zf
    from pathlib import Path as _P
    _require_owned_application(request, application_id)
    uid = _get_user_id(request)
    with get_session() as session:
        application = session.get(Application, application_id)
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")
        job = session.get(Job, application.job_id)
        from app.db.models import UserProfile
        profile = session.exec(select(UserProfile).where(UserProfile.user_id == uid)).first() if uid else None
        needs_tailoring = not (application.tailored_resume_path and application.cover_letter_path)

    # Auto-tailor on demand: autofill applications often haven't been through the
    # tailoring pipeline. Generate a tailored resume + cover letter now so the
    # extension always has a fresh, role-specific resume to upload.
    if needs_tailoring:
        try:
            from app.tailoring.tailor import tailor_for_application
            tailor_for_application(application_id)
            with get_session() as session:
                application = session.get(Application, application_id)
        except Exception as e:
            log.warning("Auto-tailor failed for app %d: %s", application_id, e)

    cover_text = ""
    if application.cover_letter_path:
        try:
            cover_text = _P(application.cover_letter_path).read_text(encoding="utf-8")
        except Exception:
            pass

    resume_text = ""
    if application.tailored_resume_path:
        try:
            resume_text = _P(application.tailored_resume_path).read_text(encoding="utf-8")
        except Exception:
            pass

    p = profile
    pack = {
        "app_id": application_id,
        "job_title": job.title if job else "",
        "company": job.company if job else "",
        "apply_url": application.apply_url or (job.url if job else ""),
        "first_name": p.first_name if p else "",
        "last_name": p.last_name if p else "",
        "email": p.email if p else "",
        "phone": p.phone if p else "",
        "location": p.location if p else "",
        "linkedin_url": p.linkedin_url if p else "",
        "github_url": p.github_url if p else "",
        "portfolio_url": p.portfolio_url if p else "",
        "current_title": p.current_title if p else "",
        "years_experience": p.years_experience if p else 0,
        "salary_min": p.salary_min if p else 0,
        "work_authorization": p.work_authorization if p else "",
        "requires_sponsorship": p.requires_sponsorship if p else False,
        "gender": p.gender if p else "Decline to self-identify",
        "ethnicity": p.ethnicity if p else "Decline to self-identify",
        "veteran_status": p.veteran_status if p else "I am not a protected veteran",
        "disability_status": p.disability_status if p else "No, I do not have a disability, or history/record of having a disability",
        "cover_letter": cover_text,
        "resume_text": resume_text,
    }

    # Add AI-generated essay answers
    try:
        from app.autofill.answer_pack import get_essay_answers
        essay_answers = get_essay_answers(application_id, user_id=uid if uid != "local" else None)
        pack["ai_answers"] = essay_answers
    except Exception as e:
        log.warning("Failed to generate essay answers for app %d: %s", application_id, e)
        pack["ai_answers"] = {}

    # Add hirepath_url and auth_token so extension can save answers back
    from app.config import settings
    # Use request.base_url so local dev hits 127.0.0.1:8000, prod hits hirepath.dev
    _base = str(request.base_url).rstrip("/")
    pack["hirepath_url"] = getattr(settings, "hirepath_url", None) or _base
    token = request.headers.get("Authorization", "").split(" ", 1)[-1]
    pack["auth_token"] = token

    # Add work experience & education (extracted from resume via LLM, cached)
    try:
        from app.autofill.answer_pack import _get_or_extract_experience_education
        exp_edu = _get_or_extract_experience_education(application, profile, user_id=uid if uid != "local" else None)
        pack["work_experience"] = exp_edu.get("work_experience", [])
        pack["education"] = exp_edu.get("education", [])
    except Exception as e:
        log.warning("Failed to extract experience/education for app %d: %s", application_id, e)
        pack["work_experience"] = []
        pack["education"] = []

    return pack


@app.get("/api/fill-pack/{application_id}/resume")
def get_tailored_resume(application_id: int, request: Request) -> dict:
    """Return the tailored resume .docx as base64 so the extension can attach it
    to a form's file input. Auto-tailors first if no resume exists yet."""
    import base64
    from pathlib import Path as _P
    _require_owned_application(request, application_id)
    with get_session() as session:
        application = session.get(Application, application_id)
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")
        path = application.tailored_resume_path

    if not path or not _P(path).exists():
        try:
            from app.tailoring.tailor import tailor_for_application
            resume_path, _ = tailor_for_application(application_id)
            path = str(resume_path)
        except Exception as e:
            log.warning("Resume tailoring failed for app %d: %s", application_id, e)
            raise HTTPException(status_code=503, detail="Could not generate resume")

    p = _P(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Resume file not found")

    data = p.read_bytes()
    mime = ("application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            if p.suffix == ".docx" else "application/octet-stream")
    return {
        "filename": p.name,
        "mime": mime,
        "base64": base64.b64encode(data).decode("ascii"),
    }


class SaveAnswerBody(BaseModel):
    question: str
    answer: str
    app_id: _Opt[int] = None


@app.post("/api/save-answer")
def save_answer(request: Request, body: SaveAnswerBody) -> dict:
    """Save a user-typed answer back to AnswerMemory so it's used next time."""
    from datetime import datetime
    from app.db.models import AnswerMemory
    uid = _require_user(request)
    question = body.question.strip()
    answer = body.answer.strip()
    if not question or not answer:
        raise HTTPException(status_code=400, detail="question and answer required")
    norm = question.lower().strip()
    user_id_arg = uid if uid != "local" else None
    with get_session() as session:
        q = select(AnswerMemory).where(AnswerMemory.label_normalized == norm)
        if user_id_arg:
            q = q.where(AnswerMemory.user_id == user_id_arg)
        existing = session.exec(q).first()
        if existing:
            existing.answer = answer
            existing.use_count += 1
            existing.last_used_at = datetime.utcnow()
            session.add(existing)
        else:
            session.add(AnswerMemory(
                user_id=user_id_arg,
                label_normalized=norm,
                label_original=question,
                answer=answer,
            ))
        session.commit()
    return {"ok": True}


class RecallAnswersBody(BaseModel):
    labels: list[str]


@app.post("/api/recall-answers")
def recall_answers(request: Request, body: RecallAnswersBody) -> dict:
    """Return remembered answers for any of the given field labels.

    Lets the extension pre-fill fields on a NEW application using answers the
    user typed by hand on PREVIOUS applications. Pure cache lookup — free.
    """
    from app.db.models import AnswerMemory
    uid = _require_user(request)
    user_id_arg = uid if uid != "local" else None
    labels = [l.strip() for l in (body.labels or []) if l and l.strip()]
    if not labels:
        return {"answers": {}}
    # Map normalized label -> original label so we can return by the caller's key
    norm_to_orig = {l.lower().strip(): l for l in labels}
    answers: dict[str, str] = {}
    with get_session() as session:
        q = select(AnswerMemory).where(
            AnswerMemory.label_normalized.in_(list(norm_to_orig.keys()))
        )
        if user_id_arg:
            q = q.where(AnswerMemory.user_id == user_id_arg)
        for mem in session.exec(q).all():
            orig = norm_to_orig.get(mem.label_normalized)
            if orig and mem.answer:
                answers[orig] = mem.answer
    return {"answers": answers}


class AskQuestionBody(BaseModel):
    question: str
    app_id: int


@app.post("/api/answer-question")
def answer_question_endpoint(request: Request, body: AskQuestionBody) -> dict:
    """Generate (or retrieve cached) answer for a single essay question.

    Called by the extension only when it finds an unanswered textarea on the
    live form. Cache-hit = free. Cache-miss = ~$0.002 (Haiku, stored forever).
    """
    uid = _require_user(request)
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question required")
    _require_owned_application(request, body.app_id)
    from app.autofill.answer_pack import answer_question
    user_id_arg = uid if uid != "local" else None
    answer = answer_question(question, body.app_id, user_id=user_id_arg)
    return {"answer": answer, "cached": bool(answer)}


@app.get("/api/extension/download")
def download_extension():
    """Bundle the extension/ folder as a downloadable zip for Chrome."""
    import io, zipfile as _zf, os as _os, traceback as _tb
    from fastapi.responses import StreamingResponse
    try:
        # extension/ is two levels up from app/api/server.py  →  repo_root/extension/
        ext_dir = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "extension"))
        if not _os.path.isdir(ext_dir):
            raise HTTPException(status_code=404, detail=f"Extension folder not found: {ext_dir}")
        buf = io.BytesIO()
        with _zf.ZipFile(buf, "w", _zf.ZIP_DEFLATED) as zf:
            for root, _, files in _os.walk(ext_dir):
                for fname in sorted(files):
                    fpath = _os.path.join(root, fname)
                    arcname = "hirepath-extension/" + _os.path.relpath(fpath, ext_dir)
                    zf.write(fpath, arcname)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=hirepath-extension.zip"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build zip: {exc}\n{_tb.format_exc()}")


@app.get("/extension")
def extension_page(request: Request):
    return templates.TemplateResponse(request=request, name="extension.html", context={})


@app.post("/application/{application_id}/submit")
def mark_as_submitted(application_id: int, request: Request) -> dict:
    from datetime import datetime
    _require_owned_application(request, application_id)
    with get_session() as session:
        application = session.get(Application, application_id)
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")
        application.status = ApplicationStatus.SUBMITTED
        application.submitted_at = datetime.utcnow()
        session.add(application)
        session.commit()
    return {"success": True, "application_id": application_id}


@app.post("/application/{application_id}/skip")
def skip_application(application_id: int, request: Request) -> dict:
    _require_owned_application(request, application_id)
    with get_session() as session:
        application = session.get(Application, application_id)
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")
        application.status = ApplicationStatus.SKIPPED
        session.add(application)
        session.commit()
    return {"success": True, "application_id": application_id}



@app.post("/application/{application_id}/reject")
def mark_as_rejected(application_id: int, request: Request) -> dict:
    """Manually mark an application as rejected (you received a rejection)."""
    from datetime import datetime
    _require_owned_application(request, application_id)
    with get_session() as session:
        application = session.get(Application, application_id)
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")
        application.status = ApplicationStatus.REJECTED
        application.updated_at = datetime.utcnow()
        application.notes = (application.notes or "") + f"\nMarked rejected on {datetime.utcnow():%Y-%m-%d}."
        session.add(application)
        session.commit()
    return {"success": True, "application_id": application_id}


def _discover_then_match(user_id) -> None:
    """Discover → rank, tracking staged status (discovering → ranking → done)
    in a DiscoveryRun row so the UI can show live progress + a final summary."""
    from app.discovery.pipeline import create_discovery_run, finish_discovery_run
    run_id = create_discovery_run(user_id)
    # Tailor the keyword search to the user's saved Target Roles when set.
    roles = _get_target_roles(user_id) or None
    # Load the profile once for keyword augmentation (cap-exempt + internships +
    # department fallback). Bounded so discovery cost never balloons.
    _prof = None
    try:
        from app.autofill.answer_pack import _get_or_create_profile
        _prof = _get_or_create_profile(user_id=user_id if user_id and user_id != "local" else None)
    except Exception:
        _prof = None

    def _add_kw(lst, term):
        if term and term.lower() not in [r.lower() for r in lst]:
            lst.append(term)

    try:
        # If the user has no explicit roles, fall back to department-aware keywords.
        if not roles:
            roles = _department_keywords(_prof) or None
        if roles:
            # Sponsorship-needing users: nudge toward cap-exempt (no-lottery) roles.
            if _user_needs_sponsorship(user_id):
                primary = roles[0]
                _add_kw(roles, f"research {primary}")
                _add_kw(roles, f"{primary} university")
            # Internship toggle: append internship variants of the primary roles.
            if _prof is not None and getattr(_prof, "include_internships_in_discovery", False):
                for base in roles[:3]:
                    _add_kw(roles, f"{base} intern")
                _add_kw(roles, "internship")
    except Exception as _se:
        log.debug("discovery keyword augmentation skipped: %s", _se)
    try:
        run_discovery(user_id, run_id=run_id, keywords=roles)   # marks the row 'ranking' + per-source counts
    except Exception as e:
        log.exception("Discovery failed: %s", e)
        finish_discovery_run(run_id, "error", error=str(e))
        return
    try:
        shortlisted = run_matching(user_id)
        finish_discovery_run(run_id, "done", total_shortlisted=len(shortlisted or []))
    except Exception as e:
        log.exception("Matching failed: %s", e)
        finish_discovery_run(run_id, "error", error=str(e))


@app.post("/run/discovery")
@_rate_limit("10/minute")
def trigger_discovery(request: Request, bg: BackgroundTasks) -> dict:
    uid = _get_user_id(request)
    # Gate: no resume → no discovery. Without a resume there is nothing to match
    # against, so scraping jobs into the user's pool would only show noise.
    if not _user_has_resume(uid):
        raise HTTPException(
            status_code=400,
            detail="Upload your resume (or fill in your profile) before discovering jobs.",
        )
    # Gate: no target roles → no discovery. Without roles we don't know what
    # titles to search for, so the keyword sources would fall back to a generic
    # list that may not match the user at all.
    if not _get_target_roles(uid):
        raise HTTPException(
            status_code=400,
            detail="Add at least one target role before discovering jobs.",
        )
    # Gate: prevent overlapping runs + enforce a cooldown so repeated clicks
    # don't waste API calls / LLM tokens (discovery also auto-runs every 6h).
    allowed, detail = _discovery_gate(uid)
    if not allowed:
        raise HTTPException(status_code=429, detail=detail)
    bg.add_task(_discover_then_match, uid if uid != "local" else None)
    return {"started": "discovery"}


@app.delete("/run/discovery")
def cancel_discovery(request: Request) -> dict:
    """Mark the active discovery run as cancelled so the poller stops tracking it."""
    from app.db.models import DiscoveryRun
    uid = _get_user_id(request)
    with get_session() as session:
        q = select(DiscoveryRun).order_by(desc(DiscoveryRun.id))
        if uid and uid != "local":
            q = q.where(DiscoveryRun.user_id == uid)
        run = session.exec(q).first()
        if run and run.status in ("discovering", "ranking"):
            run.status = "cancelled"
            run.error = "Cancelled by user"
            session.add(run)
            session.commit()
            return {"cancelled": True}
    return {"cancelled": False}


def _discovery_gate(uid) -> tuple[bool, str]:
    """Block a manual discovery if one is in progress or within the cooldown."""
    from datetime import datetime
    from app.config import settings
    from app.db.models import DiscoveryRun
    with get_session() as session:
        q = select(DiscoveryRun).order_by(desc(DiscoveryRun.id))
        if uid and uid != "local":
            q = q.where(DiscoveryRun.user_id == uid)
        run = session.exec(q).first()
    if not run:
        return True, ""
    now = datetime.utcnow()
    # In progress — block unless it looks stale (likely crashed/hung).
    if run.status in ("discovering", "ranking"):
        age = (now - (run.started_at or now)).total_seconds()
        if age < 1800:  # 30 min
            return False, "A discovery is already running — please wait for it to finish."
        return True, ""  # stale, allow a fresh run
    # Cooldown since the last completed run.
    cooldown = max(0, settings.discovery_cooldown_hours) * 3600
    ref = run.finished_at or run.started_at
    if cooldown and ref:
        elapsed = (now - ref).total_seconds()
        if elapsed < cooldown:
            remaining = int(cooldown - elapsed)
            h, m = remaining // 3600, (remaining % 3600) // 60
            when = f"{h}h {m}m" if h else f"{m}m"
            return False, f"Discovery already ran recently. Next run available in ~{when} (it also auto-runs every {settings.discovery_cooldown_hours}h)."
    return True, ""


@app.post("/run/matching")
def trigger_matching(request: Request, bg: BackgroundTasks) -> dict:
    uid = _get_user_id(request)
    bg.add_task(run_matching, uid if uid != "local" else None)
    return {"started": "matching"}


# ── Usage / Plan helpers ─────────────────────────────────────────────────────

def _get_user_plan(uid: str) -> PlanTier:
    """Return the user's current plan tier. Defaults to FREE."""
    with get_session() as session:
        sub = session.exec(
            select(UserSubscription).where(UserSubscription.user_id == uid)
        ).first()
    if not sub:
        return PlanTier.FREE
    return sub.plan


def _get_or_create_usage(session, uid: str):
    from datetime import date, timedelta
    today = date.today()
    week_start = today - timedelta(days=today.weekday())
    row = session.exec(
        select(UserUsage).where(
            UserUsage.user_id == uid,
            UserUsage.usage_date == today,
        )
    ).first()
    if not row:
        row = UserUsage(user_id=uid, usage_date=today, week_start=week_start)
        session.add(row)
        session.flush()
    return row


def _get_week_autofill_count(session, uid: str) -> int:
    """Sum autofill_count_week across all rows in the current Mon–Sun window."""
    from datetime import date, timedelta
    today = date.today()
    week_start = today - timedelta(days=today.weekday())
    rows = session.exec(
        select(UserUsage).where(
            UserUsage.user_id == uid,
            UserUsage.week_start == week_start,
        )
    ).all()
    return sum(r.autofill_count_week for r in rows)


def _check_tailor_limit(uid: str) -> tuple[bool, str, dict]:
    """Returns (allowed, detail_msg, usage_info)."""
    if uid == "local":
        return True, "", {}
    plan = _get_user_plan(uid)
    limits = PLAN_LIMITS[plan]
    daily_limit = limits["tailor_daily"]
    if daily_limit is None:
        return True, "", {"plan": plan, "daily_limit": None}
    with get_session() as session:
        row = _get_or_create_usage(session, uid)
        used = row.tailor_count
        session.commit()
    if used >= daily_limit:
        upgrade = "Basic ($19/mo)" if plan == PlanTier.FREE else "Pro ($49/mo)"
        return False, (
            f"Daily tailoring limit reached ({used}/{daily_limit}). "
            f"Resets at midnight UTC. Upgrade to {upgrade} for more."
        ), {"plan": plan, "used": used, "daily_limit": daily_limit}
    return True, "", {"plan": plan, "used": used, "daily_limit": daily_limit}


def _increment_tailor(uid: str):
    if uid == "local":
        return
    with get_session() as session:
        row = _get_or_create_usage(session, uid)
        row.tailor_count += 1
        session.add(row)
        session.commit()


def _check_autofill_limit(uid: str) -> tuple[bool, str, dict]:
    if uid == "local":
        return True, "", {}
    plan = _get_user_plan(uid)
    limits = PLAN_LIMITS[plan]
    weekly_limit = limits["autofill_weekly"]
    if weekly_limit is None:
        return True, "", {"plan": plan, "weekly_limit": None}
    if weekly_limit == 0:
        return False, (
            "Auto-fill is not available on the Free plan. "
            "Upgrade to Basic ($19/mo) to get 10 auto-fills per week."
        ), {"plan": plan, "weekly_limit": 0}
    with get_session() as session:
        used = _get_week_autofill_count(session, uid)
    if used >= weekly_limit:
        upgrade = "Pro ($49/mo)" if plan == PlanTier.BASIC else "Agency ($99/mo)"
        return False, (
            f"Weekly auto-fill limit reached ({used}/{weekly_limit}). "
            f"Resets Monday midnight UTC. Upgrade to {upgrade} for more."
        ), {"plan": plan, "used": used, "weekly_limit": weekly_limit}
    return True, "", {"plan": plan, "used": used, "weekly_limit": weekly_limit}


def _increment_autofill(uid: str):
    if uid == "local":
        return
    with get_session() as session:
        row = _get_or_create_usage(session, uid)
        row.autofill_count_week += 1
        session.add(row)
        session.commit()


@app.get("/api/usage")
def get_usage(request: Request) -> dict:
    """Return current plan + usage counters for the dashboard meter."""
    from datetime import date, timedelta
    from app.config import settings
    uid = _get_user_id(request)
    # Fail closed: an expired/invalid session resolves to uid=None in SaaS mode;
    # return a clean 401 instead of crashing on a NOT NULL user_id constraint.
    if settings.use_supabase and not uid:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if uid == "local":
        return {"plan": "local", "tailor_used": 0, "tailor_daily_limit": None,
                "autofill_used_week": 0, "autofill_weekly_limit": None}
    plan = _get_user_plan(uid)
    limits = PLAN_LIMITS[plan]
    today = date.today()
    week_start = today - timedelta(days=today.weekday())
    with get_session() as session:
        row = _get_or_create_usage(session, uid)
        tailor_used = row.tailor_count
        autofill_used = _get_week_autofill_count(session, uid)
        session.commit()
    return {
        "plan": plan,
        "tailor_used": tailor_used,
        "tailor_daily_limit": limits["tailor_daily"],
        "autofill_used_week": autofill_used,
        "autofill_weekly_limit": limits["autofill_weekly"],
        "week_start": week_start.isoformat(),
    }


@app.post("/run/tailor")
def trigger_tailor(request: Request, bg: BackgroundTasks) -> dict:
    uid = _get_user_id(request)
    bg.add_task(tailor_all_shortlisted, uid if uid != "local" else None)
    return {"started": "tailoring"}


@app.post("/run/tailor/{application_id}")
@_rate_limit("20/minute")
def trigger_tailor_single(application_id: int, request: Request, bg: BackgroundTasks) -> dict:
    """Tailor resume + cover letter for one specific application."""
    uid = _require_owned_application(request, application_id)
    allowed, detail, usage = _check_tailor_limit(uid)
    if not allowed:
        raise HTTPException(status_code=429, detail=detail)
    from app.tailoring.tailor import tailor_for_application
    bg.add_task(tailor_for_application, application_id)
    _increment_tailor(uid)
    return {"started": "tailoring", "application_id": application_id, "usage": usage}


@app.post("/run/autofill/{application_id}")
@_rate_limit("20/minute")
def trigger_autofill(application_id: int, request: Request, bg: BackgroundTasks) -> dict:
    uid = _require_owned_application(request, application_id)
    allowed, detail, usage = _check_autofill_limit(uid)
    if not allowed:
        raise HTTPException(status_code=429, detail=detail)
    bg.add_task(autofill, application_id, bypass_delay=True)
    _increment_autofill(uid)
    return {"started": "autofill", "application_id": application_id, "usage": usage}


@app.post("/run/preview/{application_id}")
def trigger_preview(application_id: int, request: Request, bg: BackgroundTasks) -> dict:
    """Re-open the filled form in a visible Playwright browser for user review."""
    _require_owned_application(request, application_id)
    bg.add_task(preview, application_id)
    return {"started": "preview", "application_id": application_id}


# ── User Profile endpoints ──────────────────────────────────────────────────

# Full UserProfile column set, kept in lock-step with app/db/models.py.
# (column, sqlite_type, postgres_type)
_USERPROFILE_COLUMNS = [
    ("user_id", "VARCHAR", "VARCHAR"),
    ("first_name", "VARCHAR DEFAULT ''", "VARCHAR DEFAULT ''"),
    ("last_name", "VARCHAR DEFAULT ''", "VARCHAR DEFAULT ''"),
    ("email", "VARCHAR DEFAULT ''", "VARCHAR DEFAULT ''"),
    ("phone", "VARCHAR DEFAULT ''", "VARCHAR DEFAULT ''"),
    ("location", "VARCHAR DEFAULT ''", "VARCHAR DEFAULT ''"),
    ("linkedin_url", "VARCHAR DEFAULT ''", "VARCHAR DEFAULT ''"),
    ("github_url", "VARCHAR DEFAULT ''", "VARCHAR DEFAULT ''"),
    ("portfolio_url", "VARCHAR DEFAULT ''", "VARCHAR DEFAULT ''"),
    ("work_authorization", "VARCHAR DEFAULT ''", "VARCHAR DEFAULT ''"),
    ("requires_sponsorship", "BOOLEAN DEFAULT 0", "BOOLEAN DEFAULT FALSE"),
    ("visa_status", "VARCHAR DEFAULT ''", "VARCHAR DEFAULT ''"),
    ("current_title", "VARCHAR DEFAULT ''", "VARCHAR DEFAULT ''"),
    ("years_experience", "INTEGER DEFAULT 0", "INTEGER DEFAULT 0"),
    ("salary_min", "INTEGER DEFAULT 0", "INTEGER DEFAULT 0"),
    ("salary_max", "INTEGER DEFAULT 0", "INTEGER DEFAULT 0"),
    ("salary_currency", "VARCHAR DEFAULT 'USD'", "VARCHAR DEFAULT 'USD'"),
    ("degree", "VARCHAR DEFAULT ''", "VARCHAR DEFAULT ''"),
    ("university", "VARCHAR DEFAULT ''", "VARCHAR DEFAULT ''"),
    ("graduation_year", "INTEGER", "INTEGER"),
    ("gender", "VARCHAR DEFAULT 'Decline to self-identify'", "VARCHAR DEFAULT 'Decline to self-identify'"),
    ("ethnicity", "VARCHAR DEFAULT 'Decline to self-identify'", "VARCHAR DEFAULT 'Decline to self-identify'"),
    ("veteran_status", "VARCHAR DEFAULT 'I am not a protected veteran'", "VARCHAR DEFAULT 'I am not a protected veteran'"),
    ("disability_status",
     "VARCHAR DEFAULT 'No, I do not have a disability, or history/record of having a disability'",
     "VARCHAR DEFAULT 'No, I do not have a disability, or history/record of having a disability'"),
    ("professional_summary", "TEXT DEFAULT ''", "TEXT DEFAULT ''"),
    ("key_skills", "TEXT DEFAULT ''", "TEXT DEFAULT ''"),
    ("target_roles", "TEXT DEFAULT ''", "TEXT DEFAULT ''"),
    ("job_type_preference", "VARCHAR DEFAULT 'full_time'", "VARCHAR DEFAULT 'full_time'"),
    ("work_auth_status", "VARCHAR DEFAULT ''", "VARCHAR DEFAULT ''"),
    ("include_internships_in_discovery", "BOOLEAN DEFAULT 0", "BOOLEAN DEFAULT FALSE"),
    ("industry", "VARCHAR DEFAULT ''", "VARCHAR DEFAULT ''"),
    ("created_at", "DATETIME", "TIMESTAMP"),
    ("updated_at", "DATETIME", "TIMESTAMP"),
]


def _repair_userprofile_schema() -> None:
    """Idempotently ensure every UserProfile column exists on the live DB.

    Unlike init_db()'s migration helper (which swallows DDL errors and only
    *prints* them), this raises on a genuine failure — so a real problem
    (permissions, connection pooler, etc.) surfaces in the API response/logs
    instead of silently leaving a column missing and looping us back to a 500.

    Postgres uses ``ADD COLUMN IF NOT EXISTS`` (idempotent). SQLite — which has
    no IF NOT EXISTS for columns — is guarded by an inspector lookup.
    """
    from sqlalchemy import text as _text, inspect as _inspect
    from app.db.init_db import engine as _engine
    from app.config import settings as _settings

    if _settings.use_supabase:
        with _engine.begin() as conn:
            for col, _sq, pg in _USERPROFILE_COLUMNS:
                conn.execute(_text(
                    f'ALTER TABLE userprofile ADD COLUMN IF NOT EXISTS {col} {pg}'))
    else:
        insp = _inspect(_engine)
        if not insp.has_table("userprofile"):
            return
        existing = {c["name"].lower() for c in insp.get_columns("userprofile")}
        with _engine.begin() as conn:
            for col, sq, _pg in _USERPROFILE_COLUMNS:
                if col.lower() not in existing:
                    conn.execute(_text(
                        f'ALTER TABLE userprofile ADD COLUMN {col} {sq}'))


@app.get("/api/profile")
def get_profile(request: Request) -> dict:
    """Return the current user profile (seeds from .env on first call)."""
    from app.autofill.answer_pack import _get_or_create_profile
    from app.config import settings
    uid = _get_user_id(request)
    # In multi-tenant mode an unverified/expired token resolves to uid=None.
    # Return a clean 401 (frontend can prompt re-login) instead of a 500.
    if settings.use_supabase and not uid:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        profile = _get_or_create_profile(user_id=uid if uid != "local" else None)
    except Exception as e:
        # Schema drift self-heal: a not-yet-migrated column makes the read fail.
        log.exception("Profile read failed; repairing schema + retrying: %s", e)
        _repair_userprofile_schema()
        profile = _get_or_create_profile(user_id=uid if uid != "local" else None)
    return {
        "id": profile.id,
        "first_name": profile.first_name,
        "last_name": profile.last_name,
        "email": profile.email,
        "phone": profile.phone,
        "location": profile.location,
        "linkedin_url": profile.linkedin_url,
        "github_url": profile.github_url,
        "portfolio_url": profile.portfolio_url,
        "work_authorization": profile.work_authorization,
        "requires_sponsorship": profile.requires_sponsorship,
        "visa_status": profile.visa_status,
        "current_title": profile.current_title,
        "years_experience": profile.years_experience,
        "salary_min": profile.salary_min,
        "salary_max": profile.salary_max,
        "salary_currency": profile.salary_currency,
        "degree": profile.degree,
        "university": profile.university,
        "graduation_year": profile.graduation_year,
        "gender": profile.gender,
        "ethnicity": profile.ethnicity,
        "veteran_status": profile.veteran_status,
        "disability_status": profile.disability_status,
        "professional_summary": profile.professional_summary,
        "key_skills": profile.key_skills,
        "target_roles": profile.target_roles,
        "job_type_preference": getattr(profile, "job_type_preference", "full_time"),
        "work_auth_status": getattr(profile, "work_auth_status", ""),
        "include_internships_in_discovery": getattr(profile, "include_internships_in_discovery", False),
        "industry": getattr(profile, "industry", ""),
    }


class ProfileUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    work_authorization: Optional[str] = None
    requires_sponsorship: Optional[bool] = None
    visa_status: Optional[str] = None
    current_title: Optional[str] = None
    years_experience: Optional[int] = None
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    salary_currency: Optional[str] = None
    degree: Optional[str] = None
    university: Optional[str] = None
    graduation_year: Optional[int] = None
    gender: Optional[str] = None
    ethnicity: Optional[str] = None
    veteran_status: Optional[str] = None
    disability_status: Optional[str] = None
    professional_summary: Optional[str] = None
    key_skills: Optional[str] = None
    target_roles: Optional[str] = None
    job_type_preference: Optional[str] = None
    work_auth_status: Optional[str] = None
    include_internships_in_discovery: Optional[bool] = None
    industry: Optional[str] = None


from datetime import datetime as _dt


@app.put("/api/profile")
def update_profile(request: Request, update: ProfileUpdate) -> dict:
    """Update user profile fields."""
    from app.db.models import UserProfile
    from app.config import settings

    uid = _get_user_id(request)
    # Fail closed: an unauthenticated/expired token in multi-tenant mode would
    # otherwise resolve to user_id=None, and the query below (no WHERE clause)
    # would update the FIRST profile in the table — i.e. some other user's data.
    if settings.use_supabase and not uid:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user_id_arg = uid if uid != "local" else None

    def _do_update():
        # Get-or-create inside the same session to avoid detached-instance issues
        with get_session() as session:
            q = select(UserProfile)
            if user_id_arg:
                q = q.where(UserProfile.user_id == user_id_arg)
            db_profile = session.exec(q).first()
            if not db_profile:
                db_profile = UserProfile(user_id=user_id_arg)
                session.add(db_profile)
                session.flush()
            for field, value in update.model_dump(exclude_none=True).items():
                setattr(db_profile, field, value)
            db_profile.updated_at = _dt.utcnow()
            session.add(db_profile)
            session.commit()

    try:
        _do_update()
    except Exception as e:
        # Most common cause is schema drift — a model column (e.g. target_roles)
        # that hasn't been migrated onto this database yet, which makes the
        # SELECT * fail. Repair the schema (surfacing any real DDL error rather
        # than swallowing it) and retry once before giving up.
        log.exception("Profile update failed; repairing schema + retrying: %s", e)
        try:
            _repair_userprofile_schema()
            _do_update()
        except Exception as e2:
            log.exception("Profile update failed after schema repair: %s", e2)
            raise HTTPException(status_code=500, detail=f"Could not save profile: {e2}")
    return {"success": True}


# ── Target Roles endpoints ──────────────────────────────────────────────────
# Target Roles are the job titles we search & rank for. They live separately
# from the profile so a user can target roles different from their current one,
# and so discovery can be gated on "roles set + resume uploaded".

# A small curated pool used to seed suggestions when we can't infer much.
_ROLE_SUGGESTION_POOL = [
    "Software Engineer", "Senior Software Engineer", "Backend Engineer",
    "Frontend Engineer", "Full Stack Engineer", "Machine Learning Engineer",
    "AI Engineer", "Data Scientist", "Data Engineer", "Data Analyst",
    "DevOps Engineer", "Platform Engineer", "Product Manager",
    "Engineering Manager", "Cloud Engineer", "Security Engineer",
    "Mobile Engineer", "QA Engineer", "Site Reliability Engineer",
]

# Department / industry → role pool, so suggestions + discovery work for
# non-CS fields. Keyed by lowercase signals found in degree/industry/title.
_DEPARTMENT_ROLES = {
    "civil": ["Civil Engineer", "Structural Engineer", "Project Engineer",
              "Transportation Engineer", "Geotechnical Engineer", "Construction Engineer"],
    "mechanical": ["Mechanical Engineer", "Design Engineer", "Manufacturing Engineer",
                   "HVAC Engineer", "Mechanical Design Engineer", "Product Engineer"],
    "aerospace": ["Aerospace Engineer", "Mechanical Engineer", "Systems Engineer",
                  "Propulsion Engineer", "Flight Test Engineer"],
    "electrical": ["Electrical Engineer", "Power Systems Engineer", "Controls Engineer",
                   "Hardware Engineer", "Electronics Engineer", "Embedded Engineer"],
    "chemical": ["Chemical Engineer", "Process Engineer", "Process Safety Engineer",
                 "Manufacturing Engineer", "Production Engineer"],
    "biomedical": ["Biomedical Engineer", "R&D Engineer", "Clinical Engineer",
                   "Quality Engineer", "Validation Engineer"],
    "industrial": ["Industrial Engineer", "Process Engineer", "Operations Engineer",
                   "Supply Chain Analyst", "Manufacturing Engineer"],
    "environmental": ["Environmental Engineer", "Civil Engineer", "Sustainability Analyst",
                      "Water Resources Engineer"],
    "finance": ["Financial Analyst", "Investment Analyst", "Financial Planning Analyst",
                "Corporate Finance Analyst"],
    "accounting": ["Accountant", "Staff Accountant", "Financial Analyst", "Auditor"],
    "marketing": ["Marketing Analyst", "Digital Marketing Specialist", "Brand Manager",
                  "Marketing Coordinator"],
    "data": ["Data Analyst", "Data Scientist", "Business Analyst", "Data Engineer"],
    "biology": ["Research Associate", "Lab Technician", "Scientist", "Research Scientist"],
    "chemistry": ["Chemist", "Research Associate", "Analytical Chemist", "QC Chemist"],
}

_DEPARTMENT_SIGNALS = {
    "civil": ("civil",),
    "mechanical": ("mechanical", "mech eng"),
    "aerospace": ("aerospace", "aeronautic", "aero eng"),
    "electrical": ("electrical", "electronic", "power systems", "ece"),
    "chemical": ("chemical eng", "chem eng", "chemical engineering"),
    "biomedical": ("biomedical", "bioengineer", "bme"),
    "industrial": ("industrial eng", "industrial engineering", "operations research"),
    "environmental": ("environmental eng", "environmental engineering"),
    "finance": ("finance", "financial"),
    "accounting": ("accounting", "accountant"),
    "marketing": ("marketing",),
    "data": ("data science", "statistics", "analytics"),
    "biology": ("biology", "biological", "biotech"),
    "chemistry": ("chemistry", "chemist"),
}


def _detect_department(profile) -> str | None:
    """Detect the user's department/field from industry + degree + current title."""
    if not profile:
        return None
    blob = " ".join([
        (getattr(profile, "industry", "") or ""),
        (getattr(profile, "degree", "") or ""),
        (getattr(profile, "current_title", "") or ""),
    ]).lower()
    if not blob.strip():
        return None
    for dept, signals in _DEPARTMENT_SIGNALS.items():
        if any(s in blob for s in signals):
            return dept
    return None


def _department_keywords(profile) -> list[str]:
    """Discovery keyword fallback adapted to the user's department (empty if CS/unknown)."""
    dept = _detect_department(profile)
    return list(_DEPARTMENT_ROLES.get(dept, [])) if dept else []


def _suggest_roles(profile, limit: int = 6) -> list[str]:
    """Suggest target roles from the user's department + current title + skills,
    padded with sensible defaults. Pure string work — no LLM, instant + free."""
    out: list[str] = []
    seen: set[str] = set()

    def _add(r: str):
        r = (r or "").strip()
        if r and r.lower() not in seen:
            seen.add(r.lower())
            out.append(r)

    if profile:
        _add(profile.current_title)
        # Skills that read like roles (contain "engineer/scientist/developer/...")
        role_words = ("engineer", "scientist", "developer", "manager", "analyst", "designer")
        for s in (profile.key_skills or "").split(","):
            s = s.strip()
            if s and any(w in s.lower() for w in role_words):
                _add(s)

    # Department-specific pool first (non-CS fields), then the generic pool.
    pool = _department_keywords(profile) + _ROLE_SUGGESTION_POOL
    for r in pool:
        if len(out) >= limit:
            break
        _add(r)
    return out[:limit]


@app.get("/api/target-roles")
def get_target_roles(request: Request) -> dict:
    """Return the user's saved target roles + suggestions to pick from."""
    from app.autofill.answer_pack import _get_or_create_profile
    from app.config import settings
    uid = _get_user_id(request)
    if settings.use_supabase and not uid:
        raise HTTPException(status_code=401, detail="Not authenticated")
    profile = _get_or_create_profile(user_id=uid if uid != "local" else None)
    roles = [r.strip() for r in (profile.target_roles or "").split(",") if r.strip()]
    return {
        "roles": roles,
        "suggestions": [s for s in _suggest_roles(profile) if s not in roles],
        "has_resume": _user_has_resume(uid),
    }


class TargetRolesUpdate(BaseModel):
    roles: list[str]


@app.put("/api/target-roles")
def update_target_roles(request: Request, body: TargetRolesUpdate) -> dict:
    """Save the user's target roles (deduped, trimmed, max 12)."""
    from app.db.models import UserProfile
    from app.config import settings
    uid = _get_user_id(request)
    if settings.use_supabase and not uid:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user_id_arg = uid if uid != "local" else None

    # Clean + dedupe (case-insensitive), preserve order, cap the list.
    cleaned: list[str] = []
    seen: set[str] = set()
    for r in (body.roles or []):
        r = (r or "").strip()
        if r and r.lower() not in seen:
            seen.add(r.lower())
            cleaned.append(r)
    cleaned = cleaned[:12]

    with get_session() as session:
        q = select(UserProfile)
        if user_id_arg:
            q = q.where(UserProfile.user_id == user_id_arg)
        db_profile = session.exec(q).first()
        if not db_profile:
            db_profile = UserProfile(user_id=user_id_arg)
            session.add(db_profile)
            session.flush()
        db_profile.target_roles = ", ".join(cleaned)
        db_profile.updated_at = _dt.utcnow()
        session.add(db_profile)
        session.commit()
    return {"success": True, "roles": cleaned}


@app.get("/api/profile/memory")
def get_profile_memory(request: Request) -> dict:
    """Latest weekly recruiter brief(s) for the current user (own data only)."""
    from app.config import settings
    from app.db.models import UserPersonalMemory
    uid = _get_user_id(request)
    if settings.use_supabase and not uid:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user_id_arg = uid if uid and uid != "local" else None
    with get_session() as session:
        q = select(UserPersonalMemory)
        if user_id_arg:
            q = q.where(UserPersonalMemory.user_id == user_id_arg)
        else:
            q = q.where(UserPersonalMemory.user_id == None)  # noqa: E711
        q = q.order_by(UserPersonalMemory.created_at.desc())
        rows = session.exec(q).all()[:5]
    return {
        "entries": [
            {"id": r.id, "source": r.source, "created_at": r.created_at.isoformat(),
             "recommendations": r.recommendations, "parsed_updates": r.parsed_updates}
            for r in rows
        ]
    }


@app.post("/api/profile/memory/refresh")
@_rate_limit("3/minute")
def refresh_profile_memory(request: Request) -> dict:
    """Trigger a GitHub harvest + recruiter brief now (own data only)."""
    from app.config import settings
    uid = _get_user_id(request)
    if settings.use_supabase and not uid:
        raise HTTPException(status_code=401, detail="Not authenticated")
    from app.intelligence.harvester import run_harvest
    try:
        return run_harvest(user_id=uid if uid and uid != "local" else None, notify=False)
    except Exception as e:
        log.exception("Profile memory refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


def _user_needs_sponsorship(uid) -> bool:
    """True when this user will need visa sponsorship (drives cap-exempt boost)."""
    try:
        from app.autofill.answer_pack import _get_or_create_profile
        from app.intelligence.work_auth import assess_profile
        p = _get_or_create_profile(user_id=uid if uid and uid != "local" else None)
        return bool(assess_profile(p).needs_future_sponsorship)
    except Exception:
        return False


def _get_target_roles(uid) -> list[str]:
    """Load saved target roles for a user as a list (empty if none)."""
    from app.db.models import UserProfile
    user_id_arg = uid if uid and uid != "local" else None
    with get_session() as session:
        q = select(UserProfile)
        if user_id_arg:
            q = q.where(UserProfile.user_id == user_id_arg)
        p = session.exec(q).first()
    if not p or not p.target_roles:
        return []
    return [r.strip() for r in p.target_roles.split(",") if r.strip()]


# ── Profile Avatar endpoints ────────────────────────────────────────────────

def _sign_avatar(bucket, path: str) -> str | None:
    """Return a signed URL for an avatar object, or None if it doesn't exist."""
    try:
        signed = bucket.create_signed_url(path, 3600)
    except Exception:
        return None
    return (signed or {}).get("signedURL") or (signed or {}).get("signedUrl")


@app.get("/api/profile/avatar")
def get_avatar(request: Request) -> dict:
    """Return the signed URL for the user's profile avatar, or null."""
    from app.config import settings
    uid = _get_user_id(request)
    if settings.use_supabase and uid and uid != "local":
        try:
            from app.db.supabase_client import service_client
            bucket = service_client().storage.from_("avatars")
            # Upload always stores {uid}/avatar.<ext> — try known extensions
            # directly. This avoids relying on list(), which can return empty.
            for ext in ("jpg", "jpeg", "png", "webp", "gif"):
                url = _sign_avatar(bucket, f"{uid}/avatar.{ext}")
                if url:
                    return {"url": url}
            # Fallback: list the folder for any unexpected filename.
            try:
                files = bucket.list(uid) or []
                names = [f.get("name", "") for f in files if f.get("name", "").startswith("avatar.")]
                if names:
                    url = _sign_avatar(bucket, f"{uid}/{names[0]}")
                    if url:
                        return {"url": url}
            except Exception:
                pass
        except Exception:
            pass
    return {"url": None}


@app.post("/api/profile/avatar")
async def upload_avatar(request: Request, file: UploadFile = File(...)) -> dict:
    """Upload a profile photo and store in Supabase storage (avatars bucket)."""
    from app.config import settings
    uid = _get_user_id(request)
    ext = (file.filename or "avatar.jpg").rsplit(".", 1)[-1].lower()
    if ext not in {"jpg", "jpeg", "png", "webp", "gif"}:
        raise HTTPException(status_code=400, detail="Unsupported image type")
    content = await file.read()
    if len(content) > 3 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large — max 3 MB")

    if settings.use_supabase and uid and uid != "local":
        try:
            from app.db.supabase_client import service_client
            sb = service_client()
            path = f"{uid}/avatar.{ext}"
            mime = file.content_type or "image/jpeg"
            # Try upsert first; fall back to remove+upload for older supabase-py
            try:
                sb.storage.from_("avatars").upload(path, content, {"content-type": mime, "upsert": "true"})
            except Exception:
                try:
                    sb.storage.from_("avatars").remove([path])
                except Exception:
                    pass
                sb.storage.from_("avatars").upload(path, content, {"content-type": mime})
            signed = sb.storage.from_("avatars").create_signed_url(path, 3600)
            url = (signed or {}).get("signedURL") or (signed or {}).get("signedUrl")
            return {"url": url}
        except Exception as exc:
            log.exception("Avatar upload failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")
    return {"url": None}


# ── Answer Pack endpoint ────────────────────────────────────────────────────

@app.get("/application/{application_id}/referral")
@_rate_limit("20/minute")
def get_referral_drafts(application_id: int, request: Request) -> dict:
    """Draft referral / hiring-manager / visa-alumni outreach for one application.
    Drafts only — the user sends them. Strictly scoped to the owning user."""
    _require_owned_application(request, application_id)
    uid = _get_user_id(request)
    from app.intelligence.referral import generate_referral_drafts
    try:
        return generate_referral_drafts(application_id, user_id=uid if uid != "local" else None)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.exception("Referral draft generation failed for app %d: %s", application_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/application/{application_id}/answer-pack")
def get_answer_pack(application_id: int, request: Request) -> dict:
    """Generate (or return cached) answer pack for one application."""
    from app.autofill.answer_pack import generate_answer_pack
    _require_owned_application(request, application_id)
    try:
        return generate_answer_pack(application_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.exception("Answer pack generation failed for app %d: %s", application_id, e)
        raise HTTPException(status_code=500, detail=str(e))


class ExtractLinkRequest(BaseModel):
    url: str


@app.post("/run/extract-link")
@_rate_limit("10/minute")
async def trigger_extract_link(req: ExtractLinkRequest, request: Request, bg: BackgroundTasks) -> dict:
    from app.discovery.extractor import extract_and_rank_job
    from app.tailoring.tailor import tailor_for_application

    uid = _get_user_id(request)
    log.info("Extracting manual link: %s", req.url)
    try:
        app_id = await extract_and_rank_job(req.url, user_id=uid if uid != "local" else None)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Tailor in the background
    bg.add_task(tailor_for_application, app_id)

    return {"success": True, "application_id": app_id}


# ── GDPR / Account deletion ──────────────────────────────────────────────────

@app.delete("/api/account")
def delete_account(request: Request) -> dict:
    """Permanently delete all data for the authenticated user."""
    uid = _require_user(request)
    from app.db.models import (
        UserProfile, PendingQuestion, AnswerMemory, DiscoveryRun,
        UserSubscription, UserUsage,
    )
    from app.db.init_db import get_session
    from sqlmodel import select, delete as sql_delete

    with get_session() as session:
        # Collect application IDs for this user
        if uid != "local":
            app_ids = [
                a.id for a in session.exec(
                    select(Application).where(Application.user_id == uid)
                ).all()
            ]
            # Delete pending questions linked to those applications
            if app_ids:
                session.exec(sql_delete(PendingQuestion).where(PendingQuestion.application_id.in_(app_ids)))
            # Delete every user-scoped row so no orphaned data remains (GDPR).
            session.exec(sql_delete(Application).where(Application.user_id == uid))
            session.exec(sql_delete(Job).where(Job.user_id == uid))
            session.exec(sql_delete(UserProfile).where(UserProfile.user_id == uid))
            session.exec(sql_delete(AnswerMemory).where(AnswerMemory.user_id == uid))
            session.exec(sql_delete(DiscoveryRun).where(DiscoveryRun.user_id == uid))
            session.exec(sql_delete(UserSubscription).where(UserSubscription.user_id == uid))
            session.exec(sql_delete(UserUsage).where(UserUsage.user_id == uid))
            session.commit()

    # Delete resume files from Supabase Storage
    from app.config import settings
    if settings.use_supabase and uid and uid != "local":
        try:
            from app.db.supabase_client import service_client
            sb = service_client()
            files = sb.storage.from_("resume").list(uid) or []
            paths = [f"{uid}/{f['name']}" for f in files if f.get("name")]
            if paths:
                sb.storage.from_("resume").remove(paths)
            # Also delete Supabase Auth user
            sb.auth.admin.delete_user(uid)
        except Exception:
            pass

    return {"success": True, "message": "All account data deleted."}


# ── CSV export ───────────────────────────────────────────────────────────────

@app.get("/api/export/applications.csv")
def export_applications_csv(request: Request):
    """Download all applications as a CSV file."""
    import csv, io
    from fastapi.responses import StreamingResponse

    uid = _require_user(request)
    _uid_filter = uid and uid != "local"

    with get_session() as session:
        q = select(Application, Job).join(Job)
        if _uid_filter:
            q = q.where(Application.user_id == uid)
        results = session.exec(q).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "company", "title", "location", "remote", "source",
        "status", "apply_track", "rerank_score", "blended_score",
        "url", "created_at", "updated_at", "submitted_at", "notes"
    ])
    for app_row, job_row in results:
        writer.writerow([
            app_row.id,
            job_row.company,
            job_row.title,
            job_row.location,
            job_row.remote,
            job_row.source.value if job_row.source else "",
            app_row.status.value if app_row.status else "",
            app_row.apply_track or "",
            job_row.rerank_score,
            job_row.blended_score,
            job_row.url,
            app_row.created_at.isoformat() if app_row.created_at else "",
            app_row.updated_at.isoformat() if app_row.updated_at else "",
            app_row.submitted_at.isoformat() if app_row.submitted_at else "",
            (app_row.notes or "").replace("\n", " "),
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=applications.csv"},
    )
