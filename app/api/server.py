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

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile

log = logging.getLogger(__name__)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import select
from sqlalchemy import func, desc

from app.autofill.agent import autofill, preview
from app.db.init_db import get_session
from app.db.models import Application, ApplicationStatus, Job
from app.discovery.pipeline import run_discovery
from app.matching.pipeline import run_matching
from app.tailoring.tailor import tailor_all_shortlisted

app = FastAPI(title="JobAgent")


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


# ── Auth helpers ─────────────────────────────────────────────────────────────

def _get_user_id(request: Request) -> str | None:
    """Extract user_id from Bearer token. Returns None if not authenticated."""
    from app.config import settings
    if not settings.use_supabase:
        return "local"   # single-user mode — no auth
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth.split(" ", 1)[1]
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
            files = sb.storage.from_("resumes").list(uid)
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
    """Run discovery → matching every 6 hours for all users."""
    import asyncio
    import logging
    _log = logging.getLogger("scheduler")
    INTERVAL = 6 * 60 * 60  # 6 hours in seconds
    # Wait 2 min after boot before first run so the server is fully ready
    await asyncio.sleep(120)
    while True:
        try:
            _log.info("Scheduler: starting discovery run")
            await asyncio.to_thread(run_discovery)
            _log.info("Scheduler: discovery done, starting matching")
            await asyncio.to_thread(run_matching)
            _log.info("Scheduler: matching done, sleeping %dh", INTERVAL // 3600)
        except Exception as e:
            _log.exception("Scheduler error: %s", e)
        await asyncio.sleep(INTERVAL)


@app.get("/")
def index_redirect():
    return RedirectResponse(url="/dashboard")


try:
    templates = Jinja2Templates(directory="app/templates")
except Exception:
    import os
    templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "..", "templates"))


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

    from app.config import settings
    if settings.use_supabase:
        try:
            from app.db.supabase_client import service_client
            sb = service_client()
            path = f"{uid}/resume.{ext}"
            sb.storage.from_("resumes").upload(path, content, {"upsert": "true", "content-type": file.content_type})
            public_url = sb.storage.from_("resumes").get_public_url(path)
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
key_skills (comma-separated string), professional_summary (2-3 sentence summary of their background)

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

    # Save to profile — reuse same single-session logic
    from app.db.models import UserProfile
    import datetime as _datetime
    with get_session() as session:
        q = select(UserProfile)
        q = q.where(UserProfile.user_id == uid)
        db_profile = session.exec(q).first()
        if not db_profile:
            db_profile = UserProfile(user_id=uid)
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

    return {"success": True, "extracted": {k: extracted.get(k) for k in field_map}}


@app.get("/api/resume/status")
def resume_status(request: Request) -> dict:
    """Whether the current user has a resume on file. Drives the Discover gate."""
    uid = _get_user_id(request)
    return {"has_resume": _user_has_resume(uid)}


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
            sb.storage.from_("resumes").upload(
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
def shortlist():
    """Top-scored jobs not yet processed."""
    with get_session() as session:
        jobs = session.exec(
            select(Job).where(Job.rerank_score >= 70).order_by(Job.rerank_score.desc())
        ).all()
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
    uid = _get_user_id(request)
    _uid_filter = uid and uid != "local"
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

    def _priority(job) -> float:
        """Rank by blended score (fit + hiring intent) when available, else fall back to rerank."""
        if job.blended_score is not None:
            return job.blended_score
        return job.rerank_score or 0

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

    from app.config import settings
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


@app.post("/run/discovery")
def trigger_discovery(request: Request, bg: BackgroundTasks) -> dict:
    uid = _get_user_id(request)
    # Gate: no resume → no discovery. Without a resume there is nothing to match
    # against, so scraping jobs into the user's pool would only show noise.
    if not _user_has_resume(uid):
        raise HTTPException(
            status_code=400,
            detail="Upload your resume (or fill in your profile) before discovering jobs.",
        )
    bg.add_task(run_discovery, uid if uid != "local" else None)
    return {"started": "discovery"}


@app.post("/run/matching")
def trigger_matching(request: Request, bg: BackgroundTasks) -> dict:
    uid = _get_user_id(request)
    bg.add_task(run_matching, uid if uid != "local" else None)
    return {"started": "matching"}


@app.post("/run/tailor")
def trigger_tailor(request: Request, bg: BackgroundTasks) -> dict:
    uid = _get_user_id(request)
    bg.add_task(tailor_all_shortlisted, uid if uid != "local" else None)
    return {"started": "tailoring"}


@app.post("/run/autofill/{application_id}")
def trigger_autofill(application_id: int, request: Request, bg: BackgroundTasks) -> dict:
    _require_owned_application(request, application_id)
    bg.add_task(autofill, application_id, bypass_delay=True)
    return {"started": "autofill", "application_id": application_id}


@app.post("/run/preview/{application_id}")
def trigger_preview(application_id: int, request: Request, bg: BackgroundTasks) -> dict:
    """Re-open the filled form in a visible Playwright browser for user review."""
    _require_owned_application(request, application_id)
    bg.add_task(preview, application_id)
    return {"started": "preview", "application_id": application_id}


from pydantic import BaseModel


# ── User Profile endpoints ──────────────────────────────────────────────────

@app.get("/api/profile")
def get_profile(request: Request) -> dict:
    """Return the current user profile (seeds from .env on first call)."""
    from app.autofill.answer_pack import _get_or_create_profile
    uid = _get_user_id(request)
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


from typing import Optional as _Opt
from datetime import datetime as _dt


@app.put("/api/profile")
def update_profile(request: Request, update: ProfileUpdate) -> dict:
    """Update user profile fields."""
    from app.autofill.answer_pack import _get_or_create_profile
    from app.db.models import UserProfile

    uid = _get_user_id(request)
    user_id_arg = uid if uid != "local" else None
    # Get-or-create inside the same session to avoid detached-instance issues
    with get_session() as session:
        q = select(UserProfile)
        if user_id_arg:
            q = q.where(UserProfile.user_id == user_id_arg)
        db_profile = session.exec(q).first()
        if not db_profile:
            # Create it fresh
            db_profile = UserProfile(user_id=user_id_arg)
            session.add(db_profile)
            session.flush()
        for field, value in update.model_dump(exclude_none=True).items():
            setattr(db_profile, field, value)
        db_profile.updated_at = _dt.utcnow()
        session.add(db_profile)
        session.commit()
    return {"success": True}


# ── Profile Avatar endpoints ────────────────────────────────────────────────

@app.get("/api/profile/avatar")
def get_avatar(request: Request) -> dict:
    """Return the signed URL for the user's profile avatar, or null."""
    from app.config import settings
    uid = _get_user_id(request)
    if settings.use_supabase and uid and uid != "local":
        try:
            from app.db.supabase_client import service_client
            sb = service_client()
            files = sb.storage.from_("avatars").list(uid)
            if files:
                names = [f.get("name", "") for f in files if f.get("name", "").startswith("avatar.")]
                if names:
                    signed = sb.storage.from_("avatars").create_signed_url(f"{uid}/{names[0]}", 3600)
                    url = (signed or {}).get("signedURL") or (signed or {}).get("signedUrl")
                    return {"url": url}
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
