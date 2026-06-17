"""End-to-end matching pipeline.

1. Load resume from disk.
2. Rebuild FAISS index over all jobs (cheap, runs in seconds for <10k jobs).
3. Search top-K by cosine similarity.
4. Rerank with Claude, store score + reasoning back on Job rows.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from sqlmodel import select

from app.config import settings
from app.db.init_db import get_session
from app.db.models import Application, ApplicationStatus, Job, JobSource
from app.matching.matcher import Matcher
from app.matching.reranker import Reranker
from app.matching.filters import RuleFilter, EmbeddingFilter, score_ghost
from app.matching.hire_probability import score_hire_probability, blended_score as compute_blended
from app.intelligence.senior_reviewer import SeniorReviewer

# Sources where the bot can fill the form automatically
_AUTOFILL_SOURCES = {JobSource.GREENHOUSE, JobSource.LEVER, JobSource.ASHBY, JobSource.WORKDAY, JobSource.SMARTRECRUITERS}

# ── Company cap (job-based, not company-based) ────────────────────────────────
# Statuses that count as an "active" application toward the per-company cap.
# IMPORTANT: SUBMITTED and INTERVIEWING are included — otherwise submitting an
# application would drop its company count to 0 and let the company flood the
# shortlist again, which is exactly the bug we are fixing.
_CAP_ACTIVE_STATUSES = [
    ApplicationStatus.SHORTLISTED,
    ApplicationStatus.TAILORED,
    ApplicationStatus.AUTOFILLED,
    ApplicationStatus.AWAITING_USER,
    ApplicationStatus.READY_TO_SUBMIT,
    ApplicationStatus.SUBMITTED,
    ApplicationStatus.INTERVIEWING,
]
_COMPANY_CAP = 2              # default max active applications per company (overridable via settings.company_cap)
_COMPANY_COOLDOWN_DAYS = 40   # once at the cap, a company is locked until its
                              # existing applications are this many days old

log = logging.getLogger(__name__)


def _app_age_days(app: Application) -> float:
    """Days since this application became active (submission date if known)."""
    ref = app.submitted_at or app.created_at
    if ref is None:
        return 0.0
    return (datetime.utcnow() - ref).total_seconds() / 86400.0


def _check_and_enforce_company_cap(session, job: Job, score: float) -> bool:
    """Decide whether a new application may be created for ``job`` under the
    per-company cap + cooldown rule.

    Rules:
      • At most ``_COMPANY_CAP`` (2) active applications per company at a time.
      • Once a company is at the cap, NO new role from that company is shortlisted
        until its existing active applications are ``_COMPANY_COOLDOWN_DAYS`` (40)
        days old. Applications past that age are treated as expired: they are
        marked SKIPPED to reopen the slot, and the new (fresher) role is allowed.

    Returns True if a new application is allowed (slots have been freed if needed),
    False if the company is still within cooldown and the job must be skipped.
    """
    from app.config import settings as _cap_settings
    cap = _cap_settings.company_cap or _COMPANY_CAP

    active = session.exec(
        select(Application).join(Job, Application.job_id == Job.id)
        .where(Job.company == job.company)
        .where(Application.status.in_(_CAP_ACTIVE_STATUSES))
    ).all()

    if len(active) < cap:
        return True  # room available

    # At the cap — only proceed if enough existing apps have aged out (>=40d).
    expired = [a for a in active if _app_age_days(a) >= _COMPANY_COOLDOWN_DAYS]
    if not expired:
        log.info(
            "Company cap: %s already has %d active app(s), none older than %dd — "
            "skipping '%s' until cooldown expires.",
            job.company, len(active), _COMPANY_COOLDOWN_DAYS, job.title,
        )
        return False

    # Free up slots by expiring the oldest aged-out applications.
    expired.sort(key=_app_age_days, reverse=True)  # oldest first
    slots_to_free = len(active) - cap + 1  # need at least 1 free slot
    for a in expired[:slots_to_free]:
        a.status = ApplicationStatus.SKIPPED
        a.notes = (a.notes or "") + (
            f"\nExpired after {_app_age_days(a):.0f}d (>{_COMPANY_COOLDOWN_DAYS}d cooldown) "
            f"— slot reopened for newer '{job.title}'."
        )
        session.add(a)
        log.info("Company cap: expired app %d at %s (age %.0fd) to reopen a slot.",
                 a.id, job.company, _app_age_days(a))
    return True


def _load_resume(user_id: str | None = None) -> str:
    """Load resume — checks Supabase Storage per user first, falls back to local file."""
    from app.config import settings as _s
    if user_id and _s.use_supabase:
        for ext in ("md", "txt", "pdf", "docx"):
            try:
                from app.db.supabase_client import service_client
                sb = service_client()
                path = f"{user_id}/resume.{ext}"
                data = sb.storage.from_("resume").download(path)
                if data:
                    if ext in ("md", "txt"):
                        return data.decode("utf-8", errors="ignore")
                    # For PDF/DOCX, extract text
                    if ext == "docx":
                        import io
                        from docx import Document
                        doc = Document(io.BytesIO(data))
                        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
                    if ext == "pdf":
                        import io
                        from pypdf import PdfReader
                        reader = PdfReader(io.BytesIO(data))
                        return "\n".join((page.extract_text() or "") for page in reader.pages)
            except Exception:
                continue
    p: Path = settings.resume_path
    if not p.exists():
        raise FileNotFoundError(
            f"Resume not found at {p}. Put a markdown version of your resume there."
        )
    return p.read_text(encoding="utf-8")


def _run_senior_review(reviewer: SeniorReviewer, job_id: int, app_id: int) -> None:
    """Call SeniorReviewer for one job and write results back to the Application row."""
    try:
        with get_session() as session:
            job = session.get(Job, job_id)
            if not job:
                return
            result = reviewer.review(job)
            if result is None:
                return
            app = session.get(Application, app_id)
            if not app:
                return
            app.profile_variant = result.recommended_resume_variant
            app.senior_fit_score = float(result.fit_score)
            app.senior_verdict = result.senior_reviewer_verdict
            app.custom_highlight_block = result.custom_highlight_block or None
            session.add(app)
            session.commit()
            log.info(
                "SeniorReview job %d app %d: variant=%s score=%d genuine=%s",
                job_id, app_id, result.recommended_resume_variant,
                result.fit_score, result.is_genuine_match,
            )
    except Exception as e:
        log.exception("SeniorReview failed for job %d app %d: %s", job_id, app_id, e)


def run_matching(user_id: str | None = None) -> List[int]:
    resume = _load_resume(user_id=user_id)
    matcher = Matcher()
    matcher.rebuild(user_id=user_id)

    candidates = matcher.search_for_resume(resume, k=settings.top_k_rerank, user_id=user_id)
    candidates = [(jid, score) for jid, score in candidates if score >= settings.min_match_score]
    log.info("%d candidates above cross-encoder threshold %.2f", len(candidates), settings.min_match_score)

    rule_filter = RuleFilter()
    embedding_filter = EmbeddingFilter(matcher=matcher)
    reranker = Reranker()
    senior_reviewer = SeniorReviewer()
    shortlisted: List[int] = []

    # Count applications already created today (scoped to user)
    with get_session() as session:
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        q = select(Application).where(Application.created_at >= today_start)
        if user_id:
            q = q.where(Application.user_id == user_id)
        today_count = len(session.exec(q).all())

    for jid, sim in candidates:
        with get_session() as session:
            job = session.get(Job, jid)
            if not job:
                continue

            # Skip closed/purged jobs — they must never be (re-)shortlisted.
            if job.is_closed:
                continue

            # Check if job is already scored to avoid wasting LLM tokens and time
            if job.rerank_score is not None:
                if job.rerank_score >= settings.shortlist_score_threshold:
                    existing = session.exec(
                        select(Application).where(Application.job_id == job.id)
                    ).first()
                    if not existing:
                        if today_count < settings.daily_shortlist_limit:
                            # Company cap + 40-day cooldown applies here too —
                            # previously this already-scored path skipped the cap,
                            # which let a single company flood the shortlist.
                            if not _check_and_enforce_company_cap(session, job, job.rerank_score):
                                session.commit()
                                continue
                            _track = "autofill" if job.source in _AUTOFILL_SOURCES else "manual"
                            session.add(
                                Application(
                                    job_id=job.id,
                                    status=ApplicationStatus.SHORTLISTED,
                                    apply_url=job.url,
                                    apply_track=_track,
                                    user_id=user_id,
                                )
                            )
                            shortlisted.append(job.id)
                            today_count += 1
                            session.commit()
                            log.info("Job '%s' @ '%s' already scored (%d) — %s track. Shortlisted.", job.title, job.company, job.rerank_score, _track)
                        else:
                            log.info("Daily apply limit reached — skipping application creation for already scored job %s.", job.title)
                    else:
                        log.info("Job '%s' @ '%s' already scored (%d) and has application. Skipping.", job.title, job.company, job.rerank_score)
                else:
                    log.info("Job '%s' @ '%s' already scored (%d). Skipping.", job.title, job.company, job.rerank_score)
                continue

            # 1. Rule Filter
            rule_res = rule_filter.filter(job)
            if not rule_res.passed:
                log.info("Job '%s' @ '%s' filtered by rules: %s", job.title, job.company, rule_res.reason)
                job.similarity_score = sim
                job.rerank_score = float(rule_res.score_override or 10.0)
                job.rerank_reasoning = f"Rule filtered: {rule_res.reason}"
                session.add(job)
                session.commit()
                continue

            # 2. Ghost Job Detection — cheap DB+text check, runs before LLM/embedding to save cost
            ghost_res = score_ghost(job, session)
            job.ghost_score = ghost_res.ghost_score
            job.ghost_flags = ghost_res.flags_json
            if ghost_res.is_ghost:
                log.info(
                    "Job '%s' @ '%s' flagged as likely ghost (score=%.2f flags=%s) — skipping",
                    job.title, job.company, ghost_res.ghost_score, ghost_res.flags,
                )
                job.similarity_score = sim
                job.rerank_score = 5.0
                job.rerank_reasoning = f"Ghost filtered (score={ghost_res.ghost_score:.2f}): {', '.join(ghost_res.flags)}"
                session.add(job)
                session.commit()
                continue
            elif ghost_res.ghost_score >= 0.3:
                log.warning(
                    "Job '%s' @ '%s' is suspicious (ghost_score=%.2f flags=%s) — proceeding with caution",
                    job.title, job.company, ghost_res.ghost_score, ghost_res.flags,
                )
            session.add(job)  # persist ghost_score even for passing jobs

            # 3. Embedding Filter
            emb_passed, emb_score, emb_reason = embedding_filter.filter(job, resume)
            if not emb_passed:
                log.info("Job '%s' @ '%s' filtered by embedding similarity: %s", job.title, job.company, emb_reason)
                job.similarity_score = sim
                job.rerank_score = 15.0
                job.rerank_reasoning = f"Embedding filtered: {emb_reason}"
                session.add(job)
                session.commit()
                continue

            # 3. LLM Reranking
            score, reason, concerns = reranker.score(resume, job)
            job.similarity_score = sim
            job.rerank_score = score
            job.rerank_reasoning = reason + (
                ("\nConcerns: " + "; ".join(concerns)) if concerns else ""
            )

            # 5. Hire Probability Scoring — no LLM, uses DB + description signals
            import json as _json
            hp_result = score_hire_probability(job, session)
            job.hire_probability_score = hp_result.score
            job.hire_probability_signals = _json.dumps(hp_result.signals)
            job.blended_score = compute_blended(score, hp_result.score)
            log.info(
                "HireProb job '%s' @ '%s': hp=%.2f signals=%s blended=%.1f",
                job.title, job.company, hp_result.score, hp_result.signals[:3], job.blended_score,
            )
            session.add(job)

            # If rerank ≥ threshold, create an Application row in SHORTLISTED state
            new_app_id: int | None = None
            if score >= settings.shortlist_score_threshold:
                existing = session.exec(
                    select(Application).where(Application.job_id == job.id)
                ).first()
                if not existing:
                    if today_count < settings.daily_shortlist_limit:
                        # ── Company cap (max 2 active) + 40-day cooldown ──
                        if not _check_and_enforce_company_cap(session, job, score):
                            session.commit()
                            continue

                        _track = "autofill" if job.source in _AUTOFILL_SOURCES else "manual"
                        new_app = Application(
                            job_id=job.id,
                            status=ApplicationStatus.SHORTLISTED,
                            apply_url=job.url,
                            apply_track=_track,
                            user_id=user_id,
                        )
                        session.add(new_app)
                        session.flush()  # populate new_app.id before commit
                        new_app_id = new_app.id
                        shortlisted.append(job.id)
                        today_count += 1
                    else:
                        log.info("Daily shortlist limit (%d) reached — skipping application creation for job %s.", settings.daily_shortlist_limit, job.title)

            session.commit()
            log.info("Job %s @ %s: sim=%.3f rerank=%.0f — %s",
                     job.title, job.company, sim, score, reason)

        # Run SeniorReviewer outside the session (LLM call — don't hold a lock)
        if new_app_id is not None:
            _run_senior_review(senior_reviewer, jid, new_app_id)
    return shortlisted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    new_ids = run_matching()
    print(f"Shortlisted {len(new_ids)} new applications.")
