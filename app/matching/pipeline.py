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

    Rules (evaluated per user — one tenant's applications never consume
    another tenant's slots):
      • At most ``settings.company_cap`` active applications per company at a
        time (falls back to ``_COMPANY_CAP`` when unset).
      • Once a company is at the cap, NO new role from that company is shortlisted
        until its existing active applications are ``_COMPANY_COOLDOWN_DAYS`` (40)
        days old. Applications past that age are treated as expired: they are
        marked SKIPPED to reopen the slot, and the new (fresher) role is allowed.

    Returns True if a new application is allowed (slots have been freed if needed),
    False if the company is still within cooldown and the job must be skipped.
    """
    from app.config import settings as _cap_settings
    cap = _cap_settings.company_cap or _COMPANY_CAP

    q = (
        select(Application).join(Job, Application.job_id == Job.id)
        .where(Job.company == job.company)
        .where(Application.status.in_(_CAP_ACTIVE_STATUSES))
    )
    # Scope to this job's owner: NULL/"local" rows are the single-user dev data.
    q = q.where(Application.user_id == job.user_id) if job.user_id else q.where(Application.user_id.is_(None))
    active = session.exec(q).all()

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
        raise ValueError("No resume found. Please upload your resume in the Profile page first.")
    
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


def _reset_stale_sponsorship_scores(user_id: str | None) -> int:
    """Self-heal: jobs hard-blocked by the OLD right-to-work boilerplate rule
    ("Sponsorship pre-filtered: matches 'must be authorized to work in...'")
    keep their stale score-10 forever, even though that boilerplate no longer
    blocks. Clear those scores so this run re-ranks them under the fixed logic.
    Idempotent — once re-scored, the old reasoning text is gone. Explicit
    refusals (NO_SPONSORSHIP_HARD) are left untouched."""
    from app.matching.filters.constants import WORK_AUTH_BOILERPLATE
    cleared = 0
    try:
        with get_session() as session:
            q = select(Job).where(
                Job.rerank_reasoning.like("Rule filtered: Sponsorship pre-filtered:%"),  # type: ignore[union-attr]
                Job.user_id == user_id,
            )
            for job in session.exec(q).all():
                reason_low = (job.rerank_reasoning or "").lower()
                if any(f"'{p}'" in reason_low for p in WORK_AUTH_BOILERPLATE):
                    job.rerank_score = None
                    job.rerank_reasoning = None
                    session.add(job)
                    cleared += 1
            session.commit()
    except Exception as e:
        log.warning("Stale sponsorship-score reset failed (non-fatal): %s", e)
    if cleared:
        log.info("Cleared %d stale boilerplate sponsorship scores for re-ranking", cleared)
    return cleared


def run_matching(user_id: str | None = None) -> List[int]:
    resume = _load_resume(user_id=user_id)
    # Give jobs unfairly blocked by the old sponsorship boilerplate rule a
    # fresh scoring pass under the fixed logic.
    _reset_stale_sponsorship_scores(user_id)
    matcher = Matcher()
    matcher.rebuild(user_id=user_id)

    # Per-user filtering: load this tenant's profile so retrieval and the rule
    # filter target their country / experience / salary band / skills /
    # sponsorship need (not a baked-in single candidate). Falls back to legacy
    # defaults if unavailable.
    _user_profile = None
    try:
        from app.autofill.answer_pack import _get_or_create_profile
        _user_profile = _get_or_create_profile(user_id=user_id)
    except Exception as _pe:
        log.debug("RuleFilter profile unavailable (using legacy defaults): %s", _pe)

    candidates = matcher.search_for_resume(resume, k=settings.top_k_rerank, user_id=user_id,
                                           profile=_user_profile)
    candidates = [(jid, score) for jid, score in candidates if score >= settings.min_match_score]
    log.info("%d candidates above cross-encoder threshold %.2f", len(candidates), settings.min_match_score)

    rule_filter = RuleFilter(profile=_user_profile)
    candidate = None
    if _user_profile:
        try:
            from app.intelligence.door_match import CandidateProfile
            candidate = CandidateProfile.from_user_profile(_user_profile)
        except Exception as _ce:
            log.warning("Could not create CandidateProfile for door filter: %s", _ce)

    embedding_filter = EmbeddingFilter(matcher=matcher)
    reranker = Reranker(profile=_user_profile)
    shortlisted: List[int] = []

    # Count applications already created today (scoped to user)
    with get_session() as session:
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        q = select(Application).where(Application.created_at >= today_start)
        if user_id:
            q = q.where(Application.user_id == user_id)
        today_count = len(session.exec(q).all())

    import json as _json
    from concurrent.futures import ThreadPoolExecutor

    # ── Phase 1: cheap serial pre-filtering ──────────────────────────────────
    # Run the non-LLM gates (already-scored / rule / ghost / embedding). Jobs that
    # survive all gates are collected for parallel LLM scoring in Phase 2. The
    # already-scored shortlist path still runs here so behavior is unchanged.
    to_rerank: list[tuple[int, float]] = []
    _rerank_priority: dict[int, float] = {}  # jid -> source/freshness-weighted CE score
    for jid, sim in candidates:
        with get_session() as session:
            job = session.get(Job, jid)
            if not job:
                continue
            if job.is_closed:
                continue

            # Already scored — (re)shortlist without spending another LLM call.
            if job.rerank_score is not None:
                if job.rerank_score >= settings.shortlist_score_threshold:
                    existing = session.exec(
                        select(Application).where(Application.job_id == job.id)
                    ).first()
                    if not existing:
                        if today_count < settings.daily_shortlist_limit:
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

            # Persist lightweight intelligence flags for filtering/querying.
            try:
                from app.matching.filters.rule_filter import classify_job_type
                from app.intelligence.sponsorship import assess as _assess_spons
                from app.intelligence.urgency import assess as _assess_urg
                job.job_type = classify_job_type(job.title, job.description)
                _sp = _assess_spons(company=job.company or "", description=job.description or "",
                                    url=job.url or "", location=job.location or "")
                job.is_cap_exempt = bool(_sp.cap_exempt)
                job.urgency_score = float(_assess_urg(job).score)
            except Exception as _ie:
                log.debug("intelligence flag tagging skipped: %s", _ie)

            # 2. Ghost Job Detection — cheap DB+text check, runs before LLM to save cost
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

            # 3. Embedding Filter
            emb_passed, emb_score, emb_reason = embedding_filter.filter(job, resume, profile=_user_profile)
            if not emb_passed:
                log.info("Job '%s' @ '%s' filtered by embedding similarity: %s", job.title, job.company, emb_reason)
                job.similarity_score = sim
                job.rerank_score = 15.0
                job.rerank_reasoning = f"Embedding filtered: {emb_reason}"
                session.add(job)
                session.commit()
                continue

            # 4. Door Filter (Cheap JD-only check)
            if candidate:
                try:
                    from app.intelligence.role_bar import build_role_bar
                    from app.intelligence.door_match import classify_door
                    bar = build_role_bar(job.title, job.description or "")
                    verdict = classify_door(candidate, bar, winners_n=0, data_quality="thin")
                    if verdict.wrong_door:
                        log.info("Job '%s' @ '%s' filtered by door match: %s", job.title, job.company, verdict.top_reason)
                        job.similarity_score = sim
                        job.rerank_score = 20.0
                        job.rerank_reasoning = f"Wrong Door: {verdict.top_reason}"
                        session.add(job)
                        session.commit()
                        continue
                except Exception as de:
                    log.warning("Door filter check failed for job %d: %s", job.id, de)

            # Survived all cheap gates — persist similarity + ghost score, queue for LLM.
            job.similarity_score = sim
            session.add(job)
            session.commit()
            to_rerank.append((jid, sim))
            # Priority for the LLM budget: CE score weighted by source quality
            # (direct ATS > boards > remote feeds > redirect aggregators) with a
            # bonus for postings fresher than 48h.
            from app.matching.filters.constants import (
                FRESH_POSTING_BONUS, FRESH_POSTING_HOURS, source_quality,
            )
            prio = sim * source_quality(job.source)
            posted = job.posted_at or job.first_seen
            if posted and (datetime.utcnow() - posted).total_seconds() < FRESH_POSTING_HOURS * 3600:
                prio *= FRESH_POSTING_BONUS
            _rerank_priority[jid] = prio

    # ── Phase 2: parallel LLM scoring (I/O-bound on the model API) ────────────
    # Each worker uses its own DB session + the shared (thread-safe) reranker
    # client, so the slow network calls overlap instead of running one-by-one.
    rerank_results: dict[int, tuple] = {}

    def _rerank_one(item):
        jid, _sim = item
        try:
            with get_session() as s:
                job = s.get(Job, jid)
                if not job:
                    return jid, None
                return jid, reranker.score(resume, job)
        except Exception as e:
            log.warning("Parallel rerank failed for job %s: %s", jid, e)
            return jid, None

    # LLM gate: only the top-N candidates reach the expensive LLM, ranked by
    # source/freshness-weighted cross-encoder priority — a fresh direct-ATS
    # posting beats an equal-scoring stale aggregator redirect for the budget.
    # The rest keep their cheap-filter scores and can be promoted later.
    if len(to_rerank) > settings.llm_rerank_cap:
        log.info(
            "LLM gate: %d candidates survived cheap filters — capping to top %d by weighted priority",
            len(to_rerank), settings.llm_rerank_cap,
        )
        to_rerank.sort(key=lambda t: _rerank_priority.get(t[0], t[1]), reverse=True)
        to_rerank = to_rerank[: settings.llm_rerank_cap]

    if to_rerank:
        with ThreadPoolExecutor(max_workers=settings.llm_rerank_workers) as ex:
            for jid, res in ex.map(_rerank_one, to_rerank):
                if res is not None:
                    rerank_results[jid] = res

    # ── Phase 3: serial store + shortlist creation (caps/cooldown/limits) ─────
    for jid, sim in to_rerank:
        res = rerank_results.get(jid)
        if res is None:
            continue
        score, reason, concerns, breakdown = res
        new_app_id: int | None = None
        with get_session() as session:
            job = session.get(Job, jid)
            if not job:
                continue
            job.rerank_score = score
            job.rerank_reasoning = reason + (
                ("\nConcerns: " + "; ".join(concerns)) if concerns else ""
            )
            job.rerank_breakdown = _json.dumps(breakdown) if breakdown else None

            # Hire Probability Scoring — no LLM, uses DB + description signals
            hp_result = score_hire_probability(job, session)
            job.hire_probability_score = hp_result.score
            job.hire_probability_signals = _json.dumps(hp_result.signals)
            job.blended_score = compute_blended(score, hp_result.score)
            session.add(job)

            # If rerank ≥ threshold, create an Application row in SHORTLISTED state
            if score >= settings.shortlist_score_threshold:
                existing = session.exec(
                    select(Application).where(Application.job_id == job.id)
                ).first()
                if not existing:
                    if today_count < settings.daily_shortlist_limit:
                        # ── Company cap + cooldown ──
                        if not _check_and_enforce_company_cap(session, job, score):
                            session.commit()
                            continue

                        # ── Link liveness — non-ATS links only (direct boards are
                        # covered by mark_ghost_jobs at scrape time). A dead link
                        # must not consume a shortlist slot or the user's time.
                        if (getattr(settings, "verify_links_on_shortlist", True)
                                and job.source not in _AUTOFILL_SOURCES):
                            from app.discovery.verify import check_job_alive
                            alive, dead_reason = check_job_alive(job.url)
                            if not alive:
                                job.is_closed = True
                                job.closed_reason = f"Deactivated ({dead_reason})"
                                session.add(job)
                                session.commit()
                                log.info("Job %s @ %s dead at shortlist time: %s",
                                         job.title, job.company, dead_reason)
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

                        # Create notification for high-fit matched jobs
                        if score >= 75:
                            try:
                                from app.db.models import UserNotification
                                notif = UserNotification(
                                    user_id=user_id,
                                    title="Perfect Job Match! 🎯",
                                    message=f"{job.title} at {job.company} matches your profile with a score of {int(score)}%.",
                                    type="high_match",
                                    link="/dashboard",
                                )
                                session.add(notif)
                            except Exception as ne:
                                log.warning("Failed to create high match notification: %s", ne)
                    else:
                        log.info("Daily shortlist limit (%d) reached — skipping application creation for job %s.", settings.daily_shortlist_limit, job.title)

            session.commit()
            log.info("Job %s @ %s: sim=%.3f rerank=%.0f — %s",
                     job.title, job.company, sim, score, reason)

        # NOTE: SeniorReviewer is NOT run here anymore — it was a second serial
        # LLM call per shortlisted job, doubling matching time and cost. It now
        # runs on demand when the user opens a job (see /application/{id}/senior-review).
    return shortlisted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    new_ids = run_matching()
    print(f"Shortlisted {len(new_ids)} new applications.")
