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

from sqlalchemy.orm import load_only
from sqlmodel import select

from app.config import settings
from app.db.init_db import get_session
from app.db.models import Application, ApplicationStatus, Job, JobSource
from app.matching.matcher import Matcher
from app.matching.reranker import Reranker
from app.matching.filters import RuleFilter, EmbeddingFilter, score_ghost
from app.matching.hire_probability import score_hire_probability, blended_score as compute_blended
from app.matching.fresh_budget import freshness_tier, order_fresh_first, order_fit_first
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


def _displace_weaker_shortlisted(session, job: Job, score: float, active) -> bool:
    """Free ONE company-cap slot by evicting the weakest cap-holder that is
    still just SHORTLISTED and loses to the new job by at least the configured
    margin. Only ever displaces untouched shortlist entries — anything the user
    or agent has acted on (TAILORED and beyond) keeps its slot. Returns True
    when a slot was freed. The displaced job keeps its (SKIPPED) application
    row, so the re-shortlist backstop can never bounce it straight back."""
    from app.config import settings as _cap_settings
    margin = max(0, _cap_settings.company_cap_displace_margin)
    candidates = [a for a in active if a.status == ApplicationStatus.SHORTLISTED]
    if not candidates:
        return False

    def _holder_score(a: Application) -> float:
        j = session.get(Job, a.job_id)
        return float(j.rerank_score) if (j and j.rerank_score is not None) else 0.0

    weakest = min(candidates, key=_holder_score)
    old_score = _holder_score(weakest)
    if float(score) < old_score + margin:
        return False
    weakest.status = ApplicationStatus.SKIPPED
    weakest.notes = (weakest.notes or "") + (
        f"\nDisplaced by higher-scoring '{job.title}' "
        f"({score:.0f} vs {old_score:.0f}) under the per-company cap."
    )
    session.add(weakest)
    log.info("Company cap: displaced shortlisted app %s at %s (score %.0f) for "
             "stronger '%s' (%.0f)", weakest.id, job.company, old_score, job.title, float(score))
    return True


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
      • DISPLACEMENT: when still at the cap, a new job that outscores the weakest
        cap-holder which is merely SHORTLISTED (untouched — no tailoring, no
        submission) by ``company_cap_displace_margin`` points evicts it (→
        SKIPPED). A better role at the same company should not sit at "Reviewed"
        for 40 days behind a weaker one nobody acted on. Applications with any
        invested effort (TAILORED and beyond) are NEVER displaced.

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
    if not expired and _cap_settings.company_cap_displace_enabled:
        displaced = _displace_weaker_shortlisted(session, job, score, active)
        if displaced:
            return True
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


def _load_verified_achievements(user_id: str | None) -> str:
    """Candidate-confirmed metrics captured via the résumé metric-gap flow. Stored
    as an AnswerMemory row and appended to the résumé so tailoring can use the real
    numbers AND grounding accepts them (they're now part of the source of truth)."""
    try:
        from app.db.models import AnswerMemory
        _uid = user_id if user_id and user_id != "local" else None
        with get_session() as session:
            row = session.exec(
                select(AnswerMemory).where(
                    AnswerMemory.user_id == _uid,
                    AnswerMemory.label_normalized == "__verified_achievements",
                )
            ).first()
        if row and (row.answer or "").strip():
            lines = [ln.strip() for ln in row.answer.split("\n") if ln.strip()]
            if lines:
                return ("\n\n## Additional Verified Achievements (candidate-confirmed)\n"
                        + "\n".join(f"- {ln}" for ln in lines))
    except Exception as e:
        log.debug("verified achievements load skipped: %s", e)
    return ""


def _load_resume(user_id: str | None = None) -> str:
    """Load resume — checks Supabase Storage per user first, falls back to local file."""
    text = _load_resume_file(user_id)
    extra = _load_verified_achievements(user_id)
    return text + extra if extra else text


def _load_resume_file(user_id: str | None = None) -> str:
    """The raw résumé text (Supabase Storage per user, else local file)."""
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
            app = session.get(Application, app_id)
            if not app:
                return
            result = reviewer.review(job, user_id=app.user_id)
            if result is None:
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
    refusals (NO_SPONSORSHIP_HARD) are left untouched.

    Guarded by a persisted per-user marker so it runs ONCE, not on every matching
    pass: the offending rule is already fixed, so no NEW job gets this stamp — but
    the unguarded full-row LIKE scan re-ran every pass, needless per-pass DB work
    and egress. Non-fatal."""
    import json as _json
    from app.db.models import FunnelEvent
    from app.matching.filters.constants import WORK_AUTH_BOILERPLATE
    cleared = 0
    try:
        with get_session() as session:
            already = session.exec(
                select(FunnelEvent.id).where(
                    FunnelEvent.stage == "stale_sponsorship_reset",
                    FunnelEvent.reason == f"user={user_id}",
                ).limit(1)
            ).first()
            if already:
                return 0
            q = select(Job).options(load_only(Job.id, Job.rerank_reasoning)).where(
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
            # Persist the marker even when nothing cleared, so a clean pool
            # doesn't re-scan on every pass.
            session.add(FunnelEvent(
                job_id=None, stage="stale_sponsorship_reset", passed=True,
                reason=f"user={user_id}",
                metadata_json=_json.dumps({"cleared": cleared}),
            ))
            session.commit()
    except Exception as e:
        log.warning("Stale sponsorship-score reset failed (non-fatal): %s", e)
    if cleared:
        log.info("Cleared %d stale boilerplate sponsorship scores for re-ranking", cleared)
    return cleared


def _reset_incident_frozen_scores(user_id: str | None) -> int:
    """One-time self-heal for the Supabase-incident scoring backlog.

    While the DB was timing out, the cheap gates stamped RECENT jobs with reject
    scores against a degraded / mid-rebuild FAISS index:
        "Rule filtered: …" (10) · "Ghost filtered …" (5) ·
        "Embedding filtered: …" (15) · "Wrong Door: …" (20)
    A stamped score removes a job from the ``only_unscored`` retrieval corpus for
    good, so those jobs are frozen out of scoring even after the DB recovers —
    which is why fresh jobs never reach the shortlist.

    Clear the cheap-gate stamps on jobs seen in the last few days so they re-enter
    scoring under the healthy index. Guarded by a persisted per-user marker
    (a ``frozen_score_reset`` FunnelEvent) so container restarts don't re-churn:
    genuinely low-fit jobs simply get re-stamped once and stay out. Non-fatal."""
    import json as _json
    from app.db.models import FunnelEvent
    _REJECT_PREFIXES = ("Embedding filtered:%", "Ghost filtered%",
                        "Rule filtered:%", "Wrong Door:%")
    cleared = 0
    try:
        with get_session() as session:
            already = session.exec(
                select(FunnelEvent.id).where(
                    FunnelEvent.stage == "frozen_score_reset",
                    FunnelEvent.reason == f"user={user_id}",
                ).limit(1)
            ).first()
            if already:
                return 0
            cutoff = datetime.utcnow() - timedelta(days=3)
            reason_col = Job.rerank_reasoning
            q = select(Job).where(
                Job.user_id == user_id,
                Job.rerank_score.isnot(None),  # type: ignore[union-attr]
                (reason_col.like(_REJECT_PREFIXES[0])  # type: ignore[union-attr]
                 | reason_col.like(_REJECT_PREFIXES[1])
                 | reason_col.like(_REJECT_PREFIXES[2])
                 | reason_col.like(_REJECT_PREFIXES[3])),
                (Job.first_seen >= cutoff) | (Job.posted_at >= cutoff),
            )
            for job in session.exec(q).all():
                job.rerank_score = None
                job.rerank_reasoning = None
                session.add(job)
                cleared += 1
            # Persist the marker even when nothing cleared, so a healthy pool
            # doesn't re-scan every matching run.
            session.add(FunnelEvent(
                job_id=None, stage="frozen_score_reset", passed=True,
                reason=f"user={user_id}",
                metadata_json=_json.dumps({"cleared": cleared}),
            ))
            session.commit()
    except Exception as e:
        log.warning("Incident frozen-score reset failed (non-fatal): %s", e)
        return 0
    if cleared:
        log.info("Reset %d incident-frozen cheap-gate scores for re-ranking (user=%s)",
                 cleared, user_id)
    return cleared


def _reshortlist_scored_jobs(user_id: str | None, today_count: int) -> tuple[List[int], int]:
    """(Re)shortlist jobs that are ALREADY LLM-scored above the threshold but
    have no Application yet (e.g. the daily limit was hit when they were scored,
    or their application was cleaned up). Direct DB query — these jobs used to
    be re-shortlisted only if they happened to win a retrieval slot, which also
    let them crowd fresh jobs out of the cross-encoder budget."""
    shortlisted: List[int] = []
    with get_session() as session:
        # Any existing application blocks a re-shortlist, regardless of the
        # application row's owner (legacy rows can carry a NULL user_id while
        # the job row was adopted by a tenant) — matches the old behavior of
        # checking Application.job_id directly.
        applied_rows = session.exec(
            select(Application.job_id).join(Job, Application.job_id == Job.id)
            .where(Job.user_id == user_id)
        ).all()
        applied_ids = {r[0] if isinstance(r, tuple) else r for r in applied_rows}
        applied_ids.discard(None)
        q = (
            select(Job)
            # Load only the columns this loop actually uses — deferring the job
            # description and the big JSON blobs (rerank_reasoning/breakdown/
            # hire_probability_signals) keeps this per-pass, up-to-500-row query
            # from streaming megabytes out of Postgres every matching pass.
            .options(load_only(
                Job.id, Job.url, Job.source, Job.company, Job.title,
                Job.rerank_score, Job.user_id,
            ))
            .where(
                Job.is_closed == False,  # noqa: E712
                Job.user_id == user_id,
                Job.rerank_score >= settings.shortlist_score_threshold,
            )
            .order_by(Job.rerank_score.desc())
            .limit(500)
        )
        for job in session.exec(q).all():
            if job.id in applied_ids:
                continue
            if today_count >= settings.daily_shortlist_limit:
                log.info("Daily shortlist limit reached — stopping re-shortlist of already-scored jobs.")
                break
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
            session.commit()
            shortlisted.append(job.id)
            today_count += 1
            log.info("Job '%s' @ '%s' already scored (%d) — %s track. Shortlisted.",
                     job.title, job.company, job.rerank_score, _track)
    return shortlisted, today_count


def _persist_prescore_rejects(rejects: list[tuple[int, float, str]]) -> None:
    """Stamp Tier-1-rejected jobs with their prescore + reason so they leave the
    unscored corpus (backlog drain). Uses a chunked bulk UPDATE (no per-row read,
    so this does not add to egress) — Supabase cancels a single UPDATE over
    thousands of rows via statement_timeout, so commit in small batches. The
    scores are below the shortlist threshold by construction, so the re-shortlist
    query never picks these up."""
    if not rejects:
        return
    mappings = [
        {"id": int(jid),
         "rerank_score": float(score),
         "rerank_reasoning": f"Pre-screened (Tier-1 fit {int(score)}): {reason}"[:500]}
        for jid, score, reason in rejects
    ]
    for start in range(0, len(mappings), 500):
        batch = mappings[start:start + 500]
        try:
            with get_session() as session:
                session.bulk_update_mappings(Job, batch)
                session.commit()
        except Exception as e:
            log.warning("Prescore-reject stamp batch %d failed (non-fatal): %s", start, e)
    log.info("Cascade Tier-1: stamped %d drained job(s)", len(mappings))


def run_matching(user_id: str | None = None) -> List[int]:
    resume = _load_resume(user_id=user_id)
    # Give jobs unfairly blocked by the old sponsorship boilerplate rule a
    # fresh scoring pass under the fixed logic.
    _reset_stale_sponsorship_scores(user_id)
    # One-time: un-freeze jobs the cheap gates stamped during the Supabase
    # incident (degraded index) so fresh postings can reach the shortlist.
    _reset_incident_frozen_scores(user_id)
    matcher = Matcher(user_id=user_id)
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

    # Retrieval runs over UNSCORED jobs only (newest first): scored jobs never
    # need another retrieval pass, and letting them compete starved fresh
    # postings out of the cross-encoder budget — the "no new jobs" bug.
    candidates = matcher.search_for_resume(resume, k=settings.top_k_rerank, user_id=user_id,
                                           profile=_user_profile, only_unscored=True)
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

    # Interaction learning: revealed preferences (dismissals/applies) calibrate
    # the LLM scoring and gate shortlisting of repeatedly-dismissed companies.
    _pref = None
    _feedback = ""
    try:
        from app.matching.preference_learning import build_preference_profile
        _pref = build_preference_profile(user_id)
        _feedback = _pref.feedback_note()
        if not _pref.has_signal:
            _pref = None
    except Exception as _fe:
        log.debug("preference learning unavailable: %s", _fe)

    reranker = Reranker(profile=_user_profile, feedback=_feedback)
    shortlisted: List[int] = []

    # Count applications already created today (scoped to user)
    with get_session() as session:
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        q = select(Application).where(Application.created_at >= today_start)
        if user_id:
            q = q.where(Application.user_id == user_id)
        today_count = len(session.exec(q).all())

    # (Re)shortlist already-scored jobs via a direct query — they no longer
    # pass through retrieval at all.
    _re_ids, today_count = _reshortlist_scored_jobs(user_id, today_count)
    shortlisted.extend(_re_ids)

    import json as _json
    from concurrent.futures import ThreadPoolExecutor

    # ── Phase 1: cheap serial pre-filtering ──────────────────────────────────
    # Run the non-LLM gates (already-scored / rule / ghost / embedding). Jobs that
    # survive all gates are collected for parallel LLM scoring in Phase 2. The
    # already-scored shortlist path still runs here so behavior is unchanged.
    to_rerank: list[tuple[int, float]] = []
    _rerank_priority: dict[int, float] = {}  # jid -> source/freshness-weighted CE score
    _rerank_tier: dict[int, int] = {}        # jid -> freshness tier (0 = freshest)
    for jid, sim in candidates:
        with get_session() as session:
            job = session.get(Job, jid)
            if not job:
                continue
            if job.is_closed:
                continue

            # Safety: retrieval is unscored-only, but guard against a job that
            # got scored between retrieval and now (re-shortlisting of scored
            # jobs is handled by _reshortlist_scored_jobs above).
            if job.rerank_score is not None:
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
            _rerank_tier[jid] = freshness_tier(job)

    # ── Phase 2: parallel LLM scoring (I/O-bound on the model API) ────────────
    # Each worker uses its own DB session + the shared (thread-safe) reranker
    # client, so the slow network calls overlap instead of running one-by-one.
    rerank_results: dict[int, tuple] = {}

    def _rerank_one(item):
        jid, _sim = item
        from app.common.inflight import claim
        try:
            with claim(jid) as ok:
                if not ok:
                    return jid, None  # another lane is scoring it right now
                with get_session() as s:
                    job = s.get(Job, jid)
                    if not job:
                        return jid, None
                    if job.rerank_score is not None:
                        # Another lane (90s scoring lane / pulse fast path) scored
                        # it since this work list was built — don't pay twice.
                        return jid, None
                    return jid, reranker.score(resume, job)
        except Exception as e:
            log.warning("Parallel rerank failed for job %s: %s", jid, e)
            return jid, None

    # Order candidates FRESH-FIRST (by freshness tier, newest postings first,
    # breaking ties within a tier by source/freshness-weighted cross-encoder
    # priority) so that when any cap bites, brand-new postings win the scarce
    # scoring slots. This is the freshness fix: previously the cap went to the
    # highest-similarity survivors, so stale-but-similar jobs crowded brand-new
    # postings out and the feed went stale.
    # NOTE: this orders WHICH jobs get scored, not the shortlist-creation order —
    # Phase 3 re-sorts by resulting fit so freshness never steals a scarce
    # daily/company-cap slot from a stronger match.
    to_rerank = order_fresh_first(to_rerank, _rerank_tier, _rerank_priority)

    # ── Cascade Tier-1: cheap bulk prescore ──────────────────────────────────
    # A cheap/fast model scores up to prescore_cap candidates. Only those whose
    # rough fit clears the advance gate reach Tier-2 (Claude — the authoritative
    # score that drives shortlisting); the clear misfits are stamped with their
    # prescore below, so they LEAVE the unscored corpus instead of being re-read
    # and re-considered every pass. Net effect: we look at ~prescore_cap jobs per
    # pass (not just llm_rerank_cap), the backlog actually drains (far less
    # repeated full-row egress), and fresh on-role jobs reach Claude sooner.
    # The effective gate is clamped to the shortlist threshold so any job that
    # could plausibly shortlist always reaches Claude. Fail-open: a None prescore
    # (cheap-model error) advances the job rather than dropping it.
    prescore_rejects: list[tuple[int, float, str]] = []  # (jid, score, reason)
    if (settings.prescore_enabled and to_rerank
            and reranker.has_prescore_backend()):
        advance_gate = min(settings.prescore_advance_threshold,
                           settings.shortlist_score_threshold)
        prescore_pool = to_rerank[: settings.prescore_cap]
        log.info("Cascade Tier-1: prescoring %d candidate(s) (gate=%d)",
                 len(prescore_pool), advance_gate)

        def _prescore_one(item):
            jid, _sim = item
            try:
                with get_session() as s:
                    job = s.get(Job, jid)
                    if not job:
                        return jid, None
                    return jid, reranker.prescore(resume, job)
            except Exception as e:
                log.debug("Prescore worker failed for job %s: %s", jid, e)
                return jid, None

        with ThreadPoolExecutor(max_workers=settings.prescore_workers) as ex:
            _pre_by_jid = dict(ex.map(_prescore_one, prescore_pool))

        advanced: list[tuple[int, float]] = []
        for jid, sim in prescore_pool:  # preserve fresh-first order
            pr = _pre_by_jid.get(jid)
            if pr is None:              # cheap-model failure → fail-open to Claude
                advanced.append((jid, sim))
            elif pr[0] >= advance_gate:
                advanced.append((jid, sim))
            else:
                prescore_rejects.append((jid, pr[0], pr[1]))
        log.info("Cascade Tier-1: %d advanced to Claude, %d drained (below gate)",
                 len(advanced), len(prescore_rejects))
        to_rerank = advanced

    # ── Cascade Tier-2: Claude (authoritative) score, capped at llm_rerank_cap ──
    if len(to_rerank) > settings.llm_rerank_cap:
        log.info(
            "LLM gate: %d candidates for Claude — capping to top %d (fresh-first)",
            len(to_rerank), settings.llm_rerank_cap,
        )
        to_rerank = to_rerank[: settings.llm_rerank_cap]

    if to_rerank:
        with ThreadPoolExecutor(max_workers=settings.llm_rerank_workers) as ex:
            for jid, res in ex.map(_rerank_one, to_rerank):
                if res is not None:
                    rerank_results[jid] = res

    # Stamp Tier-1 rejects so they exit the unscored corpus (backlog drain). Their
    # score is below the shortlist threshold by construction, so the re-shortlist
    # query will not pick them up.
    _persist_prescore_rejects(prescore_rejects)

    # ── Phase 3: serial store + shortlist creation (caps/cooldown/limits) ─────
    # Shortlist-creation order is FIT-FIRST: among the freshly-scored cohort, the
    # best-fit roles claim the scarce daily-limit + per-company-cap slots first.
    # (Iterating fresh-first here would let a marginal newer role take a company's
    # last cap slot and block a stronger same-company role for the cooldown
    # window, or spend the last daily slots on marginal jobs — so we sort by the
    # LLM score just produced. Unscored jobs sort last.)
    _scores = {jid: res[0] for jid, res in rerank_results.items() if res is not None}
    _phase3_order = order_fit_first(to_rerank, _scores)
    _liveness_checks = 0  # bound the serial link-liveness network calls per pass (lock-held)
    for jid, sim in _phase3_order:
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
                # Learning gate: the user repeatedly ✕-dismissed this company —
                # keep the score but don't shortlist another of their roles.
                if _pref is not None and (job.company or "").strip().lower() in _pref.disliked_companies:
                    log.info("Preference gate: user keeps dismissing %s — not shortlisting '%s'.",
                             job.company, job.title)
                    session.commit()
                    continue
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
                        # BOUNDED: each check is a serial ~2.5s network call made
                        # WHILE this pass holds the matching lock, so an unbounded
                        # count (up to daily_shortlist_limit) could hold the lock
                        # ~10+ min and starve every other lane. Cap the checks per
                        # pass; beyond the cap, shortlist without verifying (the
                        # link is re-checked when the user opens the job).
                        if (getattr(settings, "verify_links_on_shortlist", True)
                                and job.source not in _AUTOFILL_SOURCES
                                and _liveness_checks < settings.max_liveness_checks_per_run):
                            from app.discovery.verify import check_job_alive
                            _liveness_checks += 1
                            alive, dead_reason = check_job_alive(job.url, timeout=2.5)
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
