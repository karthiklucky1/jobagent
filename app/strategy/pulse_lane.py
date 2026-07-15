"""Pulse lane — the freshness guarantee.

Replaces the hot lane's rotating fixed-size batches with a per-board schedule
(``CompanyRegistry.next_poll_at``) that enforces two promises:

  * FAST (minutes): boards for companies any user follows ("My Companies")
    and boards that posted a new job in the last ``pulse_active_days`` days are
    polled every ``pulse_fast_interval_minutes``.
  * FLOOR (within the hour): every other LIVE board is polled at least every
    ``pulse_floor_interval_minutes`` — no company can go stale for hours just
    because it hasn't posted lately.
  * Dead weight (404s / boards that have never held a job) decays to a daily
    retry so the budget goes to boards that can actually produce.

A board that posts anything auto-promotes to FAST (``last_new_job_at`` moves),
so a "random" company only ever pays the floor price once.

Cheap-by-default: each poll computes a signature of the board's posting list
(ids + titles). Unchanged board → zero downstream work (no upserts, no per-user
routing). Only changed boards touch the DB. Description-only edits don't move
the signature — the fresh/full lanes' content-hash upserts still catch those.

Brand-new jobs take a PER-JOB FAST PATH: role match → cheap-tier prescore →
Claude score → shortlist (daily limit + company cap respected) → fresh alert —
no waiting for the next batch matching tick, and no discovery lock (the fast
path never loads FAISS or the embedding model).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeout
from datetime import datetime, timedelta
from typing import Optional

from sqlmodel import select

from app.config import settings
from app.db.init_db import get_session
from app.db.models import (
    Application, ApplicationStatus, CompanyRegistry, FunnelEvent, Job, UserProfile,
)

log = logging.getLogger(__name__)


def _norm(s: str) -> str:
    """Alphanumeric-only lowercase form for fuzzy company/slug comparison."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _watchlist_terms() -> set[str]:
    """Union of every user's followed companies, normalized. One shared fast
    lane serves all tenants (scrape once, serve many)."""
    terms: set[str] = set()
    try:
        with get_session() as session:
            rows = session.exec(select(UserProfile.target_companies)).all()
        for tc in rows:
            for part in (tc or "").split(","):
                n = _norm(part)
                if n:
                    terms.add(n)
    except Exception as e:
        log.debug("pulse: watchlist load failed: %s", e)
    return terms


def _is_watched(row: CompanyRegistry, terms: set[str]) -> bool:
    if not terms:
        return False
    slug_n = _norm(row.slug)
    name_n = _norm(row.company_name or "")
    for t in terms:
        if t and (t == slug_n or t == name_n
                  or (len(t) >= 4 and (t in slug_n or t in name_n))):
            return True
    return False


def _cadence(row: CompanyRegistry, terms: set[str], now: datetime) -> timedelta:
    """How long until this board's next poll, from its signals."""
    if _is_watched(row, terms):
        return timedelta(minutes=settings.pulse_fast_interval_minutes)
    if row.last_new_job_at and \
            (now - row.last_new_job_at) <= timedelta(days=settings.pulse_active_days):
        return timedelta(minutes=settings.pulse_fast_interval_minutes)
    if (row.job_count or 0) > 0:
        return timedelta(minutes=settings.pulse_floor_interval_minutes)
    # Never held a job (or 404s en route to retirement) — daily retry.
    return timedelta(hours=settings.pulse_dead_interval_hours)


def _due_boards(now: datetime, limit: int) -> list[CompanyRegistry]:
    """Boards whose next_poll_at has arrived (never-scheduled boards first, then
    least-recently-polled), capped so a backlog stretches the floor honestly
    instead of stampeding the container."""
    with get_session() as session:
        return session.exec(
            select(CompanyRegistry)
            .where(CompanyRegistry.is_active == True,  # noqa: E712
                   (CompanyRegistry.next_poll_at == None)  # noqa: E711
                   | (CompanyRegistry.next_poll_at <= now))
            .order_by(CompanyRegistry.next_poll_at.asc().nulls_first(),
                      CompanyRegistry.last_seen.asc().nulls_first())
            .limit(limit)
        ).all()


def _board_signature(raw: list) -> str:
    """Signature of a board's posting list: which jobs exist (id + title)."""
    keys = sorted(f"{r.external_id}|{(r.title or '')[:80]}" for r in raw)
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def _set_schedule(slug: str, ats, next_at: datetime, poll_hash: Optional[str]) -> None:
    try:
        with get_session() as session:
            row = session.exec(
                select(CompanyRegistry).where(
                    CompanyRegistry.slug == slug, CompanyRegistry.ats == ats)
            ).first()
            if not row:
                return
            row.next_poll_at = next_at
            if poll_hash is not None:
                row.poll_hash = poll_hash
            session.add(row)
            session.commit()
    except Exception as e:
        log.debug("pulse: schedule update failed for %s: %s", slug, e)


# ── Per-job fast path ─────────────────────────────────────────────────────────

def _fast_path_user(uid: str, score_budget: int,
                    deadline: Optional[float] = None) -> tuple[int, int, int]:
    """Score this user's brand-new unscored jobs RIGHT NOW (bounded), shortlist
    the fits, and fire fresh alerts. Returns (scored, shortlisted, alerts).
    Stops early once ``deadline`` (monotonic) passes so it can't overrun the tick.

    Deliberately lock-free: rule filter + ghost check + LLM cascade only — no
    FAISS/embedding model. Anything left unscored (budget, errors) is picked up
    by the 5-min matching lane, so this can only make things faster, never drop
    a job."""
    from app.matching.pipeline import (
        _AUTOFILL_SOURCES, _check_and_enforce_company_cap, _load_resume,
    )
    from app.matching.reranker import Reranker
    from app.matching.filters import score_ghost
    from app.matching.hire_probability import (
        blended_score as compute_blended, score_hire_probability,
    )
    from app.strategy.fresh_alerts import dispatch_fresh_alerts

    uid_arg = None if (not uid or uid == "local") else uid
    try:
        resume = _load_resume(user_id=uid_arg)
    except Exception as e:
        log.debug("pulse fast-path: no resume for %s (%s) — skipping", uid, e)
        return 0, 0, 0

    profile = None
    try:
        from app.autofill.answer_pack import _get_or_create_profile
        profile = _get_or_create_profile(user_id=uid_arg)
    except Exception:
        pass
    reranker = Reranker(profile=profile)

    cutoff = datetime.utcnow() - timedelta(minutes=15)
    posted_cut = datetime.utcnow() - timedelta(hours=48)
    roles = [r.strip().lower()
             for r in (getattr(profile, "target_roles", "") or "").split(",") if r.strip()]
    with get_session() as session:
        # Pull a WIDER slice of fresh postings than we can score, so we can pick
        # the on-role ones. Board dumps (e.g. Rippling serving a whole company's
        # departments) flood the newest-first list with off-target titles
        # (Mechanical Engineer, Contact Center Analyst); scoring newest-first
        # could spend the whole LLM budget on those while a genuinely-fresh
        # AI/ML match waits for the slower matching lane.
        rows = session.exec(
            select(Job.id, Job.title).where(
                Job.user_id == uid_arg,
                Job.rerank_score == None,  # noqa: E711
                Job.is_closed == False,  # noqa: E712
                Job.first_seen >= cutoff,
                # Fast-path LLM spend is reserved for genuinely fresh postings.
                # Jobs first-seen now but POSTED long ago (e.g. the scheduler
                # bootstrap adopting weeks of backlog) wait for the regular
                # matching lane — they can't produce a valid fresh alert anyway.
                (Job.posted_at == None) | (Job.posted_at >= posted_cut),  # noqa: E711
            ).order_by(Job.first_seen.desc()).limit(max(score_budget * 6, 60))
        ).all()
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        q = select(Application).where(Application.created_at >= today_start)
        q = q.where(Application.user_id == uid_arg) if uid_arg \
            else q.where(Application.user_id.is_(None))
        today_count = len(session.exec(q).all())

    # Relevance-first: score titles matching the user's target roles before the
    # off-role remainder (still newest-first within each group). Off-role fresh
    # jobs aren't dropped — the matching lane scores them next pass.
    if roles:
        from app.discovery.title_filter import role_title_match
        on_role = [jid for jid, t in rows if role_title_match(t or "", roles)]
        on_set = set(on_role)
        off_role = [jid for jid, t in rows if jid not in on_set]
        fresh_ids = (on_role + off_role)[:score_budget]
    else:
        fresh_ids = [jid for jid, _t in rows][:score_budget]
    if not fresh_ids:
        return 0, 0, 0

    use_prescore = settings.prescore_enabled and reranker.has_prescore_backend()
    gate = min(settings.prescore_advance_threshold, settings.shortlist_score_threshold)
    scored = 0
    shortlisted: list[int] = []
    for jid in fresh_ids:
        if deadline is not None and time.monotonic() >= deadline:
            break  # out of tick budget — the matching lane scores the rest
        with get_session() as session:
            job = session.get(Job, jid)
            if not job or job.rerank_score is not None or job.is_closed:
                continue
            # Ghost gate (cheap, DB+text) before any LLM spend.
            try:
                g = score_ghost(job, session)
                job.ghost_score = g.ghost_score
                job.ghost_flags = g.flags_json
                if g.is_ghost:
                    job.rerank_score = 5.0
                    job.rerank_reasoning = f"Ghost filtered (score={g.ghost_score:.2f}): {', '.join(g.flags)}"
                    session.add(job)
                    session.commit()
                    continue
            except Exception as e:
                log.debug("pulse fast-path ghost check failed for %d: %s", jid, e)

            # Cascade Tier-1: drain clear misfits without touching Claude.
            if use_prescore:
                pre = reranker.prescore(resume, job)
                if pre is not None and pre[0] < gate:
                    job.rerank_score = float(pre[0])
                    job.rerank_reasoning = f"Pre-screened (Tier-1 fit {int(pre[0])}): {pre[1]}"[:500]
                    session.add(job)
                    session.commit()
                    scored += 1
                    continue

            # Tier-2: authoritative score (includes the rule pre-filter).
            try:
                score, reason, concerns, breakdown = reranker.score(resume, job)
            except Exception as e:
                log.debug("pulse fast-path score failed for %d (left for matching lane): %s", jid, e)
                session.rollback()
                continue
            scored += 1
            job.rerank_score = score
            job.rerank_reasoning = reason + (("\nConcerns: " + "; ".join(concerns)) if concerns else "")
            job.rerank_breakdown = json.dumps(breakdown) if breakdown else None
            try:
                hp = score_hire_probability(job, session)
                job.hire_probability_score = hp.score
                job.hire_probability_signals = json.dumps(hp.signals)
                job.blended_score = compute_blended(score, hp.score)
            except Exception:
                pass
            session.add(job)

            if score >= settings.shortlist_score_threshold \
                    and today_count < settings.daily_shortlist_limit:
                existing = session.exec(
                    select(Application).where(Application.job_id == job.id)
                ).first()
                if not existing and _check_and_enforce_company_cap(session, job, score):
                    track = "autofill" if job.source in _AUTOFILL_SOURCES else "manual"
                    session.add(Application(
                        job_id=job.id, status=ApplicationStatus.SHORTLISTED,
                        apply_url=job.url, apply_track=track, user_id=uid_arg,
                    ))
                    shortlisted.append(job.id)
                    today_count += 1
            session.commit()

    alerts = 0
    if shortlisted:
        try:
            alerts = dispatch_fresh_alerts(uid, shortlisted)
        except Exception as e:
            log.warning("pulse fast-path alerts failed for %s: %s", uid, e)
    return scored, len(shortlisted), alerts


# ── One scheduler tick ────────────────────────────────────────────────────────

# One tick at a time — but SELF-HEALING. A plain lock froze the whole lane once
# a single slow tick (serial LLM scoring for 20+ min) held it. Now: a tick is
# hard-bounded by pulse_tick_max_seconds (it stops taking new work and releases
# promptly), and if a holder ever overruns a generous grace window it's treated
# as dead and the next tick proceeds anyway — the lane can never freeze forever.
_TICK_LOCK = threading.Lock()
_TICK_DEADLINE = [0.0]  # monotonic time by which the current holder must be done


def run_pulse_tick() -> dict:
    """Poll every board that's due, route changes, fast-path new jobs to alerts.
    Returns tick stats; records a ``pulse_tick`` FunnelEvent when work was done."""
    now_m = time.monotonic()
    got = _TICK_LOCK.acquire(blocking=False)
    if not got:
        if now_m < _TICK_DEADLINE[0]:
            log.info("Pulse tick skipped — previous tick still running")
            return {"boards": 0, "skipped": "previous tick still running"}
        # Holder blew past its grace window → hung/abandoned thread. Proceed
        # without the lock so the lane recovers instead of freezing forever.
        log.warning("Pulse tick: prior tick overran its grace window — proceeding")
    work_deadline = now_m + settings.pulse_tick_max_seconds
    # Grace: work deadline + a cushion for in-flight ops to drain before a steal.
    _TICK_DEADLINE[0] = work_deadline + 90
    try:
        return _run_pulse_tick_locked(work_deadline)
    finally:
        if got:
            _TICK_LOCK.release()


def _run_pulse_tick_locked(deadline: float) -> dict:
    from app.discovery.pipeline import SHARED_POOL_USER, _upsert, scraper_for
    from app.strategy.hot_lane import (
        _active_users, _mark_polled, _retire_unsupported, _title_matches,
    )

    now = datetime.utcnow()
    boards = _due_boards(now, settings.pulse_max_boards_per_tick)
    stats = {"boards": len(boards), "changed": 0, "fetched_jobs": 0,
             "new_jobs": 0, "scored": 0, "shortlisted": 0, "alerts": 0}
    # RESERVE ~40% of the budget for SCORING. During the bootstrap backlog the
    # fetch/route phase would otherwise eat the whole tick (deferring hundreds of
    # boards) and score ZERO — so fresh jobs land but sit "Queued" until the
    # slower matching lane reaches them. Stopping fetch early leaves guaranteed
    # time for the fast path to score the freshest on-role jobs each tick. When
    # fetch finishes early (steady state) the fast path just gets more time.
    fetch_deadline = deadline - max(0.0, settings.pulse_tick_max_seconds * 0.4)
    if not boards:
        return stats

    users = _active_users()
    terms = _watchlist_terms()
    all_roles = sorted({r for u in users for r in (u["roles"] or [])})

    def _fetch(board):
        scraper = scraper_for(board.ats, board.slug, board.career_url)
        if scraper is None:
            return board, None, "unsupported"
        try:
            return board, scraper.fetch(), None
        except Exception as e:
            return board, None, str(e)

    users_touched: set[str] = set()
    pool = ThreadPoolExecutor(max_workers=min(settings.pulse_fetch_workers,
                                              max(1, len(boards))))
    # Submit all fetches, then collect each with a deadline-bounded wait. A board
    # that hasn't returned by the tick's wall-clock deadline is rescheduled (it's
    # simply due again) rather than blocked on — so one slow board/host can never
    # stall the tick. shutdown(wait=False) means stragglers finish on their own.
    futures = {pool.submit(_fetch, b): b for b in boards}
    deferred = 0
    for fut in list(futures):
        remaining = fetch_deadline - time.monotonic()
        board = futures[fut]
        if remaining <= 0:
            deferred += 1
            _set_schedule(board.slug, board.ats,
                          datetime.utcnow() + _cadence(board, terms, now), None)
            continue
        try:
            board, raw, err = fut.result(timeout=remaining)
        except _FutureTimeout:
            deferred += 1
            _set_schedule(board.slug, board.ats,
                          datetime.utcnow() + _cadence(board, terms, now), None)
            continue
        except Exception as e:
            _mark_polled(board.slug, board.ats, job_count=None, ok=False, error=str(e))
            _set_schedule(board.slug, board.ats,
                          datetime.utcnow() + _cadence(board, terms, now), None)
            continue
        if raw is None:
            if err == "unsupported":
                _retire_unsupported(board.slug, board.ats)
            else:
                _mark_polled(board.slug, board.ats, job_count=None, ok=False, error=err)
                _set_schedule(board.slug, board.ats,
                              datetime.utcnow() + _cadence(board, terms, now), None)
            continue

        stats["fetched_jobs"] += len(raw)
        sig = _board_signature(raw)
        if raw and sig == (board.poll_hash or ""):
            # Unchanged board — zero downstream work. This is the common case
            # that makes the hourly floor affordable.
            _mark_polled(board.slug, board.ats, job_count=len(raw), ok=True)
            _set_schedule(board.slug, board.ats,
                          datetime.utcnow() + _cadence(board, terms, now), sig)
            continue

        stats["changed"] += 1
        new_here = 0
        try:
            new_here += _upsert(raw, user_id=SHARED_POOL_USER, user_keywords=all_roles or None)
        except Exception as e:
            log.debug("pulse shared upsert failed %s: %s", board.slug, e)
        for u in users:
            relevant = [r for r in raw if _title_matches(r.title, u["roles"])]
            if not relevant:
                continue
            try:
                n = _upsert(relevant, user_id=(None if u["user_id"] == "local" else u["user_id"]))
                if n:
                    users_touched.add(u["user_id"])
                    new_here += n
            except Exception as e:
                log.debug("pulse user upsert failed %s/%s: %s", board.slug, u["user_id"], e)
        stats["new_jobs"] += new_here
        _mark_polled(board.slug, board.ats, job_count=len(raw), ok=True, new_jobs=new_here)
        # Re-read cadence AFTER _mark_polled: a board that just posted has a
        # fresh last_new_job_at, so it lands on the fast lane immediately.
        board.last_new_job_at = datetime.utcnow() if new_here else board.last_new_job_at
        _set_schedule(board.slug, board.ats,
                      datetime.utcnow() + _cadence(board, terms, now), sig)
    pool.shutdown(wait=False)
    if deferred:
        stats["deferred"] = deferred

    # Per-job fast path for every user who just received something new — best
    # effort within whatever wall-clock time the tick has left. Anything not
    # scored here is picked up by the 5-min matching lane, so this only speeds
    # alerts, never blocks the tick (which used to run serial LLM for 20+ min).
    budget = settings.pulse_fast_path_score_cap
    for uid in users_touched:
        if budget <= 0 or time.monotonic() >= deadline:
            break
        try:
            scored, short, alerts = _fast_path_user(uid, budget, deadline)
            budget -= scored
            stats["scored"] += scored
            stats["shortlisted"] += short
            stats["alerts"] += alerts
        except Exception as e:
            log.warning("pulse fast-path failed for %s: %s", uid, e)

    try:
        with get_session() as session:
            session.add(FunnelEvent(
                job_id=None, stage="pulse_tick", passed=True,
                reason=f"boards={stats['boards']} new={stats['new_jobs']}",
                metadata_json=json.dumps(stats),
            ))
            session.commit()
    except Exception as e:
        log.debug("pulse tick event write failed: %s", e)
    log.info("Pulse tick: %s", stats)
    return stats
