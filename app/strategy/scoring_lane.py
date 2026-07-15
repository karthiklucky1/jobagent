"""Scoring lane — decoupled, parallel, cross-user job scoring.

The freshness lanes (pulse / fresh / full discovery) PRODUCE unscored "Queued"
jobs; this lane CONSUMES them. It drains the GLOBAL queue of unscored on-role
jobs across ALL users at once with a bounded pool of LLM workers, so scoring
throughput is bounded by the LLM rate limit — NOT by the number of users.

The old matching lane scored users one-at-a-time (``for uid in users:
run_matching(uid)``) — O(users), ~130s/user — so with 9 users a full cycle took
~20 min and fresh jobs sat "Queued". This lane is the fix: worker count is
independent of user count. 9 users or 10,000, the scorer runs the same 20
workers flat-out; a longer queue just means adding workers (or providers).

Design (producer/consumer):

    discovery lanes ──▶ [ Job.rerank_score IS NULL ] ──▶ scoring lane
       (produce Queued)      the DB IS the queue        (consume, in parallel)

  Phase A — SCORE (parallel, I/O-bound on the LLM): a bounded pool scores work
    items (user, job) — cheap ghost gate → GPT prescore (drain misfits) → Claude
    final. Each user's résumé/reranker is loaded ONCE and shared (thread-safe
    cache). Lock-free: no FAISS / embedding rebuild, so it runs continuously
    alongside discovery.
  Phase B — SHORTLIST (serial per user, cap-safe): jobs that cleared the bar
    become SHORTLISTED applications under the daily limit + per-company cap, then
    fresh alerts fire. Serial-per-user so the caps can't race.

Anything not reached in a cycle stays Queued and is drained next cycle (or by the
5-min matching lane's FAISS backstop) — so this only speeds scoring, never drops
a job.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeout
from datetime import datetime
from typing import List, Optional, Tuple

from sqlmodel import select

from app.config import settings
from app.db.init_db import get_session
from app.db.models import Application, ApplicationStatus, FunnelEvent, Job

log = logging.getLogger(__name__)

_LANE_LOCK = threading.Lock()  # one scoring cycle at a time in this process


def _scorable_user_ids(limit: int = 1000) -> List[Optional[str]]:
    """Distinct owners that currently have at least one unscored open job."""
    with get_session() as session:
        rows = session.exec(
            select(Job.user_id).where(
                Job.rerank_score == None,  # noqa: E711
                Job.is_closed == False,    # noqa: E712
            ).distinct().limit(limit)
        ).all()
    return [r[0] if isinstance(r, tuple) else r for r in rows]


def _user_queue(user_id: Optional[str], cap: int) -> List[int]:
    """A user's queued (unscored) job ids, freshest first, capped."""
    with get_session() as session:
        return [r[0] if isinstance(r, tuple) else r for r in session.exec(
            select(Job.id).where(
                Job.user_id == user_id,
                Job.rerank_score == None,  # noqa: E711
                Job.is_closed == False,    # noqa: E712
            ).order_by(Job.first_seen.desc()).limit(cap)
        ).all()]


class _Ctx:
    __slots__ = ("resume", "reranker", "use_prescore", "gate")

    def __init__(self, resume, reranker, use_prescore, gate):
        self.resume = resume
        self.reranker = reranker
        self.use_prescore = use_prescore
        self.gate = gate


def _score_job(jid: int, ctx: _Ctx) -> Optional[Tuple[str, int, Optional[float]]]:
    """Score one queued job. Returns ("scored"|"drained", jid, score) or None.
    Idempotent: a job already scored (by another worker / lane) is skipped."""
    from app.matching.filters import score_ghost
    from app.matching.hire_probability import (
        blended_score as compute_blended, score_hire_probability,
    )
    with get_session() as session:
        job = session.get(Job, jid)
        if not job or job.rerank_score is not None or job.is_closed:
            return None
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
                return ("drained", jid, None)
        except Exception as e:
            log.debug("scoring ghost check failed for %d: %s", jid, e)

        # Cascade Tier-1: drain clear misfits without touching Claude.
        if ctx.use_prescore:
            pre = ctx.reranker.prescore(ctx.resume, job)
            if pre is not None and pre[0] < ctx.gate:
                job.rerank_score = float(pre[0])
                job.rerank_reasoning = f"Pre-screened (Tier-1 fit {int(pre[0])}): {pre[1]}"[:500]
                session.add(job)
                session.commit()
                return ("drained", jid, None)

        # Tier-2: authoritative score (the rule pre-filter runs inside .score()).
        try:
            score, reason, concerns, breakdown = ctx.reranker.score(ctx.resume, job)
        except Exception as e:
            log.debug("scoring failed for %d (left for next cycle): %s", jid, e)
            session.rollback()
            return None
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
        session.commit()
        return ("scored", jid, float(score))


def _shortlist_user(uid, scored: List[Tuple[int, float]], stats: dict) -> None:
    """Serial, cap-safe: shortlist a user's freshly-scored fits + fire alerts."""
    from app.matching.pipeline import _AUTOFILL_SOURCES, _check_and_enforce_company_cap
    from app.strategy.fresh_alerts import dispatch_fresh_alerts
    uid_arg = None if (not uid or uid == "local") else uid

    with get_session() as session:
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        q = select(Application).where(Application.created_at >= today_start)
        q = q.where(Application.user_id == uid_arg) if uid_arg \
            else q.where(Application.user_id.is_(None))
        today_count = len(session.exec(q).all())

    shortlisted: List[int] = []
    for jid, score in sorted(scored, key=lambda x: -x[1]):  # best first
        if score < settings.shortlist_score_threshold:
            continue
        if today_count >= settings.daily_shortlist_limit:
            break
        with get_session() as session:
            job = session.get(Job, jid)
            if not job:
                continue
            if session.exec(select(Application).where(Application.job_id == jid)).first():
                continue
            if not _check_and_enforce_company_cap(session, job, score):
                session.commit()
                continue
            track = "autofill" if job.source in _AUTOFILL_SOURCES else "manual"
            session.add(Application(
                job_id=jid, status=ApplicationStatus.SHORTLISTED,
                apply_url=job.url, apply_track=track, user_id=uid_arg,
            ))
            session.commit()
            shortlisted.append(jid)
            today_count += 1
    stats["shortlisted"] += len(shortlisted)
    if shortlisted:
        try:
            stats["alerts"] += dispatch_fresh_alerts(uid, shortlisted)
        except Exception as e:
            log.warning("scoring lane alerts failed for %s: %s", uid, e)


def run_scoring_lane(deadline: Optional[float] = None) -> dict:
    """One scoring cycle: drain the global unscored queue in parallel, then
    shortlist + alert. Returns cycle stats. Skips if a cycle is already running."""
    if not _LANE_LOCK.acquire(blocking=False):
        return {"skipped": "cycle already running"}
    try:
        return _run_scoring_cycle(deadline)
    finally:
        _LANE_LOCK.release()


def _run_scoring_cycle(deadline: Optional[float]) -> dict:
    from app.matching.pipeline import _load_resume
    from app.matching.reranker import Reranker
    stats = {"users": 0, "queued": 0, "scored": 0, "drained": 0,
             "shortlisted": 0, "alerts": 0}

    users = _scorable_user_ids()
    if not users:
        return stats

    # Build the global work list: (uid, jid), freshest-first per user, capped.
    items: List[Tuple[Optional[str], int]] = []
    for uid in users:
        for jid in _user_queue(uid, settings.scoring_per_user_cap):
            items.append((uid, jid))
        if len(items) >= settings.scoring_global_cap:
            break
    items = items[: settings.scoring_global_cap]
    stats["users"] = len({u for u, _ in items})
    stats["queued"] = len(items)
    if not items:
        return stats

    # Per-user context (résumé + reranker), loaded ONCE and shared across workers.
    ctx_cache: dict = {}
    ctx_lock = threading.Lock()

    def _ctx_for(uid) -> Optional[_Ctx]:
        with ctx_lock:
            if uid in ctx_cache:
                return ctx_cache[uid]
        uid_arg = None if (not uid or uid == "local") else uid
        try:
            resume = _load_resume(user_id=uid_arg)
        except Exception as e:
            log.debug("scoring: no résumé for %s (%s) — skipping", uid, e)
            with ctx_lock:
                ctx_cache[uid] = None
            return None
        profile = None
        try:
            from app.autofill.answer_pack import _get_or_create_profile
            profile = _get_or_create_profile(user_id=uid_arg)
        except Exception:
            pass
        reranker = Reranker(profile=profile)
        ctx = _Ctx(resume, reranker,
                   settings.prescore_enabled and reranker.has_prescore_backend(),
                   min(settings.prescore_advance_threshold, settings.shortlist_score_threshold))
        with ctx_lock:
            ctx_cache[uid] = ctx
        return ctx

    def _work(item):
        uid, jid = item
        ctx = _ctx_for(uid)
        if ctx is None:
            return None
        res = _score_job(jid, ctx)
        return (uid, res) if res else None

    scored_by_user: dict = defaultdict(list)
    pool = ThreadPoolExecutor(max_workers=min(settings.scoring_workers, max(1, len(items))))
    futures = [pool.submit(_work, it) for it in items]
    for fut in futures:
        remaining = (deadline - time.monotonic()) if deadline else None
        if remaining is not None and remaining <= 0:
            break  # out of budget — the rest stay Queued for the next cycle
        try:
            out = fut.result(timeout=remaining if remaining else None)
        except _FutureTimeout:
            break
        except Exception:
            continue
        if not out:
            continue
        uid, (kind, jid, score) = out
        if kind == "scored":
            stats["scored"] += 1
            scored_by_user[uid].append((jid, score))
        elif kind == "drained":
            stats["drained"] += 1
    pool.shutdown(wait=False)

    # Phase B — shortlist + alert, serial per user (cap-safe).
    for uid, results in scored_by_user.items():
        try:
            _shortlist_user(uid, results, stats)
        except Exception as e:
            log.warning("scoring lane shortlist failed for %s: %s", uid, e)

    try:
        with get_session() as session:
            session.add(FunnelEvent(
                job_id=None, stage="scoring_cycle", passed=True,
                reason=f"users={stats['users']} scored={stats['scored']}",
                metadata_json=json.dumps(stats),
            ))
            session.commit()
    except Exception as e:
        log.debug("scoring cycle event write failed: %s", e)
    log.info("Scoring cycle: %s", stats)
    return stats
