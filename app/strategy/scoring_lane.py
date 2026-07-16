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

# ── Per-job attempt ceiling ───────────────────────────────────────────────────
# A job whose final score keeps failing (provider outage, poison payload) used
# to be re-selected EVERY 90s cycle forever — re-paying the prescore each time.
# After scoring_fail_max_attempts failures the job is deferred in memory for
# scoring_fail_defer_hours instead. Process-local by design: no schema change,
# and a restart merely retries a few times before deferring again.
_fail_counts: dict = {}
_deferred_until: dict = {}
_fail_lock = threading.Lock()


def _note_score_failure(jid: int) -> None:
    with _fail_lock:
        n = _fail_counts.get(jid, 0) + 1
        if n >= max(1, settings.scoring_fail_max_attempts):
            _fail_counts.pop(jid, None)
            _deferred_until[jid] = time.time() + settings.scoring_fail_defer_hours * 3600
            log.warning("Scoring: job %d failed %d attempts — deferred %.1fh",
                        jid, n, settings.scoring_fail_defer_hours)
        else:
            _fail_counts[jid] = n


def _note_score_success(jid: int) -> None:
    with _fail_lock:
        _fail_counts.pop(jid, None)
        _deferred_until.pop(jid, None)


def _drop_deferred(jids: List[int]) -> List[int]:
    now = time.time()
    with _fail_lock:
        # purge expired deferrals so the dict stays small
        for j, until in list(_deferred_until.items()):
            if until <= now:
                _deferred_until.pop(j, None)
        return [j for j in jids if j not in _deferred_until]


def _deferred_ids() -> set:
    """Currently-deferred job ids (expired deferrals purged first)."""
    now = time.time()
    with _fail_lock:
        for j, until in list(_deferred_until.items()):
            if until <= now:
                _deferred_until.pop(j, None)
        return set(_deferred_until)


def _transient_llm_stall() -> bool:
    """True when NO job-specific work can succeed right now — the hourly/daily
    final budget is exhausted or every provider is in circuit-breaker cooldown.
    Failures under this condition are not the job's fault and must not count
    against its attempt ceiling."""
    from app.matching.reranker import any_provider_available, llm_budget_exhausted
    return llm_budget_exhausted() or not any_provider_available()


def _scorable_user_ids(limit: int = 1000) -> List[Optional[str]]:
    """Distinct owners that currently have at least one unscored open job.
    The shared pool ('__shared__') is a corpus, not a user — its rows are never
    scored directly (they're adopted into per-user pools first), so it must not
    consume work slots."""
    from app.discovery.pipeline import SHARED_POOL_USER
    with get_session() as session:
        rows = session.exec(
            select(Job.user_id).where(
                Job.rerank_score == None,  # noqa: E711
                Job.is_closed == False,    # noqa: E712
            ).distinct().limit(limit)
        ).all()
    users = [r[0] if isinstance(r, tuple) else r for r in rows]
    return [u for u in users if u != SHARED_POOL_USER]


def _user_queue(user_id: Optional[str], cap: int) -> List[int]:
    """A user's queued (unscored) job ids, freshest first, capped. Attempt-ceiling
    deferred jobs are excluded IN THE QUERY (not after the LIMIT) so a window of
    deferred fresh jobs can't crowd valid older jobs out of the capped freshest-
    first slice and starve them indefinitely."""
    deferred = _deferred_ids()
    with get_session() as session:
        q = select(Job.id).where(
            Job.user_id == user_id,
            Job.rerank_score == None,  # noqa: E711
            Job.is_closed == False,    # noqa: E712
        )
        # Exclude deferred ids in-SQL for the common (small) set; fall back to
        # post-filtering only if the deferred set is pathologically large.
        if deferred and len(deferred) <= 2000:
            q = q.where(Job.id.notin_(deferred))
        q = q.order_by(Job.first_seen.desc()).limit(cap)
        jids = [r[0] if isinstance(r, tuple) else r for r in session.exec(q).all()]
    return jids if len(deferred) <= 2000 else _drop_deferred(jids)


class _Ctx:
    __slots__ = ("resume", "reranker", "use_prescore", "gate")

    def __init__(self, resume, reranker, use_prescore, gate):
        self.resume = resume
        self.reranker = reranker
        self.use_prescore = use_prescore
        self.gate = gate


def _pick_provider(jid: int, ctx: "_Ctx") -> Optional[str]:
    """Option A: split the FINAL score across providers by job id — a stable,
    balanced ~claude_share/rest partition. Both providers score against the same
    rubric, so throughput becomes Claude's rate limit + GPT's, not just Claude's.
    Returns None (default priority order) when dual mode is off or only one
    provider is configured."""
    if not settings.dual_score_enabled or not ctx.reranker.has_dual():
        return None
    share = max(0.0, min(1.0, settings.dual_score_claude_share))
    return "anthropic" if (jid % 100) < int(round(share * 100)) else "openai"


def _stamp_job(jid: int, ghost: Optional[Tuple],
               score: float, reasoning: str,
               extras: Optional[Tuple] = None) -> bool:
    """Short write-back session. Re-checks idempotency (another lane may have
    scored the job while we were on the LLM). Returns False if it lost the race."""
    with get_session() as session:
        job = session.get(Job, jid)
        if not job or job.rerank_score is not None or job.is_closed:
            return False
        if ghost is not None:
            job.ghost_score, job.ghost_flags = ghost
        job.rerank_score = score
        job.rerank_reasoning = reasoning
        if extras is not None:
            breakdown, hp_fn = extras
            job.rerank_breakdown = breakdown
            try:
                hp_fn(job, session)
            except Exception:
                pass
        session.add(job)
        session.commit()
    return True


# Tier-1 results for jobs whose FINAL score failed: the retry (next cycle, or
# after a deferral) reuses the prescore instead of paying the cheap model again.
# Entries are dropped the moment the job is scored/drained; the safety clear
# only guards against a pathological pile-up.
_prescore_memo: dict = {}


def _score_job(jid: int, ctx: _Ctx) -> Optional[Tuple[str, int, Optional[float], Optional[str]]]:
    """Score one queued job. Returns ("scored"|"drained", jid, score, provider)
    or None. ``provider`` is which backend produced the final score (for the
    dual-provider split stats); None for drained/ghost/single-provider jobs.
    Idempotent: a job already scored (by another worker / lane) is skipped, and
    the cross-lane in-flight claim stops two lanes paying to score the same job
    at the same time.

    DB DISCIPLINE: a connection is held only for the short read/write phases —
    NEVER across an LLM call. The old shape kept one session open for the whole
    function, so 20 workers × multi-second LLM latency pinned 20 connections and
    starved the pool for everything else (funnel/registry/web all timed out with
    "QueuePool limit reached")."""
    from app.common.inflight import claim
    with claim(jid) as ok:
        if not ok:
            return None  # another lane is scoring this job right now
        return _score_job_owned(jid, ctx)


def _score_job_owned(jid: int, ctx: _Ctx) -> Optional[Tuple[str, int, Optional[float], Optional[str]]]:
    from app.matching.filters import score_ghost
    from app.matching.hire_probability import (
        blended_score as compute_blended, score_hire_probability,
    )

    # Phase 1 — short session: load + idempotency + ghost gate (cheap, DB+text).
    ghost: Optional[Tuple] = None
    with get_session() as session:
        job = session.get(Job, jid)
        if not job or job.rerank_score is not None or job.is_closed:
            return None
        try:
            g = score_ghost(job, session)
            ghost = (g.ghost_score, g.flags_json)
            if g.is_ghost:
                job.ghost_score, job.ghost_flags = ghost
                job.rerank_score = 5.0
                job.rerank_reasoning = f"Ghost filtered (score={g.ghost_score:.2f}): {', '.join(g.flags)}"
                session.add(job)
                session.commit()
                return ("drained", jid, None, None)
        except Exception as e:
            log.debug("scoring ghost check failed for %d: %s", jid, e)
        # Detach with its loaded attributes — the LLM phase reads job fields
        # only, so it must not keep the session (and its connection) alive.
        session.expunge(job)

    # Phase 2 — NO session held: the slow LLM calls.
    # Cascade Tier-1: drain clear misfits without touching the final scorer.
    # A memoized prescore (from an attempt whose FINAL call failed) is reused so
    # retries only re-pay the step that actually failed.
    if ctx.use_prescore:
        pre = _prescore_memo.get(jid)
        if pre is None:
            pre = ctx.reranker.prescore(ctx.resume, job)
        if pre is not None and pre[0] < ctx.gate:
            reasoning = f"Pre-screened (Tier-1 fit {int(pre[0])}): {pre[1]}"[:500]
            if _stamp_job(jid, ghost, float(pre[0]), reasoning):
                _prescore_memo.pop(jid, None)
                return ("drained", jid, None, None)
            return None  # lost the race to another lane
    else:
        pre = None

    # Tier-2: authoritative score (dual routing when enabled; the rule
    # pre-filter runs inside .score()).
    provider = _pick_provider(jid, ctx)
    try:
        score, reason, concerns, breakdown = ctx.reranker.score(ctx.resume, job, provider=provider)
    except Exception as e:
        log.debug("scoring failed for %d (left for next cycle): %s", jid, e)
        if pre is not None:
            if len(_prescore_memo) > 10000:  # pathological pile-up guard
                _prescore_memo.clear()
            _prescore_memo[jid] = pre  # retry skips Tier-1
        # Transient, non-job-specific failures — the hourly/daily budget cap
        # tripping mid-cycle (the LLM_HOURLY_FINAL_CAP smoother firing) or every
        # provider being in circuit-breaker cooldown — must NOT count against
        # this job's attempt ceiling. Otherwise perfectly scorable fresh jobs
        # get deferred for scoring_fail_defer_hours purely because the budget
        # guard fired. Leave them Queued for the next eligible cycle, unpenalized.
        if _transient_llm_stall():
            return None
        _note_score_failure(jid)  # real per-job failure: attempt ceiling applies
        return None
    _note_score_success(jid)
    _prescore_memo.pop(jid, None)

    # Phase 3 — short session: idempotent write-back + hire-probability blend.
    def _hp(job, session):
        hp = score_hire_probability(job, session)
        job.hire_probability_score = hp.score
        job.hire_probability_signals = json.dumps(hp.signals)
        job.blended_score = compute_blended(score, hp.score)

    reasoning = reason + (("\nConcerns: " + "; ".join(concerns)) if concerns else "")
    extras = (json.dumps(breakdown) if breakdown else None, _hp)
    if not _stamp_job(jid, ghost, score, reasoning, extras):
        return None  # another lane scored it while we were on the LLM

    # Distillation shadow mode: run the local model beside this fresh LLM final
    # and record agreement (best-effort, ~50ms CPU, zero user-facing effect).
    try:
        from app.matching.local_scorer import shadow_score
        shadow_score(jid, ctx.resume, job, float(score))
    except Exception:
        pass
    return ("scored", jid, float(score), provider)


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
    from app.matching.reranker import (
        Reranker, any_provider_available, llm_budget_exhausted,
    )
    stats = {"users": 0, "queued": 0, "scored": 0, "drained": 0,
             "shortlisted": 0, "alerts": 0, "by_claude": 0, "by_gpt": 0}

    # Fast-exit guards: when every provider is cooling down (credit/quota) or
    # the daily spend cap is hit, a cycle would only burn CPU and log noise —
    # jobs stay Queued and the next eligible cycle picks them up.
    if not any_provider_available():
        return {**stats, "skipped": "all LLM providers cooling down"}
    if llm_budget_exhausted():
        return {**stats, "skipped": "LLM budget reached (hourly/daily cap)"}

    users = _scorable_user_ids()
    if not users:
        return stats

    # Build the global work list, freshest-first per user, ROUND-ROBIN across
    # users. Interleaving (vs. the old user-by-user fill) does two things:
    # 1. fairness — when the global cap bites, every user gets a share of the
    #    cycle instead of the first users taking all 600 slots;
    # 2. cache efficiency — same-user finals share a cached résumé prefix, and a
    #    cache entry is only readable after the first call finishes. Interleaved
    #    items keep the 20 workers on DIFFERENT users' prefixes, so job #2 for a
    #    user usually finds the cache its job #1 just wrote.
    # Every user's queue is fetched (tiny indexed id-only selects) BEFORE the
    # cap is applied — stopping the fetch at the cap would hand all slots to
    # whichever users happened to come first, which is the unfairness this
    # rewrite removes.
    queues: List[List[Tuple[Optional[str], int]]] = []
    for uid in users:
        q = [(uid, jid) for jid in _user_queue(uid, settings.scoring_per_user_cap)]
        if q:
            queues.append(q)
    items: List[Tuple[Optional[str], int]] = []
    depth = 0
    while len(items) < settings.scoring_global_cap and any(depth < len(q) for q in queues):
        for q in queues:
            if depth < len(q):
                items.append(q[depth])
                if len(items) >= settings.scoring_global_cap:
                    break
        depth += 1
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
        uid, (kind, jid, score, provider) = out
        if kind == "scored":
            stats["scored"] += 1
            scored_by_user[uid].append((jid, score))
            if provider == "anthropic":
                stats["by_claude"] += 1
            elif provider == "openai":
                stats["by_gpt"] += 1
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
