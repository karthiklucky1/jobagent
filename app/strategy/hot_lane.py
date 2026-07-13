"""Hot lane — poll the boards that matter to active users every few minutes so
brand-new postings reach shortlists (and fresh alerts) within minutes, not the
days a full-registry rotation takes.

Architecture — **fetch once, match many** (the cost-efficient model):

    Naive per-user discovery fetches the SAME ATS board once per user, so HTTP
    cost is O(boards × users). The hot lane fetches each board ONCE per cycle
    (shared), then routes each posting only to the users whose target roles it
    matches — HTTP cost is O(boards), independent of user count. Matching and
    per-user job rows stay per-user (scores are personal), but those are cheap,
    local operations; the expensive, rate-limit-bearing part (the network fetch)
    is done a single time.

Board selection is skills-aware at two levels:
  - Board level: we poll active boards that actually produce jobs, least-recently
    polled first, capped at ``hot_lane_max_boards`` — a rotating hot set.
  - Job level: each fetched posting is distributed only to users whose target
    roles appear in its title (cheap keyword routing), so a user's pool only
    grows with jobs relevant to their skills.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlmodel import select

from app.config import settings
from app.db.init_db import get_session
from app.db.models import CompanyRegistry, UserProfile

log = logging.getLogger(__name__)


def _active_users() -> list[dict]:
    """Users with a resume + target roles — {user_id, roles:[lowercased]}."""
    from app.api.server import _user_has_resume, _get_target_roles
    out = []
    with get_session() as session:
        profiles = session.exec(select(UserProfile)).all()
    for p in profiles:
        uid = p.user_id
        if not uid or not _user_has_resume(uid):
            continue
        roles = [r.lower() for r in (_get_target_roles(uid) or [])]
        out.append({"user_id": uid, "roles": roles})
    return out


def select_hot_boards(limit: int) -> list[CompanyRegistry]:
    """The rotating hot set, prioritized by proven YIELD so each cycle spends its
    budget on boards that actually post — not on dead never-polled slugs.

    Order of claim on the per-cycle budget:
      1. Proven yielders — boards that have produced a NEW posting before
         (``last_new_job_at`` set) — rotated stalest-first so all get swept.
      2. Other productive boards (``job_count > 0``) — stalest-first.
      3. A small bootstrap slice of never-polled boards
         (``hot_lane_bootstrap_frac`` of the budget, default 20%) so freshly
         seeded companies are still discovered — WITHOUT letting tens of
         thousands of dead seeded slugs eat half the cycle. The old 50/50 split
         did exactly that: every cycle burned 200 slots on never-polled slugs
         that mostly 404'd (the '404 storm'), starving boards that actually post.
      4. Backfill from any remaining active boards, stalest-first, if thin (few
         productive boards yet, or a bootstrap-heavy fresh registry).

    Dead boards are retired on the first 404 (see ``_mark_polled``), so the
    never-polled backlog drains as it is swept."""
    frac = min(max(float(getattr(settings, "hot_lane_bootstrap_frac", 0.2) or 0.0), 0.0), 1.0)
    bootstrap = int(limit * frac)
    main = limit - bootstrap

    boards: list[CompanyRegistry] = []
    seen: set = set()

    def _take(rows, cap: int) -> None:
        """Append rows until ``len(boards)`` reaches ``cap`` (skipping dupes)."""
        for b in rows:
            if len(boards) >= cap:
                break
            if b.id not in seen:
                boards.append(b)
                seen.add(b.id)

    with get_session() as session:
        # 1. Proven yielders first — boards that have ever produced a NEW posting.
        #    Capped at 'main' so the bootstrap slice is reserved.
        _take(session.exec(
            select(CompanyRegistry)
            .where(CompanyRegistry.is_active == True,  # noqa: E712
                   CompanyRegistry.last_new_job_at != None)  # noqa: E711
            .order_by(CompanyRegistry.last_seen.asc().nulls_first())
            .limit(main)
        ).all(), main)

        # 2. Other productive boards (known to hold jobs), stalest-first, filling
        #    the rest of the 'main' (non-bootstrap) budget.
        if len(boards) < main:
            _take(session.exec(
                select(CompanyRegistry)
                .where(CompanyRegistry.is_active == True,  # noqa: E712
                       CompanyRegistry.job_count > 0,
                       CompanyRegistry.last_seen != None)  # noqa: E711
                .order_by(CompanyRegistry.last_seen.asc())
                .limit(main)
            ).all(), main)

        # 3. Small bootstrap slice: never-polled boards, so new companies are
        #    still discovered — but capped so they can't dominate the cycle.
        #    Ordered (oldest-registered first) so drainage is deterministic
        #    across SQLite/Postgres and the accumulated seed backlog is swept
        #    methodically instead of in unspecified DB heap order.
        if bootstrap > 0 and len(boards) < limit:
            _take(session.exec(
                select(CompanyRegistry)
                .where(CompanyRegistry.is_active == True,  # noqa: E712
                       CompanyRegistry.last_seen == None)  # noqa: E711
                .order_by(CompanyRegistry.id.asc())
                .limit(bootstrap)
            ).all(), min(limit, len(boards) + bootstrap))

        # 4. Backfill if still short (thin productive set / bootstrap-heavy start).
        if len(boards) < limit:
            _take(session.exec(
                select(CompanyRegistry)
                .where(CompanyRegistry.is_active == True)  # noqa: E712
                .order_by(CompanyRegistry.last_seen.asc().nulls_first())
                .limit(limit)
            ).all(), limit)

    return boards[:limit]


def _users_with_pending_fresh(users: list[dict], hours: int = 24) -> set:
    """Active users who have UNSCORED postings first-seen in the last ``hours``.

    These are fresh jobs waiting to be scored — typically inserted during a
    cycle whose matching phase was skipped (discovery lock busy). Including them
    in the match set means the next unlocked cycle scores their fresh jobs even
    if THIS cycle routed nothing new to them, closing the 'inserted but never
    scored until the 2h fresh lane' gap. Cheap: one indexed LIMIT-1 existence
    check per active user (a handful of users). Self-bounding — once a user's
    fresh jobs are scored, rerank_score is set and they drop out of the set."""
    from datetime import timedelta
    from app.db.models import Job
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    pending: set = set()
    with get_session() as session:
        for u in users:
            uid = u["user_id"]
            row = session.exec(
                select(Job.id).where(
                    Job.user_id == uid,
                    Job.rerank_score == None,  # noqa: E711
                    Job.is_closed == False,  # noqa: E712
                    Job.first_seen >= cutoff,
                ).limit(1)
            ).first()
            if row is not None:
                pending.add(uid)
    return pending


def _title_matches(title: str, roles: list[str]) -> bool:
    """Skills-aware routing: alias- and token-based (see role_title_match), so
    'Senior ML Engineer' reaches a 'Machine Learning Engineer' user. The old
    exact-substring check dropped most relevant fresh postings at fetch time.
    Empty roles → accept (user hasn't narrowed), matching current discovery."""
    from app.discovery.title_filter import role_title_match
    return role_title_match(title, roles)


def run_hot_lane() -> dict:
    """Run one hot-lane cycle with a GUARANTEED heartbeat.

    The dashboard's 'Hot Lane: idle' tile is driven by ``hot_lane_run`` events.
    The cycle body only reached ``_finish_cycle`` on its success/early-exit
    paths, so if it raised anywhere in between (a bad board fetch, an upsert, a
    matching error, or even ``_active_users``/``select_hot_boards``) NO event
    was written and the lane looked permanently idle even though it was firing
    every interval and crashing. This wrapper records a heartbeat with the error
    reason on ANY failure, turning a silent 'idle' into a visible, diagnosable
    error while keeping the loop alive for the next cycle."""
    try:
        return _run_hot_lane_cycle()
    except Exception as e:  # never let a cycle die without leaving a trace
        log.exception("Hot lane cycle crashed: %s", e)
        return _finish_cycle({
            "boards": 0, "users": 0, "fetched_jobs": 0, "inserted_jobs": 0,
            "alerts": 0, "reason": f"error: {type(e).__name__}: {e}"[:280],
        })


def _run_hot_lane_cycle() -> dict:
    """One hot-lane cycle. Returns a small stats dict for logging/telemetry.

    Phase A — fetch + distribute (NO lock): board polling is lightweight
    (concurrent HTTP + slim DB upserts; no embedding model, no job-pool load),
    so it runs even while a full/fresh discovery pass holds the discovery
    lock. Previously the WHOLE cycle skipped whenever the lock was busy, and
    multi-hour full runs starved the lane indefinitely — 'Hot Lane: idle',
    'New Jobs (24h): 0'.

    Phase B — match + alert (lock, skip-if-busy): matching loads the model +
    job pool, so it still serializes (OOM guard). When the lock is busy the
    freshly inserted jobs simply wait for the next matching pass.

    A heartbeat is recorded for EVERY cycle (including early exits) with a
    reason, so the dashboard tile shows why instead of a silent 'idle'."""
    from app.discovery.pipeline import scraper_for, _upsert
    from app.common.discovery_lock import discovery_guard

    limit = int(getattr(settings, "hot_lane_max_boards", 400))
    users = _active_users()
    if not users:
        return _finish_cycle({"boards": 0, "users": 0, "reason": "no active users"})

    boards = select_hot_boards(limit)
    if not boards:
        return _finish_cycle({"boards": 0, "users": len(users),
                              "reason": "no active boards"})

    now = datetime.utcnow()
    fetched_jobs = 0
    matched_jobs = 0    # postings that routed to at least one user's roles
    inserted_jobs = 0   # NEW rows actually written (post-dedupe), all users
    shared_inserted = 0  # NEW rows written to the shared pool (scrape once)
    users_touched: set = set()
    # Union of every user's roles: lets department users' titles survive the
    # non-tech gate when writing to the SHARED pool, which serves everyone.
    _all_roles: list = sorted({r for u in users for r in (u["roles"] or [])})

    # Fetch boards CONCURRENTLY — the fetches are pure I/O, and doing 400 of
    # them sequentially held the global discovery lock for the whole sweep,
    # blocking the fresh lane, the 6h scheduler, and the manual Discover
    # button for many minutes at a time. DB writes stay serial below.
    from concurrent.futures import ThreadPoolExecutor as _Pool

    def _fetch_board(board):
        scraper = scraper_for(board.ats, board.slug, board.career_url)
        if scraper is None:
            return board, None, "unsupported"
        try:
            return board, scraper.fetch(), None  # ONE network call, shared across all users
        except Exception as e:
            return board, None, str(e)

    with _Pool(max_workers=min(12, max(1, len(boards)))) as _pool:
        fetch_results = list(_pool.map(_fetch_board, boards))

    for board, raw, err in fetch_results:
        if raw is None:
            if err == "unsupported":
                # No scraper for this ATS (e.g. iCIMS/Jobvite/Comeet/Wellfound) —
                # it can NEVER produce jobs, yet stays is_active with last_seen
                # NULL, so it re-clogged the never-polled bootstrap slice every
                # cycle. Retire it so the slice goes to boards that can post.
                _retire_unsupported(board.slug, board.ats)
            else:
                log.debug("hot lane fetch failed %s/%s: %s", board.ats, board.slug, err)
                _mark_polled(board.slug, board.ats, job_count=None, ok=False, error=err)
            continue
        fetched_jobs += len(raw)
        if not raw:
            _mark_polled(board.slug, board.ats, job_count=0, ok=True)
            continue

        # Scrape once: persist EVERY fetched posting to the shared pool, so
        # future users (and role edits) can adopt them without re-scraping.
        try:
            from app.discovery.pipeline import SHARED_POOL_USER
            shared_inserted += _upsert(raw, user_id=SHARED_POOL_USER,
                                       user_keywords=_all_roles)
        except Exception as e:
            log.debug("hot lane shared-pool upsert failed: %s", e)

        # Distribute each posting only to users whose target roles it matches.
        routed_ids: set = set()
        board_new = 0
        for u in users:
            relevant = [r for r in raw if _title_matches(r.title, u["roles"])]
            if not relevant:
                continue
            routed_ids.update(r.external_id for r in relevant)
            try:
                new = _upsert(relevant, user_id=u["user_id"])
                inserted_jobs += new
                board_new += new
                if new:
                    users_touched.add(u["user_id"])
            except Exception as e:
                log.debug("hot lane upsert failed for %s: %s", u["user_id"], e)
        matched_jobs += len(routed_ids)
        _mark_polled(board.slug, board.ats, job_count=len(raw), ok=True,
                     new_jobs=board_new)

    # Phase B: match + alert. Needs the discovery lock (embedding model + job
    # pool in memory); skip rather than wait when busy.
    #
    # The match set is NOT just users touched THIS cycle: a cycle whose Phase B
    # was skipped (lock busy) inserts fresh jobs but never scores them, and the
    # next cycle only touches a user again if MORE new jobs happen to route to
    # them — so a batch of fresh postings could sit unscored until the 2h fresh
    # lane. We also match any active user with UNSCORED postings first-seen in
    # the last day, so a deferred batch gets scored on the very next unlocked
    # cycle. run_matching's freshness reserve then guarantees those fresh jobs a
    # scoring slot. Bounded: once scored, a user drops out of the pending set.
    pending = _users_with_pending_fresh(users)
    to_match = set(users_touched) | pending
    alerts = 0
    matching = "no new jobs"
    if to_match:
        with discovery_guard(blocking=False, label="hot lane matching") as ran:
            if ran:
                matching = "ran"
                from app.matching.pipeline import run_matching
                from app.strategy.fresh_alerts import dispatch_fresh_alerts
                for uid in to_match:
                    try:
                        shortlisted = run_matching(uid) or []
                        alerts += dispatch_fresh_alerts(uid, shortlisted)
                    except Exception as e:
                        log.warning("hot lane match/alert failed for %s: %s", uid, e)
            else:
                matching = "skipped — discovery lock busy (jobs inserted, scoring deferred)"

    stats = {
        "boards": len(boards),
        "users": len(users),
        "fetched_jobs": fetched_jobs,
        "matched_jobs": matched_jobs,
        "inserted_jobs": inserted_jobs,
        "shared_inserted": shared_inserted,
        "users_with_new_jobs": len(users_touched),
        "users_matched": len(to_match),
        "users_pending_fresh": len(pending),
        "matching": matching,
        "alerts": alerts,
        "at": now.isoformat(),
    }
    return _finish_cycle(stats)


def _finish_cycle(stats: dict) -> dict:
    """Log + heartbeat for every cycle outcome, so the dashboard can always
    answer 'is the hot lane running, and if not why?'."""
    log.info("Hot lane: %s", stats)
    try:
        import json as _json
        from app.db.models import FunnelEvent
        with get_session() as session:
            session.add(FunnelEvent(
                job_id=None, stage="hot_lane_run",
                passed=(stats.get("fetched_jobs") or 0) > 0,
                reason=(stats.get("reason")
                        or f"boards={stats.get('boards')} jobs={stats.get('fetched_jobs')} "
                           f"inserted={stats.get('inserted_jobs')} alerts={stats.get('alerts')}"),
                metadata_json=_json.dumps(stats),
            ))
            session.commit()
    except Exception as e:
        log.debug("hot lane heartbeat write failed: %s", e)
    return stats


def _retire_unsupported(slug: str, ats) -> None:
    """Deactivate a board whose ATS has no scraper. It can never yield jobs, and
    leaving it active with last_seen=NULL keeps it permanently in the hot-lane
    bootstrap slice — a small but perpetual re-run of the 404 storm."""
    try:
        with get_session() as session:
            row = session.exec(
                select(CompanyRegistry).where(
                    CompanyRegistry.slug == slug, CompanyRegistry.ats == ats)
            ).first()
            if row and row.is_active:
                row.is_active = False
                row.last_seen = datetime.utcnow()
                row.inactive_reason = "unsupported ATS (no scraper)"
                session.add(row)
                session.commit()
    except Exception:
        pass


def _mark_polled(slug: str, ats, job_count: Optional[int], ok: bool,
                 new_jobs: int = 0, error: str = "") -> None:
    """Record a poll so board rotation stays fair, failures decay a board, and
    yield (new postings produced) steers future polling toward active boards.
    Dead boards (404 = board gone; or repeated failures) are retired so the
    per-cycle budget stops burning slots on companies that no longer exist."""
    from app.discovery.pipeline import BOARD_DEACTIVATE_AFTER_FAILURES
    try:
        with get_session() as session:
            row = session.exec(
                select(CompanyRegistry).where(
                    CompanyRegistry.slug == slug, CompanyRegistry.ats == ats)
            ).first()
            if not row:
                return
            row.last_seen = datetime.utcnow()
            if ok:
                if job_count is not None:
                    row.job_count = job_count
                row.failure_count = 0
                row.new_jobs_last_poll = new_jobs
                if new_jobs > 0:
                    row.last_new_job_at = datetime.utcnow()
            else:
                row.failure_count = (row.failure_count or 0) + 1
                row.last_error = (error or "")[:300]
                is_404 = "404" in (error or "")
                if is_404 or row.failure_count >= BOARD_DEACTIVATE_AFTER_FAILURES:
                    row.is_active = False
                    row.inactive_reason = (
                        "board_not_found (404)" if is_404
                        else f"unreachable x{row.failure_count}"
                    )
            session.add(row)
            session.commit()
    except Exception:
        pass
