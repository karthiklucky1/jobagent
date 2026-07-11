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
    """The rotating hot set: active boards known to produce jobs, least-recently
    polled first so every board is swept over successive cycles."""
    # Split each cycle so BOTH goals are served without starving either:
    #  - half to never-polled boards (last_seen IS NULL) → bootstrap the tens of
    #    thousands of freshly-seeded companies so they start producing jobs fast.
    #    (The old "productive boards first" order left new boards permanently at
    #    the back of a 400-cap queue, so they were never scraped at all.)
    #  - half to productive + stale boards → keep fresh jobs flowing from boards
    #    already known to post. A board scraped once gets last_seen set, so dead
    #    boards fall out after a single attempt.
    half = max(1, limit // 2)
    with get_session() as session:
        never = session.exec(
            select(CompanyRegistry)
            .where(CompanyRegistry.is_active == True,  # noqa: E712
                   CompanyRegistry.last_seen == None)  # noqa: E711
            .limit(half)
        ).all()
        need = limit - len(never)
        productive = session.exec(
            select(CompanyRegistry)
            .where(CompanyRegistry.is_active == True,  # noqa: E712
                   CompanyRegistry.last_seen != None,  # noqa: E711
                   CompanyRegistry.job_count > 0)
            .order_by(CompanyRegistry.last_seen.asc())
            .limit(need)
        ).all()
        boards = list(never) + list(productive)
        # Backfill if either bucket was thin (e.g. few productive boards yet).
        if len(boards) < limit:
            seen = {b.id for b in boards}
            extra = session.exec(
                select(CompanyRegistry)
                .where(CompanyRegistry.is_active == True)  # noqa: E712
                .order_by(CompanyRegistry.last_seen.asc().nulls_first())
                .limit(limit)
            ).all()
            for b in extra:
                if b.id not in seen and len(boards) < limit:
                    boards.append(b)
                    seen.add(b.id)
    return boards[:limit]


def _title_matches(title: str, roles: list[str]) -> bool:
    """True if any of the user's target-role keywords appears in the job title.
    Empty roles → accept (user hasn't narrowed), matching current discovery."""
    if not roles:
        return True
    t = (title or "").lower()
    return any(r in t for r in roles)


def run_hot_lane() -> dict:
    """One hot-lane cycle. Returns a small stats dict for logging/telemetry."""
    from app.discovery.pipeline import scraper_for, _upsert
    from app.matching.pipeline import run_matching
    from app.strategy.fresh_alerts import dispatch_fresh_alerts

    limit = int(getattr(settings, "hot_lane_max_boards", 400))
    users = _active_users()
    if not users:
        return {"boards": 0, "users": 0, "reason": "no active users"}

    boards = select_hot_boards(limit)
    if not boards:
        return {"boards": 0, "users": len(users), "reason": "no active boards"}

    now = datetime.utcnow()
    fetched_jobs = 0
    users_touched: set = set()

    for board in boards:
        scraper = scraper_for(board.ats, board.slug, board.career_url)
        if scraper is None:
            continue
        try:
            raw = scraper.fetch()  # ONE network call, shared across all users
        except Exception as e:
            log.debug("hot lane fetch failed %s/%s: %s", board.ats, board.slug, e)
            _mark_polled(board.slug, board.ats, job_count=None, ok=False)
            continue
        fetched_jobs += len(raw)
        _mark_polled(board.slug, board.ats, job_count=len(raw), ok=True)
        if not raw:
            continue

        # Distribute each posting only to users whose target roles it matches.
        for u in users:
            relevant = [r for r in raw if _title_matches(r.title, u["roles"])]
            if not relevant:
                continue
            try:
                new = _upsert(relevant, user_id=u["user_id"])
                if new:
                    users_touched.add(u["user_id"])
            except Exception as e:
                log.debug("hot lane upsert failed for %s: %s", u["user_id"], e)

    # Match + alert only for users who actually received new postings.
    alerts = 0
    for uid in users_touched:
        try:
            shortlisted = run_matching(uid) or []
            alerts += dispatch_fresh_alerts(uid, shortlisted)
        except Exception as e:
            log.warning("hot lane match/alert failed for %s: %s", uid, e)

    stats = {
        "boards": len(boards),
        "users": len(users),
        "fetched_jobs": fetched_jobs,
        "users_with_new_jobs": len(users_touched),
        "alerts": alerts,
        "at": now.isoformat(),
    }
    log.info("Hot lane: %s", stats)
    return stats


def _mark_polled(slug: str, ats, job_count: Optional[int], ok: bool) -> None:
    """Record a poll so board rotation stays fair and failures decay a board."""
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
            else:
                row.failure_count = (row.failure_count or 0) + 1
            session.add(row)
            session.commit()
    except Exception:
        pass
