"""Adoption — fill a user's pool from the SHARED job pool with a cheap DB copy.

The shared pool (see ``SHARED_POOL_USER`` in discovery/pipeline.py) holds every
posting any lane has fetched, once. Adoption copies the subset matching a
user's target roles + location preferences into their own pool — no HTTP, no
scraping, just database reads and the standard ``_upsert`` dedupe path. This is
what makes onboarding instant: a brand-new user (or a user who just edited
their roles) gets weeks of already-collected matching jobs in seconds, then the
regular matching pass scores them.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from sqlmodel import select

from app.db.init_db import get_session
from app.db.models import Job

log = logging.getLogger(__name__)

# How far back to adopt by default: old enough to fill a board, young enough
# that postings are still worth applying to.
ADOPT_MAX_AGE_DAYS = 21
# Cap per adoption pass — a full matching pass only LLM-scores a slice per run
# anyway, and the next cycles keep draining.
ADOPT_MAX_JOBS = 400


def adopt_shared_jobs(user_id: str | None, max_age_days: int = ADOPT_MAX_AGE_DAYS,
                      limit: int = ADOPT_MAX_JOBS) -> int:
    """Copy recent, role-matching shared-pool postings into ``user_id``'s pool.

    Reuses ``_upsert`` so per-user dedupe (source+external_id and cross-source
    slug), the country gate, and direct-ATS upgrades behave exactly as if the
    jobs had been scraped for this user. Returns the number of NEW rows."""
    from app.api.server import _get_target_roles
    from app.discovery.base import RawJob
    from app.discovery.pipeline import SHARED_POOL_USER, _upsert
    from app.discovery.title_filter import role_title_match

    roles = [r.lower() for r in (_get_target_roles(user_id or "local") or [])]

    # Location preferences — same defaults run_discovery uses.
    country, remote_ok = "United States", True
    try:
        from app.autofill.answer_pack import _get_or_create_profile
        p = _get_or_create_profile(user_id=user_id)
        if p:
            country = (getattr(p, "preferred_country", "") or "United States").strip() or "United States"
            remote_ok = bool(getattr(p, "remote_ok", True))
    except Exception as e:
        log.debug("adoption: profile unavailable (default US): %s", e)

    cutoff = datetime.utcnow() - timedelta(days=max_age_days)
    with get_session() as session:
        shared = session.exec(
            select(Job)
            .where(Job.user_id == SHARED_POOL_USER,
                   Job.is_closed == False,  # noqa: E712
                   Job.first_seen != None)  # noqa: E711
            .order_by(Job.first_seen.desc())
            .limit(5000)
        ).all()

    def _fresh_enough(j: Job) -> bool:
        ref = j.posted_at or j.first_seen
        if ref is None:
            return False
        if ref.tzinfo is not None:
            ref = ref.replace(tzinfo=None)
        return ref >= cutoff

    candidates = [j for j in shared
                  if _fresh_enough(j) and role_title_match(j.title, roles)]
    candidates = candidates[:limit]
    if not candidates:
        return 0

    raw = [RawJob(
        source=j.source.value if hasattr(j.source, "value") else str(j.source),
        external_id=j.external_id,
        company=j.company,
        title=j.title,
        location=j.location,
        remote=bool(j.remote),
        url=j.url,
        description=j.description or "",
        posted_at=j.posted_at,
    ) for j in candidates]

    inserted = _upsert(raw, user_id=user_id, preferred_country=country,
                       remote_ok=remote_ok, user_keywords=roles or None)
    log.info("Adoption: %d shared candidates → %d new jobs for user %s",
             len(candidates), inserted, user_id or "local")
    return inserted


def adopt_and_match(user_id: str | None) -> int:
    """Adoption + a matching pass — the 'instant feed' used after resume upload
    and role edits. Matching waits politely on the discovery lock; adoption
    itself never needs it (pure DB copy)."""
    adopted = adopt_shared_jobs(user_id)
    try:
        from app.common.discovery_lock import discovery_guard
        from app.matching.pipeline import run_matching
        with discovery_guard(label="instant feed") as ran:
            if ran:
                run_matching(user_id)
    except Exception as e:
        log.warning("instant-feed matching failed for %s: %s", user_id, e)
    return adopted
