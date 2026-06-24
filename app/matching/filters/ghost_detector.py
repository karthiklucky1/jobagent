"""Ghost job detection - score how likely a posting is fake or already filled.

Ghost score bands:
  0.0 - 0.3  : Looks real - proceed normally
  0.3 - 0.6  : Suspicious - proceed but log warning
  0.6 - 1.0  : Likely ghost - skip to save LLM tokens
  1.0        : Confirmed closed (is_closed=True)
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from sqlalchemy import func
from sqlmodel import Session, select

from app.db.models import Job

log = logging.getLogger(__name__)

# Known aggregator/redirect domains that don't host real job postings.
# Jobs whose URL passes through these are scored as likely ghost/stale.
_AGGREGATOR_REDIRECT_DOMAINS = {
    "lensa.com",
    "jobrapido.com",
    "jora.com",
    "talent.com",
    "jobsora.com",
    "jobtensor.com",
    "careerjet.com",
    "adzuna.com",
    "joblist.com",
    "jobted.com",
    "neuvoo.com",   # rebranded to talent.com
    "findwork.dev",
    "bebee.com",
    "jobisit.com",
}

_AGGREGATOR_REDIRECT_RE = re.compile(
    r'https?://(?:www\.)?(' + '|'.join(re.escape(d) for d in _AGGREGATOR_REDIRECT_DOMAINS) + r')/',
    re.IGNORECASE,
)

_SALARY_SIGNAL_RE = re.compile(
    r'\$[\d,]+\s*[kK]?\b'
    r'|(?:salary|compensation|pay range|base pay|annual pay)\s*:?\s*\$?[\d,]+',
    re.IGNORECASE,
)

_MIN_DESCRIPTION_WORDS = 150
_STALE_DAYS = 60
_AGING_DAYS = 45
_LAST_SEEN_STALE_DAYS = 14


@dataclass
class GhostResult:
    ghost_score: float
    flags: List[str] = field(default_factory=list)
    flags_json: str = "[]"

    @property
    def is_ghost(self) -> bool:
        return self.ghost_score >= 0.6


def _has_salary(description: str) -> bool:
    return bool(_SALARY_SIGNAL_RE.search(description))


def _posting_age_days(job: Job) -> int:
    ref = job.posted_at or job.first_seen
    if ref is None:
        return 0
    return max(0, (datetime.utcnow() - ref).days)


def score_ghost(job: Job, session: Session) -> GhostResult:
    """Compute a ghost likelihood score for a single job.

    Runs entirely on local data (no LLM / network calls).
    Cheap enough to run before every embedding + LLM filter step.
    """
    flags: List[str] = []
    score = 0.0

    if job.is_closed:
        return GhostResult(ghost_score=1.0, flags=["job_closed"], flags_json='["job_closed"]')

    age_days = _posting_age_days(job)
    if age_days >= _STALE_DAYS:
        score += 0.40
        flags.append(f"stale_{age_days}d")
    elif age_days >= _AGING_DAYS:
        score += 0.20
        flags.append(f"aging_{age_days}d")

    if job.last_seen is not None:
        stale_days = (datetime.utcnow() - job.last_seen).days
        if stale_days >= _LAST_SEEN_STALE_DAYS:
            score += 0.15
            flags.append(f"not_refreshed_{stale_days}d")

    job_url = job.url or ""
    if _AGGREGATOR_REDIRECT_RE.search(job_url):
        matched = _AGGREGATOR_REDIRECT_RE.search(job_url).group(1)
        score += 0.35
        flags.append(f"aggregator_redirect_{matched}")

    desc = job.description or ""
    word_count = len(desc.split())
    if word_count < _MIN_DESCRIPTION_WORDS:
        score += 0.15
        flags.append(f"thin_desc_{word_count}w")

    if not _has_salary(desc):
        score += 0.10
        flags.append("no_salary")

    try:
        duplicate_count = session.exec(
            select(func.count(Job.id)).where(
                Job.company == job.company,
                Job.title == job.title,
                Job.is_closed == False,
                Job.id != job.id,
            )
        ).one()
        if duplicate_count >= 2:
            score += 0.20
            flags.append(f"duplicate_postings_{duplicate_count}")
    except Exception as exc:
        log.debug("Ghost detector: duplicate-count query failed: %s", exc)

    score = min(score, 1.0)
    flags_json = json.dumps(flags)
    return GhostResult(ghost_score=round(score, 3), flags=flags, flags_json=flags_json)
