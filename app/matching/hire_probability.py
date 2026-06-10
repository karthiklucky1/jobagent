"""Hire Probability Scorer — how likely is this company *actively* hiring right now?

Combines signals already available in the DB + job description text.
No external API calls — runs fast, before the expensive LLM rerank.

Score bands (0.0–1.0):
  0.0 – 0.3  : Low hiring intent (stale, tiny company with no recent activity)
  0.3 – 0.6  : Moderate — proceed normally
  0.6 – 1.0  : Strong hiring intent — boost final ranking

Final blended score (stored separately, used in pipeline):
  blended = 0.65 * rerank_score + 0.35 * hire_probability_score * 100
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from sqlalchemy import func
from sqlmodel import Session, select

from app.db.models import Job

log = logging.getLogger(__name__)


# ── Funding / growth keywords in description ──────────────────────────────────
_FUNDING_SIGNALS = [
    "series a", "series b", "series c", "series d",
    "seed round", "recently funded", "just raised", "raised $",
    "yc ", "y combinator", "ycombinator",
    "techcrunch", "just closed",
]

_GROWTH_SIGNALS = [
    "rapidly growing", "fast-growing", "hypergrowth", "scaling fast",
    "growing team", "expanding team", "we're growing", "we are growing",
    "doubling our team", "tripling our team", "aggressive hiring",
    "multiple openings", "several openings",
]

_URGENCY_SIGNALS = [
    "immediate start", "start asap", "starting immediately",
    "urgent hire", "we need someone now", "join us today",
    "open to interview immediately",
]

# Company size bracket from description — startup-scale = more hiring velocity
_STARTUP_SIGNALS = [
    "early stage", "founding engineer", "founding team", "employee #",
    "join as one of the first", "small but mighty", "lean team",
    "we're a team of", "team of ~", "team of about",
]

_ENTERPRISE_BLOAT_SIGNALS = [
    "fortune 500", "global corporation", "multinational",
    "tens of thousands of employees", "over 10,000 employees",
    "over 50,000 employees",
]

# Posting freshness weights (days since posted_at)
_FRESH_DAYS = 14       # +0.25
_RECENT_DAYS = 30      # +0.15
_NORMAL_DAYS = 60      # 0


@dataclass
class HireProbabilityResult:
    score: float                    # 0.0–1.0
    signals: List[str] = field(default_factory=list)
    company_velocity: int = 0       # number of open jobs at same company in DB


def score_hire_probability(job: Job, session: Session) -> HireProbabilityResult:
    """Score how actively a company is hiring based on in-DB signals.

    Designed to be fast (no LLM / HTTP calls).
    """
    signals: List[str] = []
    score = 0.0
    desc = (job.description or "").lower()

    # ── 1. Posting freshness ──────────────────────────────────────────────────
    ref = job.posted_at or job.first_seen
    if ref is not None:
        age_days = max(0, (datetime.utcnow() - ref).days)
        if age_days <= _FRESH_DAYS:
            score += 0.25
            signals.append(f"fresh_posting_{age_days}d")
        elif age_days <= _RECENT_DAYS:
            score += 0.15
            signals.append(f"recent_posting_{age_days}d")
        elif age_days <= _NORMAL_DAYS:
            score += 0.05
            signals.append(f"normal_age_{age_days}d")
        # older than 60d = 0 contribution (ghost detector handles skipping)
    else:
        score += 0.05  # no date = neutral
        signals.append("no_date")

    # ── 2. Company velocity — how many open jobs at same company in DB ─────────
    try:
        open_count = session.exec(
            select(func.count(Job.id)).where(
                Job.company == job.company,
                Job.is_closed == False,
            )
        ).one() or 0
    except Exception:
        open_count = 1

    if open_count >= 10:
        score += 0.25
        signals.append(f"high_velocity_{open_count}_openings")
    elif open_count >= 5:
        score += 0.15
        signals.append(f"medium_velocity_{open_count}_openings")
    elif open_count >= 2:
        score += 0.08
        signals.append(f"low_velocity_{open_count}_openings")

    # ── 3. Funding / growth language in description ───────────────────────────
    funding_hits = [s for s in _FUNDING_SIGNALS if s in desc]
    if funding_hits:
        score += min(0.20, 0.08 * len(funding_hits))
        signals.append(f"funding_language:{','.join(funding_hits[:2])}")

    growth_hits = [s for s in _GROWTH_SIGNALS if s in desc]
    if growth_hits:
        score += min(0.15, 0.06 * len(growth_hits))
        signals.append(f"growth_language:{','.join(growth_hits[:2])}")

    urgency_hits = [s for s in _URGENCY_SIGNALS if s in desc]
    if urgency_hits:
        score += 0.10
        signals.append(f"urgency:{urgency_hits[0]}")

    # ── 4. Startup signals — higher hiring velocity per headcount ─────────────
    startup_hits = [s for s in _STARTUP_SIGNALS if s in desc]
    if startup_hits:
        score += 0.10
        signals.append(f"startup_stage:{startup_hits[0]}")

    # ── 5. Enterprise bloat penalty — slower hiring, more bureaucracy ─────────
    if any(s in desc for s in _ENTERPRISE_BLOAT_SIGNALS):
        score -= 0.10
        signals.append("enterprise_scale_penalty")

    score = round(min(max(score, 0.0), 1.0), 3)
    return HireProbabilityResult(
        score=score,
        signals=signals,
        company_velocity=open_count,
    )


def blended_score(rerank_score: float, hire_prob_score: float) -> float:
    """Combine LLM fit score with hire probability.

    rerank_score: 0–100 (from Reranker)
    hire_prob_score: 0.0–1.0 (from score_hire_probability)
    Returns: 0–100 blended score
    """
    return round(0.65 * rerank_score + 0.35 * hire_prob_score * 100, 1)
