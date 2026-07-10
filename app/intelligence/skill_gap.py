"""Skill-gap analysis: what the user's matched jobs demand vs what they can prove.

For each skill demanded across the user's top matches, classify:
  - ``matched``        — the phrase appears verbatim on the résumé.
  - ``add_visibility`` — not on the résumé, but proof exists on the user's own
                          GitHub (repo language/topic/description) or in their
                          pasted LinkedIn text. Advice: add it to your résumé
                          and LinkedIn *yourself* — we never inject skills.
  - ``learn``          — no evidence anywhere. Advice: build a small project,
                          push it to GitHub, add the skill to LinkedIn.

Deterministic (no LLM): JD phrases come from ats_keywords, evidence checks are
verbatim-phrase lookups against the résumé / GitHub / LinkedIn text blobs. The
only network call is a one-time GitHub harvest when the user has a GitHub URL
but no cached harvest yet (stored to UserPersonalMemory for next time).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from sqlmodel import select

from app.db.init_db import get_session
from app.db.models import Job, UserPersonalMemory, UserProfile
from app.tailoring.ats_keywords import _normalize, _phrase_present, extract_jd_phrases

log = logging.getLogger(__name__)

# Phrases per job to consider; demand floor keeps one-off JD noise out of the
# dashboard once there are enough scanned jobs to make counts meaningful.
_PHRASES_PER_JOB = 14
_MIN_DEMAND_WITH_MANY_JOBS = 2


def _user_arg(user_id: Optional[str]) -> Optional[str]:
    return user_id if user_id and user_id != "local" else None


def _top_jobs(user_id: Optional[str], limit: int) -> list[Job]:
    uid = _user_arg(user_id)
    with get_session() as session:
        q = select(Job).where(
            Job.user_id == uid,  # noqa: E711
            Job.rerank_score != None,  # noqa: E711
            Job.is_closed == False,  # noqa: E712
        )
        jobs = session.exec(q).all()
    jobs.sort(key=lambda j: (j.blended_score or j.rerank_score or 0), reverse=True)
    return jobs[:limit]


def _github_evidence(user_id: Optional[str], profile: Optional[UserProfile]) -> dict:
    """Return {connected, username, blob, repos:[{name, text}], harvested_at}.

    Uses the latest cached UserPersonalMemory(github) row; harvests live once
    when the user has a GitHub URL but no cached row yet.
    """
    uid = _user_arg(user_id)
    out = {"connected": False, "username": "", "blob": "", "repos": [], "harvested_at": None}

    row = None
    with get_session() as session:
        q = (
            select(UserPersonalMemory)
            .where(UserPersonalMemory.user_id == uid,  # noqa: E711
                   UserPersonalMemory.source == "github")
            .order_by(UserPersonalMemory.created_at.desc())  # type: ignore[attr-defined]
        )
        row = session.exec(q).first()

    gh: dict = {}
    if row and row.raw_content:
        try:
            raw = json.loads(row.raw_content)
            gh = raw.get("github", raw) or {}
            out["harvested_at"] = row.created_at.isoformat() if row.created_at else None
        except Exception:
            gh = {}

    github_url = (getattr(profile, "github_url", "") or "").strip() if profile else ""
    if not gh.get("repos") and github_url:
        # First use: harvest now and cache, so later loads are DB-only.
        try:
            from app.intelligence.harvester import harvest_github
            gh = harvest_github(github_url)
            if gh.get("ok"):
                with get_session() as session:
                    session.add(UserPersonalMemory(
                        user_id=uid, source="github",
                        raw_content=json.dumps({"github": gh}),
                        parsed_updates="", recommendations="",
                        created_at=datetime.utcnow(),
                    ))
                    session.commit()
                out["harvested_at"] = datetime.utcnow().isoformat()
        except Exception as e:
            log.warning("Skill-gap GitHub harvest failed: %s", e)
            gh = gh or {}

    repos = gh.get("repos") or []
    if not repos:
        return out

    out["connected"] = True
    out["username"] = gh.get("username") or ""
    parts = []
    for r in repos:
        text = " ".join([
            str(r.get("name") or ""),
            str(r.get("description") or ""),
            str(r.get("language") or ""),
            " ".join(r.get("topics") or []),
        ])
        out["repos"].append({"name": r.get("name") or "", "text": _normalize(text)})
        parts.append(text)
    for e in gh.get("events") or []:
        parts.append(str(e.get("message") or ""))
    out["blob"] = _normalize(" ".join(parts))
    return out


def _linkedin_evidence(user_id: Optional[str]) -> dict:
    """Latest pasted-LinkedIn text (user-supplied — we never scrape LinkedIn)."""
    uid = _user_arg(user_id)
    with get_session() as session:
        q = (
            select(UserPersonalMemory)
            .where(UserPersonalMemory.user_id == uid,  # noqa: E711
                   UserPersonalMemory.source == "linkedin")
            .order_by(UserPersonalMemory.created_at.desc())  # type: ignore[attr-defined]
        )
        row = session.exec(q).first()
    if not row or not (row.raw_content or "").strip():
        return {"connected": False, "blob": ""}
    return {"connected": True, "blob": _normalize(row.raw_content)}


def _load_resume_text(user_id: Optional[str], profile: Optional[UserProfile]) -> tuple[str, bool]:
    try:
        from app.matching.pipeline import _load_resume
        return _load_resume(user_id=_user_arg(user_id)), True
    except Exception as e:
        log.info("Skill-gap: no résumé loaded (%s) — falling back to profile skills", e)
        fallback = " ".join([
            getattr(profile, "key_skills", "") or "",
            getattr(profile, "professional_summary", "") or "",
        ]) if profile else ""
        return fallback, False


def _advice(skill: str, status: str, demand: int, repo: str = "") -> str:
    if status == "add_visibility":
        where = f" (your GitHub repo “{repo}” already shows it)" if repo else " (found in your LinkedIn text)"
        return (
            f"You already have proof of {skill}{where}, but it's invisible to "
            f"recruiters scanning your résumé — add it to your résumé and your "
            f"LinkedIn skills yourself."
        )
    return (
        f"{demand} of your top matches want {skill} and you have no proof of it "
        f"yet. Build a small project using {skill}, push it to GitHub with a "
        f"clear README, and add the skill to your LinkedIn — recruiters verify "
        f"there, not on claims."
    )


def compute_skill_gap(user_id: Optional[str], top_n_jobs: int = 30) -> dict:
    """Aggregate skill demand across the user's top matches and classify each
    skill by the strongest evidence the user can show for it."""
    uid = _user_arg(user_id)
    profile = None
    with get_session() as session:
        q = select(UserProfile).where(UserProfile.user_id == uid)  # noqa: E711
        profile = session.exec(q).first()

    jobs = _top_jobs(user_id, top_n_jobs)
    if not jobs:
        return {"scanned_jobs": 0, "matched": [], "add_visibility": [], "learn": [],
                "resume_loaded": False, "github": {"connected": False},
                "linkedin": {"connected": False}}

    resume_text, resume_loaded = _load_resume_text(user_id, profile)
    resume_norm = _normalize(resume_text)
    gh = _github_evidence(user_id, profile)
    li = _linkedin_evidence(user_id)

    # skill -> {demand, jobs:[(title, company)]}
    demand: dict[str, dict] = {}
    for job in jobs:
        jd = job.description or ""
        if not jd.strip():
            continue
        try:
            phrases = extract_jd_phrases(jd, top_n=_PHRASES_PER_JOB)
        except Exception:
            continue
        seen_this_job = set()
        for phrase in phrases:
            key = phrase.lower().strip()
            if not key or key in seen_this_job:
                continue
            seen_this_job.add(key)
            entry = demand.setdefault(key, {"demand": 0, "jobs": []})
            entry["demand"] += 1
            if len(entry["jobs"]) < 3:
                entry["jobs"].append({"title": job.title, "company": job.company})

    min_demand = _MIN_DEMAND_WITH_MANY_JOBS if len(jobs) >= 10 else 1
    matched, add_visibility, learn = [], [], []
    for skill, entry in demand.items():
        if entry["demand"] < min_demand:
            continue
        item = {
            "skill": skill,
            "demand": entry["demand"],
            "pct": round(100 * entry["demand"] / len(jobs)),
            "example_jobs": entry["jobs"],
        }
        if resume_norm and _phrase_present(skill, resume_norm):
            item["status"] = "matched"
            matched.append(item)
            continue
        repo_hit = next((r["name"] for r in gh["repos"] if _phrase_present(skill, r["text"])), "")
        if repo_hit or (gh["blob"] and _phrase_present(skill, gh["blob"])):
            item["status"] = "add_visibility"
            item["evidence"] = {"source": "github", "repo": repo_hit}
            item["advice"] = _advice(skill, "add_visibility", entry["demand"], repo_hit)
            add_visibility.append(item)
            continue
        if li["blob"] and _phrase_present(skill, li["blob"]):
            item["status"] = "add_visibility"
            item["evidence"] = {"source": "linkedin", "repo": ""}
            item["advice"] = _advice(skill, "add_visibility", entry["demand"])
            add_visibility.append(item)
            continue
        item["status"] = "learn"
        item["advice"] = _advice(skill, "learn", entry["demand"])
        learn.append(item)

    for bucket in (matched, add_visibility, learn):
        bucket.sort(key=lambda x: x["demand"], reverse=True)

    return {
        "scanned_jobs": len(jobs),
        "resume_loaded": resume_loaded,
        "github": {"connected": gh["connected"], "username": gh["username"],
                   "harvested_at": gh["harvested_at"]},
        "linkedin": {"connected": li["connected"]},
        "matched": matched[:40],
        "add_visibility": add_visibility[:40],
        "learn": learn[:40],
    }
