"""Rejection Autopsy orchestrator — composes the pieces into one result.

Pipeline:  X-ray "who actually holds this role" (linkedin_xray.find_champions)
        -> build_role_bar (the hidden bar)
        -> classify_door (right vs wrong door + reason + right door)

`finder` is dependency-injected so this is fully unit-testable without SerpAPI /
network. In production it defaults to the real X-ray. This function is ADDITIVE:
it returns a diagnosis dict and never hides/filters jobs.
"""
from __future__ import annotations

from typing import Callable, Optional

from app.intelligence.door_match import CandidateProfile, classify_door
from app.intelligence.role_bar import build_role_bar


def _default_finder(company: str, role: str) -> dict:
    from app.intelligence.linkedin_xray import find_champions
    return find_champions(company, role)


def run_autopsy(company: str, role: str, jd_text: str,
                candidate: CandidateProfile,
                finder: Optional[Callable[[str, str], dict]] = None) -> dict:
    finder = finder or _default_finder
    res = finder(company, role) or {}
    winners = res.get("people", []) if res.get("ok") else []
    n = len(winners)

    bar = build_role_bar(role, jd_text, winners)
    verdict = classify_door(
        candidate, bar, winners_n=n,
        data_quality="rich" if n >= 3 else "thin",
    )
    return {
        "ok": True,
        "company": company,
        "role": role,
        "winners_n": n,
        "winners": winners[:5],
        "data_ok": bool(res.get("ok")),
        "data_note": res.get("note") or res.get("reason"),
        "bar": {
            "axis": bar.axis, "years": bar.years, "domain": bar.domain,
            "onsite": bar.onsite, "onsite_metro": bar.onsite_metro,
            "pedigree": bar.pedigree, "elite_outlier": bar.elite_outlier,
        },
        "wrong_door": verdict.wrong_door,
        "top_reason": verdict.top_reason,
        "right_door": verdict.right_door,
        "confidence": verdict.confidence,
        "findings": [{"dim": f.dim, "status": f.status, "note": f.note}
                     for f in verdict.findings],
    }
