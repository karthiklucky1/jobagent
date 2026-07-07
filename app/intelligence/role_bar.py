"""Build a RoleBar (the observed hidden bar of a role) from public signals.

Inputs that are cheap + public:
  - the role title,
  - the JD text (already stored on Job.description),
  - the "winner" snippets from the X-ray (find_champions in linkedin_xray.py):
    a list of {"name", "headline", "url"} where `headline` is the Google snippet.

Output: a RoleBar (see app/intelligence/door_match.py) that feeds classify_door.

This first version is DETERMINISTIC (keyword/regex) so it is fully testable with no
API keys. An LLM refinement can override `axis` later; the shape stays the same.
"""
from __future__ import annotations

import re
from typing import List, Optional

from app.intelligence.door_match import RoleBar

# ── Axis signals ──────────────────────────────────────────────────────────────
_RESEARCH = re.compile(
    r"\b(research scientist|research engineer|member of technical staff|\bmts\b|"
    r"ph\.?\s?d|published|publication|foundations|post-?training|reinforcement learning|"
    r"\brl\b|iclr|neurips|acl|emnlp)\b", re.I)
_ENTERPRISE = re.compile(
    r"\b(s\.?v\.?p\.?|a\.?v\.?p\.?|assistant vice president|vice president|"
    r"financial services|\bbank\b|banking|insurance|fortune \d00|"
    r"director,? (data|analytics|engineering|infrastructure|platform|ai)|"
    r"chief \w+ officer)\b", re.I)
_ELITE = re.compile(
    r"(\$?\s?[2-9]\d0\s?k)|\bfounding (engineer|team)\b|prove you'?re better than ai|"
    r"generational|\bepoch\b|olympiad|top 1%", re.I)
_ONSITE = re.compile(r"\b(on-?site|in-?office|in person|on campus)\b", re.I)
_REMOTE = re.compile(r"\b(fully remote|100% remote|remote-first|work from anywhere)\b", re.I)
_YEARS = re.compile(r"(\d{1,2})\s*\+?\s*years?", re.I)
_METRO = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?),\s*([A-Z]{2})\b")

_DOMAINS = {
    "genai": r"gen[- ]?ai|generative ai|llm|prompt",
    "rag": r"\brag\b|retrieval[- ]augmented|vector (search|db|database)",
    "nlp": r"\bnlp\b|natural language",
    "cv": r"computer vision|\bcv\b|image",
    "financial services": r"financial|banking|trading|fintech|insurance",
    "ml": r"machine learning|\bml\b|mlops|model",
}


def _count(pattern: re.Pattern, texts: List[str]) -> int:
    return sum(1 for t in texts if pattern.search(t or ""))


def _detect_axis(role_title: str, jd: str, headlines: List[str]) -> str:
    blob = " ".join([role_title, jd])
    # Elite-outlier: explicit high-comp/founding/"prove" signals dominate.
    if _ELITE.search(blob):
        return "elite-outlier"
    # Research: needs to show up in the JD/title AND in ≥2 winner headlines
    # (one PhD leader isn't enough to call the whole bar "research").
    if _RESEARCH.search(blob) or _count(_RESEARCH, headlines) >= 2:
        if _RESEARCH.search(blob) and _count(_RESEARCH, headlines) >= 1:
            return "research"
        if _count(_RESEARCH, headlines) >= 2:
            return "research"
    # Enterprise: enterprise/AVP/analytics/financial signals.
    if _ENTERPRISE.search(blob) or _count(_ENTERPRISE, headlines) >= 1:
        return "enterprise"
    return "applied"


def _detect_years(jd: str, headlines: List[str]) -> Optional[int]:
    nums = [int(m) for m in _YEARS.findall(jd or "")]
    nums += [int(m) for m in _YEARS.findall(" ".join(headlines))]
    nums = [n for n in nums if 0 < n <= 20]
    return max(nums) if nums else None


def _detect_onsite(jd: str) -> (bool, str):
    if _REMOTE.search(jd or "") and not _ONSITE.search(jd or ""):
        return False, ""
    if _ONSITE.search(jd or ""):
        m = _METRO.search(jd or "")
        return True, (f"{m.group(1)}, {m.group(2)}" if m else "")
    return False, ""


def _detect_domain(jd: str, role_title: str) -> Optional[str]:
    blob = f"{role_title} {jd}"
    for label, pat in _DOMAINS.items():
        if re.search(pat, blob, re.I):
            return label
    return None


def _detect_pedigree(axis: str, headlines: List[str]) -> Optional[str]:
    if axis == "research" or _count(_RESEARCH, headlines) >= 2:
        return "PhD / published research"
    return None


def build_role_bar(role_title: str, jd_text: str = "",
                   winners: Optional[List[dict]] = None) -> RoleBar:
    headlines = [(w.get("headline") or "") for w in (winners or [])]
    axis = _detect_axis(role_title, jd_text, headlines)
    onsite, metro = _detect_onsite(jd_text)
    return RoleBar(
        years=_detect_years(jd_text, headlines),
        axis=axis,
        domain=_detect_domain(jd_text, role_title),
        onsite=onsite,
        onsite_metro=metro,
        pedigree=_detect_pedigree(axis, headlines),
        level=None,
        elite_outlier=(axis == "elite-outlier"),
    )
