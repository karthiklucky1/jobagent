"""Door-match engine — the "right door vs wrong door" verdict.

Given a candidate and the *observed hidden bar* of a role (the shared profile of
people who actually got in), decide whether the candidate is even aiming at a door
that can say yes — and if not, why, and where the right door is.

This is the deterministic core validated in the Phase-1 prototype. It is ADDITIVE
and side-effect free: it does NOT hide or filter jobs. The "who actually got in"
data (the RoleBar) is produced upstream by the X-ray/GitHub enrichment step
(app/intelligence/linkedin_xray.py); this module only does the comparison + phrasing.

Key principle (from the Royal Caribbean test): weighting is USER-PREFERENCE-DRIVEN.
Location is only a blocker for a remote-only candidate; if the candidate is open to
relocation/hybrid, an on-site role is at most a stretch — never a blanket block.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# Profile axes a role can demand. A mismatch here is the classic "wrong door".
AXES = ("applied", "research", "enterprise", "elite-outlier")

_RIGHT_DOOR = {
    "research": "Applied ML / Forward-Deployment / Solutions roles where shipping a "
                "production system IS the bar (not publications).",
    "enterprise": "Mid-level Applied/GenAI roles at product companies & startups that "
                  "value shipped proof over enterprise tenure.",
    "elite-outlier": "Normal applied-AI startups/scaleups hiring for shipped work — "
                     "not moonshot 'prove-you're-a-genius' roles.",
    "applied": "You fit — this is your door.",
}


@dataclass
class CandidateProfile:
    years: int = 0
    axis: str = "applied"               # applied | research | enterprise | elite-outlier
    domains: List[str] = field(default_factory=list)
    remote_ok: bool = True              # candidate wants remote
    open_to_relocation: bool = False    # candidate will relocate / do hybrid on-site
    work_auth: str = ""                 # e.g. "OPT", "Citizen", "H1B"
    home_metro: str = ""                # e.g. "Cincinnati, OH"

    @classmethod
    def from_user_profile(cls, p) -> "CandidateProfile":
        """Build from a UserProfile (app/db/models.py) — tolerant of missing attrs."""
        g = lambda name, default: getattr(p, name, default)
        skills = (g("key_skills", "") or "").lower()
        roles = (g("target_roles", "") or g("current_title", "") or "").lower()
        domains = [k for k in ("genai", "llm", "rag", "ml", "backend", "nlp", "cv", "data")
                   if k in skills or k in roles]
        work_auth = (g("work_authorization", "") or g("work_auth_status", "")
                     or g("visa_status", "") or "").strip()
        return cls(
            years=int(g("years_experience", 0) or 0),
            axis="applied",  # default; an upstream classifier may override
            domains=domains or ["ml"],
            remote_ok=bool(g("remote_ok", True)),
            open_to_relocation=bool(g("open_to_relocation", False)),
            work_auth=work_auth,
            home_metro=(g("location", "") or "").strip(),
        )


@dataclass
class RoleBar:
    """The observed hidden bar of a role (from people who actually got in)."""
    years: Optional[int] = None
    axis: Optional[str] = None
    domain: Optional[str] = None
    onsite: bool = False
    onsite_metro: str = ""              # where the role is on-site, if any
    pedigree: Optional[str] = None      # e.g. "PhD / published"
    level: Optional[str] = None         # e.g. "senior (AVP)", "MTS"
    elite_outlier: bool = False


@dataclass
class DoorFinding:
    dim: str          # AXIS | SENIORITY | DOMAIN | PEDIGREE | LOCATION | WORK-AUTH | BAR
    status: str       # MATCH | STRETCH | BLOCKER | SILENT-LEAK
    note: str


@dataclass
class DoorVerdict:
    wrong_door: bool
    top_reason: str
    right_door: str
    confidence: str                  # HIGH | MEDIUM | LOW (...)
    findings: List[DoorFinding] = field(default_factory=list)


def _confidence(winners_n: int, data_quality: str) -> str:
    if data_quality == "thin" or winners_n < 3:
        return f"LOW (N={winners_n}; using founder/investor signal)"
    return f"{'HIGH' if winners_n >= 6 else 'MEDIUM'} (N={winners_n})"


def _location_finding(cand: "CandidateProfile", bar: "RoleBar") -> Optional[DoorFinding]:
    """Preference-driven: on-site only blocks a remote-only candidate in another metro."""
    if not bar.onsite:
        return None
    if cand.open_to_relocation or not cand.remote_ok:
        # Candidate will go on-site / is not remote-only -> not a blocker.
        return DoorFinding("LOCATION", "MATCH", "on-site is fine — you're open to it.")
    # Remote-only candidate. Same metro -> fine; different metro -> hard blocker.
    same_metro = bool(bar.onsite_metro) and bool(cand.home_metro) and \
        bar.onsite_metro.split(",")[0].strip().lower() in cand.home_metro.lower()
    if same_metro:
        return DoorFinding("LOCATION", "STRETCH", "on-site in your metro — commute, but doable.")
    where = f" ({bar.onsite_metro})" if bar.onsite_metro else ""
    return DoorFinding("LOCATION", "BLOCKER",
                       f"role is strictly on-site{where}; you want remote and aren't open to "
                       f"relocating — likely filtered on location before skills are read.")


def classify_door(cand: "CandidateProfile", bar: "RoleBar",
                  winners_n: int = 0, data_quality: str = "rich") -> DoorVerdict:
    f: List[DoorFinding] = []

    if bar.axis and bar.axis != cand.axis:
        f.append(DoorFinding("AXIS", "BLOCKER",
                             f"role wants an **{bar.axis}** profile; you are **{cand.axis}** — "
                             f"a different door (tailoring can't fix an axis mismatch)."))
    elif bar.axis:
        f.append(DoorFinding("AXIS", "MATCH", f"your {cand.axis} axis matches."))

    if bar.years is not None:
        gap = bar.years - cand.years
        if gap >= 2:
            f.append(DoorFinding("SENIORITY", "BLOCKER",
                                 f"winners ~{bar.years}y vs your {cand.years}y ({gap}y gap) "
                                 f"for a {bar.level or 'senior'} role."))
        elif gap >= 1:
            f.append(DoorFinding("SENIORITY", "STRETCH", f"slightly under ({cand.years} vs ~{bar.years}y)."))
        else:
            f.append(DoorFinding("SENIORITY", "MATCH", "years are in range."))

    if bar.domain:
        fit = any(d in bar.domain for d in cand.domains)
        f.append(DoorFinding("DOMAIN", "MATCH" if fit else "STRETCH",
                             f"winners cluster in **{bar.domain}**" + ("" if fit else " — not in your history.")))

    if bar.pedigree:
        f.append(DoorFinding("PEDIGREE", "BLOCKER", f"winners have **{bar.pedigree}**; you don't."))

    loc = _location_finding(cand, bar)
    if loc:
        f.append(loc)

    if bar.elite_outlier:
        f.append(DoorFinding("BAR", "BLOCKER",
                             "a 'junior' title masks an elite-outlier bar — needs an outlier "
                             "signal, not just solid skills."))

    if cand.work_auth and cand.work_auth.upper().startswith("OPT"):
        f.append(DoorFinding("WORK-AUTH", "SILENT-LEAK",
                             "OPT is often read as 'future sponsorship risk' even when you say "
                             "'no sponsorship needed' — reframe: STEM OPT = 3y runway."))

    blockers = [x for x in f if x.status == "BLOCKER"]
    wrong = bool(blockers)
    top = blockers[0].note if blockers else "You fit the observed bar."
    right = _RIGHT_DOOR.get(bar.axis or cand.axis, _RIGHT_DOOR["enterprise"]) if wrong \
        else "You fit — rejection is likely visibility/positioning, not skills. Lead with proof, go warm."
    return DoorVerdict(
        wrong_door=wrong, top_reason=top, right_door=right,
        confidence=_confidence(winners_n, data_quality), findings=f,
    )
