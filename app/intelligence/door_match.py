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

    # Relocation / Hybrid preference-driven analysis
    if cand.open_to_relocation:
        if bar.onsite_metro and cand.home_metro:
            same_metro = bar.onsite_metro.split(",")[0].strip().lower() in cand.home_metro.lower()
            if same_metro:
                return DoorFinding("LOCATION / METRO", "MATCH", "on-site/hybrid in your metro — matches relocation/commute preference.")
            else:
                return DoorFinding("LOCATION / METRO", "STRETCH", f"role is on-site/hybrid in {bar.onsite_metro} — requires relocation (stretch).")
        return DoorFinding("LOCATION / METRO", "MATCH", "on-site is fine — you're open to relocation.")

    if not cand.remote_ok:
        # Candidate is not remote-only (prefers on-site/hybrid, but does not relocate)
        if bar.onsite_metro and cand.home_metro:
            same_metro = bar.onsite_metro.split(",")[0].strip().lower() in cand.home_metro.lower()
            if same_metro:
                return DoorFinding("LOCATION / METRO", "MATCH", "on-site/hybrid in your metro.")
            else:
                return DoorFinding("LOCATION / METRO", "BLOCKER", f"role is strictly on-site in {bar.onsite_metro} and you aren't open to relocation.")
        return DoorFinding("LOCATION / METRO", "MATCH", "on-site is fine.")
    
    # Remote-only candidate (remote_ok is True, open_to_relocation is False).
    if not bar.onsite_metro:
        return DoorFinding("LOCATION / METRO", "STRETCH", "role mentions on-site but metro is unspecified.")
    if not cand.home_metro:
        return DoorFinding("LOCATION / METRO", "STRETCH", f"role is on-site in {bar.onsite_metro}; your home metro is unspecified.")

    same_metro = bar.onsite_metro.split(",")[0].strip().lower() in cand.home_metro.lower()
    if same_metro:
        return DoorFinding("LOCATION / METRO", "STRETCH", "on-site in your metro — commute, but doable.")
    
    # Known different metro -> blocker.
    return DoorFinding("LOCATION / METRO", "BLOCKER",
                       f"role is strictly on-site ({bar.onsite_metro}); you want remote and aren't open to "
                       f"relocating — likely filtered on location before skills are read.")


def classify_door(cand: "CandidateProfile", bar: "RoleBar",
                  winners_n: int = 0, data_quality: str = "rich") -> DoorVerdict:
    f: List[DoorFinding] = []

    if bar.axis and bar.axis != cand.axis:
        f.append(DoorFinding("ROLE FIT", "BLOCKER",
                             f"This position requires a **{bar.axis}** background, but your profile is **{cand.axis}** — "
                             f"tailoring your resume won't bridge an axis mismatch."))
    elif bar.axis:
        f.append(DoorFinding("ROLE FIT", "MATCH", f"Your target role focus matches perfectly."))

    if bar.years is not None:
        gap = bar.years - cand.years
        if gap >= 2:
            f.append(DoorFinding("EXPERIENCE LEVEL", "BLOCKER",
                                 f"hiring bar targets ~{bar.years}y of experience vs your {cand.years}y ({gap}y gap) "
                                 f"for this {bar.level or 'senior'} role."))
        elif gap >= 1:
            f.append(DoorFinding("EXPERIENCE LEVEL", "STRETCH", f"slightly under the targeted experience ({cand.years}y vs target ~{bar.years}y)."))
        else:
            f.append(DoorFinding("EXPERIENCE LEVEL", "MATCH", "years of experience match target range."))

    if bar.domain:
        fit = any(d in bar.domain for d in cand.domains)
        f.append(DoorFinding("DOMAIN MATCH", "MATCH" if fit else "STRETCH",
                             f"requires domain expertise in **{bar.domain}**" + ("" if fit else " — which is missing from your history.")))

    if bar.pedigree:
        f.append(DoorFinding("CREDENTIALS / DEGREE", "BLOCKER", f"typically requires a credential pedigree (like **{bar.pedigree}**), which is not listed in your resume."))

    loc = _location_finding(cand, bar)
    if loc:
        f.append(loc)

    if bar.elite_outlier:
        f.append(DoorFinding("HIRING BAR", "BLOCKER",
                             "This role has a very high hiring bar requiring elite outlier credentials or background signal."))

    if cand.work_auth and cand.work_auth.upper().startswith("OPT"):
        f.append(DoorFinding("WORK AUTHORIZATION", "SILENT-LEAK",
                             "Note: OPT candidates are often filtered early due to sponsorship rules. Action: Reframe as 'STEM OPT = 3y visa runway'."))

    blockers = [x for x in f if x.status == "BLOCKER"]
    wrong = bool(blockers)
    top = blockers[0].note if blockers else "You fit the observed bar."
    
    if wrong:
        blocker_dims = {x.dim for x in blockers}
        if "ROLE FIT" in blocker_dims:
            right = _RIGHT_DOOR.get(bar.axis, _RIGHT_DOOR["enterprise"])
        elif "EXPERIENCE LEVEL" in blocker_dims:
            right = f"Roles matching your experience level ({cand.years}y) rather than senior/lead roles."
        elif "LOCATION / METRO" in blocker_dims:
            right = "Remote-friendly roles or roles located in your metro."
        elif "CREDENTIALS / DEGREE" in blocker_dims:
            right = "Roles at product companies/startups that value shipped code over academic credentials."
        else:
            right = _RIGHT_DOOR.get(bar.axis or cand.axis, _RIGHT_DOOR["enterprise"])
    else:
        right = "You fit — rejection is likely visibility/positioning, not skills. Lead with proof, go warm."
    return DoorVerdict(
        wrong_door=wrong, top_reason=top, right_door=right,
        confidence=_confidence(winners_n, data_quality), findings=f,
    )
