import re
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from app.db.models import Job
from app.matching.filters.constants import NON_US_LOCATIONS, NO_SPONSORSHIP_PATTERNS, STAFF_TITLES

log = logging.getLogger(__name__)

# Salary targeting: $80k–$150k/yr
_SALARY_TOO_HIGH_MIN = 150_000   # reject if advertised minimum >= this (e.g. "$160k-$200k")
_SALARY_TOO_LOW_MAX  = 80_000    # reject if advertised maximum <= this (e.g. "$50k-$75k")

# Pre-compile the range pattern once
_SALARY_RANGE_RE = re.compile(
    r'\$([\d,]+)\s*(k)?\s*[-–to]+\s*\$([\d,]+)\s*(k)?',
    re.IGNORECASE,
)
_SALARY_SINGLE_RE = re.compile(r'\$([\d,]+)\s*(k)?')
_SALARY_CONTEXT_RE = re.compile(
    r'(?:salary|base pay|compensation|annual pay|pay range|total pay)'
)


def _extract_salary_range(text: str) -> Optional[Tuple[float, float]]:
    """Return (min, max) from an explicit salary range like $80k–$120k.

    Only uses values that form an actual range to avoid picking up
    bonus, equity, or signing-bonus figures as salary anchors.
    Falls back to a single dollar amount near a salary keyword.
    """
    # Primary: explicit range pattern $X–$Y
    for m in _SALARY_RANGE_RE.finditer(text):
        try:
            lo = float(m.group(1).replace(',', '')) * (1000 if m.group(2) else 1)
            hi = float(m.group(3).replace(',', '')) * (1000 if m.group(4) else 1)
            if lo >= 30_000 and hi >= 30_000:
                return lo, hi
        except ValueError:
            pass

    # Fallback: single salary figure within 80 chars of a salary keyword
    for kw in _SALARY_CONTEXT_RE.finditer(text):
        window = text[max(0, kw.start() - 10): kw.start() + 80]
        for m in _SALARY_SINGLE_RE.finditer(window):
            try:
                raw = float(m.group(1).replace(',', '')) * (1000 if m.group(2) else 1)
                if raw >= 30_000:
                    return raw, raw
            except ValueError:
                pass

    return None


@dataclass
class FilterResult:
    passed: bool
    reason: str
    score_override: Optional[int] = None


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_INTERNSHIP_SIGNALS = (
    "intern", "internship", "co-op", "co op", "coop", "summer analyst",
    "industrial placement", "working student", "praktikum",
)


def classify_job_type(title: str, description: str = "") -> str:
    """Return 'internship' or 'full_time' from the title/description text.
    Title is weighted strongly; description only confirms when the title is
    ambiguous (so a full-time JD that merely mentions an internship program
    isn't misclassified)."""
    t = (title or "").lower()
    if any(s in t for s in _INTERNSHIP_SIGNALS):
        return "internship"
    d = (description or "").lower()[:600]   # only the opening lines
    if any(s in d for s in ("intern position", "internship position", "this internship",
                            "summer intern", "co-op position", "is an internship")):
        return "internship"
    return "full_time"


class RuleFilter:
    """Rule-based pre-filter.

    Per-user when given a ``profile`` (years of experience, salary band, skills,
    sponsorship need drive the thresholds); falls back to the original
    single-user defaults when no profile is supplied, so existing callers and
    local/dev runs behave exactly as before.
    """

    def __init__(self, profile=None):
        self.profile = profile
        legacy = profile is None

        # Candidate experience → drives the "requires N+ years" gap filter and
        # whether senior/staff titles are filtered out.
        self.cand_years = _safe_int(getattr(profile, "years_experience", None), 5)
        self.block_senior_titles = legacy or self.cand_years < 6

        # Salary band: only filter on a bound the user actually expressed. With
        # no profile we keep the original $80k–$150k targeting band.
        smin = _safe_int(getattr(profile, "salary_min", None), 0)
        smax = _safe_int(getattr(profile, "salary_max", None), 0)
        self.salary_floor = smin if smin > 0 else (_SALARY_TOO_LOW_MAX if legacy else None)
        self.salary_ceiling = smax if smax > 0 else (_SALARY_TOO_HIGH_MIN if legacy else None)

        # Only block jobs that refuse sponsorship when the user needs it. A
        # citizen / green-card holder should NOT lose "must be US citizen" roles.
        self.requires_sponsorship = True if legacy else bool(getattr(profile, "requires_sponsorship", False))

        # Skills the user actually has → don't reject roles that need them.
        self.user_skills = (getattr(profile, "key_skills", "") or "").lower()
        self.user_degree = (getattr(profile, "degree", "") or "").lower()

        # Job-type preference: "full_time" | "internship" | "both". Only enforced
        # when a profile is present (legacy single-user runs are unfiltered).
        self.job_type_pref = None if legacy else (getattr(profile, "job_type_preference", "full_time") or "full_time")
        # Internships are surfaced ONLY when the user explicitly opts in via the
        # discovery toggle (or an "internship" job-type preference). "both" or an
        # unset preference does NOT silently pull in internships.
        self.enforce_job_type = not legacy
        self.include_internships = False if legacy else bool(
            getattr(profile, "include_internships_in_discovery", False))

    def _has_skill(self, *needles: str) -> bool:
        return any(n in self.user_skills for n in needles)

    def filter(self, job: Job) -> FilterResult:
        desc_low = job.description.lower()
        title_low = job.title.lower()
        loc_low = (job.location or "").lower()

        # 0. Job-type preference (students: internship vs full-time).
        if self.enforce_job_type:
            jtype = classify_job_type(job.title, job.description)
            internships_wanted = self.include_internships or self.job_type_pref == "internship"
            # Internships only appear when the user opted in — fixes internships
            # leaking in for users who never asked for them.
            if jtype == "internship" and not internships_wanted:
                return FilterResult(
                    passed=False,
                    reason="Internship filtered: user did not opt into internships",
                    score_override=10,
                )
            # An internship-only seeker shouldn't get full-time roles.
            if self.job_type_pref == "internship" and jtype != "internship":
                return FilterResult(
                    passed=False,
                    reason="Full-time filtered: user wants internships only",
                    score_override=10,
                )

        # 1. Non-US Location Filter — skip entirely for remote jobs since
        #    "US/Canada Remote" or "Remote (EU)" are still valid remote roles.
        if not job.remote:
            if loc_low:
                for loc in NON_US_LOCATIONS:
                    pattern = rf"\b{re.escape(loc)}\b"
                    if re.search(pattern, loc_low):
                        return FilterResult(
                            passed=False,
                            reason=f"Location pre-filtered: job location '{job.location}' matches '{loc}' (outside the US)",
                            score_override=10
                        )
            else:
                # If location is empty, check title for explicit non-US tags
                for loc in NON_US_LOCATIONS:
                    pattern = rf"\b{re.escape(loc)}\b"
                    if re.search(pattern, title_low):
                        return FilterResult(
                            passed=False,
                            reason=f"Location pre-filtered: title '{job.title}' indicates outside the US ('{loc}')",
                            score_override=10
                        )

        # 2. Work Authorization / Sponsorship Blocker — only relevant when the
        #    user actually needs sponsorship. Citizens / GC holders keep these jobs.
        if self.requires_sponsorship:
            for pattern in NO_SPONSORSHIP_PATTERNS:
                if pattern in desc_low:
                    return FilterResult(
                        passed=False,
                        reason=f"Sponsorship pre-filtered: matches '{pattern}'",
                        score_override=10
                    )

        # 3. Experience Gap Filter — only reject when the JD *requires* (not merely
        #    prefers) well beyond the candidate's experience (their years + 4).
        #    Gap is +4 (not +2) to avoid blocking stretch roles.
        _exp_cutoff = self.cand_years + 4
        _preferred_words = ("preferred", "nice to have", "plus", "ideally", "bonus")
        for m in re.finditer(r'(\d+)\+?\s*years?', desc_low):
            years = int(m.group(1))
            context = desc_low[max(0, m.start() - 20): m.start() + 80]
            if 'experience' in context and years >= _exp_cutoff:
                # Don't reject if this is a "preferred" mention, not a requirement
                if any(w in context for w in _preferred_words):
                    continue
                return FilterResult(
                    passed=False,
                    reason=f"Experience pre-filtered: requires {years}+ years (candidate has {self.cand_years})",
                    score_override=15
                )

        # 4. Senior/staff titles — only filtered for non-senior candidates.
        if self.block_senior_titles:
            for t in STAFF_TITLES:
                if title_low.startswith(t) or f" {t}" in title_low:
                    return FilterResult(
                        passed=False,
                        reason=f"Title pre-filtered: '{job.title}' is a senior/staff-level role",
                        score_override=15
                    )

        # 5. Salary Range Filter — only enforce a bound the user expressed.
        sal_range = _extract_salary_range(desc_low)
        if sal_range:
            min_sal, max_sal = sal_range
            if self.salary_ceiling is not None and min_sal >= self.salary_ceiling:
                return FilterResult(
                    passed=False,
                    reason=f"Salary too high: starts at ${min_sal:,.0f} (target ceiling ${self.salary_ceiling:,.0f})",
                    score_override=20
                )
            if self.salary_floor is not None and max_sal <= self.salary_floor:
                return FilterResult(
                    passed=False,
                    reason=f"Salary too low: up to ${max_sal:,.0f} (target floor ${self.salary_floor:,.0f})",
                    score_override=20
                )

        # 6. Hire-probability filter — block roles the candidate can't credibly
        #    fill, but ONLY when the required skill isn't in their stack.
        #    a) Low-level systems / GPU kernel engineering
        systems_signals = [
            "cuda kernel", "gpu kernel", "write cuda", "triton kernel",
            "systems programming", "kernel developer", "kernel engineer",
            "bare metal", "memory allocator", "compiler engineer", "llvm", "mlir",
        ]
        if not self._has_skill("cuda", "gpu", "kernel", "compiler", "llvm", "systems programming"):
            if any(s in desc_low for s in systems_signals):
                return FilterResult(
                    passed=False,
                    reason="Hire-probability: GPU/kernel/compiler systems role — not in candidate stack",
                    score_override=12
                )

        #    b) C++ or Rust listed as a hard requirement (not nice-to-have)
        if not self._has_skill("c++", "cpp", "rust"):
            cpp_rust_required = [
                "c++ required", "proficiency in c++", "strong c++", "expert in c++",
                "rust required", "proficiency in rust", "strong rust", "expert in rust",
                "primary language is c++", "primary language is rust",
            ]
            if any(pat in desc_low for pat in cpp_rust_required):
                return FilterResult(
                    passed=False,
                    reason="Hire-probability: C++/Rust listed as required — not in candidate stack",
                    score_override=12
                )

        #    c) Pure research / PhD roles — skip the block for users who hold a PhD.
        if "phd" not in self.user_degree and "doctor" not in self.user_degree:
            research_signals = [
                "phd required", "phd preferred", "doctoral degree required",
                "publishing research", "publish original research",
                "first-author publication", "neurips", "icml", "iclr publication",
            ]
            if sum(1 for s in research_signals if s in desc_low) >= 2:
                return FilterResult(
                    passed=False,
                    reason="Hire-probability: pure research role requiring publications/PhD",
                    score_override=12
                )

        return FilterResult(passed=True, reason="Passed all rule filters")
