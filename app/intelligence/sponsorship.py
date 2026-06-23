"""Sponsorship intelligence — legal, public-data driven.

Assesses whether a company is likely to sponsor a work visa (H-1B), and whether
it is *cap-exempt* (universities / non-profit research / affiliated hospitals),
which can sponsor H-1Bs year-round with NO lottery.

The curated seed lists below are a stand-in for the public USCIS H-1B Employer
Data Hub and DOL OFLC LCA disclosure data — both freely downloadable. Drop a CSV
of {employer -> approvals} at ``settings.h1b_employer_csv`` and ``load_employer_hub``
will override the curated list with real numbers. Nothing here scrapes private
data; it only reasons over the posting text + public sponsorship records.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger(__name__)


class SponsorshipLikelihood(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


# ── Public-record-informed seed lists ────────────────────────────────────────
# Consistent top H-1B sponsors per public USCIS/DOL disclosure data (big tech,
# the major consultancies/IT services, and banks that file in volume every year).
KNOWN_SPONSORS = {
    # Big tech / product
    "google", "alphabet", "meta", "facebook", "amazon", "apple", "microsoft",
    "netflix", "nvidia", "intel", "qualcomm", "oracle", "salesforce", "adobe",
    "ibm", "cisco", "vmware", "uber", "lyft", "airbnb", "stripe", "block",
    "paypal", "linkedin", "snap", "pinterest", "doordash", "instacart",
    "databricks", "snowflake", "palantir", "twilio", "workday", "servicenow",
    "atlassian", "dropbox", "tesla", "spacex", "bloomberg", "intuit", "ebay",
    "walmart", "capital one", "visa", "mastercard",
    # Major consultancies / IT services (highest-volume sponsors)
    "deloitte", "accenture", "cognizant", "infosys", "tata consultancy", "tcs",
    "wipro", "capgemini", "hcl", "tech mahindra", "ernst & young", "pwc",
    "pricewaterhousecoopers", "kpmg", "mckinsey", "boston consulting",
    "ltimindtree", "mphasis", "persistent systems", "epam",
    # Banks / finance
    "jpmorgan", "jp morgan", "goldman sachs", "morgan stanley", "citigroup",
    "citi", "bank of america", "wells fargo", "american express", "blackrock",
    "two sigma", "citadel", "jane street", "de shaw",
}

# Defense / government-linked employers that typically require US persons.
KNOWN_NON_SPONSORS = {
    "lockheed", "raytheon", "rtx", "boeing", "northrop", "general dynamics",
    "l3harris", "leidos", "booz allen", "saic", "draper", "mitre", "anduril",
    "palantir usg", "caci", "peraton",
}

# Cap-exempt signals: institution of higher ed, non-profit research, hospitals.
CAP_EXEMPT_NAME_SIGNALS = (
    "university", "college", "institute of technology", "polytechnic",
    "school of medicine", "medical center", "medical school", "health system",
    "hospital", "research institute", "research center", "national laboratory",
    "national lab", "cancer center", "children's hospital", "state university",
)
CAP_EXEMPT_DESC_SIGNALS = (
    "cap-exempt", "cap exempt", "h-1b cap-exempt", "h1b cap exempt",
    "institution of higher education", "non-profit research", "nonprofit research",
)

# Kept in sync with app/matching/filters/constants.py (inlined to keep this
# module import-light — the filters package pulls in numpy via its __init__).
NO_SPONSORSHIP_PATTERNS = [
    "not offer visa sponsorship", "unable to sponsor", "do not sponsor",
    "will not sponsor", "cannot sponsor", "no visa sponsorship", "no sponsorship",
    "does not sponsor", "must be us citizen", "us citizen or permanent resident",
    "us citizenship required", "active security clearance required",
    "must hold an active secret", "must possess an active ts/sci",
]


@dataclass
class SponsorshipAssessment:
    likelihood: SponsorshipLikelihood
    cap_exempt: bool
    reason: str
    badge: str                 # short UI label
    explicitly_refuses: bool = False

    @property
    def tone(self) -> str:
        """UI colour hint."""
        if self.explicitly_refuses or self.likelihood == SponsorshipLikelihood.LOW:
            return "bad"
        if self.cap_exempt or self.likelihood == SponsorshipLikelihood.HIGH:
            return "good"
        return "unknown"


def _norm(s) -> str:
    return (s or "").lower().strip()


def _is_cap_exempt(name: str, desc: str, url: str) -> bool:
    if any(sig in name for sig in CAP_EXEMPT_NAME_SIGNALS):
        return True
    if any(sig in desc for sig in CAP_EXEMPT_DESC_SIGNALS):
        return True
    # .edu domains are institutions of higher education
    host = url.split("//")[-1].split("/")[0]
    if host.endswith(".edu") or ".edu" in host:
        return True
    return False


def assess(company: str = "", description: str = "", url: str = "") -> SponsorshipAssessment:
    """Return a legal, explainable sponsorship assessment for a posting."""
    name, desc, u = _norm(company), _norm(description), _norm(url)

    cap_exempt = _is_cap_exempt(name, desc, u)
    explicitly_refuses = any(p in desc for p in NO_SPONSORSHIP_PATTERNS)

    # Cap-exempt is the strongest positive — overrides a generic refusal phrase
    # only when the employer truly is an institution of higher ed / non-profit.
    if cap_exempt:
        return SponsorshipAssessment(
            SponsorshipLikelihood.HIGH, True,
            "Cap-exempt employer (university / non-profit research / hospital) — "
            "can sponsor H-1B year-round with no lottery.",
            "No-lottery sponsor",
        )

    # Hard public-record override: if we've ingested USCIS H-1B data and this
    # employer is in it, use the real approval numbers (data beats curated lists).
    try:
        from app.intelligence.h1b_data import lookup as _h1b_lookup
        rec = _h1b_lookup(company)
        if rec and (rec["approvals"] + rec["denials"]) >= 1 and not explicitly_refuses:
            rate = rec["rate"]
            yr = rec["year"] or ""
            if rec["approvals"] >= 5 and rate >= 0.5:
                return SponsorshipAssessment(
                    SponsorshipLikelihood.HIGH, False,
                    f"USCIS record: {rec['approvals']} H-1B approvals"
                    f"{f' (FY{yr})' if yr else ''}, {int(rate*100)}% approval rate.",
                    "Sponsors H-1B",
                )
            if rec["approvals"] >= 1:
                return SponsorshipAssessment(
                    SponsorshipLikelihood.MEDIUM, False,
                    f"USCIS record: {rec['approvals']} H-1B approval(s)"
                    f"{f' (FY{yr})' if yr else ''} — has sponsored before.",
                    "Has sponsored",
                )
    except Exception:
        pass

    if explicitly_refuses:
        return SponsorshipAssessment(
            SponsorshipLikelihood.LOW, False,
            "This posting explicitly states it will not sponsor a work visa.",
            "No sponsorship", explicitly_refuses=True,
        )

    if any(b in name for b in KNOWN_NON_SPONSORS):
        return SponsorshipAssessment(
            SponsorshipLikelihood.LOW, False,
            "Defense / government-linked employer — usually requires US persons.",
            "Rarely sponsors",
        )

    if any(b in name for b in KNOWN_SPONSORS):
        return SponsorshipAssessment(
            SponsorshipLikelihood.HIGH, False,
            "Established H-1B sponsor — files regularly per public USCIS/DOL records.",
            "Sponsors H-1B",
        )

    return SponsorshipAssessment(
        SponsorshipLikelihood.UNKNOWN, False,
        "No public sponsorship record found — worth a quick check before applying.",
        "Sponsorship unknown",
    )


# ── Backward-compatible facade ───────────────────────────────────────────────
class SponsorshipChecker:
    def __init__(self):
        log.info("SponsorshipChecker initialized (curated public-record lists).")

    def check_company(self, company_name: str) -> SponsorshipLikelihood:
        return assess(company=company_name).likelihood
