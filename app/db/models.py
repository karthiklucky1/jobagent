"""SQLModel schema. Single source of truth for job + application state."""
from __future__ import annotations

from datetime import datetime, date
from enum import Enum
from typing import Optional

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


class JobSource(str, Enum):
    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    ASHBY = "ashby"
    WORKDAY = "workday"
    SMARTRECRUITERS = "smartrecruiters"
    WORKABLE = "workable"
    RECRUITEE = "recruitee"           # Recruitee public offers API — EU-heavy
    PERSONIO = "personio"             # Personio public XML feed — DACH-heavy
    BAMBOOHR = "bamboohr"
    ICIMS = "icims"
    JOBVITE = "jobvite"
    COMEET = "comeet"
    TEAMTAILOR = "teamtailor"
    RIPPLING = "rippling"             # Rippling public ATS board API
    BREEZY = "breezy"                 # Breezy HR public JSON
    PINPOINT = "pinpoint"             # Pinpoint public postings.json
    JOIN = "join"                     # join.com public 2-step API — EU-heavy
    WELLFOUND = "wellfound"
    OTTA = "otta"
    MANUAL = "manual"
    # Aggregator / board sources
    LINKEDIN = "linkedin"
    INDEED = "indeed"
    SERPAPI = "serpapi"    # Google Jobs via SerpAPI (LinkedIn/Indeed/Glassdoor)
    REMOTIVE = "remotive"
    REMOTEOK = "remoteok"
    THEMUSE = "themuse"               # The Muse public API — free, no key
    ARBEITNOW = "arbeitnow"           # Arbeitnow job-board API — free, no key
    JOBICY = "jobicy"                 # Jobicy public API — free, no key
    WEWORKREMOTELY = "weworkremotely" # WeWorkRemotely RSS feeds — free, no key
    ADZUNA = "adzuna"                 # Adzuna API — free, 50 searches/day
    REED = "reed"                     # Reed.co.uk API — free, 5000 calls/day
    JOOBLE = "jooble"                 # Jooble API — free, 500 calls/day
    INDEEDRSS = "indeed_rss"          # Indeed public RSS — no key, ~10/query
    CROWDSOURCED = "crowdsourced"     # Crowdsourced via browser extension


class ApplicationStatus(str, Enum):
    DISCOVERED = "discovered"           # just scraped
    MATCHED = "matched"                  # passed similarity threshold
    SHORTLISTED = "shortlisted"          # passed Claude rerank
    TAILORED = "tailored"                # resume + cover letter generated
    AUTOFILLED = "autofilled"            # form filled, awaiting user
    AWAITING_USER = "awaiting_user"      # Telegram prompt pending
    READY_TO_SUBMIT = "ready_to_submit"  # all fields filled, preview link sent
    SUBMITTED = "submitted"              # applicant clicked submit
    REJECTED = "rejected"                # heard back: no
    INTERVIEWING = "interviewing"
    OFFER = "offer"                      # received an offer
    ACCEPTED = "accepted"               # accepted an offer (hired)
    SKIPPED = "skipped"                  # user declined
    ERROR = "error"


class Job(SQLModel, table=True):
    """A single job posting, scoped per user.

    Multi-tenant: each user gets their OWN copy of a posting so that per-user
    match scores (similarity/rerank/blended) and lifecycle status never collide
    across tenants. Uniqueness is therefore (user_id, source, external_id).
    """
    __table_args__ = (
        UniqueConstraint("user_id", "source", "external_id", name="uq_job_user_source_external_id"),
    )
    id: Optional[int] = Field(default=None, primary_key=True)
    # Multi-tenant: Supabase user UUID. NULL = legacy single-user SQLite row.
    user_id: Optional[str] = Field(default=None, index=True)
    source: JobSource
    external_id: str = Field(index=True)
    company: str
    title: str
    location: str = ""
    remote: bool = False
    url: str
    description: str = ""
    posted_at: Optional[datetime] = None
    discovered_at: datetime = Field(default_factory=datetime.utcnow)
    # New lifecycle and tracking fields
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    is_closed: bool = Field(default=False)
    closed_reason: Optional[str] = Field(default=None)
    content_hash: Optional[str] = Field(default=None, index=True)
    # Matching outputs
    embedding_id: Optional[int] = Field(default=None, index=True)  # FAISS index position
    similarity_score: Optional[float] = None
    rerank_score: Optional[float] = None
    rerank_reasoning: Optional[str] = None
    # JSON: per-factor breakdown {skills,experience,location,work_auth:{score,note}}
    rerank_breakdown: Optional[str] = Field(default=None)
    corporate_insights: Optional[str] = Field(default=None)  # JSON: pain point, reporting line, culture decode, leverage hook, salary/work model
    cross_source_slug: Optional[str] = Field(default=None, index=True)

    # Ghost job detection score (0.0 = definitely real, 1.0 = likely ghost)
    ghost_score: float = Field(default=0.0)
    ghost_flags: Optional[str] = Field(default=None)  # JSON list of flag strings
    # Hire probability score (0.0 = company not actively hiring, 1.0 = strong hiring intent)
    hire_probability_score: Optional[float] = Field(default=None)
    hire_probability_signals: Optional[str] = Field(default=None)  # JSON list
    # Blended final score combining rerank fit + hire probability
    blended_score: Optional[float] = Field(default=None)

    # Classified job type: "full_time" | "internship" (set during matching).
    job_type: str = Field(default="full_time")
    # Visa intelligence (persisted so we can filter/query, not just display).
    is_cap_exempt: bool = Field(default=False)
    urgency_score: float = Field(default=0.0)

    class Config:
        arbitrary_types_allowed = True


class Application(SQLModel, table=True):
    """One application per (job, attempt). Tracks lifecycle."""
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[str] = Field(default=None, index=True)
    job_id: int = Field(foreign_key="job.id", index=True)
    status: ApplicationStatus = ApplicationStatus.DISCOVERED
    tailored_resume_path: Optional[str] = None
    cover_letter_path: Optional[str] = None
    apply_url: Optional[str] = None  # may differ from job.url after redirects
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    notes: Optional[str] = None
    # "autofill" = Greenhouse/Lever/Ashby (bot fills the form)
    # "manual"   = LinkedIn/Indeed/etc. (bot preps materials, human applies)
    apply_track: str = Field(default="autofill")

    # Instrumentation fields
    resume_variant: Optional[str] = None  # e.g., "variant_a", "variant_b"
    profile_variant: Optional[str] = None  # "backend" | "ai_agents" | "fullstack"
    senior_fit_score: Optional[float] = None  # conservative score from SeniorReviewer (0-100)
    senior_verdict: Optional[str] = None  # 2-sentence blunt verdict from SeniorReviewer
    custom_highlight_block: Optional[str] = None  # 3-bullet markdown framing missing JD gaps
    response_type: str = Field(default="none")  # none, auto_rejected, screening, phone_screen, interview, offer
    rejection_analysis: Optional[str] = None  # JSON string containing forensic analysis (reason, gaps, tip)


class PendingQuestion(SQLModel, table=True):
    """A question the autofill agent needs answered via Telegram."""
    id: Optional[int] = Field(default=None, primary_key=True)
    application_id: int = Field(foreign_key="application.id", index=True)
    field_label: str            # e.g. "Years of experience with PyTorch"
    field_selector: str         # CSS selector or DOM ref
    field_type: str             # text, select, radio, file, etc.
    options: Optional[str] = None  # JSON list for select/radio
    answer: Optional[str] = None
    asked_at: datetime = Field(default_factory=datetime.utcnow)
    answered_at: Optional[datetime] = None


class AnswerMemory(SQLModel, table=True):
    """Cached answers to common application questions, keyed by normalized label.
    Personal answer memory for repeated application fields.
    """
    __table_args__ = (
        UniqueConstraint("user_id", "label_normalized", name="uq_answer_memory_user_label"),
    )
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[str] = Field(default=None, index=True)
    label_normalized: str = Field(index=True)
    label_original: str
    answer: str
    last_used_at: datetime = Field(default_factory=datetime.utcnow)
    use_count: int = 1


class CompanyRegistry(SQLModel, table=True):
    """Registry of harvested company slugs for Greenhouse, Lever, and Ashby."""
    __table_args__ = (
        UniqueConstraint("slug", "ats", name="uq_slug_ats"),
    )
    id: Optional[int] = Field(default=None, primary_key=True)
    slug: str = Field(index=True)
    ats: JobSource = Field(index=True)
    company_name: Optional[str] = Field(default=None)
    career_url: Optional[str] = Field(default=None)
    source: str = Field(default="seed")  # e.g., seed, common_crawl, yc_startup, dork
    confidence_score: int = Field(default=100, index=True)
    target_fit_score: float = Field(default=0.0, index=True)
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: Optional[datetime] = Field(default=None, index=True)
    last_validated_at: Optional[datetime] = Field(default=None)
    is_active: bool = Field(default=True, index=True)
    job_count: int = Field(default=0)
    failure_count: int = Field(default=0)
    # Yield tracking — when this board last produced a NEW posting (not just
    # re-served old ones). The hot lane prioritizes boards that actually post.
    new_jobs_last_poll: int = Field(default=0)
    last_new_job_at: Optional[datetime] = Field(default=None, index=True)
    sponsorship_signal: Optional[str] = Field(default=None)
    last_error: Optional[str] = Field(default=None)
    inactive_reason: Optional[str] = Field(default=None)
    next_retry_at: Optional[datetime] = Field(default=None, index=True)
    # Pulse lane scheduling — when this board is next due for a poll, and a
    # signature of its last-seen posting list (skip all downstream work when
    # the board hasn't changed).
    next_poll_at: Optional[datetime] = Field(default=None, index=True)
    poll_hash: Optional[str] = Field(default=None)


class UserProfile(SQLModel, table=True):
    """One row per user. Stores all fields needed to fill any job application form."""
    id: Optional[int] = Field(default=None, primary_key=True)
    # Supabase user UUID — links this profile to the authenticated user
    user_id: Optional[str] = Field(default=None, index=True, unique=True)
    # Identity
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin_url: str = ""
    github_url: str = ""
    portfolio_url: str = ""
    # Work authorization
    work_authorization: str = ""        # e.g. "US Citizen", "OPT", "H1B"
    requires_sponsorship: bool = False
    visa_status: str = ""               # free-text for edge cases
    # Professional
    current_title: str = ""
    years_experience: int = 0
    salary_min: int = 0
    salary_max: int = 0
    salary_currency: str = "USD"
    # Education (legacy single-entry fields — kept in sync with the FIRST entry
    # of education_json so autofill/matching keep working unchanged)
    degree: str = ""
    university: str = ""
    graduation_year: Optional[int] = None
    # Structured multi-entry history (JSON arrays; edited in the profile UI).
    # education_json:  [{"degree","field","university","start_year","end_year","gpa"}, ...]
    # experience_json: [{"title","company","location","start","end","current","summary"}, ...]
    education_json: Optional[str] = None
    experience_json: Optional[str] = None
    # EEOC (decline to answer by default — safest)
    gender: str = "Decline to self-identify"
    ethnicity: str = "Decline to self-identify"
    veteran_status: str = "I am not a protected veteran"
    disability_status: str = "No, I do not have a disability, or history/record of having a disability"
    # Free-text bio used to generate essay answers
    professional_summary: str = ""
    key_skills: str = ""                # comma-separated
    # Job titles the user wants us to search & rank for — comma-separated.
    # Separate from current_title: a user can target roles different from their
    # current one (e.g. an analyst targeting "Data Scientist"). Drives discovery.
    target_roles: str = ""              # comma-separated
    # Companies the user follows ("My Companies") — comma-separated. Boards for
    # these companies are polled on the pulse lane's fast (minutes) cadence.
    target_companies: str = ""          # comma-separated
    # ── Student / work-authorization / department preferences ────────────────
    # "full_time" | "internship" | "both" — drives job-type filtering.
    job_type_preference: str = "full_time"
    # Free-text work-authorization category: "OPT" | "CPT" | "STEM OPT" |
    # "H1B" | "Citizen" | "Green Card" | ... (complements work_authorization).
    work_auth_status: str = ""
    # When true, discovery also searches for internships even if target roles
    # are full-time titles (useful for OPT/CPT students).
    include_internships_in_discovery: bool = False
    # Department / industry for non-CS fields (e.g. "Civil Engineering").
    # Drives role suggestions and the discovery keyword fallback.
    industry: str = ""
    # ── Location preferences (drive discovery + scoring) ──────────────────────
    # Country the user wants jobs in. Discovery + reranker filter to this country
    # (plus remote when remote_ok). Defaults to US to preserve legacy behavior.
    preferred_country: str = "United States"
    # When true, fully-remote roles are kept even if located in another country.
    remote_ok: bool = True
    # ── Referral program ─────────────────────────────────────────────────────
    referral_code: Optional[str] = Field(default=None, index=True)   # this user's own code
    referred_by_id: Optional[str] = Field(default=None, index=True)  # user_id who referred them
    # ── Trust Profile (Phase 0) — evidence-based professional identity ─────────
    # Five 0-100 dimensions (shown as star levels, not one magic number) plus an
    # overall tier label and a JSON evidence graph ("why we believe this").
    account_type: str = "candidate"     # "candidate" | "recruiter" — drives role separation
    email_verified: bool = False
    phone_verified: bool = False
    public_handle: Optional[str] = Field(default=None, index=True)   # hirepath.dev/u/<handle>
    # Work Readiness Passport — recruiter-facing "can they start?" answers
    availability: str = ""              # "Immediately" | "2 weeks" | "Interviewing" | "Not looking"
    open_to_relocation: bool = False
    # Articulation proof (optional booster) — short video explaining own real PR
    articulation_video_url: str = ""
    articulation_pr: str = ""           # which PR/repo the video explains
    trust_identity_score: int = 0          # email/phone/edu verification
    trust_technical_score: int = 0         # GitHub repos/commits/PRs (harvester)
    trust_consistency_score: int = 0       # resume <-> reality (grounding)
    trust_activity_score: int = 0          # open-source recency/volume
    trust_completeness_score: int = 0      # profile fields filled
    trust_tier: str = ""                   # "" | Starter | Verified | Verified Pro | Elite
    trust_evidence: Optional[str] = None   # JSON: per-dimension evidence items
    resume_grounded_ratio: Optional[float] = None  # real grounding ratio (set by tailoring); None = not yet checked
    trust_computed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class PlanTier(str, Enum):
    FREE    = "free"
    BASIC   = "basic"      # $19/mo
    PRO     = "pro"        # $49/mo
    AGENCY  = "agency"     # $99/mo


# Per-plan limits
PLAN_LIMITS = {
    PlanTier.FREE:   {"tailor_daily": 5,  "autofill_weekly": 2},
    PlanTier.BASIC:  {"tailor_daily": 20, "autofill_weekly": 10},
    PlanTier.PRO:    {"tailor_daily": None, "autofill_weekly": None},   # None = unlimited
    PlanTier.AGENCY: {"tailor_daily": None, "autofill_weekly": None},
}

PLAN_PRICES = {
    PlanTier.FREE:   0,
    PlanTier.BASIC:  19,
    PlanTier.PRO:    49,
    PlanTier.AGENCY: 99,
}


class UserSubscription(SQLModel, table=True):
    """One row per user — tracks their current plan."""
    __tablename__ = "user_subscription"
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True, unique=True)
    plan: PlanTier = Field(default=PlanTier.FREE)
    stripe_customer_id: Optional[str] = Field(default=None)
    stripe_subscription_id: Optional[str] = Field(default=None)
    current_period_end: Optional[datetime] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class UserUsage(SQLModel, table=True):
    """Daily + weekly usage counters per user. One row per user per day."""
    __tablename__ = "user_usage"
    __table_args__ = (
        UniqueConstraint("user_id", "usage_date", name="uq_user_usage_date"),
    )
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True)
    usage_date: date = Field(index=True)          # the calendar day (UTC)
    week_start: date = Field(index=True)          # Monday of the current week
    tailor_count: int = Field(default=0)          # tailors used today
    autofill_count_week: int = Field(default=0)   # autofills used this week (stored on Monday row)


class FunnelEvent(SQLModel, table=True):
    """Event log for job application funnel analysis."""
    __tablename__ = "funnel_events"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: Optional[int] = Field(default=None, foreign_key="job.id", index=True)
    stage: str = Field(index=True)  # discovered, rule_filtered, embedding_filtered, scored, shortlisted, tailored, applied, responded
    passed: bool
    reason: Optional[str] = None
    metadata_json: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DiscoveryRun(SQLModel, table=True):
    """Per-source summary of one discovery run, for visibility into where jobs come from."""
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[str] = Field(default=None, index=True)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None
    total_fetched: int = 0
    total_inserted: int = 0
    total_shortlisted: int = 0
    source_counts: str = ""   # JSON: {"<source name>": {"fetched": n, "error": "..."}}
    status: str = "discovering"   # discovering | ranking | done | error
    error: Optional[str] = None


class UserPersonalMemory(SQLModel, table=True):
    """Per-user 'recruiter memory' — weekly harvested facts from the user's own
    GitHub / LinkedIn, plus LLM-written recruiter recommendations.

    Multi-tenant: always scoped by user_id. Stores only the user's OWN public
    profile data (self-harvest), never third parties.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[str] = Field(default=None, index=True)
    source: str = Field(default="github")      # "github" | "linkedin"
    raw_content: str = ""                       # raw harvested text/JSON
    parsed_updates: str = ""                    # JSON: structured deltas
    recommendations: str = ""                   # LLM recruiter brief (markdown)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TrialGrant(SQLModel, table=True):
    """Founding-user trial: the first N users get a budget of fully processed
    jobs (discover → match → tailor → apply) with all Pro features unlocked.
    One grant per user_id."""
    __table_args__ = (
        UniqueConstraint("user_id", name="uq_trial_user"),
    )
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True)
    jobs_quota: int = 100
    jobs_used: int = 0
    granted_at: datetime = Field(default_factory=datetime.utcnow)


class H1BSponsor(SQLModel, table=True):
    """Global public-record sponsor registry. NOT tenant-scoped.

    Started as the USCIS H-1B Employer Data Hub table (hence the name — kept
    for migration stability) and now holds one row per employer per country:
    - country="united states", record_type="stats": USCIS approval/denial stats
    - other countries, record_type="license": licensed-sponsor registers
      (UK Register of Licensed Sponsors, Canada LMIA employers, NL IND public
      register, etc.) where being listed means "authorised to sponsor".
    Populated by app/intelligence/h1b_data.py from the public CSVs.
    """
    __table_args__ = (
        UniqueConstraint("employer_key", "fiscal_year", "country",
                         name="uq_h1b_employer_year_country"),
    )
    id: Optional[int] = Field(default=None, primary_key=True)
    employer_key: str = Field(index=True)       # normalized lowercase name
    employer_name: str = ""                      # display name
    fiscal_year: Optional[int] = Field(default=None)
    approvals: int = 0
    denials: int = 0
    approval_rate: float = 0.0                    # approvals / (approvals+denials)
    typical_wage_level: str = ""                 # e.g. "Level II"
    is_cap_exempt: bool = False
    country: str = Field(default="united states", index=True)  # geo.norm_country form
    record_type: str = "stats"                   # "stats" (numbers) | "license" (listed = can sponsor)
    detail: str = ""                             # e.g. UK routes "Skilled Worker", LMIA stream
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class RecruiterProfile(SQLModel, table=True):
    """A demand-side account — recruiter / staffing vendor / client.

    Verified by corporate-domain match + (optional) public H-1B filing history,
    with a hard "never charge candidates" rule. This is the side that *pulls*
    from the verified candidate pool (the moat).
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[str] = Field(default=None, index=True, unique=True)  # Supabase auth user
    full_name: str = ""
    work_email: str = ""                  # must match company_domain to verify
    company_name: str = ""
    company_domain: str = ""              # e.g. "acme.com"
    title: str = ""                       # e.g. "Senior Technical Recruiter"
    specialties: str = ""                 # comma-separated roles they place
    # Verification
    verified: bool = False
    verification_notes: str = ""
    h1b_filings: int = 0                  # cached from public data (sponsorship cred)
    # Trust / safety
    charges_candidates: bool = False      # if ever true -> banned (illegal)
    banned: bool = False
    placements: int = 0                   # self-reported / platform-tracked
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class CandidateIntro(SQLModel, table=True):
    """A recruiter's request to connect with a candidate. Candidate-controlled:
    the recruiter requests, the candidate accepts before contact opens (no
    resume dump). Phase 3 adds messaging on top of this."""
    id: Optional[int] = Field(default=None, primary_key=True)
    recruiter_user_id: Optional[str] = Field(default=None, index=True)
    candidate_user_id: Optional[str] = Field(default=None, index=True)
    job_context: str = ""                 # role the recruiter is hiring for
    status: str = "requested"             # requested | accepted | declined
    created_at: datetime = Field(default_factory=datetime.utcnow)


class IntroMessage(SQLModel, table=True):
    """A message exchanged after an intro is accepted. Keeps contact on-platform
    (no email/phone leak) until both sides choose to take it off-platform."""
    id: Optional[int] = Field(default=None, primary_key=True)
    intro_id: int = Field(index=True)
    sender_user_id: Optional[str] = Field(default=None, index=True)
    body: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class IntroRating(SQLModel, table=True):
    """Two-way rating after an interaction. Candidates rate recruiters (real job?
    responsive? honest pay?), recruiters rate candidates (skills real? showed up?).
    Reputation that's earned, not self-reported."""
    id: Optional[int] = Field(default=None, primary_key=True)
    intro_id: int = Field(index=True)
    rater_user_id: Optional[str] = Field(default=None, index=True)
    ratee_user_id: Optional[str] = Field(default=None, index=True)
    stars: int = 0                        # 1-5
    note: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TrustHistory(SQLModel, table=True):
    """Append-only snapshot of a user's Trust Profile over time.

    Written on every recompute so we can show Momentum ("Trust 61 -> 84, +220
    commits in 6 months") later — a signal almost nobody tracks. Tenant-scoped.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[str] = Field(default=None, index=True)
    overall: int = 0
    tier: str = ""
    identity: int = 0
    technical: int = 0
    consistency: int = 0
    activity: int = 0
    completeness: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class UserReferralReward(SQLModel, table=True):
    """A reward unlocked when a referrer reaches the referral threshold."""
    __tablename__ = "user_referral_reward"
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True)            # the referrer who earned it
    referred_count: int = 0                       # count at time of unlock
    status: str = "active"                        # active | claimed | expired
    reward_plan: str = "pro"
    unlocked_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


class UserNotification(SQLModel, table=True):
    """Real-time in-app notifications for users (e.g. discovery runs finished, perfect jobs matched)."""
    __tablename__ = "user_notifications"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[str] = Field(default=None, index=True)
    title: str
    message: str
    type: str = "general"                     # e.g., "discovery_completed", "high_match", "general"
    read: bool = Field(default=False, index=True)
    link: Optional[str] = None                # clickable link in the UI
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Coupon(SQLModel, table=True):
    """Admin-created promo codes that grant free plan days (e.g. LAUNCH50 → 30 days PRO)."""
    __tablename__ = "coupon"
    id: Optional[int] = Field(default=None, primary_key=True)
    code: str = Field(index=True, unique=True)       # e.g. "LAUNCH50" — case-insensitive stored UPPER
    description: str = ""                             # shown to user on redemption
    reward_plan: str = "pro"                          # plan tier to grant
    reward_days: int = 30                             # days of the plan to grant
    max_uses: Optional[int] = None                    # None = unlimited
    uses_count: int = 0
    is_active: bool = True
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None                 # admin user_id


class CouponRedemption(SQLModel, table=True):
    """One row per (coupon, user) pair — prevents double-dipping."""
    __tablename__ = "coupon_redemption"
    id: Optional[int] = Field(default=None, primary_key=True)
    coupon_id: int = Field(index=True)
    user_id: str = Field(index=True)
    redeemed_at: datetime = Field(default_factory=datetime.utcnow)


class UserReview(SQLModel, table=True):
    """Submitted reviews from candidates to showcase on the landing page."""
    __tablename__ = "user_review"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[str] = Field(default=None, index=True)
    user_name: str = Field(default="Anonymous")
    rating: int = Field(default=5)                       # 1-5 stars
    content: str = Field(default="")
    is_public: bool = Field(default=True, index=True)
    is_featured: bool = Field(default=False, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

