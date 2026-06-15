"""SQLModel schema. Single source of truth for job + application state."""
from __future__ import annotations

from datetime import datetime
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
    BAMBOOHR = "bamboohr"
    ICIMS = "icims"
    JOBVITE = "jobvite"
    COMEET = "comeet"
    TEAMTAILOR = "teamtailor"
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
    SKIPPED = "skipped"                  # user declined
    ERROR = "error"


class Job(SQLModel, table=True):
    """A single job posting. external_id + source uniquely identifies."""
    __table_args__ = (
        UniqueConstraint("source", "external_id", name="uq_job_source_external_id"),
    )
    id: Optional[int] = Field(default=None, primary_key=True)
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
    content_hash: Optional[str] = Field(default=None, index=True)
    # Matching outputs
    embedding_id: Optional[int] = Field(default=None, index=True)  # FAISS index position
    similarity_score: Optional[float] = None
    rerank_score: Optional[float] = None
    rerank_reasoning: Optional[str] = None
    cross_source_slug: Optional[str] = Field(default=None, index=True)

    # Ghost job detection score (0.0 = definitely real, 1.0 = likely ghost)
    ghost_score: float = Field(default=0.0)
    ghost_flags: Optional[str] = Field(default=None)  # JSON list of flag strings
    # Hire probability score (0.0 = company not actively hiring, 1.0 = strong hiring intent)
    hire_probability_score: Optional[float] = Field(default=None)
    hire_probability_signals: Optional[str] = Field(default=None)  # JSON list
    # Blended final score combining rerank fit + hire probability
    blended_score: Optional[float] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True


class Application(SQLModel, table=True):
    """One application per (job, attempt). Tracks lifecycle."""
    id: Optional[int] = Field(default=None, primary_key=True)
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
        UniqueConstraint("label_normalized", name="uq_answer_memory_label"),
    )
    id: Optional[int] = Field(default=None, primary_key=True)
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
    sponsorship_signal: Optional[str] = Field(default=None)
    last_error: Optional[str] = Field(default=None)
    inactive_reason: Optional[str] = Field(default=None)
    next_retry_at: Optional[datetime] = Field(default=None, index=True)


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

