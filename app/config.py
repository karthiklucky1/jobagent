"""Centralized config loaded from .env."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    tavily_api_key: str = ""
    exa_api_key: str = ""
    magic_api_key: str = ""

    # Job board APIs
    serpapi_key: str = ""            # serpapi.com — Google Jobs (LinkedIn/Indeed/Glassdoor). Free: 100/mo
    remotive_enabled: bool = True    # Remotive public API — no key needed
    remoteok_enabled: bool = True    # RemoteOK public API — no key needed
    hn_whoishiring_enabled: bool = True  # HN monthly "Who is hiring?" thread — no key, early signal
    themuse_enabled: bool = True     # The Muse public API — no key needed; optional THEMUSE_API_KEY for higher rate limit
    themuse_api_key: str = ""        # optional — raises rate limits
    arbeitnow_enabled: bool = True   # Arbeitnow public API — no key needed
    jobicy_enabled: bool = True      # Jobicy public API — no key needed
    weworkremotely_enabled: bool = True  # WeWorkRemotely RSS feeds — no key needed
    indeed_rss_enabled: bool = False  # Indeed killed public RSS + blocks bots (and ToS forbids it) — off by default
    adzuna_enabled: bool = False     # Adzuna API — needs app_id + app_key (free tier)
    adzuna_app_id: str = ""
    adzuna_app_key: str = ""
    reed_enabled: bool = False       # Reed.co.uk API — needs api_key (free tier)
    reed_api_key: str = ""
    jooble_enabled: bool = False     # Jooble API — needs api_key (free tier)
    jooble_api_key: str = ""
    linkedin_rapidapi_enabled: bool = False  # LinkedIn via RapidAPI (~$10/mo)
    rapidapi_key: str = ""

    @property
    def linkedin_rapidapi_active(self) -> bool:
        """Auto-enable LinkedIn RapidAPI when a key is present, like SerpAPI."""
        return self.linkedin_rapidapi_enabled or bool(self.rapidapi_key)
    scrape_company_boards: bool = False  # JOB-FIRST by default: discovery is driven purely by job
                                         # aggregators (SerpAPI/Remotive/RemoteOK/HN), NOT a fixed company list.
                                         # Set True to also scrape the bootstrap company ATS boards
                                         # (Greenhouse/Lever/Ashby) — those add direct-ATS autofill jobs but
                                         # re-introduce company-anchored discovery.
    max_jobs_per_source: int = 200   # Cap per source per discovery run (was 50)

    # Telegram
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # GitHub harvester (optional token lifts the public API rate limit)
    github_token: str = ""

    # Admin token gating the one-off H-1B CSV upload page (empty = page disabled)
    admin_token: str = ""

    # Owner-only admin dashboard: comma-separated allow-list of emails.
    admin_emails: str = "karthikamruthaluri2002@gmail.com,karthiklucky899@gmail.com"

    # Referral program
    referral_threshold: int = 10          # friends needed to unlock the reward
    referral_reward_days: int = 30        # days of the reward plan granted
    referral_reward_plan: str = "pro"     # plan tier granted on unlock

    @property
    def admin_emails_list(self) -> List[str]:
        return [e.strip().lower() for e in self.admin_emails.split(",") if e.strip()]

    # Founding-user trial: first N users get a budget of fully processed jobs
    # with all Pro features unlocked.
    trial_max_users: int = 10
    trial_job_quota: int = 100

    # Local personal dashboard
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # Paths
    data_dir: Path = Path("./data")
    resume_path: Path = Path("./data/resume_master.md")
    resume_docx_path: Path = Path("./data/resume_master.docx")
    profiles_dir: Path = Path("./data/profiles")
    faiss_index_path: Path = Path("./data/jobs.faiss")
    sqlite_path: Path = Path("./data/jobagent.db")

    # Supabase — set these to migrate from SQLite to PostgreSQL
    supabase_url: str = ""             # https://xxxx.supabase.co
    supabase_anon_key: str = ""        # public anon key (safe to expose in browser)
    supabase_service_role_key: str = ""  # service role key (server-only, never expose)
    database_url: str = ""             # postgresql://postgres:[password]@db.xxxx.supabase.co:5432/postgres

    @property
    def use_supabase(self) -> bool:
        return bool(self.database_url and self.supabase_url)

    @property
    def sqlite_url(self) -> str:
        if self.use_supabase:
            return self.database_url
        return f"sqlite:///{self.sqlite_path}"
    bootstrap_path: Path = Path("./data/bootstrap_companies.json")

    # Applicant
    applicant_first_name: str = "Karthik"
    applicant_last_name: str = ""
    applicant_email: str = ""
    applicant_phone: str = ""
    applicant_location: str = "Cincinnati, OH"
    applicant_github: str = ""
    applicant_linkedin: str = ""
    applicant_work_auth: str = ""

    # Matching
    min_match_score: float = 0.15          # lowered from 0.20 — cross-encoder floor
    top_k_rerank: int = 1000              # raised from 500 — send more to LLM
    daily_apply_limit: int = 25          # cap on actual auto-submissions per day (autofill)
    daily_shortlist_limit: int = 200     # cap on how many jobs get shortlisted onto the board per day
    shortlist_score_threshold: int = 35  # lowered from 40 — min LLM rerank score (0-100) to shortlist
    company_cap: int = 3                 # max active applications per company at once (focused, low spray-risk)
    discovery_cooldown_hours: int = 24    # min hours between manual discovery runs (saves API calls + tokens)

    # Models
    scoring_model: str = "claude-haiku-4-5-20251001"
    tailoring_model: str = "claude-sonnet-4-6"        # resume tailoring — Sonnet for quality
    cover_letter_model: str = "claude-haiku-4-5-20251001"  # cover letter — Haiku saves ~$0.012/app
    doctor_model: str = "claude-haiku-4-5-20251001"   # resume doctor quality check

    # Thresholds & Constraints
    min_embedding_score: float = 0.28    # lowered from 0.35 — was too aggressive
    qa_confidence_threshold: float = 0.7
    grounding_similarity_threshold: float = 0.5

    ghost_score_threshold: float = 0.6   # jobs at or above this score are skipped as likely ghost postings

    # Submission Delays & Limits
    submission_jitter_min: float = 180.0
    submission_jitter_max: float = 480.0
    headless: bool = True

    # Discovery
    greenhouse_boards: str = ""
    lever_boards: str = ""
    ashby_boards: str = ""
    jobs_keywords: str = "Machine Learning Engineer,AI Engineer,Python Developer,LLM Engineer,AI/ML Engineer,Backend Python Engineer,ML Engineer,Applied Scientist,NLP Engineer,GenAI Engineer"

    @property
    def jobs_keywords_list(self) -> List[str]:
        return [k.strip() for k in self.jobs_keywords.split(",") if k.strip()]

    @property
    def greenhouse_boards_list(self) -> List[str]:
        return [b.strip() for b in self.greenhouse_boards.split(",") if b.strip()]

    @property
    def lever_boards_list(self) -> List[str]:
        return [b.strip() for b in self.lever_boards.split(",") if b.strip()]

    @property
    def ashby_boards_list(self) -> List[str]:
        return [b.strip() for b in self.ashby_boards.split(",") if b.strip()]

settings = Settings()

if not settings.telegram_enabled:
    settings.telegram_bot_token = ""
    settings.telegram_chat_id = ""

