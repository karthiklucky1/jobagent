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
    serpapi_date_posted: str = "3days"  # Google Jobs freshness window: today | 3days | week | month
    serpapi_max_keywords: int = 8    # searches fired per run (each = 1 quota unit); caps the shared-run role union
    serpapi_concurrency: int = 5     # concurrent Google Jobs searches (was sequential → 45s timeout)
    remotive_enabled: bool = True    # Remotive public API — no key needed
    remoteok_enabled: bool = True    # RemoteOK public API — no key needed
    hn_whoishiring_enabled: bool = True  # HN monthly "Who is hiring?" thread — no key, early signal
    themuse_enabled: bool = True     # The Muse public API — no key needed; optional THEMUSE_API_KEY for higher rate limit
    themuse_api_key: str = ""        # optional — raises rate limits
    arbeitnow_enabled: bool = True   # Arbeitnow public API — no key needed
    jobicy_enabled: bool = True      # Jobicy public API — no key needed
    weworkremotely_enabled: bool = True  # WeWorkRemotely RSS feeds — no key needed
    indeed_rss_enabled: bool = False  # Indeed killed public RSS + blocks bots (and ToS forbids it) — off by default
    # Keyed sources auto-activate: enabled by default, but each source still
    # skips itself when its key is missing — so "add the key → it works", and
    # setting the *_ENABLED flag to false force-disables even with a key set.
    adzuna_enabled: bool = True      # Adzuna API — needs app_id + app_key (free tier)
    adzuna_app_id: str = ""
    adzuna_app_key: str = ""
    reed_enabled: bool = True        # Reed.co.uk API — needs api_key (free tier)
    reed_api_key: str = ""
    jooble_enabled: bool = True      # Jooble API — needs api_key (free tier)
    jooble_api_key: str = ""
    linkedin_rapidapi_enabled: bool = True  # LinkedIn via RapidAPI (~$10/mo) — needs rapidapi_key
    rapidapi_key: str = ""

    @property
    def linkedin_rapidapi_active(self) -> bool:
        """Active only when explicitly enabled AND a key is present, so
        LINKEDIN_RAPIDAPI_ENABLED=false reliably turns it off (e.g. while the
        RapidAPI quota is exhausted) even with a key still configured."""
        return self.linkedin_rapidapi_enabled and bool(self.rapidapi_key)
    scrape_company_boards: bool = False  # JOB-FIRST by default: discovery is driven purely by job
                                         # aggregators (SerpAPI/Remotive/RemoteOK/HN), NOT a fixed company list.
                                         # Set True to also scrape the bootstrap company ATS boards
                                         # (Greenhouse/Lever/Ashby) — those add direct-ATS autofill jobs but
                                         # re-introduce company-anchored discovery.
    max_jobs_per_source: int = 200   # Cap per source per discovery run (was 50)
    # Cap on boards a keyword-search source fetches per run. The registry now
    # holds ~56K boards; fetching all of them blows the 45s per-source timeout.
    # The direct-ATS board lanes cover the full registry, so keyword search only
    # needs the top productive boards.
    keyword_search_max_slugs: int = 250

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
    top_k_rerank: int = 600               # final candidate pool size returned by retrieval
    cross_encoder_cap: int = 120          # max pairs scored by the local CPU cross-encoder (the real CPU bottleneck)
    cross_encoder_max_length: int = 256   # token cap per cross-encoder pair (shorter = far faster on CPU)
    cross_encoder_text_chars: int = 700   # chars of profile/job text fed to each cross-encoder pair
    # Reranker backend for the retrieval rerank stage. "local" = the on-CPU
    # cross-encoder (slow on Railway). "jina" = Jina Reranker API (fast, cheap).
    # Any API failure or missing key falls back to local, then to FAISS order.
    rerank_provider: str = "local"        # "local" | "jina"
    jina_api_key: str = ""                # api.jina.ai — rerank API key
    jina_rerank_model: str = "jina-reranker-v2-base-multilingual"
    llm_rerank_cap: int = 100             # max jobs sent to the LLM reranker per run (fresh-first order); env LLM_RERANK_CAP
    llm_rerank_workers: int = 12          # concurrent LLM scoring workers (tune to Anthropic tier)
    llm_rerank_max_retries: int = 4       # retry budget on 429/overloaded before leaving job unscored
    llm_request_timeout: float = 45.0     # per-request LLM timeout (s). Bounds a matching pass so a slow API can't freeze it while it holds the matching lock. SDK default is 600s.
    max_liveness_checks_per_run: int = 25 # cap on serial link-liveness network calls per matching pass (each ~2.5s, lock-held) so one pass can't starve other lanes
    matching_lane_interval_minutes: int = 5  # INDEPENDENT matching loop cadence (env MATCHING_LANE_INTERVAL_MINUTES; 0 disables). Decouples scoring from discovery so a stalled discovery can't starve matching.
    daily_apply_limit: int = 25          # cap on actual auto-submissions per day (autofill)
    daily_shortlist_limit: int = 200     # cap on how many jobs get shortlisted onto the board per day
    shortlist_score_threshold: int = 35  # lowered from 40 — min LLM rerank score (0-100) to shortlist
    company_cap: int = 3                 # max active applications per company at once (focused, low spray-risk)
    discovery_cooldown_hours: int = 24    # min hours between manual discovery runs (saves API calls + tokens)
    discovery_interval_hours: int = 6     # scheduler cadence for automatic discovery+matching per user
    direct_ats_enabled: bool = True       # scrape active CompanyRegistry boards directly (live jobs, direct links)
    max_boards_per_run: int = 400         # cap on registry boards scraped per discovery run. Higher covers the ~56K registry faster but holds more jobs in memory per run; 400 balances coverage vs. the container memory limit (800 contributed to an OOM crash).
    # Wall-clock cap on the board phase (fetch + per-board DB work) of a
    # discovery run. Without it a run held the global discovery lock for 3+
    # hours and starved the hot lane. Deferred boards go first next run.
    board_phase_budget_minutes: int = 30
    # Fresh lane: boards-only rescan every N hours (0 disables). Applying within
    # 24-72h of posting measurably lifts response rates, so registry boards are
    # rescanned far more often than full discovery runs.
    fresh_lane_interval_hours: int = 2
    # Hot lane: poll the most productive boards every N minutes (0 disables) so
    # brand-new postings reach shortlists within minutes. Fetches each board
    # once and distributes to matching users (cost = O(boards), not O(boards×
    # users)). This is what makes "fresh within minutes of posting" real.
    hot_lane_interval_minutes: int = 20
    hot_lane_max_boards: int = 400
    # Fraction of each hot-lane cycle spent bootstrapping never-polled boards.
    # The rest goes to proven yielders + productive boards. Kept low so tens of
    # thousands of dead seeded slugs can't eat the budget (they 404 and get
    # retired) and starve boards that actually post — env HOT_LANE_BOOTSTRAP_FRAC.
    hot_lane_bootstrap_frac: float = 0.2
    # Bulk registry seed from the open ats-scrapers slug dataset (~20K companies
    # across Greenhouse/Lever/Ashby/SmartRecruiters/Workable/Recruitee/Personio).
    open_dataset_seed_enabled: bool = True
    open_slug_dataset_base: str = "https://raw.githubusercontent.com/kalil0321/ats-scrapers/main/ats-companies"
    verify_links_on_shortlist: bool = True  # HEAD-check non-ATS links before they take a shortlist slot

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

    # CORS configuration
    cors_allowed_origins: str = "http://localhost:5173,http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000,https://hirepath.dev"

settings = Settings()

if not settings.telegram_enabled:
    settings.telegram_bot_token = ""
    settings.telegram_chat_id = ""

