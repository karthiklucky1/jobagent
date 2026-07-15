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
    llm_rerank_cap: int = 100             # max jobs sent to the FINAL (Claude) reranker per run (fresh-first order); env LLM_RERANK_CAP
    llm_rerank_workers: int = 12          # concurrent LLM scoring workers (tune to Anthropic tier)
    # ── Two-tier scoring cascade ──────────────────────────────────────────────
    # Tier 1: a cheap/fast model (default GPT-4o-mini) bulk-scores many candidates
    # per pass; only those clearing prescore_advance_threshold go to Tier 2
    # (Claude, the authoritative score that drives shortlisting). Jobs Tier 1
    # clearly rejects are stamped with their prescore so they leave the unscored
    # corpus — this drains the backlog (fewer repeated full-row reads = less
    # egress) and lets far more than llm_rerank_cap jobs be looked at per pass.
    prescore_enabled: bool = True         # PRESCORE_ENABLED — turn the cascade on/off (off = old single-tier behavior)
    prescore_provider: str = "openai"     # PRESCORE_PROVIDER — "openai" | "anthropic" (Tier-1 bulk scorer; falls back to whatever key exists)
    prescore_model: str = "gpt-4o-mini"   # PRESCORE_MODEL — cheap/fast Tier-1 model
    prescore_cap: int = 600               # PRESCORE_CAP — max candidates Tier-1 scores per pass (fresh-first)
    prescore_advance_threshold: int = 35  # PRESCORE_ADVANCE_THRESHOLD — Tier-1 fit >= this advances to Claude; below is stamped and drained. The pipeline clamps the effective gate to <= shortlist_score_threshold, so 35 (== the shortlist bar) is the highest quality-safe value: anything that could plausibly shortlist still gets the authoritative look. Was 30 — the wider gap advanced ~90% of jobs, making Tier-1 a cost ADD instead of a filter.
    prescore_workers: int = 16            # PRESCORE_WORKERS — concurrent Tier-1 workers (cheap model tolerates more)
    llm_rerank_max_retries: int = 1       # LLM_RERANK_MAX_RETRIES — in-call attempts per backend on 429/overloaded. 1 = no in-call retry: a failed job stays rerank_score-NULL and the 90s scoring lane re-queues it anyway, so in-call backoff (old default 4) only stacked sleeps and hammered exhausted quotas.
    llm_provider_cooldown_minutes: int = 30  # LLM_PROVIDER_COOLDOWN_MINUTES — circuit breaker: a provider returning credit/quota errors is skipped for this long instead of being re-hit every cycle (0 disables)
    llm_daily_final_cap: int = 1500       # LLM_DAILY_FINAL_CAP — max Tier-2 (authoritative) LLM scores per UTC day, all lanes combined (~$5-8/day worst case). Jobs past the cap stay Queued for tomorrow, freshest-first. 0 = unlimited. Raise as paying users grow.
    llm_hourly_final_cap: int = 150       # LLM_HOURLY_FINAL_CAP — smoothing: max finals per clock hour (~$0.50-0.75/hr). Without it the lane burst-drains a big backlog at ~2K finals/hour and the whole daily budget can burn in under an hour. Fresh jobs still score within the hour. 0 = unlimited.
    scoring_fail_max_attempts: int = 3    # SCORING_FAIL_MAX_ATTEMPTS — after this many failed final-score attempts a job is deferred (sits out) instead of re-queued every 90s forever
    scoring_fail_defer_hours: float = 6.0 # SCORING_FAIL_DEFER_HOURS — how long a repeatedly-failing job sits out before it may be retried
    # ── DB connection pool (Postgres/Supabase only) ───────────────────────────
    # Sized for the background lanes + web traffic. Scoring workers no longer
    # hold a connection during LLM calls (they open short sessions before/after),
    # so the pool does NOT need to scale with scoring_workers — but it does need
    # headroom over the old 5+10, which starved funnel/registry/web requests
    # ("QueuePool limit ... reached" errors) whenever the lanes overlapped.
    db_pool_size: int = 10                # DB_POOL_SIZE
    db_max_overflow: int = 20             # DB_MAX_OVERFLOW
    # ── Distilled local scorer (see docs/DISTILLATION.md) ─────────────────────
    # A small cross-encoder fine-tuned on this deployment's own LLM scores.
    # Until a trained model exists at local_scorer_path everything no-ops.
    # Shadow mode runs it NEXT TO LLM finals and records agreement as
    # FunnelEvents (stage="shadow_score") — flip to local-first only after
    # scripts/shadow_report.py shows the agreement you're comfortable with.
    local_scorer_path: str = "data/models/hirepath-scorer"  # LOCAL_SCORER_PATH
    local_scorer_shadow: bool = True      # LOCAL_SCORER_SHADOW
    llm_request_timeout: float = 45.0     # per-request LLM timeout (s). Bounds a matching pass so a slow API can't freeze it while it holds the matching lock. SDK default is 600s.
    max_liveness_checks_per_run: int = 25 # cap on serial link-liveness network calls per matching pass (each ~2.5s, lock-held) so one pass can't starve other lanes
    matching_lane_interval_minutes: int = 5  # INDEPENDENT matching loop cadence (env MATCHING_LANE_INTERVAL_MINUTES; 0 disables). Decouples scoring from discovery so a stalled discovery can't starve matching.
    matching_catchup_passes: int = 4     # max scoring passes per user per lane tick when a large backlog exists (env MATCHING_CATCHUP_PASSES; 1 = old behavior). Drains a post-incident unscored backlog faster; bounded by a wall-clock budget.
    matching_catchup_backlog: int = 200  # only run extra catch-up passes while a user's unscored backlog exceeds this (env MATCHING_CATCHUP_BACKLOG)
    # ── Scoring lane (decoupled, parallel, cross-user) ────────────────────────
    # Drains the GLOBAL queue of unscored on-role jobs across ALL users at once
    # with a bounded pool of LLM workers, so scoring throughput depends on the
    # LLM rate limit — NOT on the number of users. Replaces the old O(users)
    # serial per-user matching loop as the primary "get fresh jobs scored fast"
    # engine. Lock-free (cheap gates + GPT->Claude cascade, no FAISS), so it runs
    # continuously alongside discovery. The 5-min matching lane stays as the
    # FAISS-retrieval + reshortlist + self-heal backstop.
    scoring_lane_enabled: bool = True      # SCORING_LANE_ENABLED
    scoring_lane_interval_seconds: int = 90  # cadence; 0 disables
    scoring_workers: int = 20              # GLOBAL concurrent LLM scoring workers (size to your Anthropic/OpenAI rate limit, not user count)
    scoring_per_user_cap: int = 40         # max queued jobs scored per user per cycle (fresh-first)
    scoring_global_cap: int = 200          # max total jobs scored per cycle (bounds cost + wall-clock; also bounds how many prescores can overshoot when the finals budget trips mid-cycle)
    scoring_lane_max_seconds: int = 120    # hard wall-clock cap per cycle
    # ── Dual-provider final scoring (Option A) ────────────────────────────────
    # The prescore→final cascade is a RELAY (GPT drains misfits, then Claude
    # scores the survivors) — one job flows GPT→Claude, so the two can't be
    # "split" on the same job. To lift the single-provider rate-limit ceiling we
    # instead split the FINAL score across providers by JOB: ~claude_share of
    # jobs get Claude's authoritative score, the rest go to GPT-4o — in parallel.
    # BOTH score against the SAME rubric (_get_system_prompt + _SCORE_BANDS), so
    # the numbers are comparable; a small calibration offset nudges GPT's scale
    # onto Claude's if it clusters low/high. Only active when BOTH provider keys
    # exist; with one key it's a no-op (the single provider scores everything).
    dual_score_enabled: bool = False        # DUAL_SCORE_ENABLED — split final scoring across Claude + GPT. OFF by default: the GPT final is the full gpt-4o (~2.5x Haiku's price) and in practice OpenAI rate limits spilled most of its share back onto Haiku anyway — so dual mode mostly added cost, not throughput. Enable only with an OpenAI tier that can actually absorb its share.
    dual_score_claude_share: float = 0.6    # DUAL_SCORE_CLAUDE_SHARE — fraction of finals routed to Claude (rest to GPT)
    dual_score_openai_model: str = "gpt-4o" # DUAL_SCORE_OPENAI_MODEL — the GPT FINAL scorer (full model, not mini, so it's comparable to Claude). Set to gpt-4o-mini to cut cost at some accuracy loss.
    dual_score_openai_offset: float = 0.0   # DUAL_SCORE_OPENAI_OFFSET — calibration added to GPT scores to align with Claude's scale (e.g. +5 if GPT clusters ~5 points low). Clamped to 0-100.

    # ── Observability (all dormant until set — safe to ship empty) ─────────────
    sentry_dsn: str = ""                   # SENTRY_DSN — enables error tracking when set
    sentry_environment: str = "production" # SENTRY_ENVIRONMENT
    sentry_traces_sample_rate: float = 0.0 # SENTRY_TRACES_SAMPLE_RATE (0 = errors only, cheapest)
    heartbeat_matching_url: str = ""       # HEARTBEAT_MATCHING_URL — healthchecks.io ping URL for the matching lane
    daily_apply_limit: int = 25          # cap on actual auto-submissions per day (autofill)
    daily_shortlist_limit: int = 200     # cap on how many jobs get shortlisted onto the board per day
    shortlist_score_threshold: int = 35  # lowered from 40 — min LLM rerank score (0-100) to shortlist
    shortlist_render_cap: int = 200      # max shortlist cards rendered on the dashboard. Was 100, which HID jobs: with 161 shortlisted the board showed 100 while the header/live count said 161, so 61 jobs could never appear and the "new matches" banner looped forever. 200 covers a full day's shortlisting (daily_shortlist_limit); above it the "showing X of Y" note kicks in.
    shortlist_max_age_days: int = 14     # SHORTLIST_MAX_AGE_DAYS — a posting older than this is likely filled/ghosted, so a job that has sat SHORTLISTED (never tailored/applied) this long is auto-removed from the board (→ SKIPPED, which also frees the per-company slot). Keeps the shortlist to fresh, applyable roles. 0 disables the prune.
    company_cap: int = 3                 # max active applications per company at once (focused, low spray-risk)
    # When a company is at the cap and a NEW job scores clearly higher than a
    # cap-holding application that is still just SHORTLISTED (untouched — not
    # tailored/submitted), the weaker shortlist entry is displaced (→ SKIPPED)
    # so the stronger role takes the slot. Applications the user or agent has
    # invested effort in (TAILORED and beyond) are NEVER displaced.
    company_cap_displace_enabled: bool = True  # COMPANY_CAP_DISPLACE_ENABLED
    company_cap_displace_margin: int = 5       # COMPANY_CAP_DISPLACE_MARGIN — new job must beat the weakest shortlisted holder by at least this many points (hysteresis against churn)
    discovery_cooldown_hours: int = 24    # min hours between manual discovery runs (saves API calls + tokens)
    discovery_interval_hours: int = 6     # scheduler cadence for automatic discovery+matching per user
    # Onboarding: after a new user's résumé/roles land we first fill their board
    # from the shared pool (instant DB copy). But the shared pool skews toward the
    # roles existing users already search (historically AI/ML), so a user from a
    # different domain (mechanical, finance, nursing…) adopts almost nothing and
    # sees an empty feed. When adoption leaves them under this many on-role jobs,
    # kick a targeted scrape of THEIR roles right away instead of waiting for the
    # next 6h global pass — their domain fills within minutes. 0 disables the scrape.
    onboarding_active_discovery: bool = True  # ONBOARDING_ACTIVE_DISCOVERY
    onboarding_min_jobs: int = 25             # ONBOARDING_MIN_JOBS — adopt-count floor below which onboarding actively scrapes the user's domain
    # "My roles" relevance filter (All Jobs). Title matching alone misses jobs
    # whose title is worded differently but is the same work ("Applied Scientist"
    # ≈ "ML Engineer"). We DON'T need a separate semantic cache for this — the AI
    # fit score already IS the semantic signal (it scores the job against the
    # résumé, not the title), so a differently-titled-but-relevant job scores high.
    # The filter keeps a job when its title matches a role OR it scored at/above
    # this fit floor. 0 = title-only (no semantic catch).
    roles_filter_score_floor: int = 50        # ROLES_FILTER_SCORE_FLOOR
    # Semantic adoption. When copying shared-pool jobs into a user's pool, the
    # title gate (role_title_match) can miss "same work, different title" postings
    # (an "Applied Scientist" that's really ML work). This adds a second pass: keep
    # every title match PLUS the closest résumé-neighbours by embedding cosine
    # (reuses the local MiniLM model — no API cost), so relevant postings are
    # copied and scored instead of filtered out before scoring. Bounded by the
    # adoption cap, with a title-only fallback if embeddings are unavailable.
    adoption_semantic_enabled: bool = True     # ADOPTION_SEMANTIC_ENABLED
    adoption_semantic_threshold: float = 0.30  # ADOPTION_SEMANTIC_THRESHOLD — min résumé↔job cosine for a non-title-match to be adopted
    adoption_semantic_max_candidates: int = 1500  # ADOPTION_SEMANTIC_MAX_CANDIDATES — cap on non-title jobs embedded per pass (CPU bound)
    adoption_semantic_max_extras: int = 50     # ADOPTION_SEMANTIC_MAX_EXTRAS — cap on off-title neighbours ADOPTED per user per pass. Every adopted row is a new rerank_score-NULL job the LLM scorer must pay for; adoption passes repeat every few hours, so a small per-pass budget still surfaces the same jobs — just spread out. 0 = title-only.
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
    # ── Pulse lane (freshness guarantee) ─────────────────────────────────────
    # Replaces the hot lane's rotating 400-board batches with a per-board
    # next_poll_at schedule: watchlist companies + boards that recently posted
    # are polled every PULSE_FAST_INTERVAL_MINUTES; every other LIVE board is
    # swept at least every PULSE_FLOOR_INTERVAL_MINUTES (the "within the hour"
    # promise); boards that 404 / never held a job decay to a daily retry. New
    # jobs take a per-job fast path (role match → cascade score → alert) instead
    # of waiting for the next batch matching tick. When enabled, the legacy hot
    # lane loop is not started (the fresh/full lanes stay as safety nets).
    pulse_lane_enabled: bool = True        # PULSE_LANE_ENABLED
    pulse_fast_interval_minutes: int = 5   # watchlist + recently-active boards
    pulse_floor_interval_minutes: int = 60 # every live board at least this often
    pulse_dead_interval_hours: int = 24    # 404/empty boards retry cadence
    pulse_active_days: int = 7             # "recently posted" = new job within N days
    pulse_tick_seconds: int = 60           # scheduler tick
    pulse_tick_max_seconds: int = 150      # HARD wall-clock cap per tick. The tick stops taking new work past this and reschedules the rest — so it always releases the lock promptly (a tick that ran serial LLM scoring for 20+ min once froze the whole lane). Keep < tick_seconds*3.
    pulse_max_boards_per_tick: int = 300   # hard cap per tick. Steady-state demand is ~150 boards/min, so 300 keeps headroom; the wall-clock cap above bounds it further during bootstrap.
    pulse_fetch_workers: int = 24          # concurrent board fetches per tick
    pulse_fast_path_score_cap: int = 10    # max brand-new jobs LLM-scored per tick via the fast path (kept small so the tick stays short; the rest are scored by the 5-min matching lane)
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
    # FALLBACK-ONLY keyword list. Onboarding and the global scheduled pass always
    # pass explicit keywords (the user's own roles / the union of all users'
    # roles), so this list is used only when NO roles exist yet — a fresh deploy
    # or a role-less cold start. It used to be 100% AI/ML, which meant the shared
    # pool a brand-new non-tech user adopts from was all AI/ML — a mechanical or
    # nursing candidate saw almost nothing. Now it spans the major sectors so the
    # cold-start pool is broad; each user's real feed is still driven by THEIR roles.
    jobs_keywords: str = (
        "Software Engineer,Data Analyst,Machine Learning Engineer,Product Manager,"
        "Mechanical Engineer,Civil Engineer,Electrical Engineer,Manufacturing Engineer,"
        "Financial Analyst,Accountant,Registered Nurse,Healthcare Administrator,"
        "Marketing Manager,Sales Representative,Operations Manager,Project Manager,"
        "UX Designer,Business Analyst,Customer Success Manager,Human Resources Specialist,"
        "Supply Chain Analyst,Administrative Assistant"
    )

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

