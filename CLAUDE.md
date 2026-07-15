# CLAUDE.md — HirePath

AI job-application copilot. Discovers tech roles from ATS APIs/feeds, scores each
against the user's résumé via a cascade, tailors résumé+cover letter (grounded), and
auto-fills forms. **The human always reviews and clicks Submit** — never auto-submit.

Multi-tenant web app (grew from a single-user agent). Public README has the full story;
this file is the working map for editing the code.

## Architecture
`Discover → Match cascade → Score & enrich → Tailor (grounding check) → Auto-fill → User reviews & Submits`

**Matching cascade** (`app/matching/pipeline.py`, cheapest-first so LLM cost stays low):
1. Retrieval — BM25 + FAISS (`all-MiniLM-L6-v2`) over UNSCORED jobs only (newest
   first) → top-K (`matcher.py`); scored jobs re-shortlist via direct query, not
   retrieval — letting them compete starved fresh postings of CE slots
2. Rule filter — title/seniority/location/job-type, per-company cap (`filters/`)
3. Ghost filter — drops inactive/fake postings
4. Embedding gate — cosine-similarity floor
5. LLM reranker — two-tier cascade (`reranker.py`): a cheap Tier-1 model
   (`prescore()`, GPT-4o-mini or Haiku) bulk-scores up to `prescore_cap` fresh
   candidates; only those clearing the advance gate reach Tier-2 (Claude, the
   authoritative 0–100 + reasoning). Clear misfits are stamped with their
   prescore so they exit the unscored corpus — draining the backlog instead of
   re-reading it every pass. Toggle with `PRESCORE_ENABLED`.
6. Hire probability — blends fit + hiring-intent signals (`hire_probability.py`)
7. Senior review — independent "senior engineer" verdict + score (`intelligence/senior_reviewer.py`)

## Stack
Python 3.11 · FastAPI/Uvicorn · SQLModel. **Supabase Postgres + Auth in prod; local
SQLite fallback when `SUPABASE_URL` is unset.** Claude (primary LLM) / OpenAI optional ·
sentence-transformers + FAISS + rank-bm25 · Playwright (Chromium) + MV3 Chrome extension ·
python-telegram-bot · APScheduler · Jinja + Tailwind (CDN) + Chart.js (server-rendered).

## File structure
```
app/
  api/server.py     # ALL routes, dashboard, auth, admin (single large file)
  config.py         # Settings (pydantic-settings, env-driven)
  db/models.py      # SQLModel tables; supabase_client.py = JWT/user_id; init_db.py
  discovery/        # ATS scrapers; sources/ = aggregators & feeds (~20)
  matching/         # matcher, filters/, reranker, hire_probability, pipeline
  tailoring/        # tailor, ats_keywords, grounding (anti-hallucination), doctor
  autofill/         # Playwright filler + answer_pack
  intelligence/     # sponsorship/H1B, work_auth, senior_reviewer, urgency, referral,
                    # skill_gap (JD vs resume/GitHub advice), job_check (free ghost/fit check)
  strategy/         # daily_engine (apply scoring/limits)
  analytics/        # funnel, cost_dashboard, crm, reporter
  qa_store/         # canonical answers (answers.yaml) + memory resolver
  telegram_bot/     # async approval/handoff loop
  templates/        # landing, dashboard, pricing, auth, privacy, terms, extension
extension/          # MV3: background.js, content.js, popup
scripts/            # run_discovery, run_matching, seed_registry, status_check
tests/              # pytest (matching, tailoring, grounding, autofill, funnel...)
data/               # résumé master, FAISS index, generated docs, local SQLite
```

## Key models (`app/db/models.py`)
`Job`, `Application`, `UserProfile`, `UserSubscription`/`UserUsage`/`PlanTier`,
`CompanyRegistry`, `DiscoveryRun`, `FunnelEvent`, `PendingQuestion`, `AnswerMemory`,
`UserPersonalMemory`, `H1BSponsor`, `UserNotification`, referrals/coupons.

UI-relevant `Job`/`Application` fields: `rerank_score` (0–100 fit), `rerank_reasoning`,
`blended_score` (priority), `hire_probability_signals` (JSON), `senior_fit_score`,
`senior_verdict`, `ghost_score`/`ghost_flags`, `custom_highlight_block`.

## Conventions & decisions
- **Multi-tenancy:** every query is scoped by `user_id` from the Supabase JWT
  (`_get_user_id`/`_require_user_id` in server.py). `"local"` = SQLite dev user.
  Never leak data across users; check ownership on per-application routes.
- **Scrape once, serve many:** all scheduled lanes write postings ONCE to the
  shared pool (`Job.user_id == SHARED_POOL_USER`, pipeline.py); per-user pools
  are filled by `strategy/adoption.py` (cheap DB copy by roles+country; also
  runs on resume upload + role edits = instant feeds). Scheduled discovery is
  ONE global pass with the union of all users' roles — never per-user.
- **Scheduler:** `server.py`'s asyncio scheduler runs global discovery→adopt→match
  ~every `DISCOVERY_INTERVAL_HOURS` in BOTH local and prod, plus a "fresh lane"
  every 2h (`_global_fresh_scan`, phase="fresh" = registry boards + free keyless
  feeds; quota-keyed sources stay on the full lane; env FRESH_LANE_INTERVAL_HOURS,
  0 disables) and a board-freshness lane: the "pulse lane" by default
  (`strategy/pulse_lane.py`, per-board `next_poll_at` schedule — watchlist
  `UserProfile.target_companies` + recently-posting boards every 5 min, every
  live board ≤60 min, dead boards daily; unchanged boards skipped via
  `poll_hash`; new jobs take a lock-free per-job fast path: ghost check →
  prescore cascade → Claude → shortlist → fresh alert). Set PULSE_LANE_ENABLED=0
  to fall back to the legacy 20-min "hot lane" (`strategy/hot_lane.py`) — only
  one of the two runs. Do NOT also schedule those in
  `app/main.py` (it only adds the Telegram bot + harvester/validator/report jobs)
  — double-runs otherwise.
- **Scoring lane** (`strategy/scoring_lane.py`, every `SCORING_LANE_INTERVAL_SECONDS`):
  the decoupled, PARALLEL, cross-user scorer — drains the global `rerank_score
  IS NULL` queue across ALL users with a fixed pool of `scoring_workers` (GPT
  prescore → Claude final), so throughput is bounded by LLM rate limits, not
  user count (the matching lane scores users serially = O(users)). Lock-free
  (no FAISS); the 5-min matching lane stays as the retrieval + reshortlist +
  self-heal backstop. Set `SCORING_LANE_ENABLED=0` to fall back to matching-lane-only.
- **Run modes:** prod = `uvicorn app.api.server:app`; local all-in-one = `python -m app.main`.
- **Jinja filters** (`server.py`): `fromjson`, `cleantext`, `humanize_signal`
  (turns raw signal tokens like `fresh_posting_4d` → "Posted 4 days ago").
- **Dashboard** is one big `templates/dashboard.html` (HTML + inline `<script>`). Modals
  toggle via `style.display` (not the `hidden` class — inline `display` overrides it).
  After editing, validate: parse Jinja + `node --check` the touched `<script>` block.
- **Tuning lives in env/Settings:** `shortlist_score_threshold` (default 35),
  `top_k_rerank`, `MIN_MATCH_SCORE`, `DAILY_APPLY_LIMIT`, `*_BOARDS` slugs.
- **Compliance:** public ATS/feeds only, respect robots.txt; no LinkedIn/Indeed
  automation (discovery-only links). Tailoring must stay grounded in the real résumé.

## Workflow
- Tests: `pytest` (or target files); lint: `ruff check app`.
- Validate template/python edits before committing; keep commits scoped + descriptive.
- Branch per the session's assigned feature branch; commit + push when done.

## Maintenance
Update on major architectural changes or completed modules. Keep under ~150 lines —
prune stale info rather than appending.
