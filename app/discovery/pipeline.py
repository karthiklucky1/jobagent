"""Pipeline that runs all configured scrapers and upserts into the DB."""
from __future__ import annotations

import logging
import hashlib
import re
from typing import List

from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from app.config import settings
from app.db.init_db import get_session
from app.db.models import Job, JobSource, CompanyRegistry, Application, ApplicationStatus
from app.discovery.ashby import AshbyScraper
from app.discovery.base import RawJob
# Greenhouse/Lever/Ashby are used by the .env fallback in _all_scrapers; the
# full per-ATS mapping (incl. Workday/Workable/Recruitee/Personio/Rippling/
# Breezy/Pinpoint/Teamtailor) lives in scraper_for() to keep both paths in sync.
from app.discovery.greenhouse import GreenhouseScraper
from app.discovery.lever import LeverScraper

log = logging.getLogger(__name__)

# Permissive/Negative filtering to exclude obvious non-tech roles
_TECH_TITLE_RE = re.compile(
    r'\b(engineer|scientist|developer|researcher|architect|analyst|'
    r'mlops|devops|sre|quantitative|quant|statistician|'
    r'programmer|technologist|intelligence|nlp|llm|'
    r'platform|infrastructure|backend|fullstack|full[\-\s]stack|frontend|front[\-\s]stack|'
    r'machine\s*learning|deep\s*learning|computer\s*vision|data|technical|member\s+of\s+technical\s+staff)\b',
    re.IGNORECASE,
)

_NON_TECH_TITLE_RE = re.compile(
    r'\b(sales|marketing|recruiter|hr|talent\s+acquisition|people\s+ops|'
    r'finance|accountant|accounting|payroll|billing|auditor|'
    r'legal|counsel|lawyer|compliance|'
    r'receptionist|administrative|assistant|secretary|office\s+manager|'
    r'customer\s+support|customer\s+success|sales\s+rep|account\s+exec|'
    r'copywriter|content\s+writer|editor|translator|'
    r'nurse|doctor|medical|therapist|chef|cook|driver|cashier|'
    r'facilities|janitor|security\s+guard|maintenance)\b',
    re.IGNORECASE,
)

def is_obvious_non_tech(title: str) -> bool:
    if _NON_TECH_TITLE_RE.search(title):
        if _TECH_TITLE_RE.search(title):
            return False
        return True
    return False


def _all_scrapers():
    scrapers = []
    
    # 1. Query active boards from the DB registry — least-recently-scraped first
    #    so the max_boards_per_run cap rotates fairly across runs.
    try:
        with get_session() as session:
            db_companies = session.exec(
                select(CompanyRegistry)
                .where(CompanyRegistry.is_active == True)
                .order_by(CompanyRegistry.last_seen.asc().nulls_first())
            ).all()
            
            for comp in db_companies:
                s = scraper_for(comp.ats, comp.slug, comp.career_url)
                if s is not None:
                    scrapers.append(s)
    except Exception as e:
        log.warning("Could not load scrapers from CompanyRegistry database: %s. Falling back to .env", e)

    # 2. Fallback / Merge static list from .env if the database returned nothing
    if not scrapers:
        log.info("Registry empty or failed. Seeding scrapers from .env config lists.")
        for slug in settings.greenhouse_boards_list:
            scrapers.append(GreenhouseScraper(slug))
        for slug in settings.lever_boards_list:
            scrapers.append(LeverScraper(slug))
        for slug in settings.ashby_boards_list:
            scrapers.append(AshbyScraper(slug))

    # Deduplicate scrapers by type + slug
    seen = set()
    deduped = []
    for s in scrapers:
        slug_attr = getattr(s, "board_slug", None) or getattr(s, "company_slug", None) or getattr(s, "org_slug", None)
        if slug_attr:
            key = (s.name, slug_attr.lower().strip())
            if key not in seen:
                seen.add(key)
                deduped.append(s)
            
    return deduped


def _normalize(text: str) -> str:
    """Normalize company name by stripping common suffixes and spacing."""
    text = text.lower().strip()
    text = re.sub(r'[.,\/#!$%\^&\*;:{}=\-_`~()]', '', text)
    text = re.sub(r'\b(inc|llc|corp|co|corporation|ltd|gmbh|sa)\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _normalize_title(text: str) -> str:
    """Normalize job title to standard abbreviations."""
    text = text.lower().strip()
    text = re.sub(r'[.,\/#!$%\^&\*;:{}=\-_`~()]', '', text)
    text = re.sub(r'\bsr\b', 'senior', text)
    text = re.sub(r'\bjr\b', 'junior', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _normalize_location(text: str) -> str:
    """Normalize location for slug comparison."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[.,\/#!$%\^&\*;:{}=\-_`~()]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _cross_source_slug(company: str, title: str, location: str) -> str:
    """Generate unique hash identifier for cross-source job duplication check."""
    c = _normalize(company)
    t = _normalize_title(title)
    l = _normalize_location(location)
    return hashlib.sha256(f"{c}|{t}|{l}".encode("utf-8")).hexdigest()


# ── Country detection for location filtering ────────────────────────────────
# Many aggregator feeds (Remotive, RemoteOK, Jobicy, WeWorkRemotely…) are global,
# so a user sees jobs from every country. We drop postings whose location clearly
# names a DIFFERENT country than the user wants (remote roles are kept when the
# user is remote-friendly). The detection logic lives in app.common.geo so the
# rule filter and retrieval stages agree with discovery on country handling.
from app.common.geo import location_allowed as _location_allowed  # noqa: E402


_DIRECT_ATS_SOURCES = {
    JobSource.GREENHOUSE, JobSource.LEVER, JobSource.ASHBY,
    JobSource.WORKDAY, JobSource.SMARTRECRUITERS,
}


def scraper_for(ats, slug: str, career_url: str | None = None):
    """Map a (ats, slug) to a live scraper instance, or None if unsupported.
    Shared by the full discovery pipeline and the hot lane so both stay in sync
    as new ATS scrapers are added."""
    from app.discovery.greenhouse import GreenhouseScraper
    from app.discovery.lever import LeverScraper
    from app.discovery.ashby import AshbyScraper
    from app.discovery.smartrecruiters import SmartRecruitersScraper
    from app.discovery.workday import WorkdayScraper
    from app.discovery.workable import WorkableScraper
    from app.discovery.recruitee import RecruiteeScraper
    from app.discovery.personio import PersonioScraper
    from app.discovery.rippling import RipplingScraper
    from app.discovery.breezy import BreezyScraper
    from app.discovery.pinpoint import PinpointScraper
    from app.discovery.teamtailor import TeamtailorScraper
    from app.discovery.bamboohr import BambooHRScraper
    from app.discovery.join import JoinScraper
    mapping = {
        JobSource.GREENHOUSE: lambda: GreenhouseScraper(slug),
        JobSource.LEVER: lambda: LeverScraper(slug),
        JobSource.ASHBY: lambda: AshbyScraper(slug),
        JobSource.SMARTRECRUITERS: lambda: SmartRecruitersScraper(slug),
        JobSource.WORKDAY: lambda: WorkdayScraper(slug, career_url),
        JobSource.WORKABLE: lambda: WorkableScraper(slug),
        JobSource.RECRUITEE: lambda: RecruiteeScraper(slug),
        JobSource.PERSONIO: lambda: PersonioScraper(slug),
        JobSource.RIPPLING: lambda: RipplingScraper(slug),
        JobSource.BREEZY: lambda: BreezyScraper(slug),
        JobSource.PINPOINT: lambda: PinpointScraper(slug),
        JobSource.TEAMTAILOR: lambda: TeamtailorScraper(slug),
        JobSource.BAMBOOHR: lambda: BambooHRScraper(slug),
        JobSource.JOIN: lambda: JoinScraper(slug),
    }
    factory = mapping.get(ats)
    return factory() if factory else None


def _upsert(raw_jobs: List[RawJob], user_id: str | None = None,
            preferred_country: str | None = None, remote_ok: bool = True,
            user_keywords: List[str] | None = None) -> int:
    """Insert new jobs; skip duplicates by (source, external_id) and cross-source slug.

    Fast path: ONE query snapshots this user's existing dedupe keys, and raw
    jobs that are unchanged, recently-seen duplicates are skipped with zero DB
    work. Previously every duplicate cost 1-2 SELECTs plus a commit against
    Supabase — with most of a run's fetched jobs being re-seen postings, that
    serial round-tripping was the multi-minute "stuck at 16/16 sources" gap
    between scanning and ranking. Jobs that DO need writes still go through the
    per-job logic (which re-checks inside its own session, so concurrent runs
    stay race-safe).

    When ``preferred_country`` is set, postings located in a different country are
    dropped — including remote roles anchored to another country; remote is
    kept only when same-country, truly global, or unspecified (``remote_ok``). ``user_keywords`` (the
    user's Target Roles / department roles) override the default non-tech title
    gate — an accountant's or marketer's own roles must never be dropped as
    "non-tech".
    """
    from app.analytics.funnel import FunnelTracker
    from app.discovery.title_filter import keyword_hit
    from datetime import datetime
    inserted = 0

    # Cheap in-process gates first (no DB).
    candidates: List[RawJob] = []
    for r in raw_jobs:
        # Permissive gate: only skip obvious non-tech titles before any DB work
        # — unless the user's own keywords claim the title (department users).
        if is_obvious_non_tech(r.title or "") and not keyword_hit(r.title or "", user_keywords):
            continue
        # Per-user country gate: drop jobs clearly located in another country.
        if preferred_country and not _location_allowed(
            r.location or "", bool(getattr(r, "remote", False)), preferred_country, remote_ok
        ):
            continue
        candidates.append(r)
    if not candidates:
        return 0

    # Snapshot existing dedupe keys for this user in one query.
    by_key: dict = {}    # (source, external_id) -> (content_hash, last_seen)
    by_slug: dict = {}   # cross_source_slug -> source value
    prefetched = False
    try:
        with get_session() as session:
            rows = session.exec(
                select(Job.source, Job.external_id, Job.content_hash,
                       Job.last_seen, Job.cross_source_slug)
                .where(Job.user_id == user_id)
            ).all()
        for src, ext, chash, lseen, slug in rows:
            src_v = src.value if hasattr(src, "value") else str(src)
            by_key[(src_v, ext)] = (chash, lseen)
            if slug and slug not in by_slug:
                by_slug[slug] = src_v
        prefetched = True
    except Exception as e:
        # Fall back to per-job checks — slower but identical behavior.
        log.warning("Upsert prefetch failed (using per-job dedupe): %s", e)

    _now = datetime.utcnow()
    for r in candidates:
        content_hash = hashlib.sha256((r.description or "").encode("utf-8")).hexdigest()

        if prefetched:
            hit = by_key.get((r.source, r.external_id))
            if hit:
                prev_hash, prev_seen = hit
                if prev_hash == content_hash and prev_seen is not None \
                        and (_now - prev_seen).total_seconds() <= 6 * 3600:
                    continue  # unchanged, recently-seen duplicate — no DB work
            else:
                slug_v = by_slug.get(_cross_source_slug(r.company, r.title, r.location))
                if slug_v is not None:
                    try:
                        upgrades = (JobSource(r.source) in _DIRECT_ATS_SOURCES
                                    and JobSource(slug_v) not in _DIRECT_ATS_SOURCES)
                    except ValueError:
                        upgrades = False
                    if not upgrades:
                        continue  # cross-source duplicate with nothing to upgrade

        try:
            with get_session() as session:
                # 1. Check exact source + external_id duplicate — scoped to THIS
                # user, since jobs are per-user (uniqueness is user+source+ext_id).
                existing = session.exec(
                    select(Job).where(
                        Job.user_id == user_id,
                        Job.source == JobSource(r.source),
                        Job.external_id == r.external_id,
                    )
                ).first()
                if existing:
                    # Update description/content_hash if changed
                    if existing.content_hash != content_hash:
                        existing.description = r.description
                        existing.content_hash = content_hash
                        existing.embedding_id = None  # force re-embed
                        existing.last_seen = datetime.utcnow()
                        session.add(existing)
                        session.commit()
                    else:
                        # Unchanged duplicate: only refresh last_seen when it's
                        # meaningfully stale (>6h). Touching+committing every dup
                        # on every run was thousands of round-trips to Postgres
                        # per discovery pass — the bulk of a re-run's wall time.
                        _seen = existing.last_seen
                        if _seen is None or (datetime.utcnow() - _seen).total_seconds() > 6 * 3600:
                            existing.last_seen = datetime.utcnow()
                            session.add(existing)
                            session.commit()
                    continue

                # 2. Check cross-source slug duplicate — also scoped to THIS user.
                slug = _cross_source_slug(r.company, r.title, r.location)
                existing_cross = session.exec(
                    select(Job).where(
                        Job.user_id == user_id,
                        Job.cross_source_slug == slug,
                    )
                ).first()
                if existing_cross:
                    direct_ats_sources = {
                        JobSource.GREENHOUSE,
                        JobSource.LEVER,
                        JobSource.ASHBY,
                        JobSource.WORKDAY,
                        JobSource.SMARTRECRUITERS
                    }
                    # Upgrade to direct ATS version if existing was from a manual/aggregator source
                    if existing_cross.source not in direct_ats_sources and JobSource(r.source) in direct_ats_sources:
                        log.info(
                            "Discovery: Upgrading cross-source job from manual '%s' to direct ATS '%s' for '%s' @ '%s'",
                            existing_cross.source, r.source, r.title, r.company
                        )
                        existing_cross.source = JobSource(r.source)
                        existing_cross.external_id = r.external_id
                        existing_cross.url = r.url
                        existing_cross.description = r.description
                        existing_cross.content_hash = content_hash
                        existing_cross.embedding_id = None  # force re-embed
                        existing_cross.last_seen = datetime.utcnow()
                        session.add(existing_cross)
                        
                        # Update the application track to autofill
                        existing_app = session.exec(
                            select(Application).where(Application.job_id == existing_cross.id)
                        ).first()
                        if existing_app:
                            existing_app.apply_track = "autofill"
                            existing_app.apply_url = r.url
                            session.add(existing_app)
                        session.commit()
                    continue

                job = Job(
                    source=JobSource(r.source),
                    external_id=r.external_id,
                    company=r.company,
                    title=r.title,
                    location=r.location,
                    remote=r.remote,
                    url=r.url,
                    description=r.description,
                    posted_at=r.posted_at,
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    content_hash=content_hash,
                    cross_source_slug=slug,
                    user_id=user_id,
                )
                session.add(job)
                session.commit()
                session.refresh(job)
                FunnelTracker.record(job.id, "discovered", True)
                inserted += 1
                if prefetched:
                    # Keep the snapshot current so later items in THIS batch that
                    # duplicate a just-inserted job skip without a round-trip.
                    by_key[(r.source, r.external_id)] = (content_hash, job.last_seen)
                    by_slug.setdefault(slug, r.source)
        except IntegrityError:
            log.debug("IntegrityError (concurrent duplicate) skipped for '%s' @ '%s'", r.title, r.company)

    return inserted


def mark_ghost_jobs(source: str, company: str, active_external_ids: List[str], user_id: str | None = None):
    """Mark jobs that disappeared from a direct ATS board as closed.

    Scoped to a single user — jobs are per-user, so one user's discovery run must
    never close another tenant's jobs.
    """
    from datetime import datetime
    # Safety: an empty active list means we have nothing to compare against —
    # never close every job for the company on an empty/failed fetch.
    if not active_external_ids:
        log.debug("mark_ghost_jobs: empty active_external_ids for %s (%s) — skipping", company, source)
        return
    with get_session() as session:
        # Find all open jobs for this source and company (this user only)
        q = select(Job).where(
            Job.source == JobSource(source),
            Job.company == company,
            Job.is_closed == False,
        )
        if user_id is not None:
            q = q.where(Job.user_id == user_id)
        jobs = session.exec(q).all()
        
        closed_count = 0
        for job in jobs:
            if job.external_id not in active_external_ids:
                job.is_closed = True
                source_name = source.value if hasattr(source, "value") else str(source)
                job.closed_reason = f"Removed from company {source_name} ATS board"
                session.add(job)
                closed_count += 1
                
                # Update any active applications associated with this job
                app_model = session.exec(
                    select(Application).where(Application.job_id == job.id)
                ).first()
                if app_model and app_model.status not in [ApplicationStatus.SUBMITTED, ApplicationStatus.REJECTED, ApplicationStatus.SKIPPED]:
                    app_model.status = ApplicationStatus.SKIPPED
                    app_model.notes = (app_model.notes or "") + f"\nJob closed/removed from company {source_name} ATS."
                    session.add(app_model)
                    
        session.commit()
        if closed_count > 0:
            log.info("Closed %d ghost jobs for %s (%s)", closed_count, company, source)


async def feed_companies_from_aggregators(raw_jobs: List[RawJob]):
    """Extract companies from aggregator jobs, resolve their ATS, and register them."""
    from app.discovery.registry import register_discovered_companies
    from app.discovery.sources.base import DiscoveredCompany
    from app.discovery.resolver import ATSDetector
    
    discovered = []
    seen_keys = set()
    
    for r in raw_jobs:
        if not r.company:
            continue
        
        # Check if URL itself points directly to a known ATS
        detected = ATSDetector.detect_from_url(r.url)
        if detected:
            ats_type, slug = detected
            key = (ats_type, slug.lower().strip())
            if key not in seen_keys:
                seen_keys.add(key)
                discovered.append(DiscoveredCompany(
                    name=r.company,
                    slug=slug,
                    ats=ats_type,
                    career_url=r.url,
                    source="aggregator_feeder"
                ))
        # Skip companies where ATS is not detectable from URL — probing their
        # homepages via HTTP would cost 500+ sequential requests on the hot path.
                
    if discovered:
        new_regs = await register_discovered_companies(discovered)
        log.info("Aggregator feeder: registered %d new companies from %d candidates", new_regs, len(discovered))


def run_discovery(user_id: str | None = None, run_id: int | None = None,
                  keywords: list[str] | None = None, phase: str = "all") -> int:
    """Orchestrates the self-growing company graph discovery:
    1. Seed DB from fallback/bootstrap if empty.
    2. Run pluggable sources to discover candidate companies.
    3. Resolve ATS and slugs for discovered companies, register them.
    4. Run validation loop on pending/active companies (update metadata, scores).
    5. Run scrapers for all active boards in the registry (Greenhouse, Lever, Ashby, SmartRecruiters, Workday).
    6. Return total newly inserted jobs.

    ``keywords`` — the user's Target Roles. When provided, keyword-driven sources
    search for these instead of the global default list, so discovery is tailored
    to what each user is actually looking for.
    """
    import asyncio

    # Per-user target roles drive the keyword sources; fall back to the global list.
    _keywords = keywords if keywords else settings.jobs_keywords_list

    # Per-user location preference — drives the country gate in _upsert + SerpAPI query.
    _country = "United States"
    _remote_ok = True
    try:
        from app.autofill.answer_pack import _get_or_create_profile
        _p = _get_or_create_profile(user_id=user_id if user_id and user_id != "local" else None)
        if _p:
            _country = (getattr(_p, "preferred_country", "") or "United States").strip() or "United States"
            _remote_ok = bool(getattr(_p, "remote_ok", True))
    except Exception as _ce:
        log.debug("discovery country preference unavailable (default US): %s", _ce)

    # Run async parts of the pipeline: discovery, registration, validation
    async def run_discovery_async():
        # A. Seed registry
        from app.discovery.registry import seed_registry
        seed_registry()
        
        # B. Run pluggable sources
        from app.discovery.sources.builtin_fallback import BuiltinFallbackSource
        from app.discovery.sources.yc_companies import YCCompanySource
        from app.discovery.sources.search_engine import SearchEngineSource
        
        log.info("Running pluggable company discovery sources...")
        discovered_list = []
        
        # Builtin Fallback (JSON)
        try:
            fallback_src = BuiltinFallbackSource()
            res = await fallback_src.discover()
            discovered_list.extend(res)
            log.info("Fallback source returned %d candidates", len(res))
        except Exception as e:
            log.warning("Builtin Fallback discovery failed: %s", e)
            
        # YC Startup directory
        try:
            yc_src = YCCompanySource()
            res = await yc_src.discover()
            discovered_list.extend(res)
            log.info("YC Startup source returned %d candidates", len(res))
        except Exception as e:
            log.warning("YC Startup discovery failed: %s", e)
            
        # Search Engine Source (Tavily/Exa/Google)
        try:
            search_src = SearchEngineSource(keywords=_keywords)
            res = await search_src.discover()
            discovered_list.extend(res)
            log.info("Search Engine source returned %d candidates", len(res))
        except Exception as e:
            log.warning("Search Engine discovery failed: %s", e)
            
        # C. Register and resolve discovered companies
        from app.discovery.registry import register_discovered_companies
        new_regs = await register_discovered_companies(discovered_list)
        log.info("Registered %d new companies in database.", new_regs)
        
        # D. Run validation loop
        from app.discovery.registry import run_validation_loop
        validated = await run_validation_loop(limit=100)
        log.info("Validated %d candidate company boards.", validated)

    # Execute async setup — only when company-board mode is enabled.
    # In job-first mode (scrape_company_boards=False) we skip ALL company
    # discovery/registration/validation so nothing is anchored to a fixed
    # company list; jobs come purely from the aggregators in Section F.
    if settings.scrape_company_boards and phase != "fast":
        try:
            asyncio.run(run_discovery_async())
        except Exception as e:
            log.exception("Async company discovery/validation loop failed: %s", e)
    else:
        log.info("Skipping company discovery/registration/validation (mode=%s, scrape_company_boards=%s)",
                 phase, settings.scrape_company_boards)

    # E. Run scrapers for all active boards in the registry (Greenhouse, Lever, Ashby, SmartRecruiters, Workday)
    # Gated by settings.scrape_company_boards — disable for pure job-first discovery.
    total_new = 0
    from datetime import datetime as _dtm
    _run_started = _dtm.utcnow()

    # NOTE: jobs are now per-user (uniqueness = user_id+source+external_id), so
    # discovery always writes rows owned by THIS user. We deliberately do NOT
    # claim orphaned (user_id IS NULL) rows — that would hand legacy jobs to
    # whoever runs discovery first, leaking across tenants.
    source_stats: dict[str, dict] = {}  # per-source {"fetched": n, "error": "..."} for the run summary
    
    def _save_incremental():
        if run_id:
            try:
                import json
                from app.db.models import DiscoveryRun
                with get_session() as session:
                    row = session.get(DiscoveryRun, run_id)
                    if row:
                        row.source_counts = json.dumps(source_stats)
                        session.add(row)
                        session.commit()
            except Exception as _db_err:
                log.debug("Incremental save failed: %s", _db_err)

    _boards_fetched = 0
    # Direct ATS board scraping runs whenever direct_ats_enabled — this is what
    # produces live, direct-apply jobs (and drives mark_ghost_jobs open/close
    # detection). scrape_company_boards only gates the heavyweight company
    # DISCOVERY block above (YC/search-engine slug hunting).
    if (settings.direct_ats_enabled or settings.scrape_company_boards) and phase != "fast":
        if not settings.scrape_company_boards:
            # Job-first mode never ran seed_registry above — make sure the
            # bootstrap slugs exist before asking the registry for boards.
            try:
                from app.discovery.registry import seed_registry
                with get_session() as _s:
                    _has_any = _s.exec(select(CompanyRegistry).limit(1)).first()
                if not _has_any:
                    seed_registry()
            except Exception as _se:
                log.warning("Registry seed check failed: %s", _se)
        scrapers = _all_scrapers()
        if len(scrapers) > settings.max_boards_per_run:
            # Rotate fairly across runs: least-recently-validated boards first.
            log.info("Capping board scrape to %d of %d registered boards this run.",
                     settings.max_boards_per_run, len(scrapers))
            scrapers = scrapers[: settings.max_boards_per_run]
    else:
        scrapers = []
        log.info("direct_ats_enabled=False — skipping fixed-company board scraping (aggregators only)")
    log.info("Executing job scraping for %d active boards...", len(scrapers))
    def _touch_registry(source_name: str, slug: str | None, job_count: int) -> None:
        """Record a successful scrape so board rotation stays fair across runs."""
        if not slug:
            return
        try:
            from datetime import datetime as _now_dtm
            with get_session() as session:
                row = session.exec(
                    select(CompanyRegistry)
                    .where(CompanyRegistry.slug == slug)
                    .where(CompanyRegistry.ats == JobSource(source_name))
                ).first()
                if row:
                    row.last_seen = _now_dtm.utcnow()
                    row.job_count = job_count
                    session.add(row)
                    session.commit()
        except Exception as _te:
            log.debug("registry touch failed for %s/%s: %s", source_name, slug, _te)

    # Fetch boards CONCURRENTLY — sequentially, 250+ boards × 1-2s of network
    # latency each meant discovery ran for 5-10+ minutes and looked hung. The
    # fetches are pure I/O; DB writes (upsert/ghost/registry) stay serial below.
    from concurrent.futures import ThreadPoolExecutor as _BoardPool
    import time as _bt

    def _fetch_board(scraper):
        try:
            return scraper, scraper.fetch()
        except Exception as e:
            log.exception("Scraper %s failed: %s", scraper.name, e)
            return scraper, None

    _t0 = _bt.time()
    fetched_boards: list = []
    if scrapers:
        with _BoardPool(max_workers=min(12, len(scrapers))) as _pool:
            fetched_boards = list(_pool.map(_fetch_board, scrapers))
        log.info("Fetched %d boards in %.1fs", len(fetched_boards), _bt.time() - _t0)

    for scraper, raw in fetched_boards:
        try:
            if raw is not None:
                new = _upsert(raw, user_id=user_id, preferred_country=_country, remote_ok=_remote_ok, user_keywords=_keywords)
                total_new += new
                _boards_fetched += len(raw)
                _slug = (getattr(scraper, "board_slug", None) or getattr(scraper, "company_slug", None)
                         or getattr(scraper, "org_slug", None))
                _touch_registry(scraper.name, _slug, len(raw))

                # Close ghost jobs ONLY on a successful, non-empty fetch. An empty
                # result is indistinguishable from a soft failure (rate-limit /
                # transient empty response), and ghost-closing on it would wrongly
                # close every job for the company and SKIP their applications.
                if raw and len(raw) > 0:
                    active_ids = [r.external_id for r in raw]
                    company_name = raw[0].company
                    if company_name:
                        mark_ghost_jobs(scraper.name, company_name, active_ids, user_id=user_id)
            else:
                log.warning("%s scraper fetch returned None (failed)", scraper.name)
        except Exception as e:
            log.exception("Scraper %s processing failed: %s", scraper.name, e)

    # E.5 HN "Who is hiring?" monthly thread — pre-posting intelligence.
    # Postings here often predate the big boards. We upsert them directly
    # (manual-apply track) rather than routing through company discovery,
    # because most HN comments don't map to a scrapeable ATS board.
    if settings.hn_whoishiring_enabled and phase != "boards":
        try:
            from app.discovery.sources.hn_whoishiring import HNWhoIsHiringSource
            src = HNWhoIsHiringSource(keywords=_keywords)
            hn_raw = asyncio.run(src.fetch_jobs())
            source_stats["HN Who-is-hiring"] = {"fetched": len(hn_raw or [])}
            _save_incremental()
            if hn_raw:
                hn_new = _upsert(hn_raw, user_id=user_id, preferred_country=_country, remote_ok=_remote_ok, user_keywords=_keywords)
                total_new += hn_new
                log.info("HN Who-is-hiring: %d postings fetched, %d new inserted", len(hn_raw), hn_new)
        except Exception as e:
            source_stats["HN Who-is-hiring"] = {"fetched": 0, "error": str(e)[:200]}
            _save_incremental()
            log.warning("HN Who-is-hiring source failed: %s", e)

    # F. Job-board aggregators — SerpAPI (Google Jobs: LinkedIn/Indeed/Glassdoor),
    # HN, Remotive, RemoteOK. These are JOB-FIRST sources: we upsert their postings
    # directly so discovery is driven by individual roles across many companies,
    # not by a fixed list of company boards. We also feed the company names into
    # the registry as a bonus (so a future direct-ATS scrape can upgrade the row),
    # but the jobs land in the DB regardless of whether the company has a scrapeable ATS.
    async def run_direct_sources_async() -> int:
        from app.discovery.sources.serpapi import SerpAPISource
        from app.discovery.sources.hn_jobs import HNJobsSource
        from app.discovery.sources.remotive import RemotiveSource
        from app.discovery.sources.remoteok import RemoteOKSource
        from app.discovery.sources.themuse import TheMuseSource
        from app.discovery.sources.arbeitnow import ArbeitnowSource
        from app.discovery.sources.jobicy import JobicySource
        from app.discovery.sources.weworkremotely import WeWorkRemotelySource
        from app.discovery.sources.indeed_rss import IndeedRSSSource
        from app.discovery.sources.adzuna import AdzunaSource
        from app.discovery.sources.reed import ReedSource
        from app.discovery.sources.jooble import JoobleSource
        from app.discovery.sources.linkedin_rapidapi import LinkedInRapidAPISource
        from app.discovery.sources.greenhouse_search import GreenhouseKeywordSource
        from app.discovery.sources.lever_search import LeverKeywordSource

        direct_sources = [
            ("SerpAPI Google Jobs", SerpAPISource),
            ("HN Jobs (Hacker News)", HNJobsSource),
            ("Remotive", RemotiveSource),
            ("RemoteOK", RemoteOKSource),
            ("The Muse", TheMuseSource),
            ("Arbeitnow", ArbeitnowSource),
            ("Jobicy", JobicySource),
            ("WeWorkRemotely", WeWorkRemotelySource),
            ("Indeed RSS", IndeedRSSSource),
            ("Adzuna", AdzunaSource),
            ("Reed.co.uk", ReedSource),
            ("Jooble", JoobleSource),
            ("LinkedIn (RapidAPI)", LinkedInRapidAPISource),
            ("Greenhouse (keyword search)", GreenhouseKeywordSource),
            ("Lever (keyword search)", LeverKeywordSource),
        ]

        # Fetch all aggregator sources concurrently with a per-source timeout so
        # one slow/hanging source can't stall the whole run (which would leave the
        # UI spinning with no summary).
        # Country-aware sources receive the user's preferred country for global sourcing.
        _COUNTRY_AWARE_SOURCES = {"SerpAPI Google Jobs", "Adzuna", "Reed.co.uk", "Jooble",
                                  "LinkedIn (RapidAPI)"}
        async def _fetch_one(name, src_cls):
            try:
                if name in _COUNTRY_AWARE_SOURCES:
                    src = src_cls(keywords=_keywords, country=_country)
                else:
                    src = src_cls(keywords=_keywords)
                raw_jobs = await asyncio.wait_for(src.fetch_jobs(), timeout=45)
                source_stats[name] = {"fetched": len(raw_jobs)}
                log.info("%s: fetched %d jobs", name, len(raw_jobs))
                _save_incremental()
                return raw_jobs
            except asyncio.TimeoutError:
                source_stats[name] = {"fetched": 0, "error": "timed out after 45s"}
                log.warning("Direct source '%s' timed out", name)
                _save_incremental()
            except Exception as e:
                source_stats[name] = {"fetched": 0, "error": str(e)[:200]}
                log.warning("Direct source '%s' failed: %s", name, e)
                _save_incremental()
            return []

        all_raw_jobs = []
        results = await asyncio.gather(*[_fetch_one(n, c) for n, c in direct_sources])
        for r in results:
            all_raw_jobs.extend(r)

        inserted = 0
        if all_raw_jobs:
            # JOB-FIRST: insert the postings directly into the DB.
            inserted = _upsert(all_raw_jobs, user_id=user_id, preferred_country=_country, remote_ok=_remote_ok, user_keywords=_keywords)
            log.info("Aggregator job-first upsert: %d new jobs from %d fetched", inserted, len(all_raw_jobs))
            # Stash the raw jobs so company-registration can run AFTER the run
            # summary is written (it's slow and must not delay the UI breakdown).
            _direct_raw_holder.extend(all_raw_jobs)

        return inserted

    _direct_raw_holder = []
    if phase != "boards":
        try:
            direct_new = asyncio.run(run_direct_sources_async())
            total_new += direct_new
        except Exception as e:
            log.exception("Direct job-board sources failed: %s", e)

    log.info("Discovery complete. Total new jobs inserted: %d", total_new)

    # Persist the per-source summary FIRST so the UI gets the breakdown quickly,
    # before the slower best-effort post-processing below.
    if _boards_fetched:
        source_stats["Company ATS boards"] = {"fetched": _boards_fetched}
    try:
        _write_discovery_run(user_id, source_stats, total_new, _run_started,
                             run_id=run_id, status="ranking" if run_id else "done")
    except Exception as e:
        log.warning("Could not write discovery run summary (non-fatal): %s", e)

    # Slow, best-effort post-processing — runs after the summary so it never
    # delays the UI. Registers companies for future direct-ATS upgrades, and
    # upgrades shortlisted jobs to autofill where an ATS is detectable.
    if _direct_raw_holder:
        try:
            asyncio.run(feed_companies_from_aggregators(_direct_raw_holder))
        except Exception as e:
            log.warning("Aggregator company feeder failed (non-fatal): %s", e)
    try:
        ats_upgraded = run_ats_upgrade_for_shortlisted()
        log.info("ATS upgrade: registered %d boards from shortlisted companies", ats_upgraded)
    except Exception as e:
        log.warning("Selective ATS upgrade failed (non-fatal): %s", e)

    return total_new


def _write_discovery_run(user_id, source_stats: dict, total_inserted: int, started_at,
                         run_id: int | None = None, status: str = "done") -> None:
    """Record per-source counts. Updates an existing run row if run_id is given."""
    import json
    from datetime import datetime
    from app.db.models import DiscoveryRun
    total_fetched = sum(int(v.get("fetched", 0)) for v in source_stats.values())
    with get_session() as session:
        if run_id:
            row = session.get(DiscoveryRun, run_id)
            if row:
                try:
                    prev = json.loads(row.source_counts or "{}")
                except (ValueError, TypeError):
                    prev = {}
                merged = {**prev, **source_stats}
                row.total_fetched = sum(int(v.get("fetched", 0)) for v in merged.values())
                row.total_inserted = (row.total_inserted or 0) + total_inserted
                row.source_counts = json.dumps(merged)
                row.status = status
                session.add(row)
                session.commit()
                return
        session.add(DiscoveryRun(
            user_id=user_id,
            started_at=started_at,
            finished_at=datetime.utcnow(),
            total_fetched=total_fetched,
            total_inserted=total_inserted,
            source_counts=json.dumps(source_stats),
            status=status,
        ))
        session.commit()


def create_discovery_run(user_id) -> int | None:
    """Create a 'discovering' run row up-front so the UI can show live status."""
    from datetime import datetime
    from app.db.models import DiscoveryRun
    try:
        with get_session() as session:
            row = DiscoveryRun(user_id=user_id, started_at=datetime.utcnow(),
                               status="discovering", source_counts="{}")
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id
    except Exception as e:
        log.warning("Could not create discovery run: %s", e)
        return None


def finish_discovery_run(run_id: int | None, status: str,
                         total_shortlisted: int = 0, error: str | None = None) -> None:
    """Mark a run row as done/error with the final shortlist count."""
    from datetime import datetime
    from app.db.models import DiscoveryRun, UserNotification
    if not run_id:
        return
    try:
        with get_session() as session:
            row = session.get(DiscoveryRun, run_id)
            if not row:
                return
            row.status = status
            row.total_shortlisted = total_shortlisted
            if error:
                row.error = error[:300]
            row.finished_at = datetime.utcnow()
            session.add(row)
            
            # Create in-app notification for the user
            try:
                if status == "done":
                    title = "Job Discovery Completed 🔍"
                    message = f"Found {row.total_fetched} new postings. Shortlisted {total_shortlisted} high-fit matches."
                else:
                    title = "Job Discovery Failed ⚠️"
                    message = f"Discovery run encountered an error: {error or 'Unknown error'}"
                
                notif = UserNotification(
                    user_id=row.user_id,
                    title=title,
                    message=message,
                    type="discovery_completed",
                    link="/dashboard",
                )
                session.add(notif)
            except Exception as ne:
                log.warning("Failed to create discovery run notification: %s", ne)
                
            session.commit()
    except Exception as e:
        log.warning("Could not finish discovery run: %s", e)


def run_ats_upgrade_for_shortlisted() -> int:
    """Scan shortlisted/tailored applications whose apply_track is 'manual'.
    If the job URL points to a known ATS (Greenhouse/Lever/Ashby), register
    the company board so the next scrape picks it up and upgrades the job
    to autofill-capable.

    Returns the number of boards registered.
    """
    from app.discovery.resolver import ATSDetector
    from datetime import datetime

    _AUTOFILL_ATS = {
        "greenhouse": JobSource.GREENHOUSE,
        "lever": JobSource.LEVER,
        "ashby": JobSource.ASHBY,
    }

    registered = 0
    with get_session() as session:
        # Find manual-track applications that are shortlisted or tailored
        apps = session.exec(
            select(Application, Job).join(Job, Application.job_id == Job.id).where(
                Application.apply_track == "manual",
                Application.status.in_([
                    ApplicationStatus.SHORTLISTED,
                    ApplicationStatus.TAILORED,
                    ApplicationStatus.MATCHED,
                ]),
            )
        ).all()

        for app, job in apps:
            if not job.url:
                continue

            detected = ATSDetector.detect_from_url(job.url)
            if not detected:
                continue

            ats_name, slug = detected
            if ats_name not in _AUTOFILL_ATS:
                continue

            ats_enum = _AUTOFILL_ATS[ats_name]

            # Check if this board is already registered
            existing = session.exec(
                select(CompanyRegistry).where(
                    CompanyRegistry.slug == slug,
                    CompanyRegistry.ats == ats_enum,
                )
            ).first()

            if existing:
                # Just make sure it's active
                if not existing.is_active:
                    existing.is_active = True
                    existing.inactive_reason = None
                    session.add(existing)
            else:
                # Register the new board
                new_reg = CompanyRegistry(
                    slug=slug,
                    ats=ats_enum,
                    company_name=job.company,
                    career_url=job.url,
                    source="shortlist_ats_upgrade",
                    confidence_score=100,
                    is_active=True,
                    first_seen=datetime.utcnow(),
                )
                session.add(new_reg)
                registered += 1
                log.info(
                    "ATS upgrade: registered %s board '%s' for '%s' (from shortlisted job)",
                    ats_name, slug, job.company,
                )

            # Upgrade the application track
            app.apply_track = "autofill"
            app.apply_url = job.url
            session.add(app)

        session.commit()

    return registered


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    n = run_discovery()
    print(f"Inserted {n} new jobs.")
