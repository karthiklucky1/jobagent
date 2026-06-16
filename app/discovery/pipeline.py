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
from app.discovery.greenhouse import GreenhouseScraper
from app.discovery.lever import LeverScraper
from app.discovery.smartrecruiters import SmartRecruitersScraper
from app.discovery.workday import WorkdayScraper

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
    
    # 1. Query active boards from the DB registry
    try:
        with get_session() as session:
            db_companies = session.exec(
                select(CompanyRegistry).where(CompanyRegistry.is_active == True)
            ).all()
            
            for comp in db_companies:
                if comp.ats == JobSource.GREENHOUSE:
                    scrapers.append(GreenhouseScraper(comp.slug))
                elif comp.ats == JobSource.LEVER:
                    scrapers.append(LeverScraper(comp.slug))
                elif comp.ats == JobSource.ASHBY:
                    scrapers.append(AshbyScraper(comp.slug))
                elif comp.ats == JobSource.SMARTRECRUITERS:
                    scrapers.append(SmartRecruitersScraper(comp.slug))
                elif comp.ats == JobSource.WORKDAY:
                    scrapers.append(WorkdayScraper(comp.slug, comp.career_url))
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


def _upsert(raw_jobs: List[RawJob], user_id: str | None = None) -> int:
    """Insert new jobs; skip duplicates by (source, external_id) and cross-source slug.

    Each job is committed individually so a race-condition IntegrityError from
    concurrent scrapers only drops that one duplicate rather than rolling back
    the entire batch.
    """
    from app.analytics.funnel import FunnelTracker
    from datetime import datetime
    inserted = 0

    for r in raw_jobs:
        # Permissive gate: only skip obvious non-tech titles before any DB work
        if is_obvious_non_tech(r.title or ""):
            continue
            
        content_hash = hashlib.sha256((r.description or "").encode("utf-8")).hexdigest()
        
        try:
            with get_session() as session:
                # 1. Check exact source + external_id duplicate
                existing = session.exec(
                    select(Job).where(
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
                        existing.last_seen = datetime.utcnow()
                        session.add(existing)
                        session.commit()
                    continue

                # 2. Check cross-source slug duplicate
                slug = _cross_source_slug(r.company, r.title, r.location)
                existing_cross = session.exec(
                    select(Job).where(Job.cross_source_slug == slug)
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
        except IntegrityError:
            log.debug("IntegrityError (concurrent duplicate) skipped for '%s' @ '%s'", r.title, r.company)

    return inserted


def mark_ghost_jobs(source: str, company: str, active_external_ids: List[str]):
    """Mark jobs that disappeared from a direct ATS board as closed."""
    from datetime import datetime
    with get_session() as session:
        # Find all open jobs for this source and company
        jobs = session.exec(
            select(Job).where(
                Job.source == JobSource(source),
                Job.company == company,
                Job.is_closed == False
            )
        ).all()
        
        closed_count = 0
        for job in jobs:
            if job.external_id not in active_external_ids:
                job.is_closed = True
                session.add(job)
                closed_count += 1
                
                # Update any active applications associated with this job
                app_model = session.exec(
                    select(Application).where(Application.job_id == job.id)
                ).first()
                if app_model and app_model.status not in [ApplicationStatus.SUBMITTED, ApplicationStatus.REJECTED, ApplicationStatus.SKIPPED]:
                    app_model.status = ApplicationStatus.SKIPPED
                    app_model.notes = (app_model.notes or "") + "\nJob closed/removed from ATS."
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


def run_discovery(user_id: str | None = None) -> int:
    """Orchestrates the self-growing company graph discovery:
    1. Seed DB from fallback/bootstrap if empty.
    2. Run pluggable sources to discover candidate companies.
    3. Resolve ATS and slugs for discovered companies, register them.
    4. Run validation loop on pending/active companies (update metadata, scores).
    5. Run scrapers for all active boards in the registry (Greenhouse, Lever, Ashby, SmartRecruiters, Workday).
    6. Return total newly inserted jobs.
    """
    import asyncio

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
            search_src = SearchEngineSource(keywords=settings.jobs_keywords_list)
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
    if settings.scrape_company_boards:
        try:
            asyncio.run(run_discovery_async())
        except Exception as e:
            log.exception("Async company discovery/validation loop failed: %s", e)
    else:
        log.info("scrape_company_boards=False — skipping company discovery/registration/validation (job-first mode)")

    # E. Run scrapers for all active boards in the registry (Greenhouse, Lever, Ashby, SmartRecruiters, Workday)
    # Gated by settings.scrape_company_boards — disable for pure job-first discovery.
    total_new = 0
    from datetime import datetime as _dtm
    _run_started = _dtm.utcnow()
    source_stats: dict[str, dict] = {}  # per-source {"fetched": n, "error": "..."} for the run summary
    _boards_fetched = 0
    scrapers = _all_scrapers() if settings.scrape_company_boards else []
    if not settings.scrape_company_boards:
        log.info("scrape_company_boards=False — skipping fixed-company board scraping (job-first mode)")
    log.info("Executing job scraping for %d active boards...", len(scrapers))
    for scraper in scrapers:
        try:
            raw = scraper.fetch()
            if raw is not None:
                new = _upsert(raw, user_id=user_id)
                total_new += new
                _boards_fetched += len(raw)
                
                # Close ghost jobs: any job in our DB for this source/company that was not fetched
                active_ids = [r.external_id for r in raw]
                company_name = getattr(scraper, "company_slug", None) or getattr(scraper, "board_slug", None) or getattr(scraper, "org_slug", None)
                if company_name:
                    company_name = company_name.replace("-", " ").replace("_", " ").title()
                if raw and len(raw) > 0:
                    company_name = raw[0].company
                if company_name:
                    mark_ghost_jobs(scraper.name, company_name, active_ids)
            else:
                log.warning("%s scraper fetch returned None (failed)", scraper.name)
        except Exception as e:
            log.exception("Scraper %s failed: %s", scraper.name, e)

    # E.5 HN "Who is hiring?" monthly thread — pre-posting intelligence.
    # Postings here often predate the big boards. We upsert them directly
    # (manual-apply track) rather than routing through company discovery,
    # because most HN comments don't map to a scrapeable ATS board.
    if settings.hn_whoishiring_enabled:
        try:
            from app.discovery.sources.hn_whoishiring import HNWhoIsHiringSource
            src = HNWhoIsHiringSource(keywords=settings.jobs_keywords_list)
            hn_raw = asyncio.run(src.fetch_jobs())
            source_stats["HN Who-is-hiring"] = {"fetched": len(hn_raw or [])}
            if hn_raw:
                hn_new = _upsert(hn_raw, user_id=user_id)
                total_new += hn_new
                log.info("HN Who-is-hiring: %d postings fetched, %d new inserted", len(hn_raw), hn_new)
        except Exception as e:
            source_stats["HN Who-is-hiring"] = {"fetched": 0, "error": str(e)[:200]}
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
        async def _fetch_one(name, src_cls):
            try:
                src = src_cls(keywords=settings.jobs_keywords_list)
                raw_jobs = await asyncio.wait_for(src.fetch_jobs(), timeout=45)
                source_stats[name] = {"fetched": len(raw_jobs)}
                log.info("%s: fetched %d jobs", name, len(raw_jobs))
                return raw_jobs
            except asyncio.TimeoutError:
                source_stats[name] = {"fetched": 0, "error": "timed out after 45s"}
                log.warning("Direct source '%s' timed out", name)
            except Exception as e:
                source_stats[name] = {"fetched": 0, "error": str(e)[:200]}
                log.warning("Direct source '%s' failed: %s", name, e)
            return []

        all_raw_jobs = []
        results = await asyncio.gather(*[_fetch_one(n, c) for n, c in direct_sources])
        for r in results:
            all_raw_jobs.extend(r)

        inserted = 0
        if all_raw_jobs:
            # JOB-FIRST: insert the postings directly into the DB.
            inserted = _upsert(all_raw_jobs, user_id=user_id)
            log.info("Aggregator job-first upsert: %d new jobs from %d fetched", inserted, len(all_raw_jobs))
            # Bonus: also register the companies so a direct-ATS scrape can later
            # upgrade these rows to autofill-capable boards (best-effort, non-fatal).
            try:
                await feed_companies_from_aggregators(all_raw_jobs)
            except Exception as e:
                log.warning("Aggregator company feeder failed (non-fatal): %s", e)

        return inserted

    try:
        direct_new = asyncio.run(run_direct_sources_async())
        total_new += direct_new
    except Exception as e:
        log.exception("Direct job-board sources failed: %s", e)

    log.info("Discovery complete. Total new jobs inserted: %d", total_new)

    # G. Selective ATS upgrade — for shortlisted/tailored jobs, detect if the
    # company uses Greenhouse/Lever/Ashby from the URL and register only those
    # boards. This converts apply_track from "manual" → "autofill" for jobs
    # that already matched the resume.
    try:
        ats_upgraded = run_ats_upgrade_for_shortlisted()
        log.info("ATS upgrade: registered %d boards from shortlisted companies", ats_upgraded)
    except Exception as e:
        log.warning("Selective ATS upgrade failed (non-fatal): %s", e)

    # Persist a per-source summary of this run so the UI can show where jobs came from.
    if _boards_fetched:
        source_stats["Company ATS boards"] = {"fetched": _boards_fetched}
    try:
        _write_discovery_run(user_id, source_stats, total_new, _run_started)
    except Exception as e:
        log.warning("Could not write discovery run summary (non-fatal): %s", e)

    return total_new


def _write_discovery_run(user_id, source_stats: dict, total_inserted: int, started_at) -> None:
    """Save a DiscoveryRun row summarizing per-source fetch counts."""
    import json
    from datetime import datetime
    from app.db.models import DiscoveryRun
    total_fetched = sum(int(v.get("fetched", 0)) for v in source_stats.values())
    with get_session() as session:
        session.add(DiscoveryRun(
            user_id=user_id,
            started_at=started_at,
            finished_at=datetime.utcnow(),
            total_fetched=total_fetched,
            total_inserted=total_inserted,
            source_counts=json.dumps(source_stats),
            status="done",
        ))
        session.commit()


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
