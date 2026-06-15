from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List, Optional
import httpx
from sqlmodel import select

from app.config import settings
from app.db.init_db import get_session
from app.db.models import CompanyRegistry, JobSource

log = logging.getLogger(__name__)

def seed_registry() -> int:
    """Read the .env configured boards and the bootstrap lists, and insert them if not present."""
    import json
    count = 0
    
    bootstrap = {"greenhouse": [], "lever": [], "ashby": []}
    if settings.bootstrap_path.exists():
        try:
            with open(settings.bootstrap_path, "r", encoding="utf-8") as f:
                bootstrap = json.load(f)
        except Exception as e:
            log.error("Failed to load bootstrap JSON: %s", e)
            
    # Collect all sources
    greenhouse_slugs = set(settings.greenhouse_boards_list + bootstrap.get("greenhouse", []))
    lever_slugs = set(settings.lever_boards_list + bootstrap.get("lever", []))
    ashby_slugs = set(settings.ashby_boards_list + bootstrap.get("ashby", []))

    with get_session() as session:
        # Greenhouse
        for slug in greenhouse_slugs:
            slug = slug.strip().lower()
            if not slug:
                continue
            existing = session.exec(
                select(CompanyRegistry).where(
                    CompanyRegistry.slug == slug,
                    CompanyRegistry.ats == JobSource.GREENHOUSE
                )
            ).first()
            if not existing:
                session.add(CompanyRegistry(slug=slug, ats=JobSource.GREENHOUSE, source="seed"))
                count += 1

        # Lever
        for slug in lever_slugs:
            slug = slug.strip().lower()
            if not slug:
                continue
            existing = session.exec(
                select(CompanyRegistry).where(
                    CompanyRegistry.slug == slug,
                    CompanyRegistry.ats == JobSource.LEVER
                )
            ).first()
            if not existing:
                session.add(CompanyRegistry(slug=slug, ats=JobSource.LEVER, source="seed"))
                count += 1

        # Ashby
        for slug in ashby_slugs:
            slug = slug.strip().lower()
            if not slug:
                continue
            existing = session.exec(
                select(CompanyRegistry).where(
                    CompanyRegistry.slug == slug,
                    CompanyRegistry.ats == JobSource.ASHBY
                )
            ).first()
            if not existing:
                session.add(CompanyRegistry(slug=slug, ats=JobSource.ASHBY, source="seed"))
                count += 1

        session.commit()
    log.info("Registry seeded with %d new board entries.", count)
    return count

def calculate_confidence_score(ats: JobSource, job_count: int, is_active: bool, source: str) -> int:
    score = 40
    if is_active:
        score += 30
    if job_count > 0:
        score += 20
    if source in ["seed", "google_discovery", "search_engine", "yc_directory"]:
        score += 10
    return max(0, min(100, score))


def calculate_target_fit_score(company_name: str, job_titles: List[str]) -> float:
    score = 0.0
    keywords = ["machine learning", "ml", "ai", "llm", "python", "artificial intelligence"]
    for title in job_titles:
        title_lower = title.lower()
        if any(kw in title_lower for kw in keywords):
            score += 10.0
    return min(100.0, score)


async def validate_slug(slug: str, ats: JobSource, career_url: Optional[str] = None) -> tuple[bool, int, List[str], Optional[str]]:
    """Hits public ATS API or URL once to verify if active.
    Returns (is_active, job_count, job_titles, error_msg).
    """
    client = httpx.AsyncClient(timeout=10.0, follow_redirects=True, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    is_active = False
    job_count = 0
    job_titles = []
    error_msg = None
    
    try:
        if ats == JobSource.GREENHOUSE:
            url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
            r = await client.get(url)
            if r.status_code == 200:
                payload = r.json()
                jobs = payload.get("jobs", [])
                is_active = len(jobs) > 0
                job_count = len(jobs)
                job_titles = [j.get("title", "") for j in jobs]
            else:
                error_msg = f"API returned status {r.status_code}"
        elif ats == JobSource.LEVER:
            url = f"https://api.lever.co/v0/postings/{slug}?mode=json"
            r = await client.get(url)
            if r.status_code == 200:
                payload = r.json()
                is_active = len(payload) > 0
                job_count = len(payload)
                job_titles = [j.get("title", "") for j in payload]
            else:
                error_msg = f"API returned status {r.status_code}"
        elif ats == JobSource.ASHBY:
            url = f"https://api.ashbyhq.com/posting-api/job-board/{slug}"
            r = await client.get(url)
            if r.status_code == 200:
                payload = r.json()
                jobs = payload.get("jobs", [])
                is_active = len(jobs) > 0
                job_count = len(jobs)
                job_titles = [j.get("title", "") for j in jobs]
            else:
                error_msg = f"API returned status {r.status_code}"
        elif ats == JobSource.SMARTRECRUITERS:
            url = f"https://api.smartrecruiters.com/v1/companies/{slug}/postings"
            r = await client.get(url)
            if r.status_code == 200:
                payload = r.json()
                jobs = payload.get("content", [])
                is_active = len(jobs) > 0
                job_count = len(jobs)
                job_titles = [j.get("name", "") for j in jobs]
            else:
                error_msg = f"API returned status {r.status_code}"
        elif ats == JobSource.WORKDAY:
            from app.discovery.workday import parse_workday_url
            domain, tenant, site = parse_workday_url(career_url, slug)
            url = f"https://{domain}/wday/cxs/{tenant}/{site}/jobs"
            payload = {
                "appliedFacets": {},
                "limit": 100,
                "offset": 0,
                "searchText": ""
            }
            r = await client.post(url, json=payload)
            if r.status_code == 200:
                payload = r.json()
                jobs = payload.get("jobPostings", [])
                is_active = len(jobs) > 0
                job_count = payload.get("total", len(jobs))
                job_titles = [j.get("title", "") for j in jobs]
            else:
                error_msg = f"API returned status {r.status_code}"
        else:
            # For workable, bamboohr, etc., hit their career_url
            target_url = career_url
            if not target_url:
                if ats == JobSource.WORKABLE:
                    target_url = f"https://apply.workable.com/{slug}/"
                elif ats == JobSource.BAMBOOHR:
                    target_url = f"https://{slug}.bamboohr.com/jobs/"
                elif ats == JobSource.TEAMTAILOR:
                    target_url = f"https://{slug}.teamtailor.com/"
            
            if target_url:
                r = await client.get(target_url)
                is_active = r.status_code in [200, 301, 302]
                if not is_active:
                    error_msg = f"Hiring page returned status {r.status_code}"
            else:
                error_msg = "No career URL available for manual ATS"
    except Exception as e:
        error_msg = str(e)
        log.debug("Probing slug '%s' (%s) failed: %s", slug, ats.value, e)
    finally:
        await client.aclose()
        
    return is_active, job_count, job_titles, error_msg


async def check_h1b_sponsorship(company_name: str) -> str:
    """Query Tavily/Exa to check if the company has H-1B sponsorship history."""
    if not settings.tavily_api_key:
        return "Unknown (No API Key)"
        
    query = f'"{company_name}" H-1B sponsorship history myvisajobs h1bgrader'
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": 3
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(url, json=payload)
            if r.status_code == 200:
                results = r.json().get("results", [])
                text = " ".join([res.get("content", "").lower() for res in results])
                
                # Check for positive signals
                if any(x in text for x in ["sponsored", "sponsors h1b", "h-1b visa sponsor", "h1b visa sponsor", "number of h1b", "lcas approved"]):
                    return "Yes (Historical Sponsor)"
                elif any(x in text for x in ["does not sponsor", "no h1b visa", "never sponsored"]):
                    return "No (Unlikely to Sponsor)"
                else:
                    return "Unknown (No clear visa history)"
    except Exception as e:
        log.warning("Sponsorship check failed for %s: %s", company_name, e)
    return "Unknown"


async def run_validation_loop(limit: int = 100) -> int:
    """Selects unvalidated or oldest last-seen slugs, validates them, 
    calculates scores, and implements self-healing retry policy.
    """
    log.info("Starting Company Registry Validator Job...")

    now = datetime.utcnow()
    with get_session() as session:
        rows = session.exec(
            select(CompanyRegistry)
            .where((CompanyRegistry.next_retry_at.is_(None)) | (CompanyRegistry.next_retry_at <= now))
            .order_by(CompanyRegistry.target_fit_score.desc(), CompanyRegistry.last_seen.asc())
            .limit(limit)
        ).all()
        candidates = [
            (c.id, c.slug, c.ats, c.career_url, c.company_name, c.source)
            for c in rows
        ]

    if not candidates:
        log.info("No companies require validation retry at this time.")
        return 0

    log.info("Validating %d company boards...", len(candidates))
    validated_count = 0

    for comp_id, slug, ats, career_url, company_name, source in candidates:
        is_active, job_count, job_titles, error_msg = await validate_slug(slug, ats, career_url)

        conf = calculate_confidence_score(ats, job_count, is_active, source)
        fit = calculate_target_fit_score(company_name or slug, job_titles)
        
        # Check visa sponsorship if not set or Unknown
        sponsorship = "Unknown"
        with get_session() as session:
            current_comp = session.get(CompanyRegistry, comp_id)
            if current_comp:
                sponsorship = current_comp.sponsorship_signal or "Unknown"
        
        if is_active and (sponsorship == "Unknown" or not sponsorship):
            sponsorship = await check_h1b_sponsorship(company_name or slug)
            
        with get_session() as session:
            db_comp = session.get(CompanyRegistry, comp_id)
            if db_comp:
                db_comp.is_active = is_active
                db_comp.job_count = job_count
                db_comp.confidence_score = conf
                db_comp.target_fit_score = fit
                db_comp.sponsorship_signal = sponsorship
                db_comp.last_validated_at = datetime.utcnow()
                db_comp.last_seen = datetime.utcnow()
                
                if is_active:
                    db_comp.failure_count = 0
                    db_comp.last_error = None
                    db_comp.inactive_reason = None
                    db_comp.next_retry_at = datetime.utcnow() + timedelta(days=2)
                else:
                    db_comp.failure_count += 1
                    db_comp.last_error = error_msg
                    
                    if db_comp.failure_count >= 7:
                        db_comp.is_active = False
                        db_comp.inactive_reason = f"Failed 7 consecutive validation checks. Last error: {error_msg}"
                        db_comp.next_retry_at = datetime.utcnow() + timedelta(days=30)
                        log.warning("Company Registry: Board '%s' (%s) marked INACTIVE after 7 failures.", db_comp.slug, db_comp.ats.value)
                    elif db_comp.failure_count >= 3:
                        db_comp.next_retry_at = datetime.utcnow() + timedelta(days=5)
                    else:
                        db_comp.next_retry_at = datetime.utcnow() + timedelta(days=1)
                        
                session.add(db_comp)
                session.commit()
                validated_count += 1
                
        import asyncio
        await asyncio.sleep(0.5)
        
    log.info("Validation cycle complete. Successfully validated %d companies.", validated_count)
    return validated_count


def _insert_company_if_new(comp_name: str, slug: str, ats_source: JobSource, career_url: str, source: str) -> bool:
    """Insert a company into CompanyRegistry if not already present. Returns True if inserted."""
    with get_session() as session:
        existing = session.exec(
            select(CompanyRegistry).where(
                CompanyRegistry.slug == slug,
                CompanyRegistry.ats == ats_source
            )
        ).first()
        if not existing:
            session.add(CompanyRegistry(
                slug=slug,
                ats=ats_source,
                company_name=comp_name,
                career_url=career_url,
                source=source,
                is_active=True,
                confidence_score=50,
                target_fit_score=0.0
            ))
            session.commit()
            log.info("Registered new discovered company: %s (%s) using %s", comp_name, slug, ats_source.value)
            return True
    return False


async def register_discovered_companies(discovered: List[DiscoveredCompany]) -> int:
    """Register pre-resolved companies (ATS type already known) into CompanyRegistry.

    Companies with unknown ATS are skipped — HTTP probing is deferred to the
    background validation loop to avoid blocking the discovery hot path.
    """
    import asyncio
    from app.discovery.sources.base import DiscoveredCompany

    new_added = 0

    async def _register_one(comp: DiscoveredCompany) -> int:
        ats = comp.ats
        slug = comp.slug
        career_url = comp.career_url

        if ats in ["yc_domain", "unknown"]:
            # Skip — HTTP probing removed from hot path
            return 0

        try:
            ats_source = JobSource(ats.lower().strip())
        except ValueError:
            log.debug("Unknown ATS type: %s for %s", ats, comp.name)
            return 0

        slug = slug.strip().lower()
        if not slug:
            return 0

        inserted = await asyncio.get_event_loop().run_in_executor(
            None, _insert_company_if_new, comp.name, slug, ats_source, career_url or "", comp.source
        )
        return 1 if inserted else 0

    results = await asyncio.gather(*[_register_one(c) for c in discovered], return_exceptions=True)
    for r in results:
        if isinstance(r, int):
            new_added += r
        else:
            log.debug("register_discovered_companies error: %s", r)

    return new_added

