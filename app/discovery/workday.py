"""Workday public CXS (Career Site External) API scraper.

Bypasses browser automation by hitting the JSON API directly.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import List, Tuple
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.discovery.base import RawJob

log = logging.getLogger(__name__)

# Lightweight tech title filter to avoid fetching details for obvious non-tech jobs
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

def _is_obvious_non_tech(title: str) -> bool:
    if _NON_TECH_TITLE_RE.search(title):
        if _TECH_TITLE_RE.search(title):
            return False
        return True
    return False

def _strip_html(html: str) -> str:
    return BeautifulSoup(html or "", "html.parser").get_text(separator="\n").strip()


def parse_workday_url(career_url: str | None, slug: str) -> Tuple[str, str, str]:
    """Parse career URL to extract domain, tenant, and site.
    Fallback to slug if career_url is not a workday URL.
    """
    if not career_url:
        # Fallback logic
        if "." in slug:
            tenant = slug.split(".")[0]
            domain = f"{slug}.myworkdayjobs.com"
        else:
            tenant = slug
            domain = f"{slug}.myworkdayjobs.com"
        return domain, tenant, "External"
        
    parsed = urlparse(career_url)
    hostname = parsed.hostname or f"{slug}.myworkdayjobs.com"
    
    # Extract tenant from domain (first segment before .myworkdayjobs or .wdX)
    tenant = hostname.split(".")[0]
    
    # Extract site from path
    path_parts = [p for p in parsed.path.split("/") if p]
    site = "External"
    for part in path_parts:
        if part.lower() in ["jobs", "job", "login", "wday"]:
            continue
        # Skip language codes (e.g., en-US)
        if re.match(r"^[a-z]{2}-[A-Z]{2}$", part) or re.match(r"^[a-z]{2}$", part):
            continue
        site = part
        break
        
    return hostname, tenant, site


class WorkdayScraper:
    name = "workday"

    def __init__(self, company_slug: str, career_url: str | None = None):
        self.company_slug = company_slug
        self.career_url = career_url

    def fetch(self) -> List[RawJob] | None:
        domain, tenant, site = parse_workday_url(self.career_url, self.company_slug)
        url = f"https://{domain}/wday/cxs/{tenant}/{site}/jobs"
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        jobs: List[RawJob] = []
        offset = 0
        limit = 20
        max_total = 100  # Cap total jobs fetched per company run to avoid timeouts
        
        try:
            while len(jobs) < max_total:
                payload = {
                    "appliedFacets": {},
                    "limit": limit,
                    "offset": offset,
                    "searchText": ""
                }
                r = httpx.post(url, json=payload, headers=headers, timeout=30.0)
                if r.status_code != 200:
                    log.warning("Workday fetch failed for %s: HTTP %d", tenant, r.status_code)
                    # If offset is 0, this is a fatal run error
                    return None if offset == 0 else jobs
                    
                data = r.json()
                postings = data.get("jobPostings", [])
                if not postings:
                    break
                    
                for p in postings:
                    title = p.get("title", "")
                    if _is_obvious_non_tech(title):
                        continue
                        
                    ext_path = p.get("externalPath")
                    if not ext_path:
                        continue
                        
                    # Fetch details
                    path_suffix = ext_path if ext_path.startswith("/job") else f"/job{ext_path}"
                    detail_url = f"https://{domain}/wday/cxs/{tenant}/{site}{path_suffix}"
                    try:
                        dr = httpx.get(detail_url, headers=headers, timeout=15.0)
                        if dr.status_code != 200:
                            continue
                        detail_data = dr.json()
                    except Exception as e:
                        log.debug("Workday: failed to fetch job details for %s: %s", ext_path, e)
                        continue
                        
                    info = detail_data.get("jobPostingInfo", {})
                    description = _strip_html(info.get("jobDescription", ""))
                    
                    req_id = info.get("jobReqId") or p.get("bulletFields", [None])[0] or ext_path.split("_")[-1]
                    location = info.get("location") or p.get("locationsText") or ""  # coerce null → ""
                    remote = "remote" in location.lower()
                    
                    posted = info.get("startDate")
                    posted_dt = None
                    if posted:
                        try:
                            posted_dt = datetime.strptime(posted, "%Y-%m-%d")
                        except Exception:
                            posted_dt = None
                            
                    apply_url = f"https://{domain}/{site}{ext_path}"
                    
                    jobs.append(
                        RawJob(
                            source="workday",
                            external_id=str(req_id),
                            company=tenant.replace("-", " ").replace("_", " ").title(),
                            title=title,
                            location=location,
                            remote=remote,
                            url=apply_url,
                            description=description,
                            posted_at=posted_dt,
                        )
                    )
                    
                    if len(jobs) >= max_total:
                        break
                        
                # Next page
                total = data.get("total", 0)
                offset += limit
                if offset >= total:
                    break
                    
        except httpx.HTTPError as e:
            log.warning("Workday connection failed for %s: %s", tenant, e)
            return None if offset == 0 else jobs
            
        log.info("Workday[%s]: %d tech jobs parsed successfully", tenant, len(jobs))
        return jobs
