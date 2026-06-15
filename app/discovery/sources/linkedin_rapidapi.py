"""LinkedIn Jobs via RapidAPI — ~$10/mo for real LinkedIn data.

Subscribe at https://rapidapi.com/jaypat87/api/linkedin-jobs-search/
Header: X-RapidAPI-Key + X-RapidAPI-Host
Returns real LinkedIn job listings with full descriptions.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List
import hashlib

import httpx

from app.config import settings
from app.discovery.base import RawJob

log = logging.getLogger(__name__)

_API_URL = "https://linkedin-jobs-search.p.rapidapi.com/"
_SEARCH_TERMS = [
    "machine learning engineer",
    "AI engineer",
    "python developer",
    "LLM engineer",
    "MLOps engineer",
    "GenAI engineer",
]


class LinkedInRapidAPISource:
    """Fetches LinkedIn jobs via RapidAPI (paid, ~$10/mo)."""

    def __init__(self, keywords: List[str] | None = None):
        self.keywords = [k.lower() for k in (keywords or settings.jobs_keywords_list)]

    async def fetch_jobs(self) -> List[RawJob]:
        if not settings.linkedin_rapidapi_enabled:
            return []
        api_key = settings.rapidapi_key
        if not api_key:
            log.debug("LinkedIn RapidAPI: no rapidapi_key configured — skipping")
            return []

        jobs: List[RawJob] = []
        seen: set[str] = set()
        limit = settings.max_jobs_per_source

        headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "linkedin-jobs-search.p.rapidapi.com",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                for term in _SEARCH_TERMS:
                    if len(jobs) >= limit:
                        break
                    try:
                        r = await client.post(
                            _API_URL,
                            json={
                                "search_terms": term,
                                "location": "United States",
                                "page": "1",
                            },
                            headers=headers,
                        )
                        if r.status_code == 403:
                            log.warning("LinkedIn RapidAPI: invalid key or quota exceeded")
                            break
                        if r.status_code != 200:
                            log.warning("LinkedIn RapidAPI: HTTP %d for '%s'", r.status_code, term)
                            continue

                        for item in r.json() if isinstance(r.json(), list) else []:
                            if len(jobs) >= limit:
                                break
                            try:
                                job_url = (item.get("job_url") or "").strip()
                                ext_id = hashlib.md5(job_url.encode()).hexdigest() if job_url else ""
                                if not ext_id or ext_id in seen:
                                    continue
                                title = (item.get("job_title") or "").strip()
                                if not any(kw in title.lower() for kw in self.keywords):
                                    continue
                                seen.add(ext_id)

                                location = (item.get("job_location") or "Remote").strip()
                                remote = "remote" in location.lower()

                                posted_at: datetime | None = None
                                date_str = item.get("posted_date") or ""
                                try:
                                    if date_str:
                                        posted_at = datetime.fromisoformat(
                                            date_str.replace("Z", "+00:00")
                                        ).replace(tzinfo=None)
                                except Exception:
                                    pass

                                jobs.append(RawJob(
                                    source="linkedin",
                                    external_id=ext_id,
                                    company=(item.get("company_name") or "Unknown").strip(),
                                    title=title,
                                    location=location,
                                    remote=remote,
                                    url=job_url,
                                    description=(item.get("job_description") or "").strip(),
                                    posted_at=posted_at,
                                ))
                            except Exception as e:
                                log.debug("LinkedIn RapidAPI: parse error: %s", e)
                    except Exception as e:
                        log.warning("LinkedIn RapidAPI: request failed for '%s': %s", term, e)

        except Exception as e:
            log.warning("LinkedIn RapidAPI: fetch failed: %s", e)

        log.info("LinkedInRapidAPISource: fetched %d jobs", len(jobs))
        return jobs
