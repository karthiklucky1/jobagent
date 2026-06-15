"""Reed.co.uk source — free API, 5000 calls/day.

Register at https://www.reed.co.uk/developers/jobseeker to get an API key.
Endpoint: https://www.reed.co.uk/api/1.0/search
Has US-based remote roles despite being UK-focused. Very high volume.
Auth: HTTP Basic with api_key as username, empty password.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List

import httpx

from app.config import settings
from app.discovery.base import RawJob

log = logging.getLogger(__name__)

_API_URL = "https://www.reed.co.uk/api/1.0/search"
_SEARCH_TERMS = [
    "machine learning engineer",
    "python developer",
    "AI engineer",
    "LLM engineer",
    "backend python",
]


class ReedSource:
    """Fetches jobs from the Reed.co.uk API (includes remote/US roles)."""

    def __init__(self, keywords: List[str] | None = None):
        self.keywords = [k.lower() for k in (keywords or settings.jobs_keywords_list)]

    async def fetch_jobs(self) -> List[RawJob]:
        if not settings.reed_enabled:
            return []
        api_key = settings.reed_api_key
        if not api_key:
            log.debug("Reed: no api_key configured — skipping")
            return []

        jobs: List[RawJob] = []
        seen: set[str] = set()
        limit = settings.max_jobs_per_source

        try:
            async with httpx.AsyncClient(
                timeout=15.0,
                auth=(api_key, ""),   # Reed uses Basic auth: key as username, blank password
            ) as client:
                for term in _SEARCH_TERMS:
                    if len(jobs) >= limit:
                        break
                    try:
                        r = await client.get(
                            _API_URL,
                            params={
                                "keywords": term,
                                "locationName": "Remote",
                                "resultsToTake": 100,
                                "minimumSalary": 0,
                            },
                        )
                        if r.status_code == 401:
                            log.warning("Reed: invalid API key")
                            break
                        if r.status_code != 200:
                            log.warning("Reed: HTTP %d for term '%s'", r.status_code, term)
                            continue

                        for item in r.json().get("results", []):
                            if len(jobs) >= limit:
                                break
                            try:
                                ext_id = str(item.get("jobId", ""))
                                if not ext_id or ext_id in seen:
                                    continue
                                title = (item.get("jobTitle") or "").strip()
                                if not any(kw in title.lower() for kw in self.keywords):
                                    continue
                                seen.add(ext_id)

                                location = (item.get("locationName") or "Remote").strip()
                                remote = "remote" in location.lower()

                                posted_at: datetime | None = None
                                date_str = item.get("date") or ""
                                try:
                                    if date_str:
                                        posted_at = datetime.fromisoformat(
                                            date_str.replace("Z", "+00:00")
                                        ).replace(tzinfo=None)
                                except Exception:
                                    pass

                                jobs.append(RawJob(
                                    source="reed",
                                    external_id=ext_id,
                                    company=(item.get("employerName") or "Unknown").strip(),
                                    title=title,
                                    location=location,
                                    remote=remote,
                                    url=f"https://www.reed.co.uk/jobs/{ext_id}",
                                    description=(item.get("jobDescription") or "").strip(),
                                    posted_at=posted_at,
                                ))
                            except Exception as e:
                                log.debug("Reed: parse error %s: %s", item.get("jobId"), e)
                    except Exception as e:
                        log.warning("Reed: request failed for term '%s': %s", term, e)

        except Exception as e:
            log.warning("Reed: fetch failed: %s", e)

        log.info("ReedSource: fetched %d jobs", len(jobs))
        return jobs
