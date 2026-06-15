"""Adzuna source — free API, 50 searches/day.

Register at https://developer.adzuna.com/ to get app_id + app_key.
Endpoint: https://api.adzuna.com/v1/api/jobs/us/search/1
Millions of US jobs, updated daily. Great for ML/Python roles.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List

import httpx

from app.config import settings
from app.discovery.base import RawJob

log = logging.getLogger(__name__)

_API_URL = "https://api.adzuna.com/v1/api/jobs/us/search/{page}"
_RESULTS_PER_PAGE = 50


class AdzunaSource:
    """Fetches US jobs from the Adzuna API."""

    def __init__(self, keywords: List[str] | None = None):
        self.keywords = [k.lower() for k in (keywords or settings.jobs_keywords_list)]

    async def fetch_jobs(self) -> List[RawJob]:
        if not settings.adzuna_enabled:
            return []
        app_id = settings.adzuna_app_id
        app_key = settings.adzuna_app_key
        if not app_id or not app_key:
            log.debug("Adzuna: no app_id/app_key configured — skipping")
            return []

        jobs: List[RawJob] = []
        seen: set[str] = set()
        limit = settings.max_jobs_per_source

        # Build one search per keyword group to maximize relevant results
        search_terms = [
            "machine learning engineer",
            "python developer AI",
            "LLM engineer",
            "backend python engineer",
        ]

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                for term in search_terms:
                    if len(jobs) >= limit:
                        break
                    try:
                        r = await client.get(
                            _API_URL.format(page=1),
                            params={
                                "app_id": app_id,
                                "app_key": app_key,
                                "what": term,
                                "where": "remote",
                                "results_per_page": _RESULTS_PER_PAGE,
                                "content-type": "application/json",
                                "sort_by": "date",
                            },
                        )
                        if r.status_code == 401:
                            log.warning("Adzuna: invalid credentials")
                            break
                        if r.status_code == 429:
                            log.warning("Adzuna: daily quota hit")
                            break
                        if r.status_code != 200:
                            log.warning("Adzuna: HTTP %d for term '%s'", r.status_code, term)
                            continue

                        for item in r.json().get("results", []):
                            if len(jobs) >= limit:
                                break
                            try:
                                ext_id = str(item.get("id", ""))
                                if not ext_id or ext_id in seen:
                                    continue
                                title = (item.get("title") or "").strip()
                                if not any(kw in title.lower() for kw in self.keywords):
                                    continue
                                seen.add(ext_id)

                                location_obj = item.get("location", {})
                                location = location_obj.get("display_name", "Remote")
                                remote = "remote" in location.lower()

                                posted_at: datetime | None = None
                                created = item.get("created") or ""
                                try:
                                    if created:
                                        posted_at = datetime.fromisoformat(
                                            created.replace("Z", "+00:00")
                                        ).replace(tzinfo=None)
                                except Exception:
                                    pass

                                jobs.append(RawJob(
                                    source="adzuna",
                                    external_id=ext_id,
                                    company=(item.get("company", {}).get("display_name") or "Unknown").strip(),
                                    title=title,
                                    location=location,
                                    remote=remote,
                                    url=(item.get("redirect_url") or "").strip(),
                                    description=(item.get("description") or "").strip(),
                                    posted_at=posted_at,
                                ))
                            except Exception as e:
                                log.debug("Adzuna: parse error %s: %s", item.get("id"), e)
                    except Exception as e:
                        log.warning("Adzuna: request failed for term '%s': %s", term, e)

        except Exception as e:
            log.warning("Adzuna: fetch failed: %s", e)

        log.info("AdzunaSource: fetched %d jobs", len(jobs))
        return jobs
