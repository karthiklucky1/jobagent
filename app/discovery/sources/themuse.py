"""The Muse source — free public API, no auth required.

https://www.themuse.com/developers/api/v2
Endpoint: https://www.themuse.com/api/public/jobs?category=...&page=N
Returns thousands of curated tech jobs across many companies. An optional
THEMUSE_API_KEY raises rate limits but is not required.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List

import httpx

from app.config import settings
from app.discovery.base import RawJob
from app.discovery.title_filter import matches_title

log = logging.getLogger(__name__)

_API_URL = "https://www.themuse.com/api/public/jobs"
# The Muse's own category taxonomy — we request the engineering/data ones.
_CATEGORIES = ["Software Engineering", "Data Science", "Data and Analytics"]
_MAX_PAGES = 5  # 20 results/page → up to ~100 candidates before keyword filtering


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


class TheMuseSource:
    """Fetches jobs from The Muse public API, filtered to our keywords + US/remote."""

    def __init__(self, keywords: List[str] | None = None):
        self.keywords = [k.lower() for k in (keywords or settings.jobs_keywords_list)]

    async def fetch_jobs(self) -> List[RawJob]:
        jobs: List[RawJob] = []
        seen: set[str] = set()
        limit = settings.max_jobs_per_source
        api_key = getattr(settings, "themuse_api_key", "") or ""

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                for page in range(_MAX_PAGES):
                    if len(jobs) >= limit:
                        break
                    params = {"page": page, "category": _CATEGORIES, "location": "Remote"}
                    if api_key:
                        params["api_key"] = api_key
                    r = await client.get(_API_URL, params=params)
                    if r.status_code == 429:
                        log.warning("TheMuse: rate-limited — stopping")
                        break
                    if r.status_code != 200:
                        log.warning("TheMuse: HTTP %d on page %d", r.status_code, page)
                        break

                    results = r.json().get("results", [])
                    if not results:
                        break
                    for item in results:
                        if len(jobs) >= limit:
                            break
                        try:
                            ext_id = str(item.get("id", ""))
                            if not ext_id or ext_id in seen:
                                continue
                            title = (item.get("name") or "").strip()
                            if not matches_title(title, self.keywords):
                                continue
                            seen.add(ext_id)

                            company = ((item.get("company") or {}).get("name") or "Unknown").strip()
                            locs = item.get("locations") or []
                            location = ", ".join(l.get("name", "") for l in locs) or "Remote"
                            remote = "remote" in location.lower()
                            url = ((item.get("refs") or {}).get("landing_page") or "").strip()

                            jobs.append(RawJob(
                                source="themuse",
                                external_id=ext_id,
                                company=company,
                                title=title,
                                location=location,
                                remote=remote,
                                url=url,
                                description=(item.get("contents") or "").strip(),
                                posted_at=_parse_dt(item.get("publication_date")),
                            ))
                        except Exception as e:
                            log.debug("TheMuse: parse failed for %s: %s", item.get("id"), e)
        except Exception as e:
            log.warning("TheMuse: fetch failed: %s", e)

        log.info("TheMuseSource: fetched %d jobs", len(jobs))
        return jobs
