"""Remotive source — free public API, no auth needed.

https://remotive.com/api/remote-jobs
Returns remote-only jobs. Good for AI/ML/Python roles at remote-first companies.
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

_API_URL = "https://remotive.com/api/remote-jobs"
_TARGET_CATEGORIES = {
    "software-dev", "data", "devops-sysadmin", "product", "backend",
    "machine-learning", "data-engineering", "ai"
}


class RemotiveSource:
    """Fetches remote jobs from the Remotive public API."""

    def __init__(self, keywords: List[str] | None = None):
        self.keywords = [k.lower() for k in (keywords or settings.jobs_keywords_list)]

    async def fetch_jobs(self) -> List[RawJob]:
        if not settings.remotive_enabled:
            return []

        jobs: List[RawJob] = []
        seen_ids: set[str] = set()
        limit = settings.max_jobs_per_source

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(_API_URL, params={"limit": 200})
                if r.status_code != 200:
                    log.warning("Remotive: HTTP %d", r.status_code)
                    return []

                data = r.json()
                for item in data.get("jobs", []):
                    if len(jobs) >= limit:
                        break
                    try:
                        ext_id = str(item.get("id", ""))
                        if not ext_id or ext_id in seen_ids:
                            continue

                        title = (item.get("title") or "").strip().lower()
                        if not matches_title(title, self.keywords):
                            continue

                        seen_ids.add(ext_id)

                        posted_at: datetime | None = None
                        pub_str = item.get("publication_date") or ""
                        try:
                            if pub_str:
                                posted_at = datetime.fromisoformat(pub_str.replace("Z", "+00:00")).replace(tzinfo=None)
                        except Exception:
                            pass

                        jobs.append(RawJob(
                            source="remotive",
                            external_id=ext_id,
                            company=(item.get("company_name") or "Unknown").strip(),
                            title=item.get("title", "").strip(),
                            location=(item.get("candidate_required_location") or "Remote").strip(),
                            remote=True,
                            url=(item.get("url") or "").strip(),
                            description=(item.get("description") or "").strip(),
                            posted_at=posted_at,
                        ))
                    except Exception as e:
                        log.debug("Remotive: failed to parse item %s: %s", item.get("id"), e)

        except Exception as e:
            log.warning("Remotive: fetch failed: %s", e)

        log.info("RemotiveSource: fetched %d jobs", len(jobs))
        return jobs
