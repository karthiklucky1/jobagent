"""Jobicy source — free public API, no auth needed.

https://jobicy.com/api/v2/remote-jobs
Engineering-focused remote job board with good ML/Python volume.
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

_API_URL = "https://jobicy.com/api/v2/remote-jobs"
# Jobicy uses tag-based filtering; these cover AI/ML/Python roles
_TARGET_TAGS = ["python", "machine-learning", "ai", "data-science", "backend"]


class JobicySource:
    """Fetches remote jobs from the Jobicy public API."""

    def __init__(self, keywords: List[str] | None = None):
        self.keywords = [k.lower() for k in (keywords or settings.jobs_keywords_list)]

    async def fetch_jobs(self) -> List[RawJob]:
        if not settings.jobicy_enabled:
            return []

        jobs: List[RawJob] = []
        seen_ids: set[str] = set()
        limit = settings.max_jobs_per_source

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                for tag in _TARGET_TAGS:
                    if len(jobs) >= limit:
                        break
                    try:
                        r = await client.get(
                            _API_URL,
                            params={"count": 50, "tag": tag},
                        )
                        if r.status_code != 200:
                            log.warning("Jobicy: HTTP %d for tag %s", r.status_code, tag)
                            continue

                        data = r.json()
                        for item in data.get("jobs", []):
                            if len(jobs) >= limit:
                                break
                            try:
                                ext_id = str(item.get("id", ""))
                                if not ext_id or ext_id in seen_ids:
                                    continue

                                title = (item.get("jobTitle") or "").strip()
                                if not title:
                                    continue

                                # Keyword match on title
                                title_lower = title.lower()
                                if not matches_title(title, self.keywords):
                                    continue

                                seen_ids.add(ext_id)

                                posted_at: datetime | None = None
                                pub_str = item.get("pubDate") or ""
                                try:
                                    if pub_str:
                                        posted_at = datetime.fromisoformat(
                                            pub_str.replace("Z", "+00:00")
                                        ).replace(tzinfo=None)
                                except Exception:
                                    pass

                                location = (item.get("jobGeo") or "Remote").strip()
                                remote = True  # Jobicy is a remote-jobs board

                                jobs.append(RawJob(
                                    source="jobicy",
                                    external_id=ext_id,
                                    company=(item.get("companyName") or "Unknown").strip(),
                                    title=title,
                                    location=location,
                                    remote=remote,
                                    url=(item.get("url") or "").strip(),
                                    description=(item.get("jobDescription") or "").strip(),
                                    posted_at=posted_at,
                                ))
                            except Exception as e:
                                log.debug("Jobicy: failed to parse item %s: %s", item.get("id"), e)
                    except Exception as e:
                        log.warning("Jobicy: fetch failed for tag %s: %s", tag, e)

        except Exception as e:
            log.warning("Jobicy: fetch failed: %s", e)

        log.info("JobicySource: fetched %d jobs", len(jobs))
        return jobs
