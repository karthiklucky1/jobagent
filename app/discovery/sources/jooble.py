"""Jooble source — free API, 500 calls/day.

Request API key at https://jooble.org/api/about
Endpoint: https://jooble.org/api/{api_key}  (POST with JSON body)
Aggregates jobs from thousands of boards globally. Good US remote volume.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List
import hashlib

import httpx

from app.config import settings
from app.discovery.base import RawJob
from app.discovery.title_filter import matches_title

log = logging.getLogger(__name__)

_API_URL = "https://jooble.org/api/{key}"
_SEARCH_TERMS = [
    "machine learning engineer remote",
    "python AI engineer remote",
    "LLM engineer remote",
    "backend python engineer remote",
    "MLOps engineer remote",
]


class JoobleSource:
    """Fetches jobs from the Jooble API (global aggregator)."""

    def __init__(self, keywords: List[str] | None = None):
        self.keywords = [k.lower() for k in (keywords or settings.jobs_keywords_list)]

    async def fetch_jobs(self) -> List[RawJob]:
        if not settings.jooble_enabled:
            return []
        api_key = settings.jooble_api_key
        if not api_key:
            log.debug("Jooble: no api_key configured — skipping")
            return []

        jobs: List[RawJob] = []
        seen: set[str] = set()
        limit = settings.max_jobs_per_source
        url = _API_URL.format(key=api_key)

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                for term in _SEARCH_TERMS:
                    if len(jobs) >= limit:
                        break
                    try:
                        r = await client.post(
                            url,
                            json={"keywords": term, "location": "remote", "page": 1},
                            headers={"Content-Type": "application/json"},
                        )
                        if r.status_code == 403:
                            log.warning("Jooble: invalid API key or quota exceeded")
                            break
                        if r.status_code != 200:
                            log.warning("Jooble: HTTP %d for term '%s'", r.status_code, term)
                            continue

                        for item in r.json().get("jobs", []):
                            if len(jobs) >= limit:
                                break
                            try:
                                link = (item.get("link") or "").strip()
                                if not link:
                                    continue
                                ext_id = hashlib.md5(link.encode()).hexdigest()
                                if ext_id in seen:
                                    continue
                                title = (item.get("title") or "").strip()
                                if not matches_title(title, self.keywords):
                                    continue
                                seen.add(ext_id)

                                location = (item.get("location") or "Remote").strip()
                                remote = "remote" in location.lower()

                                posted_at: datetime | None = None
                                updated = item.get("updated") or ""
                                try:
                                    if updated:
                                        posted_at = datetime.fromisoformat(
                                            updated.replace("Z", "+00:00")
                                        ).replace(tzinfo=None)
                                except Exception:
                                    pass

                                jobs.append(RawJob(
                                    source="jooble",
                                    external_id=ext_id,
                                    company=(item.get("company") or "Unknown").strip(),
                                    title=title,
                                    location=location,
                                    remote=remote,
                                    url=link,
                                    description=(item.get("snippet") or "").strip(),
                                    posted_at=posted_at,
                                ))
                            except Exception as e:
                                log.debug("Jooble: parse error: %s", e)
                    except Exception as e:
                        log.warning("Jooble: request failed for term '%s': %s", term, e)

        except Exception as e:
            log.warning("Jooble: fetch failed: %s", e)

        log.info("JoobleSource: fetched %d jobs", len(jobs))
        return jobs
