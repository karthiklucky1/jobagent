"""Indeed RSS source — deprecated publisher API but RSS feeds still work.

Indeed's RSS feeds are public and require no auth. They return the 10 most
recent jobs for a given query. We run multiple targeted queries to maximize
coverage. No key needed — but Indeed may throttle aggressive polling.
"""
from __future__ import annotations

import hashlib
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import List
from urllib.parse import quote_plus

import httpx

from app.config import settings
from app.discovery.base import RawJob

log = logging.getLogger(__name__)

_RSS_URL = "https://www.indeed.com/rss?q={query}&l=Remote&sort=date&fromage=7"

_QUERIES = [
    "machine+learning+engineer",
    "AI+engineer+python",
    "LLM+engineer",
    "MLOps+engineer",
    "backend+python+engineer",
    "GenAI+engineer",
    "applied+scientist+ML",
    "NLP+engineer",
]


class IndeedRSSSource:
    """Fetches Indeed jobs via public RSS feeds (no auth, ~10 results/query)."""

    def __init__(self, keywords: List[str] | None = None):
        self.keywords = [k.lower() for k in (keywords or settings.jobs_keywords_list)]

    async def fetch_jobs(self) -> List[RawJob]:
        if not settings.indeed_rss_enabled:
            return []

        jobs: List[RawJob] = []
        seen: set[str] = set()
        limit = settings.max_jobs_per_source

        try:
            async with httpx.AsyncClient(
                timeout=15.0,
                headers={"User-Agent": "Mozilla/5.0 (compatible; JobAgent/1.0)"},
                follow_redirects=True,
            ) as client:
                for query in _QUERIES:
                    if len(jobs) >= limit:
                        break
                    try:
                        url = _RSS_URL.format(query=query)
                        r = await client.get(url)
                        if r.status_code != 200:
                            log.warning("Indeed RSS: HTTP %d for query '%s'", r.status_code, query)
                            continue

                        root = ET.fromstring(r.text)
                        for item in root.findall(".//item"):
                            if len(jobs) >= limit:
                                break
                            try:
                                link = (item.findtext("link") or "").strip()
                                if not link:
                                    continue
                                # Use guid if available, else hash the link
                                guid = (item.findtext("guid") or link).strip()
                                ext_id = hashlib.md5(guid.encode()).hexdigest()
                                if ext_id in seen:
                                    continue

                                title = (item.findtext("title") or "").strip()
                                if not any(kw in title.lower() for kw in self.keywords):
                                    continue
                                seen.add(ext_id)

                                # Indeed RSS title format: "Job Title - Company Name"
                                company = "Unknown"
                                if " - " in title:
                                    parts = title.rsplit(" - ", 1)
                                    title, company = parts[0].strip(), parts[1].strip()

                                posted_at: datetime | None = None
                                pub = item.findtext("pubDate") or ""
                                try:
                                    if pub:
                                        posted_at = parsedate_to_datetime(pub).replace(tzinfo=None)
                                except Exception:
                                    pass

                                description = (item.findtext("description") or "").strip()

                                jobs.append(RawJob(
                                    source="indeed",
                                    external_id=ext_id,
                                    company=company,
                                    title=title,
                                    location="Remote",
                                    remote=True,
                                    url=link,
                                    description=description,
                                    posted_at=posted_at,
                                ))
                            except Exception as e:
                                log.debug("Indeed RSS: parse error: %s", e)
                    except Exception as e:
                        log.warning("Indeed RSS: request failed for query '%s': %s", query, e)

        except Exception as e:
            log.warning("Indeed RSS: fetch failed: %s", e)

        log.info("IndeedRSSSource: fetched %d jobs", len(jobs))
        return jobs
