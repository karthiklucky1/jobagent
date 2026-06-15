"""WeWorkRemotely source — free RSS feeds, no auth needed.

https://weworkremotely.com/categories/remote-programming-jobs.rss
Curated remote job board with high-quality programming and devops roles.
Uses RSS/XML feeds — no API key required.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import List
from xml.etree import ElementTree as ET

import httpx

from app.config import settings
from app.discovery.base import RawJob

log = logging.getLogger(__name__)

_RSS_FEEDS = [
    "https://weworkremotely.com/categories/remote-programming-jobs.rss",
    "https://weworkremotely.com/categories/remote-devops-sysadmin-jobs.rss",
    "https://weworkremotely.com/categories/remote-back-end-programming-jobs.rss",
]


def _strip_html(html: str) -> str:
    """Strip HTML tags for description text."""
    return re.sub(r"<[^>]+>", " ", html).strip()


class WeWorkRemotelySource:
    """Fetches remote jobs from WeWorkRemotely RSS feeds."""

    def __init__(self, keywords: List[str] | None = None):
        self.keywords = [k.lower() for k in (keywords or settings.jobs_keywords_list)]

    async def fetch_jobs(self) -> List[RawJob]:
        if not settings.weworkremotely_enabled:
            return []

        jobs: List[RawJob] = []
        seen_links: set[str] = set()
        limit = settings.max_jobs_per_source

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                for feed_url in _RSS_FEEDS:
                    if len(jobs) >= limit:
                        break
                    try:
                        r = await client.get(feed_url)
                        if r.status_code != 200:
                            log.warning("WeWorkRemotely: HTTP %d for %s", r.status_code, feed_url)
                            continue

                        root = ET.fromstring(r.text)
                        channel = root.find("channel")
                        if channel is None:
                            continue

                        for item in channel.findall("item"):
                            if len(jobs) >= limit:
                                break
                            try:
                                title_el = item.find("title")
                                link_el = item.find("link")
                                desc_el = item.find("description")
                                pub_el = item.find("pubDate")

                                title = (title_el.text or "").strip() if title_el is not None else ""
                                link = (link_el.text or "").strip() if link_el is not None else ""
                                description = _strip_html(desc_el.text or "") if desc_el is not None else ""

                                if not title or not link or link in seen_links:
                                    continue

                                # Keyword match on title
                                title_lower = title.lower()
                                if not any(kw in title_lower for kw in self.keywords):
                                    continue

                                seen_links.add(link)

                                # Extract company from title pattern "Company: Role"
                                company = "Unknown"
                                if ":" in title:
                                    company = title.split(":")[0].strip()
                                    title = ":".join(title.split(":")[1:]).strip()

                                # Parse pubDate (RFC 2822 format)
                                posted_at: datetime | None = None
                                if pub_el is not None and pub_el.text:
                                    try:
                                        from email.utils import parsedate_to_datetime
                                        posted_at = parsedate_to_datetime(pub_el.text).replace(tzinfo=None)
                                    except Exception:
                                        pass

                                # Use link as external_id (unique per posting)
                                ext_id = link.rstrip("/").split("/")[-1] or link

                                jobs.append(RawJob(
                                    source="weworkremotely",
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
                                log.debug("WeWorkRemotely: failed to parse item: %s", e)

                    except Exception as e:
                        log.warning("WeWorkRemotely: fetch failed for %s: %s", feed_url, e)

        except Exception as e:
            log.warning("WeWorkRemotely: fetch failed: %s", e)

        log.info("WeWorkRemotelySource: fetched %d jobs", len(jobs))
        return jobs
