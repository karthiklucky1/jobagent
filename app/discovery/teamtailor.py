"""Teamtailor public RSS: https://{slug}.teamtailor.com/jobs.rss

Teamtailor is the leading Nordics ATS. Every careers site can expose a public
jobs.rss feed with pubDate — cheap near-real-time freshness for Scandinavian /
European coverage. (RSS must be enabled by the tenant; a 404 raises so the
pipeline retires the dead slug — the registry validator can revive false positives.)
"""
from __future__ import annotations

import logging
import re
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import List
from xml.etree import ElementTree as ET

import httpx
from bs4 import BeautifulSoup

from app.discovery.base import RawJob

log = logging.getLogger(__name__)


def _strip_html(html: str) -> str:
    return BeautifulSoup(html or "", "html.parser").get_text(separator="\n").strip()


class TeamtailorScraper:
    name = "teamtailor"

    def __init__(self, board_slug: str):
        self.board_slug = board_slug

    def fetch(self) -> List[RawJob]:
        url = f"https://{self.board_slug}.teamtailor.com/jobs.rss"
        try:
            r = httpx.get(url, timeout=30.0, follow_redirects=True)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Permanent statuses mean the slug is wrong, gone, or private — let
            # the exception propagate so the discovery pipeline's dead-board
            # recorder retires the registry row (these long-tail ATSes are NOT
            # covered by the Greenhouse/Lever/Ashby validation loop, so a junk
            # slug would otherwise 404 on every cycle forever).
            if e.response is not None and e.response.status_code in (401, 403, 404, 410):
                raise
            log.warning("Teamtailor fetch failed for %s: %s", self.board_slug, e)
            return []
        except httpx.HTTPError as e:
            log.warning("Teamtailor fetch failed for %s: %s", self.board_slug, e)
            return []

        try:
            root = ET.fromstring(r.content)
        except ET.ParseError as e:
            log.warning("Teamtailor[%s] bad RSS: %s", self.board_slug, e)
            return []

        jobs: List[RawJob] = []
        for item in root.iter("item"):
            link = (item.findtext("link") or "").strip()
            title = (item.findtext("title") or "").strip()
            if not link or not title:
                continue
            # external id = trailing numeric/slug segment of the job URL
            ext_id = re.sub(r"[^a-zA-Z0-9_-]", "", link.rstrip("/").split("/")[-1]) or link
            posted_dt = None
            pub = item.findtext("pubDate")
            if pub:
                try:
                    dt = parsedate_to_datetime(pub)
                    posted_dt = dt.astimezone(timezone.utc) if dt else None
                except (TypeError, ValueError):
                    pass
            desc = _strip_html(item.findtext("description") or "")
            # Teamtailor RSS often puts "Title - Location" in the title
            location = ""
            if " - " in title:
                title, location = title.rsplit(" - ", 1)
            jobs.append(
                RawJob(
                    source="teamtailor",
                    external_id=str(ext_id),
                    company=self.board_slug.replace("-", " ").title(),
                    title=title.strip(),
                    location=location.strip(),
                    remote="remote" in (title + location + desc).lower(),
                    url=link,
                    description=desc,
                    posted_at=posted_dt,
                )
            )
        log.info("Teamtailor[%s]: %d jobs", self.board_slug, len(jobs))
        return jobs
