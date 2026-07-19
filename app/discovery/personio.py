"""Personio public XML job feed: https://{slug}.jobs.personio.de/xml

Every Personio-hosted careers page publishes this feed for job-board
syndication. No auth required. Personio is the dominant SMB ATS in the
DACH region — good coverage for German/Austrian/Swiss roles.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List

import httpx
from bs4 import BeautifulSoup

from app.discovery.base import RawJob

log = logging.getLogger(__name__)


def _strip_html(html: str) -> str:
    return BeautifulSoup(html or "", "html.parser").get_text(separator="\n").strip()


def _text(node, tag: str) -> str:
    el = node.find(tag)
    return (el.text or "").strip() if el is not None and el.text else ""


class PersonioScraper:
    name = "personio"

    def __init__(self, board_slug: str):
        self.board_slug = board_slug

    def fetch(self) -> List[RawJob]:
        url = f"https://{self.board_slug}.jobs.personio.de/xml"
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
            log.warning("Personio fetch failed for %s: %s", self.board_slug, e)
            return []
        except httpx.HTTPError as e:
            log.warning("Personio fetch failed for %s: %s", self.board_slug, e)
            return []

        try:
            root = ET.fromstring(r.text)
        except ET.ParseError as e:
            log.warning("Personio XML parse failed for %s: %s", self.board_slug, e)
            return []

        company = self.board_slug.replace("-", " ").title()
        jobs: List[RawJob] = []
        for pos in root.iter("position"):
            ext_id = _text(pos, "id")
            title = _text(pos, "name")
            if not ext_id or not title:
                continue
            location = _text(pos, "office")
            schedule = _text(pos, "schedule")
            remote = "remote" in location.lower() or "remote" in schedule.lower()
            posted_dt = None
            created = _text(pos, "createdAt")
            if created:
                try:
                    posted_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                except ValueError:
                    pass
            desc_parts = []
            for jd in pos.iter("jobDescription"):
                heading = _text(jd, "name")
                value = _strip_html(_text(jd, "value"))
                if value:
                    desc_parts.append(f"{heading}\n{value}" if heading else value)
            jobs.append(
                RawJob(
                    source="personio",
                    external_id=ext_id,
                    company=company,
                    title=title,
                    location=location,
                    remote=remote,
                    url=f"https://{self.board_slug}.jobs.personio.de/job/{ext_id}",
                    description="\n\n".join(desc_parts),
                    posted_at=posted_dt,
                )
            )
        log.info("Personio[%s]: %d jobs", self.board_slug, len(jobs))
        return jobs
