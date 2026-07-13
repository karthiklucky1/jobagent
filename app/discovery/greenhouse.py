"""Greenhouse public boards API: https://boards-api.greenhouse.io/v1/boards/{slug}/jobs

This is an officially exposed public endpoint companies use to power their
own careers pages. No auth, no rate limiting in practice, and the data is
explicitly meant to be consumed. Compliant.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List

import httpx
from bs4 import BeautifulSoup

from app.discovery.base import RawJob

log = logging.getLogger(__name__)

BASE = "https://boards-api.greenhouse.io/v1/boards"


def _strip_html(html: str) -> str:
    return BeautifulSoup(html or "", "html.parser").get_text(separator="\n").strip()


class GreenhouseScraper:
    name = "greenhouse"

    def __init__(self, board_slug: str):
        self.board_slug = board_slug

    def fetch(self) -> List[RawJob]:
        url = f"{BASE}/{self.board_slug}/jobs?content=true"
        try:
            r = httpx.get(url, timeout=30.0, follow_redirects=True)
            r.raise_for_status()
        except httpx.HTTPError as e:
            log.warning("Greenhouse fetch failed for %s: %s", self.board_slug, e)
            return []

        payload = r.json()
        jobs: List[RawJob] = []
        for j in payload.get("jobs", []):
            # Coerce to "" — a posting can carry {"location": {"name": null}},
            # and .get("name", "") returns None (the key exists), so .lower()
            # below would crash and take down the WHOLE board's fetch.
            location = (j.get("location") or {}).get("name") or ""
            remote = "remote" in location.lower()
            posted = j.get("updated_at")
            posted_dt = datetime.fromisoformat(posted.replace("Z", "+00:00")) if posted else None
            jobs.append(
                RawJob(
                    source="greenhouse",
                    external_id=str(j["id"]),
                    company=self.board_slug.replace("-", " ").replace("_", " ").title(),
                    title=j.get("title", ""),
                    location=location,
                    remote=remote,
                    url=j.get("absolute_url", ""),
                    description=_strip_html(j.get("content", "")),
                    posted_at=posted_dt,
                )
            )
        log.info("Greenhouse[%s]: %d jobs", self.board_slug, len(jobs))
        return jobs
