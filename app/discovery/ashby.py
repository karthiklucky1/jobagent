"""Ashby public job board API.

Ashby exposes a public JSON endpoint at:
  https://api.ashbyhq.com/posting-api/job-board/{org}?includeCompensation=true

(Some Ashby orgs use a GraphQL endpoint; the REST one is more stable.)
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List

import httpx
from bs4 import BeautifulSoup

from app.discovery.base import RawJob

log = logging.getLogger(__name__)

BASE = "https://api.ashbyhq.com/posting-api/job-board"


def _strip_html(html: str) -> str:
    return BeautifulSoup(html or "", "html.parser").get_text(separator="\n").strip()


class AshbyScraper:
    name = "ashby"

    def __init__(self, org_slug: str):
        self.org_slug = org_slug

    def fetch(self) -> List[RawJob]:
        url = f"{BASE}/{self.org_slug}?includeCompensation=true"
        try:
            r = httpx.get(url, timeout=30.0, follow_redirects=True)
            r.raise_for_status()
        except httpx.HTTPError as e:
            log.warning("Ashby fetch failed for %s: %s", self.org_slug, e)
            return []

        payload = r.json()
        jobs: List[RawJob] = []
        for j in payload.get("jobs", []):
            location = j.get("locationName") or ""  # coerce null → "" (see greenhouse.py)
            remote = j.get("isRemote", False) or "remote" in location.lower()
            published = j.get("publishedDate")
            try:
                posted_dt = datetime.fromisoformat(published.replace("Z", "+00:00")) if published else None
            except Exception:
                posted_dt = None
            jobs.append(
                RawJob(
                    source="ashby",
                    external_id=j["id"],
                    company=self.org_slug.replace("-", " ").replace("_", " ").title(),
                    title=j.get("title", ""),
                    location=location,
                    remote=remote,
                    url=j.get("jobUrl", ""),
                    description=_strip_html(j.get("descriptionHtml", "")),
                    posted_at=posted_dt,
                )
            )
        log.info("Ashby[%s]: %d jobs", self.org_slug, len(jobs))
        return jobs
