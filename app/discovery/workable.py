"""Workable public widget API: https://apply.workable.com/api/v1/widget/accounts/{slug}

Officially exposed endpoint that powers companies' own hosted careers pages
(apply.workable.com). No auth required; data is published for consumption.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List

import httpx
from bs4 import BeautifulSoup

from app.discovery.base import RawJob

log = logging.getLogger(__name__)

BASE = "https://apply.workable.com/api/v1/widget/accounts"


def _strip_html(html: str) -> str:
    return BeautifulSoup(html or "", "html.parser").get_text(separator="\n").strip()


class WorkableScraper:
    name = "workable"

    def __init__(self, board_slug: str):
        self.board_slug = board_slug

    def fetch(self) -> List[RawJob]:
        url = f"{BASE}/{self.board_slug}?details=true"
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
            log.warning("Workable fetch failed for %s: %s", self.board_slug, e)
            return []
        except httpx.HTTPError as e:
            log.warning("Workable fetch failed for %s: %s", self.board_slug, e)
            return []

        payload = r.json()
        company = (payload.get("name") or self.board_slug.replace("-", " ").title()).strip()
        jobs: List[RawJob] = []
        for j in payload.get("jobs", []):
            shortcode = str(j.get("shortcode") or j.get("code") or "").strip()
            if not shortcode:
                continue
            city = (j.get("city") or "").strip()
            country = (j.get("country") or "").strip()
            location = ", ".join(p for p in (city, country) if p)
            remote = bool(j.get("telecommuting") or j.get("remote")) or "remote" in location.lower()
            posted_dt = None
            published = j.get("published_on") or j.get("created_at")
            if published:
                try:
                    posted_dt = datetime.fromisoformat(str(published).replace("Z", "+00:00"))
                except ValueError:
                    pass
            jobs.append(
                RawJob(
                    source="workable",
                    external_id=shortcode,
                    company=company,
                    title=(j.get("title") or "").strip(),
                    location=location,
                    remote=remote,
                    url=j.get("application_url") or j.get("url")
                        or f"https://apply.workable.com/{self.board_slug}/j/{shortcode}/",
                    description=_strip_html(j.get("description") or ""),
                    posted_at=posted_dt,
                )
            )
        log.info("Workable[%s]: %d jobs", self.board_slug, len(jobs))
        return jobs
