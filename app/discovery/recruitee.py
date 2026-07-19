"""Recruitee public offers API: https://{slug}.recruitee.com/api/offers/

The JSON feed behind every Recruitee-hosted careers site. No auth required.
Recruitee is heavily used by European companies — good coverage for non-US
users.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List

import httpx
from bs4 import BeautifulSoup

from app.discovery.base import RawJob

log = logging.getLogger(__name__)


def _strip_html(html: str) -> str:
    return BeautifulSoup(html or "", "html.parser").get_text(separator="\n").strip()


class RecruiteeScraper:
    name = "recruitee"

    def __init__(self, board_slug: str):
        self.board_slug = board_slug

    def fetch(self) -> List[RawJob]:
        url = f"https://{self.board_slug}.recruitee.com/api/offers/"
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
            log.warning("Recruitee fetch failed for %s: %s", self.board_slug, e)
            return []
        except httpx.HTTPError as e:
            log.warning("Recruitee fetch failed for %s: %s", self.board_slug, e)
            return []

        payload = r.json()
        jobs: List[RawJob] = []
        for j in payload.get("offers", []):
            ext_id = str(j.get("id") or "").strip()
            if not ext_id:
                continue
            if (j.get("status") or "").lower() not in ("", "published", "open"):
                continue
            location = (j.get("location") or "").strip()
            if not location:
                location = ", ".join(p for p in ((j.get("city") or "").strip(),
                                                 (j.get("country") or "").strip()) if p)
            remote = bool(j.get("remote")) or "remote" in location.lower()
            posted_dt = None
            published = j.get("published_at") or j.get("created_at")
            if published:
                try:
                    posted_dt = datetime.fromisoformat(str(published).replace("Z", "+00:00"))
                except ValueError:
                    pass
            desc = " ".join(_strip_html(j.get(f) or "") for f in ("description", "requirements"))
            jobs.append(
                RawJob(
                    source="recruitee",
                    external_id=ext_id,
                    company=(j.get("company_name") or self.board_slug.replace("-", " ").title()).strip(),
                    title=(j.get("title") or "").strip(),
                    location=location,
                    remote=remote,
                    url=j.get("careers_url")
                        or f"https://{self.board_slug}.recruitee.com/o/{j.get('slug') or ext_id}",
                    description=desc.strip(),
                    posted_at=posted_dt,
                )
            )
        log.info("Recruitee[%s]: %d jobs", self.board_slug, len(jobs))
        return jobs
