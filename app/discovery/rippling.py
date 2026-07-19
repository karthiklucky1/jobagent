"""Rippling public ATS board API: https://api.rippling.com/platform/api/ats/v1/board/{slug}/jobs

Public JSON, no auth. Rippling's own ATS product; growing US SMB coverage.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List

import httpx
from bs4 import BeautifulSoup

from app.discovery.base import RawJob

log = logging.getLogger(__name__)

_API = "https://api.rippling.com/platform/api/ats/v1/board/{slug}/jobs"


def _strip_html(html: str) -> str:
    return BeautifulSoup(html or "", "html.parser").get_text(separator="\n").strip()


class RipplingScraper:
    name = "rippling"

    def __init__(self, board_slug: str):
        self.board_slug = board_slug

    def fetch(self) -> List[RawJob]:
        try:
            r = httpx.get(_API.format(slug=self.board_slug), timeout=30.0, follow_redirects=True)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Permanent statuses mean the slug is wrong, gone, or private — let
            # the exception propagate so the discovery pipeline's dead-board
            # recorder retires the registry row (these long-tail ATSes are NOT
            # covered by the Greenhouse/Lever/Ashby validation loop, so a junk
            # slug would otherwise 404 on every cycle forever).
            if e.response is not None and e.response.status_code in (401, 403, 404, 410):
                raise
            log.warning("Rippling fetch failed for %s: %s", self.board_slug, e)
            return []
        except httpx.HTTPError as e:
            log.warning("Rippling fetch failed for %s: %s", self.board_slug, e)
            return []

        payload = r.json()
        items = payload if isinstance(payload, list) else payload.get("items", payload.get("jobs", []))
        jobs: List[RawJob] = []
        for j in items or []:
            ext_id = str(j.get("id") or j.get("uuid") or "").strip()
            if not ext_id:
                continue
            loc = j.get("workLocation") or {}
            location = (loc.get("label") or j.get("location") or "").strip() if isinstance(loc, dict) else str(loc)
            remote = "remote" in location.lower() or bool(j.get("isRemote"))
            posted_dt = None
            for key in ("publishedAt", "createdAt", "postedDate"):
                v = j.get(key)
                if not v:
                    continue
                try:
                    if isinstance(v, (int, float)):
                        posted_dt = datetime.fromtimestamp(v / 1000 if v > 1e11 else v, tz=timezone.utc)
                    else:
                        posted_dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                    break
                except (ValueError, OSError):
                    pass
            jobs.append(
                RawJob(
                    source="rippling",
                    external_id=ext_id,
                    company=(j.get("companyName") or self.board_slug.replace("-", " ").title()).strip(),
                    title=(j.get("name") or j.get("title") or "").strip(),
                    location=location,
                    remote=remote,
                    url=j.get("url") or j.get("applyUrl")
                        or f"https://app.rippling.com/jobs/{self.board_slug}/{ext_id}",
                    description=_strip_html(j.get("description") or ""),
                    posted_at=posted_dt,
                )
            )
        log.info("Rippling[%s]: %d jobs", self.board_slug, len(jobs))
        return jobs
