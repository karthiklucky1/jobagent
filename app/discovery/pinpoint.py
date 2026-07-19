"""Pinpoint public JSON: https://{slug}.pinpointhq.com/postings.json

Single public, unauthenticated JSON endpoint per tenant. Pinpoint is the ATS
vendor behind much recruiting-industry hiring; clean structured data incl.
publish date.
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


def _text(val) -> str:
    """Coerce an API field to a clean string; some boards nest location
    fields as {"name": ...} objects instead of plain strings."""
    if isinstance(val, dict):
        val = val.get("name") or val.get("label") or ""
    return str(val or "").strip()


class PinpointScraper:
    name = "pinpoint"

    def __init__(self, board_slug: str):
        self.board_slug = board_slug

    def fetch(self) -> List[RawJob]:
        url = f"https://{self.board_slug}.pinpointhq.com/postings.json"
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
            log.warning("Pinpoint fetch failed for %s: %s", self.board_slug, e)
            return []
        except httpx.HTTPError as e:
            log.warning("Pinpoint fetch failed for %s: %s", self.board_slug, e)
            return []

        try:
            payload = r.json()
        except Exception:
            return []
        items = payload.get("data", payload) if isinstance(payload, dict) else payload
        jobs: List[RawJob] = []
        for j in items or []:
            attrs = j.get("attributes", j) if isinstance(j, dict) else {}
            ext_id = str(j.get("id") or attrs.get("id") or "").strip()
            if not ext_id:
                continue
            location = _text(attrs.get("location_name")) or _text(attrs.get("location"))
            remote = "remote" in location.lower() or bool(attrs.get("remote"))
            posted_dt = None
            published = attrs.get("published_at") or attrs.get("created_at")
            if published:
                try:
                    posted_dt = datetime.fromisoformat(str(published).replace("Z", "+00:00"))
                except ValueError:
                    pass
            jobs.append(
                RawJob(
                    source="pinpoint",
                    external_id=ext_id,
                    company=_text(attrs.get("company_name")) or self.board_slug.replace("-", " ").title(),
                    title=(attrs.get("title") or "").strip(),
                    location=location,
                    remote=remote,
                    url=attrs.get("url") or attrs.get("apply_url")
                        or f"https://{self.board_slug}.pinpointhq.com/postings/{ext_id}",
                    description=_strip_html(attrs.get("description") or ""),
                    posted_at=posted_dt,
                )
            )
        log.info("Pinpoint[%s]: %d jobs", self.board_slug, len(jobs))
        return jobs
