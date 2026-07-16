"""Breezy HR public JSON: https://{slug}.breezy.hr/json

One public, unauthenticated JSON endpoint per tenant. The list carries the
canonical job URL and publish date; descriptions are per-posting but the list
gives enough (title/location/date) for freshness + matching.
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
    """Coerce an API field to a clean string; Breezy sometimes nests
    city/state/country as {"name": ..., "id": ...} objects."""
    if isinstance(val, dict):
        val = val.get("name") or val.get("label") or ""
    return str(val or "").strip()


class BreezyScraper:
    name = "breezy"

    def __init__(self, board_slug: str):
        self.board_slug = board_slug

    def fetch(self) -> List[RawJob]:
        url = f"https://{self.board_slug}.breezy.hr/json"
        try:
            r = httpx.get(url, timeout=30.0, follow_redirects=True)
            r.raise_for_status()
        except httpx.HTTPError as e:
            log.warning("Breezy fetch failed for %s: %s", self.board_slug, e)
            return []

        try:
            payload = r.json()
        except Exception:
            return []
        items = payload if isinstance(payload, list) else payload.get("positions", [])
        jobs: List[RawJob] = []
        for j in items or []:
            ext_id = str(j.get("_id") or j.get("id") or "").strip()
            if not ext_id:
                continue
            loc = j.get("location") or {}
            if isinstance(loc, dict):
                location = ", ".join(p for p in (
                    _text(loc.get("city")),
                    _text(loc.get("state")),
                    _text(loc.get("country")),
                ) if p)
            else:
                location = str(loc)
            remote = bool(loc.get("is_remote")) if isinstance(loc, dict) else "remote" in location.lower()
            posted_dt = None
            published = j.get("published_date") or j.get("creation_date")
            if published:
                try:
                    posted_dt = datetime.fromisoformat(str(published).replace("Z", "+00:00"))
                except ValueError:
                    pass
            jobs.append(
                RawJob(
                    source="breezy",
                    external_id=ext_id,
                    company=self.board_slug.replace("-", " ").title(),
                    title=(j.get("name") or "").strip(),
                    location=location,
                    remote=remote,
                    url=j.get("url") or f"https://{self.board_slug}.breezy.hr/p/{ext_id}",
                    description=_strip_html(j.get("description") or ""),
                    posted_at=posted_dt,
                )
            )
        log.info("Breezy[%s]: %d jobs", self.board_slug, len(jobs))
        return jobs
