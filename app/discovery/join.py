"""join.com public jobs API (EU-heavy SMB ATS). Two-step:

    GET https://join.com/companies/{slug}                    → resolve company id
    GET https://join.com/api/public/companies/{id}/jobs      → paginated jobs JSON

Public, unauthenticated. join.com is the largest single ATS in the open dataset
(~23.5K companies) and is European-heavy — key coverage for a worldwide launch.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import List, Optional

import httpx
from bs4 import BeautifulSoup

from app.discovery.base import RawJob

log = logging.getLogger(__name__)

_BASE = "https://join.com"
_API = "https://join.com/api/public"
# The company object embedded in the Next.js page carries id + domain.
_COMPANY_RE = re.compile(r'"company"\s*:\s*\{[^{}]*?"id"\s*:\s*(\d+)[^{}]*?"domain"\s*:\s*"([^"]+)"')
_MAX_PAGES = 5


def _strip_html(html: str) -> str:
    return BeautifulSoup(html or "", "html.parser").get_text(separator="\n").strip()


class JoinScraper:
    name = "join"

    def __init__(self, board_slug: str):
        self.board_slug = board_slug

    def _resolve_company_id(self, client: httpx.Client) -> Optional[str]:
        try:
            r = client.get(f"{_BASE}/companies/{self.board_slug}")
        except httpx.HTTPError:
            return None
        if r.status_code == 404:
            return None
        # Validate the embedded company domain matches the slug we asked for —
        # join.com serves a placeholder company (id 233) for empty tenants.
        for cid, domain in _COMPANY_RE.findall(r.text):
            if domain.lower() == self.board_slug.lower():
                return cid
        m = _COMPANY_RE.search(r.text)
        return m.group(1) if m else None

    def fetch(self) -> List[RawJob]:
        jobs: List[RawJob] = []
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            company_id = self._resolve_company_id(client)
            if not company_id:
                return []
            page = 1
            while page <= _MAX_PAGES:
                try:
                    r = client.get(
                        f"{_API}/companies/{company_id}/jobs",
                        params={"locale": "en-us", "page": page, "pageSize": 100},
                    )
                    if r.status_code != 200:
                        break
                    payload = r.json()
                except (httpx.HTTPError, json.JSONDecodeError, ValueError):
                    break

                items = payload.get("items", []) if isinstance(payload, dict) else []
                if not items:
                    break
                for j in items:
                    ext_id = str(j.get("id") or j.get("idParam") or "").strip()
                    if not ext_id:
                        continue
                    loc = j.get("location") or {}
                    if isinstance(loc, dict):
                        location = ", ".join(p for p in (
                            (loc.get("cityName") or loc.get("city") or "").strip(),
                            (loc.get("countryName") or loc.get("country") or "").strip(),
                        ) if p)
                    else:
                        location = str(loc or "")
                    remote = (str(j.get("jobLocationType") or "").lower() == "remote"
                              or "remote" in location.lower())
                    posted_dt = None
                    published = j.get("publishedAt") or j.get("createdAt")
                    if published:
                        try:
                            posted_dt = datetime.fromisoformat(str(published).replace("Z", "+00:00"))
                        except ValueError:
                            pass
                    jobs.append(
                        RawJob(
                            source="join",
                            external_id=ext_id,
                            company=self.board_slug.replace("-", " ").title(),
                            title=(j.get("name") or j.get("title") or "").strip(),
                            location=location,
                            remote=remote,
                            url=j.get("url") or f"{_BASE}/companies/{self.board_slug}/jobs/{ext_id}",
                            description=_strip_html(j.get("description") or ""),
                            posted_at=posted_dt,
                        )
                    )
                pagination = payload.get("pagination") or {}
                if page >= int(pagination.get("totalPages", page)):
                    break
                page += 1
        log.info("Join[%s]: %d jobs", self.board_slug, len(jobs))
        return jobs
