"""Hacker News "Who is Hiring?" monthly thread — pre-posting intelligence source.

Every month the `whoishiring` bot posts an "Ask HN: Who is hiring?" thread. Each
top-level comment is a single job posting, written by the hiring team directly —
often weeks before (or instead of) the role hitting the big job boards. This is
one of the earliest, highest-signal sources of real openings.

Free, no key. Uses the HN Algolia API:
  - Find latest thread:  https://hn.algolia.com/api/v1/search_by_date?tags=story,author_whoishiring
  - Fetch with comments: https://hn.algolia.com/api/v1/items/{objectID}

Comment convention (loosely followed):
  Company | Role | Location | REMOTE | Salary | tech stack
  <free text description>
  <apply link>

We parse the first line for structured fields and keep the full text as the
description, then filter by the configured keywords and US/remote location.
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from html import unescape
from typing import List, Optional, Tuple

import httpx

from app.config import settings
from app.discovery.base import RawJob

log = logging.getLogger(__name__)

_SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
_ITEM_URL = "https://hn.algolia.com/api/v1/items/{}"

_US_TERMS = {
    "remote", "united states", "usa", "u.s.", "us-based", "us only", "san francisco",
    "new york", "seattle", "austin", "boston", "chicago", "los angeles", "denver",
    "sf", "nyc", "bay area", "washington", "atlanta", "miami", "anywhere",
}


def _strip_html(html: str) -> str:
    # HN comment HTML uses <p> and <a href>; keep the link text/URLs readable.
    text = re.sub(r"<a[^>]*href=\"([^\"]+)\"[^>]*>.*?</a>", r" \1 ", html or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _first_url(text: str) -> Optional[str]:
    m = re.search(r"https?://[^\s<>\")]+", text or "")
    return m.group(0).rstrip(".,);") if m else None


def _parse_comment(text: str) -> Tuple[str, str, str, bool]:
    """Parse (company, role, location, remote) from a Who-is-hiring comment.

    The first non-empty line carries the structured pipe-delimited header in the
    common case; fall back to best-effort heuristics otherwise.
    """
    first_line = ""
    for line in text.splitlines():
        if line.strip():
            first_line = line.strip()
            break

    remote = bool(re.search(r"\bremote\b", text[:400], re.IGNORECASE))

    parts = [p.strip() for p in first_line.split("|") if p.strip()]
    if len(parts) >= 2:
        company = parts[0][:80]
        role = parts[1][:120]
        location = ""
        for p in parts[2:]:
            if re.search(r"remote|onsite|hybrid|[A-Z]{2}\b|,|united states|usa|\bus\b", p, re.IGNORECASE):
                location = re.sub(r"https?://\S+", "", p).strip()[:80]
                break
        return company, role, location or ("Remote" if remote else ""), remote

    # No pipe header — take the leading company-ish token before a separator.
    m = re.match(r"^(.{2,60}?)\s*[–—:\-]\s*(.+)$", first_line)
    if m:
        return m.group(1).strip()[:80], m.group(2).strip()[:120], ("Remote" if remote else ""), remote

    return "HN Who-is-Hiring", first_line[:120] or "See description", ("Remote" if remote else ""), remote


def _matches_keywords(role: str, text: str, keywords: list) -> bool:
    hay = f"{role}\n{text[:800]}".lower()
    return any(kw.lower() in hay for kw in keywords)


def _is_us_or_remote(location: str, text: str) -> bool:
    blob = f"{location} {text[:300]}".lower()
    return any(t in blob for t in _US_TERMS)


def _looks_like_posting(text: str) -> bool:
    """Heuristic: real postings mention a role-ish word or an apply link."""
    if _first_url(text):
        return True
    return bool(re.search(
        r"\b(engineer|developer|scientist|hiring|role|position|fulltime|full-time|"
        r"remote|onsite|apply|backend|frontend|full[\-\s]?stack)\b",
        text[:400], re.IGNORECASE,
    ))


class HNWhoIsHiringSource:
    """Fetches postings from the latest HN 'Who is hiring?' thread."""

    name = "hn_whoishiring"

    def __init__(self, keywords: List[str] | None = None):
        self.keywords = keywords or settings.jobs_keywords_list

    async def _latest_thread_id(self, client: httpx.AsyncClient) -> Optional[str]:
        r = await client.get(
            _SEARCH_URL,
            params={"tags": "story,author_whoishiring", "hitsPerPage": 10},
        )
        if r.status_code != 200:
            log.warning("HN WhoIsHiring: thread search failed HTTP %d", r.status_code)
            return None
        for hit in r.json().get("hits", []):
            title = (hit.get("title") or "").lower()
            if "who is hiring" in title:
                return str(hit.get("objectID"))
        return None

    async def fetch_jobs(self) -> List[RawJob]:
        jobs: List[RawJob] = []
        limit = settings.max_jobs_per_source
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                thread_id = await self._latest_thread_id(client)
                if not thread_id:
                    log.warning("HN WhoIsHiring: no current thread found")
                    return []

                r = await client.get(_ITEM_URL.format(thread_id))
                if r.status_code != 200:
                    log.warning("HN WhoIsHiring: thread fetch failed HTTP %d", r.status_code)
                    return []

                thread = r.json()
                children = thread.get("children", []) or []
                log.info("HN WhoIsHiring: thread %s has %d top-level comments", thread_id, len(children))

                for c in children:
                    if len(jobs) >= limit:
                        break
                    if not c or c.get("author") is None:
                        continue  # deleted comment
                    raw_html = c.get("text") or ""
                    text = _strip_html(raw_html)
                    if len(text) < 40 or not _looks_like_posting(text):
                        continue

                    company, role, location, remote = _parse_comment(text)

                    if not _matches_keywords(role, text, self.keywords):
                        continue
                    if not _is_us_or_remote(location, text):
                        continue

                    comment_id = str(c.get("id"))
                    posted_at: Optional[datetime] = None
                    ts = c.get("created_at_i")
                    if ts:
                        try:
                            posted_at = datetime.utcfromtimestamp(int(ts))
                        except Exception:
                            pass

                    apply_url = _first_url(text) or f"https://news.ycombinator.com/item?id={comment_id}"
                    ext_id = hashlib.md5(f"hn_whoishiring_{comment_id}".encode()).hexdigest()

                    jobs.append(RawJob(
                        source="indeed",  # manual-apply bucket (no ATS autofill for HN comments)
                        external_id=ext_id,
                        company=company,
                        title=role,
                        location=location or "Remote",
                        remote=remote,
                        url=apply_url,
                        description=text[:5000],
                        posted_at=posted_at,
                    ))
        except Exception as e:
            log.warning("HN WhoIsHiring: fetch failed: %s", e)

        log.info("HNWhoIsHiringSource: %d matching postings", len(jobs))
        return jobs
