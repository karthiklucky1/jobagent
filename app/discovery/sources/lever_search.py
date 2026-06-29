"""Lever board-wide keyword search — no fixed company list.

Lever exposes public job boards at:
  https://api.lever.co/v0/postings/{slug}?mode=json

No auth needed. We search across a broad set of known Lever slugs,
plus any slugs registered in CompanyRegistry with ats=lever.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List

import httpx

from app.config import settings
from app.discovery.base import RawJob
from app.discovery.title_filter import matches_title

log = logging.getLogger(__name__)

BASE = "https://api.lever.co/v0/postings"

_SEED_SLUGS = [
    "netflix", "twitter", "x", "lyft", "pinterest",
    "cloudflare", "hashicorp", "datadog", "pagerduty", "fastly",
    "elastic", "confluent", "cockroachdb", "mongodb", "redis",
    "segment", "twilio", "sendgrid", "plaid", "marqeta",
    "chime", "robinhood", "coinbase", "kraken", "gemini",
    "palantir", "anduril", "skydio", "shield-ai",
    "huggingface", "stability", "midjourney", "runway",
    "notion", "coda", "craft", "fibery",
    "vercel", "netlify", "supabase", "neon", "turso",
    "loom", "pitch", "miro", "whimsical",
    "dbt-labs", "hightouch", "census", "rudderstack",
    "modern-treasury", "mercury", "brex", "rho",
    "benchling", "relay", "watershed", "pachama",
]


class LeverKeywordSource:
    """Searches ALL Lever boards by keyword — not limited to fixed companies."""

    def __init__(self, keywords: List[str] | None = None):
        self.keywords = [k.lower() for k in (keywords or settings.jobs_keywords_list)]

    async def fetch_jobs(self) -> List[RawJob]:
        jobs: List[RawJob] = []
        seen: set[str] = set()
        limit = settings.max_jobs_per_source

        slugs = list(_SEED_SLUGS)
        try:
            from app.db.init_db import get_session
            from app.db.models import CompanyRegistry, JobSource
            from sqlmodel import select
            with get_session() as session:
                regs = session.exec(
                    select(CompanyRegistry).where(
                        CompanyRegistry.ats == JobSource.LEVER,
                        CompanyRegistry.is_active == True,
                    )
                ).all()
                for r in regs:
                    if r.slug not in slugs:
                        slugs.append(r.slug)
        except Exception as e:
            log.debug("LeverKeyword: could not load registry slugs: %s", e)

        import asyncio

        async def _fetch_slug(slug: str, client: httpx.AsyncClient, sem: asyncio.Semaphore) -> List[RawJob]:
            async with sem:
                try:
                    r = await client.get(
                        f"{BASE}/{slug}",
                        params={"mode": "json", "limit": 250},
                    )
                    if r.status_code == 404:
                        return []
                    if r.status_code != 200:
                        log.debug("Lever: HTTP %d for %s", r.status_code, slug)
                        return []

                    slug_jobs = []
                    for item in r.json():
                        try:
                            title = (item.get("text") or "").strip()
                            if not matches_title(title, self.keywords):
                                continue

                            ext_id = item.get("id") or ""
                            if not ext_id:
                                continue

                            loc_list = item.get("categories", {}).get("location") or ""
                            location = loc_list if isinstance(loc_list, str) else "Remote"
                            remote = "remote" in location.lower() or item.get("categories", {}).get("commitment", "").lower() == "remote"

                            posted_at: datetime | None = None
                            ts = item.get("createdAt")
                            try:
                                if ts:
                                    posted_at = datetime.utcfromtimestamp(int(ts) / 1000)
                            except Exception:
                                pass

                            desc_html = ""
                            for block in (item.get("descriptionBody") or {}).get("content") or []:
                                for node in block.get("content") or []:
                                    desc_html += node.get("text", "") + " "

                            slug_jobs.append(RawJob(
                                source="lever",
                                external_id=ext_id,
                                company=(item.get("company") or slug.replace("-", " ").title()).strip(),
                                title=title,
                                location=location,
                                remote=remote,
                                url=item.get("hostedUrl") or f"https://jobs.lever.co/{slug}/{ext_id}",
                                description=desc_html.strip(),
                                posted_at=posted_at,
                            ))
                        except Exception as e:
                            log.debug("Lever: parse error for %s/%s: %s", slug, item.get("id"), e)
                    return slug_jobs
                except Exception as e:
                    log.debug("Lever: request failed for %s: %s", slug, e)
                    return []

        try:
            sem = asyncio.Semaphore(15)
            async with httpx.AsyncClient(timeout=10.0) as client:
                tasks = [_fetch_slug(slug, client, sem) for slug in slugs]
                results = await asyncio.gather(*tasks)
                for res in results:
                    for job in res:
                        if len(jobs) >= limit:
                            break
                        if job.external_id in seen:
                            continue
                        seen.add(job.external_id)
                        jobs.append(job)
                    if len(jobs) >= limit:
                        break
        except Exception as e:
            log.warning("LeverKeywordSource: fetch failed: %s", e)

        log.info("LeverKeywordSource: fetched %d jobs from %d slugs", len(jobs), len(slugs))
        return jobs
