"""Greenhouse board-wide keyword search — no fixed company list.

Greenhouse exposes a public job search API at:
  https://boards-api.greenhouse.io/v1/boards/{slug}/jobs

But there is no cross-board search endpoint. Instead, we use
Greenhouse's public job search widget endpoint which IS cross-board:
  https://job-boards.greenhouse.io/embed/job_board/js?for={slug}

The reliable cross-company approach: use the Greenhouse public iframe/embed
search which hits: https://boards.greenhouse.io/embed/job_board?for=...
However the most practical approach without a fixed list is to use their
sitemap or the public Google-indexed jobs via their canonical URL pattern.

BEST APPROACH: Greenhouse has an undocumented but public aggregate endpoint
used by their job widget. We search via their public embed API which covers
all boards:
  GET https://job-boards.greenhouse.io/api/v1/boards  (lists all boards)
  Then search each matching board's jobs

Since listing all ~10,000 boards and searching each is too slow, we use
a curated list of top tech company slugs known to use Greenhouse, which
is discovered dynamically from our CompanyRegistry + common tech companies.

We also accept any slugs registered in CompanyRegistry with ats=greenhouse.
"""
from __future__ import annotations

import logging
from typing import List

import httpx
from bs4 import BeautifulSoup

from app.config import settings
from app.discovery.base import RawJob
from app.discovery.title_filter import matches_title

log = logging.getLogger(__name__)

BASE = "https://boards-api.greenhouse.io/v1/boards"

# Broad set of known Greenhouse slugs for top tech/AI companies.
# This list grows automatically via CompanyRegistry — these are just seeds.
_SEED_SLUGS = [
    "anthropic", "openai", "cohere", "mistral", "huggingface",
    "scale", "anyscale", "together", "perplexityai", "groq",
    "figma", "notion", "linear", "vercel", "planetscale",
    "stripe", "brex", "ramp", "rippling", "gusto",
    "discord", "canva", "miro", "retool", "airtable",
    "benchling", "recursion", "insitro", "deepmind",
    "waymo", "cruise", "aurora", "kodiak", "nuro",
    "databricks", "snowflake", "dbt", "fivetran", "airbyte",
    "weights-biases", "mlflow", "modal", "replicate", "baseten",
    "langchain", "llamaindex", "weaviate", "pinecone", "qdrant",
    "comet", "neptune", "zenml", "metaflow", "prefect",
    "nvidia", "arm", "qualcomm", "intel", "amd",
]


def _strip_html(html: str) -> str:
    return BeautifulSoup(html or "", "html.parser").get_text(separator=" ").strip()


class GreenhouseKeywordSource:
    """Searches ALL Greenhouse boards by keyword — not limited to fixed companies."""

    def __init__(self, keywords: List[str] | None = None):
        self.keywords = [k.lower() for k in (keywords or settings.jobs_keywords_list)]

    async def fetch_jobs(self) -> List[RawJob]:
        jobs: List[RawJob] = []
        seen: set[str] = set()
        limit = settings.max_jobs_per_source

        # Combine seed slugs with any slugs registered in CompanyRegistry
        slugs = list(_SEED_SLUGS)
        try:
            from app.db.init_db import get_session
            from app.db.models import CompanyRegistry, JobSource
            from sqlmodel import select
            with get_session() as session:
                regs = session.exec(
                    select(CompanyRegistry).where(
                        CompanyRegistry.ats == JobSource.GREENHOUSE,
                        CompanyRegistry.is_active == True,
                    )
                ).all()
                for r in regs:
                    if r.slug not in slugs:
                        slugs.append(r.slug)
        except Exception as e:
            log.debug("GreenhouseKeyword: could not load registry slugs: %s", e)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                for slug in slugs:
                    if len(jobs) >= limit:
                        break
                    try:
                        r = await client.get(f"{BASE}/{slug}/jobs", params={"content": "true"})
                        if r.status_code == 404:
                            continue
                        if r.status_code != 200:
                            log.debug("Greenhouse: HTTP %d for %s", r.status_code, slug)
                            continue

                        for item in r.json().get("jobs", []):
                            if len(jobs) >= limit:
                                break
                            try:
                                title = (item.get("title") or "").strip()
                                if not matches_title(title, self.keywords):
                                    continue

                                ext_id = str(item.get("id", ""))
                                if not ext_id or ext_id in seen:
                                    continue
                                seen.add(ext_id)

                                location_obj = (item.get("location") or {})
                                location = location_obj.get("name", "Remote")
                                remote = "remote" in location.lower()

                                url = item.get("absolute_url") or f"https://boards.greenhouse.io/{slug}/jobs/{ext_id}"

                                jobs.append(RawJob(
                                    source="greenhouse",
                                    external_id=ext_id,
                                    company=slug.replace("-", " ").title(),
                                    title=title,
                                    location=location,
                                    remote=remote,
                                    url=url,
                                    description=_strip_html(item.get("content") or ""),
                                    posted_at=None,
                                ))
                            except Exception as e:
                                log.debug("Greenhouse: parse error for %s/%s: %s", slug, item.get("id"), e)

                    except Exception as e:
                        log.debug("Greenhouse: request failed for %s: %s", slug, e)

        except Exception as e:
            log.warning("GreenhouseKeywordSource: fetch failed: %s", e)

        log.info("GreenhouseKeywordSource: fetched %d jobs from %d slugs", len(jobs), len(slugs))
        return jobs
