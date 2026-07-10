"""BambooHR public careers widget: https://{slug}.bamboohr.com/jobs/embed2.php

BambooHR's old /careers/list JSON was deprecated in 2024. The current public
source is the embedded widget (static HTML, one <li> per job), enriched per job
from the public /careers/{id}/detail JSON (description + datePosted). Boards are
SMB-sized, so per-job enrichment is bounded and cheap.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List

import httpx
from bs4 import BeautifulSoup

from app.discovery.base import RawJob

log = logging.getLogger(__name__)

_WIDGET = "https://{slug}.bamboohr.com/jobs/embed2.php"
_DETAIL = "https://{slug}.bamboohr.com/careers/{id}/detail"
_MAX_JOBS = 60  # cap per board so enrichment cost stays bounded


def _strip_html(html: str) -> str:
    return BeautifulSoup(html or "", "html.parser").get_text(separator="\n").strip()


class BambooHRScraper:
    name = "bamboohr"

    def __init__(self, board_slug: str):
        self.board_slug = board_slug

    def fetch(self) -> List[RawJob]:
        widget_url = _WIDGET.format(slug=self.board_slug)
        try:
            r = httpx.get(widget_url, timeout=30.0, follow_redirects=True)
            r.raise_for_status()
        except httpx.HTTPError as e:
            log.warning("BambooHR widget failed for %s: %s", self.board_slug, e)
            return []

        soup = BeautifulSoup(r.text, "html.parser")
        items = soup.select("li.BambooHR-ATS-Jobs-Item")
        jobs: List[RawJob] = []
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            for li in items[:_MAX_JOBS]:
                a = li.find("a", href=True)
                if not a:
                    continue
                title = a.get_text(strip=True)
                href = a["href"]
                job_url = href if href.startswith("http") else f"https:{href}" if href.startswith("//") \
                    else f"https://{self.board_slug}.bamboohr.com{href}"
                # id from bhrPositionID_{id} or the URL tail
                ext_id = (li.get("id", "") or "").replace("bhrPositionID_", "").strip()
                if not ext_id:
                    ext_id = job_url.rstrip("/").split("/")[-1]
                loc_el = li.select_one(".BambooHR-ATS-Location")
                location = loc_el.get_text(strip=True) if loc_el else ""

                description, posted_dt, remote = "", None, "remote" in (title + location).lower()
                try:
                    d = client.get(_DETAIL.format(slug=self.board_slug, id=ext_id))
                    if d.status_code == 200:
                        detail = d.json()
                        description = _strip_html(detail.get("description") or "")
                        posted = detail.get("datePosted")
                        if posted:
                            try:
                                posted_dt = datetime.fromisoformat(str(posted).replace("Z", "+00:00"))
                            except ValueError:
                                pass
                        loc = detail.get("location") or {}
                        if isinstance(loc, dict):
                            better = ", ".join(p for p in (
                                (loc.get("city") or "").strip(),
                                (loc.get("state") or "").strip(),
                                (loc.get("country") or "").strip(),
                            ) if p)
                            location = better or location
                        remote = remote or bool(detail.get("isRemote"))
                except Exception:
                    pass  # widget row still usable without the detail enrichment

                if not title:
                    continue
                jobs.append(
                    RawJob(
                        source="bamboohr",
                        external_id=str(ext_id),
                        company=self.board_slug.replace("-", " ").title(),
                        title=title,
                        location=location,
                        remote=remote,
                        url=job_url,
                        description=description,
                        posted_at=posted_dt,
                    )
                )
        log.info("BambooHR[%s]: %d jobs", self.board_slug, len(jobs))
        return jobs
