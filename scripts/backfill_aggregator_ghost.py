"""Backfill ghost scores for existing jobs whose URLs pass through aggregator
redirect domains (e.g. lensa.com from Jooble).

Usage:
    python -m scripts.backfill_aggregator_ghost          # dry-run
    python -m scripts.backfill_aggregator_ghost --apply  # write to DB
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from sqlmodel import Session, select

from app.db.init_db import get_engine
from app.db.models import Job
from app.matching.filters.ghost_detector import _AGGREGATOR_REDIRECT_RE, score_ghost

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main(apply: bool) -> None:
    engine = get_engine()
    with Session(engine) as session:
        jobs = session.exec(
            select(Job).where(Job.is_closed == False)
        ).all()

        updated = 0
        for job in jobs:
            url = job.url or ""
            if not _AGGREGATOR_REDIRECT_RE.search(url):
                continue

            result = score_ghost(job, session)
            old_score = job.ghost_score or 0.0
            if result.ghost_score <= old_score:
                continue  # already scored higher or equal

            log.info(
                "[%s] '%s' @ '%s'  score %.2f→%.2f  flags=%s",
                job.source, job.title, job.company,
                old_score, result.ghost_score, result.flags,
            )
            if apply:
                existing_flags = json.loads(job.ghost_flags or "[]")
                merged = list(dict.fromkeys(existing_flags + result.flags))
                job.ghost_score = result.ghost_score
                job.ghost_flags = json.dumps(merged)
                session.add(job)
            updated += 1

        if apply:
            session.commit()
            log.info("Updated %d jobs.", updated)
        else:
            log.info("Dry-run: %d jobs would be updated. Pass --apply to commit.", updated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write changes to DB")
    args = parser.parse_args()
    main(apply=args.apply)
