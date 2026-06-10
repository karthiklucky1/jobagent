"""One-time cleanup for switching to job-first discovery.

Old runs scraped a fixed list of company ATS boards (Greenhouse/Lever/Ashby/
Workday/SmartRecruiters). Those jobs stay in the DB and keep getting matched and
shown even after you disable company-board scraping — which is why you still see
"company-based" jobs.

This script marks those direct-ATS jobs as closed and skips any non-submitted
applications tied to them, so the dashboard/matcher only surface job-first
(aggregator) postings going forward. SUBMITTED / INTERVIEWING applications are
left untouched so you don't lose anything you've already acted on.

Usage:
    python scripts/purge_company_board_jobs.py          # dry run (counts only)
    python scripts/purge_company_board_jobs.py --apply   # perform the cleanup
"""
from __future__ import annotations

import sys

from sqlmodel import select

from app.db.init_db import get_session, init_db
from app.db.models import Job, JobSource, Application, ApplicationStatus

# Direct-ATS / company-board sources that job-first mode replaces.
_BOARD_SOURCES = [
    JobSource.GREENHOUSE,
    JobSource.LEVER,
    JobSource.ASHBY,
    JobSource.WORKDAY,
    JobSource.SMARTRECRUITERS,
]

# Applications we must NOT disturb (already acted on).
_KEEP_APP_STATUSES = {ApplicationStatus.SUBMITTED, ApplicationStatus.INTERVIEWING}


def main(apply: bool) -> None:
    init_db()
    with get_session() as session:
        board_jobs = session.exec(
            select(Job).where(Job.source.in_(_BOARD_SOURCES), Job.is_closed == False)
        ).all()

        closed = 0
        skipped_apps = 0
        protected = 0
        for job in board_jobs:
            apps = session.exec(
                select(Application).where(Application.job_id == job.id)
            ).all()
            # If a protected application exists, leave the job alone.
            if any(a.status in _KEEP_APP_STATUSES for a in apps):
                protected += 1
                continue
            if apply:
                job.is_closed = True
                session.add(job)
                for a in apps:
                    if a.status not in _KEEP_APP_STATUSES:
                        a.status = ApplicationStatus.SKIPPED
                        a.notes = (a.notes or "") + "\nClosed: switched to job-first discovery (company board purge)."
                        session.add(a)
                        skipped_apps += 1
            closed += 1

        if apply:
            session.commit()

        verb = "Closed" if apply else "Would close"
        print(f"{verb} {closed} company-board jobs "
              f"({skipped_apps} applications skipped, {protected} protected by submitted/interviewing status).")
        if not apply:
            print("Dry run only. Re-run with --apply to perform the cleanup.")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
