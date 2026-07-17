"""Retention purge must delete ONLY closed, old, unreferenced jobs — never an
applied job, a recent job, or an open one (app/strategy/job_retention.py)."""
from datetime import datetime, timedelta

import pytest
from sqlmodel import delete

from app.db.init_db import get_session
from app.db.models import Application, ApplicationStatus, Job, JobSource
from app.strategy.job_retention import purge_old_closed_jobs

OLD = datetime.utcnow() - timedelta(days=90)
RECENT = datetime.utcnow() - timedelta(days=5)


def _mk_job(session, ext, closed, first_seen):
    j = Job(external_id=ext, source=JobSource.GREENHOUSE, title="Engineer",
            company="Acme", url=f"https://x/{ext}", is_closed=closed, first_seen=first_seen)
    session.add(j)
    session.commit()
    session.refresh(j)
    return j.id


@pytest.fixture
def scenario():
    with get_session() as s:
        ids = {
            "closed_old": _mk_job(s, "ret-co", True, OLD),        # should DELETE
            "closed_old_applied": _mk_job(s, "ret-coa", True, OLD),  # keep (has app)
            "closed_recent": _mk_job(s, "ret-cr", True, RECENT),  # keep (too new)
            "open_old": _mk_job(s, "ret-oo", False, OLD),         # keep (not closed)
        }
        app = Application(job_id=ids["closed_old_applied"], status=ApplicationStatus.SUBMITTED)
        s.add(app)
        s.commit()
    yield ids
    with get_session() as s:
        s.exec(delete(Application).where(Application.job_id.in_(list(ids.values()))))
        s.exec(delete(Job).where(Job.id.in_(list(ids.values()))))
        s.commit()


def _exists(jid):
    with get_session() as s:
        return s.get(Job, jid) is not None


def test_purge_deletes_only_dead_jobs(scenario):
    ids = scenario
    n = purge_old_closed_jobs(days=60)
    assert n >= 1
    assert not _exists(ids["closed_old"])            # deleted
    assert _exists(ids["closed_old_applied"])        # kept — has an application
    assert _exists(ids["closed_recent"])             # kept — too new
    assert _exists(ids["open_old"])                  # kept — still open


def test_purge_disabled_when_days_zero(scenario):
    assert purge_old_closed_jobs(days=0) == 0
    assert _exists(scenario["closed_old"])           # nothing deleted
