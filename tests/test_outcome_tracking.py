"""Outcome tracking: ghosted marks + response-rate funnel stats."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlmodel import delete

from app.db.init_db import get_session
from app.db.models import Application, ApplicationStatus, FunnelEvent, Job, JobSource


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from app.api.server import app
    return TestClient(app)


def _mk_app(session, i: int, status=ApplicationStatus.SUBMITTED,
            submitted_days_ago: int | None = 1, response_type: str = "none"):
    job = Job(user_id=None, source=JobSource.GREENHOUSE, external_id=f"out-{i}",
              company=f"Co{i}", title="SWE", url=f"https://x/{i}", description="jd")
    session.add(job)
    session.commit()
    session.refresh(job)
    application = Application(
        user_id=None, job_id=job.id, status=status,
        submitted_at=(datetime.utcnow() - timedelta(days=submitted_days_ago))
        if submitted_days_ago is not None else None,
        response_type=response_type,
    )
    session.add(application)
    session.commit()
    session.refresh(application)
    return application


def _clean(session):
    session.exec(delete(Application))
    session.exec(delete(FunnelEvent))
    session.exec(delete(Job))
    session.commit()


def test_mark_ghosted_keeps_submitted_status(client):
    with get_session() as session:
        _clean(session)
        application = _mk_app(session, 1)
        app_id = application.id

    res = client.post(f"/application/{app_id}/outcome?outcome=ghosted")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["response_type"] == "ghosted"
    assert body["status"] == "submitted"

    with get_session() as session:
        row = session.get(Application, app_id)
        assert row.status == ApplicationStatus.SUBMITTED
        assert row.response_type == "ghosted"
        events = session.exec(
            __import__("sqlmodel").select(FunnelEvent).where(FunnelEvent.stage == "responded")
        ).all()
        assert any(e.reason == "ghosted" and e.passed is False for e in events)


def test_mark_outcome_invalid_rejected(client):
    with get_session() as session:
        _clean(session)
        application = _mk_app(session, 2)
    res = client.post(f"/application/{application.id}/outcome?outcome=banana")
    assert res.status_code == 400


def test_funnel_reports_ghosting_and_response_rate(client):
    with get_session() as session:
        _clean(session)
        _mk_app(session, 10)                                              # fresh, silent
        _mk_app(session, 11, submitted_days_ago=30)                       # presumed ghosted
        _mk_app(session, 12, response_type="ghosted")                     # explicit ghosted
        _mk_app(session, 13, status=ApplicationStatus.INTERVIEWING)       # responded
        _mk_app(session, 14, status=ApplicationStatus.REJECTED)           # responded (no)

    res = client.get("/api/funnel")
    assert res.status_code == 200, res.text
    d = res.json()
    assert d["applied"] == 5
    assert d["ghosted"] == 1
    assert d["presumed_ghosted"] == 1
    assert d["interviewing"] == 1 and d["rejected"] == 1
    assert d["response_rate"] == 40  # 2 of 5
