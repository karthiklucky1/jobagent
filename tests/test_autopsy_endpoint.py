"""Endpoint test for GET /application/{id}/autopsy.

Runs with no SERPAPI key — the autopsy degrades to a JD-only diagnosis but still
returns a full structured result. Read-only; nothing is hidden or mutated.
"""
from sqlmodel import select
from app.db.init_db import get_session
from app.db.models import Job, JobSource, Application, ApplicationStatus


def _client():
    from fastapi.testclient import TestClient
    from app.api.server import app
    return TestClient(app)


def _seed():
    with get_session() as s:
        for j in s.exec(select(Job).where(Job.external_id.like("autopsy-test-%"))).all():
            for a in s.exec(select(Application).where(Application.job_id == j.id)).all():
                s.delete(a)
            s.delete(j)
        s.commit()
        j = Job(source=JobSource.MANUAL, external_id="autopsy-test-1",
                company="Royal Caribbean Group", title="AI Engineer",
                url="http://x", description="7+ years. Enterprise Data & Analytics. On-site Miami, FL.")
        s.add(j); s.commit(); s.refresh(j)
        a = Application(job_id=j.id, status=ApplicationStatus.SHORTLISTED)
        s.add(a); s.commit(); s.refresh(a)
        return a.id, j.id


def _cleanup():
    with get_session() as s:
        for j in s.exec(select(Job).where(Job.external_id.like("autopsy-test-%"))).all():
            for a in s.exec(select(Application).where(Application.job_id == j.id)).all():
                s.delete(a)
            s.delete(j)
        s.commit()


def test_autopsy_endpoint_returns_diagnosis():
    app_id, _ = _seed()
    try:
        r = _client().get(f"/application/{app_id}/autopsy")
        assert r.status_code == 200, r.text
        body = r.json()
        for key in ("wrong_door", "top_reason", "right_door", "confidence", "findings", "bar"):
            assert key in body, f"missing {key}"
        assert body["bar"]["axis"] in ("applied", "research", "enterprise", "elite-outlier")
        # Enterprise + on-site JD → this candidate should be flagged wrong-door.
        assert body["wrong_door"] is True
    finally:
        _cleanup()


def test_autopsy_endpoint_404_for_missing():
    r = _client().get("/application/999999999/autopsy")
    assert r.status_code == 404
