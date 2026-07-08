"""Endpoint test for the enriched GET /application/{id}/match report.

The job detail panel now renders one unified AI Match Report; this verifies
the endpoint supplies everything it needs: score, humanized company/hiring
signals, resume keyword coverage, and the vetting-pipeline trail.
"""
import json

from sqlmodel import select

from app.db.init_db import get_session
from app.db.models import Job, JobSource, Application, ApplicationStatus


def _client():
    from fastapi.testclient import TestClient
    from app.api.server import app
    return TestClient(app)


def _cleanup():
    with get_session() as s:
        for j in s.exec(select(Job).where(Job.external_id.like("match-report-test-%"))).all():
            for a in s.exec(select(Application).where(Application.job_id == j.id)).all():
                s.delete(a)
            s.delete(j)
        s.commit()


def _seed():
    _cleanup()
    with get_session() as s:
        j = Job(
            source=JobSource.GREENHOUSE, external_id="match-report-test-1",
            company="Acme AI", title="Backend Engineer", url="https://boards.greenhouse.io/acme/jobs/1",
            description="We need Python, FastAPI and PostgreSQL experience. Series A startup.",
            rerank_score=78.0,
            rerank_reasoning="Strong backend overlap.\nConcerns: no Kubernetes experience",
            blended_score=71.2,
            hire_probability_score=0.64,
            hire_probability_signals=json.dumps(
                ["fresh_posting_3d", "funding_language:series a", "high_velocity_6_openings"]),
            ghost_score=0.1,
        )
        s.add(j); s.commit(); s.refresh(j)
        a = Application(job_id=j.id, status=ApplicationStatus.SHORTLISTED)
        s.add(a); s.commit(); s.refresh(a)
        return a.id


def test_match_report_includes_signals_and_pipeline():
    app_id = _seed()
    try:
        r = _client().get(f"/application/{app_id}/match")
        assert r.status_code == 200, r.text
        d = r.json()

        assert d["score"] == 78
        assert d["blended_score"] == 71
        assert d["hire_probability"] == 64

        # Raw signal tokens must arrive humanized, funding info included.
        # Posted-time signals are excluded — they freeze at match time and go
        # stale; the UI header shows the live posted time instead.
        assert "Posted 3 days ago" not in d["signals"]
        assert "Series A funded" in d["signals"]
        assert "6 openings (actively hiring)" in d["signals"]

        # Vetting trail: every gate named, ghost + AI fit + hiring intent passed.
        steps = {st["name"]: st for st in d["pipeline"]}
        assert len(d["pipeline"]) == 7
        assert steps["Ghost-posting check"]["ok"] is True
        assert steps["AI fit review"]["ok"] is True
        assert "78/100" in steps["AI fit review"]["detail"]
        assert "64%" in steps["Hiring-intent analysis"]["detail"]
        # No senior review recorded yet → surfaced as pending, not passed.
        assert steps["Senior engineer review"]["ok"] is False

        # Keyword coverage is best-effort (needs a resume on disk); when present
        # it must carry the matched/missing structure the skill chips render.
        if d["skills"] is not None:
            for key in ("matched", "missing", "total", "coverage_pct"):
                assert key in d["skills"]
    finally:
        _cleanup()


def test_match_report_flags_ghost_posting():
    app_id = _seed()
    with get_session() as s:
        a = s.get(Application, app_id)
        j = s.get(Job, a.job_id)
        j.ghost_score = 0.8
        s.add(j); s.commit()
    try:
        d = _client().get(f"/application/{app_id}/match").json()
        steps = {st["name"]: st for st in d["pipeline"]}
        assert steps["Ghost-posting check"]["ok"] is False
    finally:
        _cleanup()
