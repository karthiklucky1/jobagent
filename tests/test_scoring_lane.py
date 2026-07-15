"""Scoring lane: decoupled parallel cross-user scoring → shortlist + alert."""
from __future__ import annotations

from datetime import datetime, timedelta

from sqlmodel import delete, select

import app.strategy.scoring_lane as sl
from app.db.init_db import get_session
from app.db.models import (
    Application, FunnelEvent, Job, JobSource, UserNotification, UserProfile,
)


def _clean(session):
    for model in (Application, UserNotification, FunnelEvent, Job, UserProfile):
        session.exec(delete(model))
    session.commit()


def _job(session, uid, ext, title="Senior ML Engineer"):
    session.add(Job(title=title, company=f"Co{ext}", location="Remote", remote=True,
                    description="LLMs", source=JobSource.GREENHOUSE, external_id=str(ext),
                    url=f"https://boards.greenhouse.io/co/jobs/{ext}", user_id=uid,
                    first_seen=datetime.utcnow(),
                    posted_at=datetime.utcnow() - timedelta(hours=1)))


class _FakeReranker:
    def __init__(self, profile=None, feedback=""):
        pass
    def has_prescore_backend(self):
        return False
    def score(self, resume, job):
        # "Forklift" scores low (drain), everything else strong.
        if "Forklift" in job.title:
            return 12.0, "unrelated", [], {}
        return 78.0, "Strong fit", [], {}


class _G:
    is_ghost = False
    ghost_score = 0.0
    flags_json = None
    flags = []


def _patch(monkeypatch):
    monkeypatch.setattr("app.matching.reranker.Reranker", _FakeReranker)
    monkeypatch.setattr("app.matching.pipeline._load_resume", lambda user_id=None: "resume")
    monkeypatch.setattr("app.matching.filters.score_ghost", lambda job, session: _G())


def test_scores_multiple_users_and_shortlists(monkeypatch):
    with get_session() as session:
        _clean(session)
        session.add(UserProfile(user_id="ua", target_roles="ml engineer"))
        session.add(UserProfile(user_id="ub", target_roles="ml engineer"))
        _job(session, "ua", 1)
        _job(session, "ua", 2)
        _job(session, "ub", 3)
        session.commit()
    _patch(monkeypatch)

    stats = sl.run_scoring_lane()
    assert stats["users"] == 2
    assert stats["scored"] == 3
    assert stats["shortlisted"] == 3
    assert stats["alerts"] >= 1

    with get_session() as session:
        assert all(j.rerank_score == 78.0 for j in session.exec(select(Job)).all())
        apps = session.exec(select(Application)).all()
        assert {a.user_id for a in apps} == {"ua", "ub"}
        # A scoring_cycle funnel event was recorded.
        assert session.exec(select(FunnelEvent).where(FunnelEvent.stage == "scoring_cycle")).all()


def test_low_scores_are_not_shortlisted(monkeypatch):
    with get_session() as session:
        _clean(session)
        session.add(UserProfile(user_id="ua", target_roles="ml engineer"))
        _job(session, "ua", 1, title="Forklift Operator")
        session.commit()
    _patch(monkeypatch)

    stats = sl.run_scoring_lane()
    assert stats["scored"] == 1 and stats["shortlisted"] == 0
    with get_session() as session:
        assert session.exec(select(Job)).first().rerank_score == 12.0
        assert session.exec(select(Application)).first() is None


def test_already_scored_jobs_are_skipped(monkeypatch):
    with get_session() as session:
        _clean(session)
        session.add(UserProfile(user_id="ua", target_roles="ml engineer"))
        j = Job(title="Senior ML Engineer", company="Co", location="Remote", remote=True,
                description="x", source=JobSource.GREENHOUSE, external_id="pre",
                url="https://x/pre", user_id="ua", rerank_score=90.0,
                first_seen=datetime.utcnow())
        session.add(j)
        session.commit()
    _patch(monkeypatch)

    stats = sl.run_scoring_lane()
    # Nothing unscored → nothing to do.
    assert stats["scored"] == 0


def test_empty_queue_is_noop(monkeypatch):
    with get_session() as session:
        _clean(session)
    _patch(monkeypatch)
    stats = sl.run_scoring_lane()
    assert stats.get("scored", 0) == 0 and stats.get("shortlisted", 0) == 0
