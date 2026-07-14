"""Pulse lane: cadence tiers, change detection, tick routing, fast-path alerts,
and the My Companies watchlist API."""
from __future__ import annotations

from datetime import datetime, timedelta

from sqlmodel import delete, select

import app.strategy.pulse_lane as pl
from app.config import settings
from app.db.init_db import get_session
from app.db.models import (
    Application, CompanyRegistry, FunnelEvent, Job, JobSource, UserNotification,
    UserProfile,
)
from app.discovery.base import RawJob


def _clean(session):
    for model in (Application, UserNotification, FunnelEvent, Job,
                  CompanyRegistry, UserProfile):
        session.exec(delete(model))
    session.commit()


def _board(session, slug, job_count=5, last_new_days=None, next_poll_at=None):
    row = CompanyRegistry(slug=slug, ats=JobSource.GREENHOUSE, is_active=True,
                          job_count=job_count, source="test",
                          next_poll_at=next_poll_at)
    if last_new_days is not None:
        row.last_new_job_at = datetime.utcnow() - timedelta(days=last_new_days)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def _raw(ext, title="Senior ML Engineer"):
    return RawJob(source="greenhouse", external_id=str(ext), company="Acme",
                  title=title, location="Remote", remote=True,
                  url=f"https://boards.greenhouse.io/acme/jobs/{ext}",
                  description="Build LLM systems with PyTorch and RAG.",
                  posted_at=datetime.utcnow())


# ── cadence / watchlist matching ──────────────────────────────────────────────

def test_cadence_tiers():
    now = datetime.utcnow()
    fast = timedelta(minutes=settings.pulse_fast_interval_minutes)
    floor = timedelta(minutes=settings.pulse_floor_interval_minutes)
    dead = timedelta(hours=settings.pulse_dead_interval_hours)

    watched = CompanyRegistry(slug="anthropic", ats=JobSource.GREENHOUSE, job_count=0)
    assert pl._cadence(watched, {"anthropic"}, now) == fast

    recent = CompanyRegistry(slug="acme", ats=JobSource.GREENHOUSE, job_count=9,
                             last_new_job_at=now - timedelta(days=1))
    assert pl._cadence(recent, set(), now) == fast

    live = CompanyRegistry(slug="quiet", ats=JobSource.GREENHOUSE, job_count=3,
                           last_new_job_at=now - timedelta(days=90))
    assert pl._cadence(live, set(), now) == floor

    empty = CompanyRegistry(slug="ghost-town", ats=JobSource.GREENHOUSE, job_count=0)
    assert pl._cadence(empty, set(), now) == dead


def test_watchlist_matching_is_normalized():
    row = CompanyRegistry(slug="fleetdatacenters", ats=JobSource.GREENHOUSE,
                          company_name="Fleet Data Centers", job_count=1)
    assert pl._is_watched(row, {pl._norm("Fleet Data Centers")})
    assert pl._is_watched(row, {pl._norm("fleet-data-centers")})
    assert not pl._is_watched(row, {pl._norm("Stripe")})
    # Short generic terms must not fuzzy-match everything.
    assert not pl._is_watched(row, {"fle"})


def test_board_signature_tracks_posting_list():
    a = [_raw(1), _raw(2)]
    b = [_raw(2), _raw(1)]           # order must not matter
    c = [_raw(1), _raw(2), _raw(3)]  # a new job must change it
    assert pl._board_signature(a) == pl._board_signature(b)
    assert pl._board_signature(a) != pl._board_signature(c)


# ── one tick end-to-end (fake scraper, no LLM) ───────────────────────────────

def test_tick_routes_new_jobs_and_schedules_next_poll(monkeypatch):
    with get_session() as session:
        _clean(session)
        _board(session, "acme")
        session.add(UserProfile(user_id="u_ml", target_roles="machine learning engineer"))
        session.commit()

    class FakeScraper:
        def fetch(self):
            return [_raw(1), _raw(2, title="Bakery Production Manager")]

    monkeypatch.setattr("app.discovery.pipeline.scraper_for",
                        lambda ats, slug, career_url=None: FakeScraper())
    monkeypatch.setattr("app.strategy.hot_lane._active_users",
                        lambda: [{"user_id": "u_ml", "roles": ["machine learning engineer"]}])
    fast_calls = []
    monkeypatch.setattr(pl, "_fast_path_user",
                        lambda uid, budget: (fast_calls.append(uid) or (1, 1, 1)))

    stats = pl.run_pulse_tick()
    assert stats["boards"] == 1 and stats["changed"] == 1
    assert stats["new_jobs"] >= 1

    with get_session() as session:
        # ML job routed to the user; the bakery title was not.
        titles = {j.title for j in session.exec(
            select(Job).where(Job.user_id == "u_ml")).all()}
        assert "Senior ML Engineer" in titles
        assert "Bakery Production Manager" not in titles
        row = session.exec(select(CompanyRegistry)).first()
        assert row.next_poll_at is not None and row.poll_hash
        # Board just posted → promoted to the fast cadence.
        due_in = row.next_poll_at - datetime.utcnow()
        assert due_in <= timedelta(minutes=settings.pulse_fast_interval_minutes + 1)
    assert fast_calls == ["u_ml"]

    # Second tick, same board content: due again → unchanged, zero new work.
    with get_session() as session:
        row = session.exec(select(CompanyRegistry)).first()
        row.next_poll_at = datetime.utcnow() - timedelta(minutes=1)
        session.add(row)
        session.commit()
    stats2 = pl.run_pulse_tick()
    assert stats2["boards"] == 1 and stats2["changed"] == 0 and stats2["new_jobs"] == 0


def test_tick_records_funnel_event(monkeypatch):
    with get_session() as session:
        _clean(session)
        _board(session, "acme")

    class FakeScraper:
        def fetch(self):
            return [_raw(1)]

    monkeypatch.setattr("app.discovery.pipeline.scraper_for",
                        lambda ats, slug, career_url=None: FakeScraper())
    monkeypatch.setattr("app.strategy.hot_lane._active_users", lambda: [])
    pl.run_pulse_tick()
    with get_session() as session:
        ev = session.exec(select(FunnelEvent).where(FunnelEvent.stage == "pulse_tick")).all()
        assert len(ev) == 1


# ── per-job fast path (fake reranker) ────────────────────────────────────────

def test_fast_path_scores_shortlists_and_alerts(monkeypatch):
    with get_session() as session:
        _clean(session)
        j = Job(title="Senior ML Engineer", company="Acme", location="Remote",
                remote=True, description="LLMs", source=JobSource.GREENHOUSE,
                external_id="fp1", url="https://boards.greenhouse.io/acme/jobs/9",
                user_id="u_ml", first_seen=datetime.utcnow(),
                posted_at=datetime.utcnow() - timedelta(hours=2))
        session.add(j)
        session.commit()
        jid = j.id

    class FakeReranker:
        def __init__(self, profile=None, feedback=""):
            pass
        def has_prescore_backend(self):
            return False
        def score(self, resume, job):
            return 74.0, "Strong fit", [], {}

    monkeypatch.setattr("app.matching.reranker.Reranker", FakeReranker)
    monkeypatch.setattr("app.matching.pipeline._load_resume", lambda user_id=None: "resume")

    class _G:
        is_ghost = False
        ghost_score = 0.0
        flags_json = None
        flags = []
    monkeypatch.setattr("app.matching.filters.score_ghost", lambda job, session: _G())

    scored, short, alerts = pl._fast_path_user("u_ml", score_budget=10)
    assert (scored, short) == (1, 1)
    assert alerts == 1

    with get_session() as session:
        job = session.get(Job, jid)
        assert job.rerank_score == 74.0
        apps = session.exec(select(Application).where(Application.job_id == jid)).all()
        assert len(apps) == 1 and apps[0].user_id == "u_ml"
        notes = session.exec(select(UserNotification)).all()
        assert len(notes) == 1 and notes[0].type == "fresh_job"


def test_fast_path_prescore_drains_misfit(monkeypatch):
    with get_session() as session:
        _clean(session)
        j = Job(title="Forklift Operator", company="Acme", location="Denver",
                remote=False, description="Warehouse", source=JobSource.GREENHOUSE,
                external_id="fp2", url="https://x/2", user_id="u_ml",
                first_seen=datetime.utcnow(), posted_at=datetime.utcnow())
        session.add(j)
        session.commit()
        jid = j.id

    class FakeReranker:
        def __init__(self, profile=None, feedback=""):
            pass
        def has_prescore_backend(self):
            return True
        def prescore(self, resume, job):
            return 8.0, "unrelated field"
        def score(self, resume, job):
            raise AssertionError("Tier-2 must not run for a clear Tier-1 misfit")

    monkeypatch.setattr("app.matching.reranker.Reranker", FakeReranker)
    monkeypatch.setattr("app.matching.pipeline._load_resume", lambda user_id=None: "resume")

    class _G:
        is_ghost = False
        ghost_score = 0.0
        flags_json = None
        flags = []
    monkeypatch.setattr("app.matching.filters.score_ghost", lambda job, session: _G())

    scored, short, alerts = pl._fast_path_user("u_ml", score_budget=10)
    assert (scored, short, alerts) == (1, 0, 0)
    with get_session() as session:
        job = session.get(Job, jid)
        assert job.rerank_score == 8.0
        assert "Pre-screened" in (job.rerank_reasoning or "")


# ── watchlist API ─────────────────────────────────────────────────────────────

def test_target_companies_api_roundtrip_and_pull_forward():
    from fastapi.testclient import TestClient
    from app.api.server import app as fastapi_app

    with get_session() as session:
        _clean(session)
        _board(session, "fleetdatacenters", job_count=4,
               next_poll_at=datetime.utcnow() + timedelta(hours=6))

    client = TestClient(fastapi_app)
    r = client.put("/api/target-companies",
                   json={"companies": ["Fleet Data Centers", "  ", "Fleet Data Centers"]})
    assert r.status_code == 200, r.text
    assert r.json()["companies"] == ["Fleet Data Centers"]

    r2 = client.get("/api/target-companies")
    assert r2.status_code == 200
    body = r2.json()
    assert body["companies"] == ["Fleet Data Centers"]
    assert body["fast_interval_minutes"] == settings.pulse_fast_interval_minutes

    # Saving pulled the matching board forward — due now, not in 6 hours.
    with get_session() as session:
        row = session.exec(select(CompanyRegistry)).first()
        assert row.next_poll_at <= datetime.utcnow() + timedelta(seconds=5)
