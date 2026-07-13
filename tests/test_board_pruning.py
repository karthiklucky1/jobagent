"""Dead-board retirement: 404 boards must be deactivated in bulk after a fetch,
regardless of the board-phase budget, so they stop 404ing every run."""
from __future__ import annotations

from sqlmodel import delete, select

from app.db.init_db import get_session
from app.db.models import CompanyRegistry, JobSource
from app.discovery.pipeline import (
    BOARD_DEACTIVATE_AFTER_FAILURES, record_board_failures_bulk,
)


def _seed(slug, active=True, fails=0):
    with get_session() as s:
        s.add(CompanyRegistry(slug=slug, ats=JobSource.GREENHOUSE, is_active=active,
                              job_count=1, source="test", failure_count=fails))
        s.commit()


def _clean():
    with get_session() as s:
        s.exec(delete(CompanyRegistry))
        s.commit()


def test_bulk_prune_retires_404_immediately_and_flaky_after_threshold():
    _clean()
    _seed("gone")                                  # 404 → retire now
    _seed("flaky", fails=BOARD_DEACTIVATE_AFTER_FAILURES - 1)  # one more kills it
    _seed("blip")                                  # single transient → survives
    _seed("healthy")                               # never in the failure list

    n = record_board_failures_bulk([
        ("gone", "greenhouse", "Client error '404 Not Found' for url '...'"),
        ("flaky", "greenhouse", "timeout"),
        ("blip", "greenhouse", "timeout"),
        (None, "greenhouse", "no slug — ignored"),
    ])
    assert n == 2  # gone + flaky

    with get_session() as s:
        rows = {r.slug: r for r in s.exec(select(CompanyRegistry)).all()}
    assert rows["gone"].is_active is False and "404" in rows["gone"].inactive_reason
    assert rows["flaky"].is_active is False
    assert rows["blip"].is_active is True and rows["blip"].failure_count == 1
    assert rows["healthy"].is_active is True and rows["healthy"].failure_count == 0


def test_bulk_prune_skips_already_inactive():
    _clean()
    _seed("already", active=False)
    n = record_board_failures_bulk([("already", "greenhouse", "404")])
    assert n == 0  # already retired — not re-counted
