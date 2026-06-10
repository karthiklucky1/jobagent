"""Unit tests for hire_probability scorer."""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from app.matching.hire_probability import score_hire_probability, blended_score, HireProbabilityResult
from app.db.models import Job, JobSource


def _make_job(**kwargs) -> Job:
    defaults = dict(
        id=1,
        source=JobSource.GREENHOUSE,
        external_id="test-1",
        company="Acme Corp",
        title="ML Engineer",
        url="https://example.com/job/1",
        description="We are building AI products.",
        posted_at=datetime.utcnow() - timedelta(days=5),
        first_seen=datetime.utcnow() - timedelta(days=5),
        last_seen=datetime.utcnow(),
        is_closed=False,
    )
    defaults.update(kwargs)
    return Job(**defaults)


def _mock_session(open_count: int = 1):
    session = MagicMock()
    # session.exec(...).one() should return open_count
    session.exec.return_value.one.return_value = open_count
    return session


class TestPostingFreshness:
    def test_fresh_posting_gets_boost(self):
        job = _make_job(posted_at=datetime.utcnow() - timedelta(days=3))
        result = score_hire_probability(job, _mock_session())
        assert result.score >= 0.25
        assert any("fresh" in s for s in result.signals)

    def test_recent_posting_gets_partial_boost(self):
        job = _make_job(posted_at=datetime.utcnow() - timedelta(days=20))
        result = score_hire_probability(job, _mock_session())
        assert any("recent" in s for s in result.signals)

    def test_old_posting_no_freshness_boost(self):
        job = _make_job(posted_at=datetime.utcnow() - timedelta(days=90))
        result = score_hire_probability(job, _mock_session())
        assert not any("fresh" in s or "recent" in s for s in result.signals)


class TestCompanyVelocity:
    def test_high_velocity_company(self):
        job = _make_job()
        result = score_hire_probability(job, _mock_session(open_count=15))
        assert any("high_velocity" in s for s in result.signals)
        assert result.company_velocity == 15

    def test_medium_velocity_company(self):
        job = _make_job()
        result = score_hire_probability(job, _mock_session(open_count=6))
        assert any("medium_velocity" in s for s in result.signals)

    def test_single_opening_no_velocity_boost(self):
        job = _make_job()
        result = score_hire_probability(job, _mock_session(open_count=1))
        assert not any("velocity" in s for s in result.signals)


class TestDescriptionSignals:
    def test_funding_language_boosts_score(self):
        job = _make_job(description="We just raised a Series B and are growing fast.")
        result = score_hire_probability(job, _mock_session())
        assert any("funding" in s for s in result.signals)

    def test_growth_language_boosts_score(self):
        job = _make_job(description="We are a rapidly growing team looking to double our headcount.")
        result = score_hire_probability(job, _mock_session())
        assert any("growth" in s for s in result.signals)

    def test_urgency_language_boosts_score(self):
        job = _make_job(description="We need someone to join us with an immediate start.")
        result = score_hire_probability(job, _mock_session())
        assert any("urgency" in s for s in result.signals)

    def test_enterprise_scale_penalizes_score(self):
        job_plain = _make_job(description="Normal job posting.")
        job_enterprise = _make_job(description="Fortune 500 company with over 50,000 employees worldwide.")
        plain_result = score_hire_probability(job_plain, _mock_session())
        ent_result = score_hire_probability(job_enterprise, _mock_session())
        assert ent_result.score < plain_result.score
        assert any("enterprise" in s for s in ent_result.signals)

    def test_startup_signals_boost_score(self):
        job = _make_job(description="Join as founding engineer on an early stage AI startup.")
        result = score_hire_probability(job, _mock_session())
        assert any("startup" in s for s in result.signals)


class TestBlendedScore:
    def test_blended_weights(self):
        # rerank=80, hp=0.8 → 0.65*80 + 0.35*80 = 80
        assert blended_score(80.0, 0.8) == pytest.approx(0.65 * 80 + 0.35 * 0.8 * 100, abs=0.2)

    def test_blended_caps_at_100(self):
        result = blended_score(100.0, 1.0)
        assert result <= 100.0

    def test_blended_floors_at_0(self):
        result = blended_score(0.0, 0.0)
        assert result >= 0.0

    def test_high_fit_low_hp_still_decent(self):
        # Good fit (85) but company shows low hiring intent (0.1) → still shortlistable
        result = blended_score(85.0, 0.1)
        assert result >= 55.0  # 0.65*85 + 0.35*10 = 55.25 + 3.5 = 58.75

    def test_score_bounds(self):
        for rs in [0, 30, 60, 90, 100]:
            for hp in [0.0, 0.3, 0.6, 1.0]:
                result = blended_score(float(rs), hp)
                assert 0.0 <= result <= 100.0, f"Out of bounds: rerank={rs} hp={hp} → {result}"
