"""Cost guards: provider circuit breaker, daily spend cap, cache-minimum
padding, per-job attempt ceiling, round-robin fairness, adoption extras cap.

These are the levers that prevent a repeat of the Jul-15 overnight spend:
every test exercises a guard with fake clients — no API keys, no spend.
"""
from __future__ import annotations

import time
from datetime import datetime

import pytest
from sqlmodel import delete

import app.matching.reranker as rr
import app.strategy.scoring_lane as sl
from app.config import settings
from app.db.init_db import get_session
from app.db.models import Application, FunnelEvent, Job, JobSource, UserNotification, UserProfile


@pytest.fixture(autouse=True)
def _reset_guard_state():
    """Module-level guard state must not leak between tests."""
    rr._provider_down_until.clear()
    rr._daily_finals["day"] = ""
    rr._daily_finals["count"] = 0
    rr._hourly_finals["hour"] = ""
    rr._hourly_finals["count"] = 0
    sl._fail_counts.clear()
    sl._deferred_until.clear()
    yield
    rr._provider_down_until.clear()
    rr._daily_finals["day"] = ""
    rr._daily_finals["count"] = 0
    rr._hourly_finals["hour"] = ""
    rr._hourly_finals["count"] = 0
    sl._fail_counts.clear()
    sl._deferred_until.clear()


def _job(title="Senior ML Engineer", jid=None):
    j = Job(title=title, company="Acme", location="Remote", remote=True,
            description="Build LLM systems in Python.", source=JobSource.GREENHOUSE,
            external_id="x1", url="https://x/1")
    if jid is not None:
        j.id = jid
    return j


def _reranker_with(anthropic=True, openai=True):
    rk = rr.Reranker.__new__(rr.Reranker)
    rk._profile = None
    rk._feedback = ""
    rk._anthropic_client = object() if anthropic else None
    rk._openai_client = object() if openai else None
    rk._active_backend = "anthropic" if anthropic else ("openai" if openai else None)
    return rk


# ── Circuit breaker ───────────────────────────────────────────────────────────
def test_credit_error_marks_provider_down_and_falls_back(monkeypatch):
    rk = _reranker_with(True, True)
    monkeypatch.setattr(rk, "_pre_filter_job", lambda job: None)
    monkeypatch.setattr(rk, "_score_anthropic", lambda rb, jb: (_ for _ in ()).throw(
        RuntimeError("Your credit balance is too low")))
    monkeypatch.setattr(rk, "_score_openai", lambda rb, jb:
                        '{"score": 70, "reason": "ok", "concerns": [], "breakdown": {}}')

    score, *_ = rk.score("resume", _job(), provider="anthropic")
    assert score == 70.0                                  # fallback served it
    assert not rr.provider_available("anthropic")         # breaker tripped
    assert rr.provider_available("openai")


def test_down_provider_is_skipped_without_api_call(monkeypatch):
    rk = _reranker_with(True, True)
    monkeypatch.setattr(rk, "_pre_filter_job", lambda job: None)
    calls = {"anthropic": 0}

    def _anthropic(rb, jb):
        calls["anthropic"] += 1
        raise AssertionError("should not be called while down")
    monkeypatch.setattr(rk, "_score_anthropic", _anthropic)
    monkeypatch.setattr(rk, "_score_openai", lambda rb, jb:
                        '{"score": 61, "reason": "ok", "concerns": [], "breakdown": {}}')

    rr._mark_provider_down("anthropic")
    score, *_ = rk.score("resume", _job(), provider="anthropic")
    assert score == 61.0 and calls["anthropic"] == 0


def test_all_providers_down_raises_before_any_call(monkeypatch):
    rk = _reranker_with(True, True)
    monkeypatch.setattr(rk, "_pre_filter_job", lambda job: None)
    rr._mark_provider_down("anthropic")
    rr._mark_provider_down("openai")
    assert not rr.any_provider_available()
    with pytest.raises(RuntimeError, match="cooling down"):
        rk.score("resume", _job())


def test_breaker_expires_after_cooldown():
    rr._mark_provider_down("anthropic")
    assert not rr.provider_available("anthropic")
    rr._provider_down_until["anthropic"] = time.time() - 1  # cooldown elapsed
    assert rr.provider_available("anthropic")


def test_breaker_disabled_when_cooldown_zero(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider_cooldown_minutes", 0)
    rr._mark_provider_down("anthropic")
    assert rr.provider_available("anthropic")


def test_prescore_credit_error_trips_breaker(monkeypatch):
    rk = _reranker_with(anthropic=False, openai=True)
    monkeypatch.setattr(rk, "_pre_filter_job", lambda job: None)
    monkeypatch.setattr(rk, "_prescore_openai", lambda prompt: (_ for _ in ()).throw(
        RuntimeError("insufficient_quota: You exceeded your current quota")))
    assert rk.prescore("resume", _job()) is None          # fail-open
    assert not rr.provider_available("openai")


# ── Daily spend cap ───────────────────────────────────────────────────────────
def test_daily_budget_blocks_finals_past_cap(monkeypatch):
    monkeypatch.setattr(settings, "llm_daily_final_cap", 2)
    rk = _reranker_with(True, False)
    monkeypatch.setattr(rk, "_pre_filter_job", lambda job: None)
    monkeypatch.setattr(rk, "_score_anthropic", lambda rb, jb:
                        '{"score": 80, "reason": "ok", "concerns": [], "breakdown": {}}')

    assert rk.score("resume", _job())[0] == 80.0
    assert rk.score("resume", _job())[0] == 80.0
    assert rr.llm_budget_exhausted()
    with pytest.raises(RuntimeError, match="budget"):
        rk.score("resume", _job())                        # third call: no spend


def test_daily_budget_resets_on_new_day(monkeypatch):
    monkeypatch.setattr(settings, "llm_daily_final_cap", 1)
    rr._daily_finals["day"] = "1999-01-01"
    rr._daily_finals["count"] = 999
    assert not rr.llm_budget_exhausted()                  # stale day ≠ today


def test_daily_budget_unlimited_when_zero(monkeypatch):
    monkeypatch.setattr(settings, "llm_daily_final_cap", 0)
    rr._daily_finals["day"] = datetime.utcnow().strftime("%Y-%m-%d")
    rr._daily_finals["count"] = 10**9
    assert not rr.llm_budget_exhausted()


# ── Cache-minimum padding ─────────────────────────────────────────────────────
def test_resume_block_padded_past_cache_minimum():
    resume = "PYTHON ML ENGINEER RESUME LINE\n" * 150   # ~4.6K chars — typical résumé
    block = rr._resume_context_block(resume)
    assert len(block) >= rr._CACHE_MIN_BLOCK_CHARS       # crosses Haiku's 4096-token floor
    assert "<resume_repeat>" in block                    # padded with the résumé itself
    assert "no new information" in block                 # pad is labeled inert


def test_long_resume_needs_no_padding():
    resume = "x" * 20000
    block = rr._resume_context_block(resume)
    assert "<resume_repeat>" not in block
    assert resume[: rr._RESUME_SLICE_CHARS] in block     # slice raised from 6000


def test_tiny_resume_is_not_padded():
    block = rr._resume_context_block("short resume")     # padding would need 100s of repeats
    assert "<resume_repeat>" not in block
    assert len(block) < rr._CACHE_MIN_BLOCK_CHARS        # stays uncached (status quo)


def test_resume_block_is_deterministic():
    resume = "PYTHON ML ENGINEER RESUME LINE\n" * 150
    assert rr._resume_context_block(resume) == rr._resume_context_block(resume)


# ── Per-job attempt ceiling ───────────────────────────────────────────────────
def test_repeated_failures_defer_job(monkeypatch):
    monkeypatch.setattr(settings, "scoring_fail_max_attempts", 3)
    for _ in range(2):
        sl._note_score_failure(42)
    assert sl._drop_deferred([42]) == [42]               # under the ceiling: retried
    sl._note_score_failure(42)                            # third strike
    assert sl._drop_deferred([42]) == []                  # deferred
    assert 42 not in sl._fail_counts


def test_success_clears_failure_state():
    sl._note_score_failure(7)
    sl._note_score_success(7)
    assert 7 not in sl._fail_counts and sl._drop_deferred([7]) == [7]


def test_deferral_expires():
    sl._deferred_until[9] = time.time() - 1
    assert sl._drop_deferred([9]) == [9]
    assert 9 not in sl._deferred_until                    # expired entries purged


# ── Scoring lane guards ───────────────────────────────────────────────────────
def _clean(session):
    for model in (Application, UserNotification, FunnelEvent, Job, UserProfile):
        session.exec(delete(model))
    session.commit()


def test_shared_pool_is_not_a_scorable_user():
    from app.discovery.pipeline import SHARED_POOL_USER
    with get_session() as session:
        _clean(session)
        session.add(Job(title="ML Engineer", company="C", location="R", remote=True,
                        description="d", source=JobSource.GREENHOUSE, external_id="s1",
                        url="https://x/s1", user_id=SHARED_POOL_USER,
                        first_seen=datetime.utcnow()))
        session.add(Job(title="ML Engineer", company="C", location="R", remote=True,
                        description="d", source=JobSource.GREENHOUSE, external_id="u1",
                        url="https://x/u1", user_id="ua", first_seen=datetime.utcnow()))
        session.commit()
    users = sl._scorable_user_ids()
    assert "ua" in users and SHARED_POOL_USER not in users


def test_cycle_skips_when_all_providers_down(monkeypatch):
    with get_session() as session:
        _clean(session)
        session.add(Job(title="ML Engineer", company="C", location="R", remote=True,
                        description="d", source=JobSource.GREENHOUSE, external_id="u1",
                        url="https://x/u1", user_id="ua", first_seen=datetime.utcnow()))
        session.commit()
    rr._mark_provider_down("anthropic")
    rr._mark_provider_down("openai")
    stats = sl.run_scoring_lane()
    assert stats.get("skipped") == "all LLM providers cooling down"
    assert stats["scored"] == 0


def test_cycle_skips_when_budget_exhausted(monkeypatch):
    monkeypatch.setattr(settings, "llm_daily_final_cap", 1)
    rr._daily_finals["day"] = datetime.utcnow().strftime("%Y-%m-%d")
    rr._daily_finals["count"] = 1
    stats = sl.run_scoring_lane()
    assert stats.get("skipped") == "LLM budget reached (hourly/daily cap)"


def test_global_cap_is_shared_fairly_round_robin(monkeypatch):
    """With the cap at 2 and two users queued, each user gets one slot — the old
    user-by-user fill gave both slots to whichever user came first."""
    with get_session() as session:
        _clean(session)
        for uid, ext in (("ua", 1), ("ua", 2), ("ub", 3), ("ub", 4)):
            session.add(Job(title="Senior ML Engineer", company=f"Co{ext}", location="R",
                            remote=True, description="LLMs", source=JobSource.GREENHOUSE,
                            external_id=str(ext), url=f"https://x/{ext}", user_id=uid,
                            first_seen=datetime.utcnow()))
        session.commit()

    class _FakeReranker:
        def __init__(self, profile=None, feedback=""):
            pass
        def has_prescore_backend(self):
            return False
        def has_dual(self):
            return False
        def score(self, resume, job, provider=None):
            return 78.0, "fit", [], {}

    class _G:
        is_ghost = False
        ghost_score = 0.0
        flags_json = None
        flags = []

    monkeypatch.setattr("app.matching.reranker.Reranker", _FakeReranker)
    monkeypatch.setattr("app.matching.pipeline._load_resume", lambda user_id=None: "resume")
    monkeypatch.setattr("app.matching.filters.score_ghost", lambda job, session: _G())
    monkeypatch.setattr(settings, "scoring_global_cap", 2)

    stats = sl.run_scoring_lane()
    assert stats["scored"] == 2
    assert stats["users"] == 2                            # one slot each, not 2+0


# ── Adoption extras cap ───────────────────────────────────────────────────────
def test_semantic_extras_bounded_by_budget(monkeypatch):
    from app.strategy import adoption

    captured = {}

    def _fake_extras(others, roles, user_id, need):
        captured["need"] = need
        return others[:need]

    monkeypatch.setattr(adoption, "_semantic_extras", _fake_extras)
    monkeypatch.setattr(settings, "adoption_semantic_enabled", True)
    monkeypatch.setattr(settings, "adoption_semantic_max_extras", 2)
    monkeypatch.setattr("app.discovery.title_filter.role_title_match",
                        lambda title, roles: False)      # nothing matches by title

    jobs = [_job(title=f"Applied Scientist {i}") for i in range(10)]
    picked = adoption._select_adoptable(jobs, ["ml engineer"], "ua", limit=400)
    assert captured["need"] == 2                          # budget, not 400 open slots
    assert len(picked) == 2


def test_semantic_extras_zero_budget_means_title_only(monkeypatch):
    from app.strategy import adoption
    monkeypatch.setattr(settings, "adoption_semantic_enabled", True)
    monkeypatch.setattr(settings, "adoption_semantic_max_extras", 0)
    monkeypatch.setattr(adoption, "_semantic_extras",
                        lambda others, roles, user_id, need: (_ for _ in ()).throw(
                            AssertionError("must not embed with zero budget")))
    monkeypatch.setattr("app.discovery.title_filter.role_title_match",
                        lambda title, roles: False)
    picked = adoption._select_adoptable([_job()], ["ml engineer"], "ua", limit=400)
    assert picked == []


# ── Cross-lane in-flight claim ────────────────────────────────────────────────
def test_inflight_claim_blocks_second_lane():
    from app.common import inflight
    inflight._inflight.clear()
    assert inflight.try_claim(101) is True
    assert inflight.try_claim(101) is False          # second lane blocked
    inflight.release(101)
    assert inflight.try_claim(101) is True           # free again after release
    inflight.release(101)


def test_inflight_context_manager_releases_on_exception():
    from app.common import inflight
    inflight._inflight.clear()
    with pytest.raises(ValueError):
        with inflight.claim(7) as ok:
            assert ok is True
            raise ValueError("boom")
    assert inflight.try_claim(7) is True             # released despite the error
    inflight.release(7)


def test_score_job_skips_job_claimed_by_another_lane(monkeypatch):
    from app.common import inflight
    inflight._inflight.clear()

    class _NeverCalled:
        def prescore(self, *a, **k):
            raise AssertionError("must not score a claimed job")
        def score(self, *a, **k):
            raise AssertionError("must not score a claimed job")
        def has_dual(self):
            return False

    ctx = sl._Ctx("resume", _NeverCalled(), True, 35)
    inflight.try_claim(999)                          # another lane owns it
    try:
        assert sl._score_job(999, ctx) is None       # skipped, zero LLM calls
    finally:
        inflight.release(999)


# ── Prescore memo (retry pays only for the step that failed) ─────────────────
def test_failed_final_reuses_memoized_prescore(monkeypatch):
    from app.common import inflight
    inflight._inflight.clear()
    sl._prescore_memo.clear()
    with get_session() as session:
        _clean(session)
        session.add(Job(title="ML Engineer", company="C", location="R", remote=True,
                        description="d", source=JobSource.GREENHOUSE, external_id="m1",
                        url="https://x/m1", user_id="ua", first_seen=datetime.utcnow()))
        session.commit()
        jid = session.exec(__import__("sqlmodel").select(Job.id)).first()
        jid = jid[0] if isinstance(jid, tuple) else jid

    calls = {"pre": 0, "final": 0}

    class _RK:
        def prescore(self, resume, job):
            calls["pre"] += 1
            return (60.0, "promising")               # advances past the gate
        def score(self, resume, job, provider=None):
            calls["final"] += 1
            if calls["final"] == 1:
                raise RuntimeError("overloaded")     # first final fails
            return 70.0, "fit", [], {}
        def has_dual(self):
            return False

    class _G:
        is_ghost = False
        ghost_score = 0.0
        flags_json = None
        flags = []

    monkeypatch.setattr("app.matching.filters.score_ghost", lambda job, session: _G())
    ctx = sl._Ctx("resume", _RK(), True, 35)

    assert sl._score_job(jid, ctx) is None           # attempt 1: final fails
    assert jid in sl._prescore_memo                  # Tier-1 result kept
    out = sl._score_job(jid, ctx)                    # attempt 2: retry
    assert out is not None and out[0] == "scored"
    assert calls["pre"] == 1                         # prescore paid ONCE, not twice
    assert calls["final"] == 2
    assert jid not in sl._prescore_memo              # cleaned up after success


# ── OpenAI prescore prefix size ───────────────────────────────────────────────
def test_prescore_prompt_resume_slice_crosses_openai_cache_minimum():
    resume = "x" * 10000
    prompt = rr._build_prescore_prompt(resume, _job())
    # résumé slice must be ≥4000 chars (~1000 tokens): with the ~200-token system
    # prompt that puts the static prefix past OpenAI's 1,024-token minimum.
    assert "x" * 4000 in prompt
    assert "x" * 4001 not in prompt                  # still bounded (cheap tier)
    assert prompt.index("<resume>") == 0             # résumé leads → static prefix


# ── Hourly smoothing cap ──────────────────────────────────────────────────────
def test_hourly_cap_blocks_burst(monkeypatch):
    monkeypatch.setattr(settings, "llm_daily_final_cap", 1000)
    monkeypatch.setattr(settings, "llm_hourly_final_cap", 2)
    rk = _reranker_with(True, False)
    monkeypatch.setattr(rk, "_pre_filter_job", lambda job: None)
    monkeypatch.setattr(rk, "_score_anthropic", lambda rb, jb:
                        '{"score": 80, "reason": "ok", "concerns": [], "breakdown": {}}')
    assert rk.score("resume", _job())[0] == 80.0
    assert rk.score("resume", _job())[0] == 80.0
    assert rr.llm_budget_exhausted()                     # hourly cap hit, daily far away
    with pytest.raises(RuntimeError, match="budget"):
        rk.score("resume", _job())


def test_hourly_cap_resets_next_hour(monkeypatch):
    monkeypatch.setattr(settings, "llm_hourly_final_cap", 1)
    rr._hourly_finals["hour"] = "1999-01-01 00"          # stale hour
    rr._hourly_finals["count"] = 999
    assert not rr.llm_budget_exhausted()


def test_hourly_cap_disabled_when_zero(monkeypatch):
    monkeypatch.setattr(settings, "llm_hourly_final_cap", 0)
    monkeypatch.setattr(settings, "llm_daily_final_cap", 0)
    rr._hourly_finals["hour"] = datetime.utcnow().strftime("%Y-%m-%d %H")
    rr._hourly_finals["count"] = 10**9
    assert not rr.llm_budget_exhausted()


# ── Daily-quota 429 trips the breaker ─────────────────────────────────────────
_OPENAI_RPD_MSG = ("Error code: 429 - {'error': {'message': 'Rate limit reached for "
                   "gpt-4o-mini in organization org-X on requests per day (RPD): "
                   "Limit 10000, Used 10000, Requested 1.', 'type': 'requests', "
                   "'code': 'rate_limit_exceeded'}}")


def test_daily_quota_429_trips_breaker_in_score(monkeypatch):
    rk = _reranker_with(True, True)
    monkeypatch.setattr(rk, "_pre_filter_job", lambda job: None)
    monkeypatch.setattr(rk, "_score_openai", lambda rb, jb: (_ for _ in ()).throw(
        RuntimeError(_OPENAI_RPD_MSG)))
    monkeypatch.setattr(settings, "dual_score_enabled", False)
    monkeypatch.setattr(rk, "_score_anthropic", lambda rb, jb:
                        '{"score": 66, "reason": "ok", "concerns": [], "breakdown": {}}')
    score, *_ = rk.score("resume", _job(), provider="openai")
    assert score == 66.0                                 # Claude picked it up
    assert not rr.provider_available("openai")           # RPD exhaustion = breaker trip


def test_daily_quota_429_trips_breaker_in_prescore(monkeypatch):
    rk = _reranker_with(anthropic=False, openai=True)
    monkeypatch.setattr(rk, "_pre_filter_job", lambda job: None)
    monkeypatch.setattr(rk, "_prescore_openai", lambda prompt: (_ for _ in ()).throw(
        RuntimeError(_OPENAI_RPD_MSG)))
    assert rk.prescore("resume", _job()) is None
    assert not rr.provider_available("openai")


def test_transient_429_does_not_trip_breaker(monkeypatch):
    rk = _reranker_with(True, True)
    monkeypatch.setattr(rk, "_pre_filter_job", lambda job: None)
    monkeypatch.setattr(rk, "_score_anthropic", lambda rb, jb: (_ for _ in ()).throw(
        RuntimeError("Error code: 429 - rate_limit_error: too many requests, retry shortly")))
    monkeypatch.setattr(rk, "_score_openai", lambda rb, jb:
                        '{"score": 55, "reason": "ok", "concerns": [], "breakdown": {}}')
    score, *_ = rk.score("resume", _job(), provider="anthropic")
    assert score == 55.0
    assert rr.provider_available("anthropic")            # per-minute 429 = transient, no trip


# ── Anthropic prescores draw from the same budget as finals ───────────────────
def test_anthropic_prescore_counts_against_budget(monkeypatch):
    monkeypatch.setattr(settings, "llm_daily_final_cap", 2)
    monkeypatch.setattr(settings, "llm_hourly_final_cap", 0)
    rk = _reranker_with(anthropic=True, openai=False)
    monkeypatch.setattr(rk, "_pre_filter_job", lambda job: None)

    class _Msgs:
        @staticmethod
        def create(**kw):
            r = type("R", (), {})()
            r.content = [type("C", (), {"text": '{"score": 20, "reason": "off-role"}'})()]
            return r
    rk._anthropic_client = type("F", (), {"messages": _Msgs()})()

    assert rk.prescore("resume", _job())[0] == 20.0      # 1st Haiku prescore
    assert rk.prescore("resume", _job())[0] == 20.0      # 2nd — budget now full
    assert rr.llm_budget_exhausted()                     # prescores consumed it
    assert rk.prescore("resume", _job()) is None         # 3rd: skipped, no API call


def test_openai_prescore_does_not_touch_budget(monkeypatch):
    monkeypatch.setattr(settings, "llm_daily_final_cap", 1)
    rk = _reranker_with(anthropic=False, openai=True)
    monkeypatch.setattr(rk, "_pre_filter_job", lambda job: None)
    monkeypatch.setattr(rk, "_prescore_openai",
                        lambda prompt: '{"score": 25, "reason": "off-role"}')
    for _ in range(5):                                    # mini is pennies — never budgeted
        assert rk.prescore("resume", _job())[0] == 25.0
    assert not rr.llm_budget_exhausted()
