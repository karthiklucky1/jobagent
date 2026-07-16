"""Regressions for two scoring-lane bugs:

1. Transient budget/cooldown failures were counted as per-job scoring failures,
   deferring perfectly-scorable fresh jobs for hours (scoring_lane.py).
2. Attempt-ceiling deferrals were filtered AFTER the SQL LIMIT, so a window of
   deferred fresh jobs could starve valid older jobs (now excluded in-query via
   _deferred_ids()).
"""
import time

import app.strategy.scoring_lane as sl


def _reset():
    sl._fail_counts.clear()
    sl._deferred_until.clear()


def test_deferred_ids_purges_expired_and_returns_set():
    _reset()
    sl._deferred_until[1] = time.time() + 3600   # live
    sl._deferred_until[2] = time.time() - 1       # expired
    ids = sl._deferred_ids()
    assert ids == {1}
    assert 2 not in sl._deferred_until            # expired entry purged


def test_transient_stall_true_when_budget_exhausted(monkeypatch):
    import app.matching.reranker as rr
    monkeypatch.setattr(rr, "llm_budget_exhausted", lambda: True)
    monkeypatch.setattr(rr, "any_provider_available", lambda: True)
    assert sl._transient_llm_stall() is True


def test_transient_stall_true_when_all_providers_cooling(monkeypatch):
    import app.matching.reranker as rr
    monkeypatch.setattr(rr, "llm_budget_exhausted", lambda: False)
    monkeypatch.setattr(rr, "any_provider_available", lambda: False)
    assert sl._transient_llm_stall() is True


def test_transient_stall_false_when_healthy(monkeypatch):
    import app.matching.reranker as rr
    monkeypatch.setattr(rr, "llm_budget_exhausted", lambda: False)
    monkeypatch.setattr(rr, "any_provider_available", lambda: True)
    assert sl._transient_llm_stall() is False
