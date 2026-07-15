"""Onboarding seed: instant adoption + a targeted domain scrape when the shared
pool doesn't cover a new user's field (the "mechanical engineer got 1 job" fix).

The active-discovery branch imports helpers from app.api.server, which pulls the
whole ML stack and can't import in this test env — so we inject a light fake
``app.api.server`` module and assert seed_new_user's branching, not the scrape.
"""
from __future__ import annotations

import sys
import types

import app.strategy.adoption as ad


def _fake_server(monkeypatch, *, roles, has_resume, sink):
    m = types.ModuleType("app.api.server")
    m._get_target_roles = lambda uid: roles
    m._user_has_resume = lambda uid: has_resume
    m._discover_then_match = lambda uid: sink.append(uid)
    monkeypatch.setitem(sys.modules, "app.api.server", m)


def _base(monkeypatch, *, adopted, pool_count):
    monkeypatch.setattr(ad, "adopt_and_match", lambda uid: adopted)
    monkeypatch.setattr(ad, "_user_pool_count", lambda uid: pool_count)
    monkeypatch.setattr(ad.settings, "onboarding_active_discovery", True)
    monkeypatch.setattr(ad.settings, "onboarding_min_jobs", 25)


def test_thin_pool_triggers_active_discovery(monkeypatch):
    """Mechanical user: adoption copies almost nothing → scrape their roles."""
    sink: list = []
    _base(monkeypatch, adopted=2, pool_count=3)
    _fake_server(monkeypatch, roles=["mechanical engineer"], has_resume=True, sink=sink)

    adopted = ad.seed_new_user("umech")
    assert adopted == 2               # returns the instant-feed count
    assert sink == ["umech"]          # domain scrape fired


def test_full_pool_skips_active_discovery(monkeypatch):
    """AI/ML user: shared pool already covers them → no extra scrape (cost-safe)."""
    sink: list = []
    _base(monkeypatch, adopted=120, pool_count=120)
    _fake_server(monkeypatch, roles=["ml engineer"], has_resume=True, sink=sink)

    adopted = ad.seed_new_user("uai")
    assert adopted == 120
    assert sink == []


def test_no_roles_skips_active_discovery(monkeypatch):
    """No roles → nothing to search for, so we never scrape."""
    sink: list = []
    _base(monkeypatch, adopted=0, pool_count=0)
    _fake_server(monkeypatch, roles=[], has_resume=True, sink=sink)

    ad.seed_new_user("unoroles")
    assert sink == []


def test_no_resume_skips_active_discovery(monkeypatch):
    """No résumé → matching would only surface noise, so we never scrape."""
    sink: list = []
    _base(monkeypatch, adopted=0, pool_count=0)
    _fake_server(monkeypatch, roles=["nurse"], has_resume=False, sink=sink)

    ad.seed_new_user("unoresume")
    assert sink == []


def test_disabled_flag_skips_active_discovery(monkeypatch):
    """Kill switch: onboarding_active_discovery=False → adoption only."""
    sink: list = []
    monkeypatch.setattr(ad, "adopt_and_match", lambda uid: 1)
    monkeypatch.setattr(ad, "_user_pool_count", lambda uid: 0)
    monkeypatch.setattr(ad.settings, "onboarding_active_discovery", False)
    _fake_server(monkeypatch, roles=["mechanical engineer"], has_resume=True, sink=sink)

    adopted = ad.seed_new_user("ux")
    assert adopted == 1
    assert sink == []


def test_zero_threshold_disables_scrape(monkeypatch):
    """onboarding_min_jobs=0 also disables the scrape."""
    sink: list = []
    monkeypatch.setattr(ad, "adopt_and_match", lambda uid: 0)
    monkeypatch.setattr(ad, "_user_pool_count", lambda uid: 0)
    monkeypatch.setattr(ad.settings, "onboarding_active_discovery", True)
    monkeypatch.setattr(ad.settings, "onboarding_min_jobs", 0)
    _fake_server(monkeypatch, roles=["mechanical engineer"], has_resume=True, sink=sink)

    ad.seed_new_user("uz")
    assert sink == []
