"""SerpAPI fetch: concurrent keyword searches (not sequential → 45s timeout)
and a per-run keyword cap so quota isn't burned on the shared-run role union."""
from __future__ import annotations

import asyncio

import pytest

import app.discovery.sources.serpapi as sp
from app.config import settings


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = ""

    def json(self):
        return self._payload


class _FakeClient:
    """Records concurrency: how many requests were in flight at once."""
    def __init__(self, per_kw_jobs=1, delay=0.05, tracker=None):
        self.per_kw_jobs = per_kw_jobs
        self.delay = delay
        self.t = tracker

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, params=None):
        if self.t is not None:
            self.t["cur"] += 1
            self.t["max"] = max(self.t["max"], self.t["cur"])
            self.t["calls"] += 1
        await asyncio.sleep(self.delay)
        kw = (params or {}).get("q", "x")
        jobs = [{
            "title": f"ML Engineer {i}", "company_name": f"Co{i}",
            "location": "Remote", "job_id": f"{kw}-{i}",
            "description": "d", "detected_extensions": {"posted_at": "2 days ago"},
            "apply_options": [{"link": "https://boards.greenhouse.io/acme/jobs/1"}],
        } for i in range(self.per_kw_jobs)]
        if self.t is not None:
            self.t["cur"] -= 1
        return _Resp(200, {"jobs_results": jobs})


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_searches_run_concurrently(monkeypatch):
    monkeypatch.setattr(settings, "serpapi_key", "test-key")
    tracker = {"cur": 0, "max": 0, "calls": 0}
    monkeypatch.setattr(sp.httpx, "AsyncClient",
                        lambda *a, **k: _FakeClient(tracker=tracker))

    src = sp.SerpAPISource(keywords=["a", "b", "c", "d", "e"], country="United States")
    jobs = _run(src.fetch_jobs())
    assert tracker["calls"] == 5           # one search per keyword
    assert tracker["max"] >= 2             # ran in parallel, not one-at-a-time
    assert len(jobs) == 5                  # one job parsed per keyword


def test_keyword_cap_limits_quota(monkeypatch):
    monkeypatch.setattr(settings, "serpapi_key", "test-key")
    monkeypatch.setattr(settings, "serpapi_max_keywords", 3)
    tracker = {"cur": 0, "max": 0, "calls": 0}
    monkeypatch.setattr(sp.httpx, "AsyncClient",
                        lambda *a, **k: _FakeClient(tracker=tracker))

    # 10 roles (e.g. the union of all users') → only 3 searches fired.
    src = sp.SerpAPISource(keywords=[f"role{i}" for i in range(10)])
    _run(src.fetch_jobs())
    assert tracker["calls"] == 3


def test_no_key_skips(monkeypatch):
    monkeypatch.setattr(settings, "serpapi_key", "")
    src = sp.SerpAPISource(keywords=["a"])
    assert _run(src.fetch_jobs()) == []
