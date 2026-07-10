"""BambooHR (widget+detail) and join.com (2-step) scrapers, mocked HTTP."""
from __future__ import annotations

import pytest


class FakeResp:
    def __init__(self, status=200, json_data=None, text=""):
        self.status_code = status
        self._json = json_data
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("err", request=None, response=None)

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class FakeClient:
    def __init__(self, routes):
        self.routes = routes

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, **kw):
        for frag, resp in self.routes.items():
            if frag in url:
                return resp
        return FakeResp(status=404)


WIDGET_HTML = """
<ul>
  <li id="bhrPositionID_101" class="BambooHR-ATS-Jobs-Item">
    <a href="//acme.bamboohr.com/careers/101">Senior Backend Engineer</a>
    <span class="BambooHR-ATS-Location">Austin, TX</span>
  </li>
</ul>
"""


def test_bamboohr(monkeypatch):
    import app.discovery.bamboohr as m
    routes = {
        "jobs/embed2.php": FakeResp(text=WIDGET_HTML),
        "careers/101/detail": FakeResp(json_data={
            "description": "<p>Python, Kafka</p>",
            "datePosted": "2026-07-10",
            "location": {"city": "Austin", "state": "TX", "country": "USA"},
        }),
    }
    # widget GET uses httpx.get; detail uses httpx.Client
    monkeypatch.setattr(m.httpx, "get", lambda *a, **k: routes["jobs/embed2.php"])
    monkeypatch.setattr(m.httpx, "Client", lambda *a, **k: FakeClient(routes))
    jobs = m.BambooHRScraper("acme").fetch()
    assert len(jobs) == 1
    j = jobs[0]
    assert j.title == "Senior Backend Engineer" and j.external_id == "101"
    assert j.source == "bamboohr" and "Kafka" in j.description
    assert j.posted_at is not None and "Austin" in j.location


def test_join(monkeypatch):
    import app.discovery.join as m
    company_html = '<script>{"company":{"id":8899,"domain":"acme","name":"Acme"}}</script>'
    routes = {
        "/companies/acme": FakeResp(text=company_html),  # resolve id
        "/api/public/companies/8899/jobs": FakeResp(json_data={
            "items": [{
                "id": "j1", "name": "Product Manager",
                "location": {"cityName": "Berlin", "countryName": "Germany"},
                "publishedAt": "2026-07-09T00:00:00Z",
                "description": "Roadmaps", "jobLocationType": "onsite",
                "url": "https://join.com/companies/acme/jobs/j1",
            }],
            "pagination": {"totalPages": 1},
        }),
    }
    monkeypatch.setattr(m.httpx, "Client", lambda *a, **k: FakeClient(routes))
    jobs = m.JoinScraper("acme").fetch()
    assert len(jobs) == 1
    j = jobs[0]
    assert j.title == "Product Manager" and j.source == "join"
    assert "Berlin" in j.location and j.posted_at is not None


def test_join_placeholder_company_rejected(monkeypatch):
    import app.discovery.join as m
    # Empty tenant: embedded company domain != requested slug → no id → 0 jobs
    routes = {"/companies/newco": FakeResp(text='<script>{"company":{"id":233,"domain":"greenteg"}}</script>')}
    monkeypatch.setattr(m.httpx, "Client", lambda *a, **k: FakeClient(routes))
    assert m.JoinScraper("newco").fetch() == []


def test_scraper_for_wires_bamboo_join():
    from app.discovery.pipeline import scraper_for
    from app.db.models import JobSource
    assert scraper_for(JobSource.BAMBOOHR, "acme") is not None
    assert scraper_for(JobSource.JOIN, "acme") is not None


def test_public_freshness_endpoint():
    from fastapi.testclient import TestClient
    import app.api.server as srv
    srv.__dict__.pop("_PUBLIC_FRESHNESS_CACHE", None)
    client = TestClient(srv.app)
    r = client.get("/api/public/freshness")
    assert r.status_code == 200
    d = r.json()
    for k in ("active_boards", "jobs_tracked_7d", "median_post_to_alert_min"):
        assert k in d
