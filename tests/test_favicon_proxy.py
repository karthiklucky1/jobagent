"""Favicon proxy: icon hits stream through, misses 204 (no console noise)."""
from __future__ import annotations

import pytest


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 120  # >90 bytes, image-shaped


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from app.api.server import app
    return TestClient(app)


class FakeResp:
    def __init__(self, status_code=200, content=b"", ctype="image/png"):
        self.status_code = status_code
        self.content = content
        self.headers = {"content-type": ctype}


def test_favicon_hit_and_cache(client, monkeypatch):
    import app.api.server as srv
    srv._FAVICON_CACHE.clear()
    calls = {"n": 0}

    def fake_get(url, **kw):
        calls["n"] += 1
        return FakeResp(200, PNG)

    import httpx
    monkeypatch.setattr(httpx, "get", fake_get)
    r = client.get("/api/favicon", params={"domain": "github.com"})
    assert r.status_code == 200
    assert r.content == PNG
    assert r.headers["cache-control"].startswith("public")
    # Second call served from cache — no new upstream fetch
    r2 = client.get("/api/favicon", params={"domain": "github.com"})
    assert r2.status_code == 200 and calls["n"] == 1


def test_favicon_miss_is_silent_204(client, monkeypatch):
    import app.api.server as srv
    srv._FAVICON_CACHE.clear()
    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: FakeResp(404, b"Not Found", "text/html"))
    r = client.get("/api/favicon", params={"domain": "nofavicon-here.com"})
    assert r.status_code == 204
    assert r.content == b""


def test_favicon_rejects_garbage_domain(client):
    for bad in ("<script>", "a", "", "foo..bar", "x" * 200):
        r = client.get("/api/favicon", params={"domain": bad})
        assert r.status_code == 204, bad
