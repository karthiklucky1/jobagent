"""Tests for the extension LinkedIn profile import endpoint.

The browser extension reads the user's OWN LinkedIn profile text (manual click,
own logged-in tab) and POSTs it to /api/profile/memory/linkedin — the same legal
"paste" path the dashboard uses. These tests verify the backend stores that text
as a UserPersonalMemory row and rejects empty input.
"""
from __future__ import annotations

from sqlmodel import select

from app.db.init_db import get_session
from app.db.models import UserPersonalMemory

# Unique marker so cleanup only touches rows this test created.
_MARKER = "HirePath LinkedIn import test — Senior ML Engineer at Acme"


def _client():
    from fastapi.testclient import TestClient
    from app.api.server import app
    return TestClient(app)


def _cleanup():
    with get_session() as s:
        rows = s.exec(
            select(UserPersonalMemory).where(UserPersonalMemory.source == "linkedin")
        ).all()
        for row in rows:
            if _MARKER in (row.raw_content or ""):
                s.delete(row)
        s.commit()


def test_linkedin_import_stores_memory():
    """A posted profile text is saved as a linkedin-source UserPersonalMemory row."""
    _cleanup()
    try:
        text = (
            _MARKER + "\n"
            "Skills: Python, PyTorch, FastAPI, RAG pipelines, vector search.\n"
            "About: 4 years building production LLM systems."
        )
        r = _client().post("/api/profile/memory/linkedin", json={"text": text})
        assert r.status_code == 200, r.text

        body = r.json()
        assert "id" in body
        assert "recommendations" in body

        with get_session() as s:
            row = s.get(UserPersonalMemory, body["id"])
            assert row is not None
            assert row.source == "linkedin"
            assert _MARKER in row.raw_content
    finally:
        _cleanup()


def test_linkedin_import_rejects_empty_text():
    """Blank profile text is rejected with a 400 (nothing to import)."""
    r = _client().post("/api/profile/memory/linkedin", json={"text": "   "})
    assert r.status_code == 400
