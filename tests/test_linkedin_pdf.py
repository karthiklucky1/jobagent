"""Tests for the LinkedIn 'Save to PDF' import endpoint.

The user exports their OWN profile via LinkedIn's Profile → More → Save to PDF,
then uploads it. The endpoint extracts the text and stores it via the same legal
path as the paste/extension import (UserPersonalMemory, source="linkedin").
"""
from __future__ import annotations

import pytest
from sqlmodel import select

from app.db.init_db import get_session
from app.db.models import UserPersonalMemory

# ASCII-only marker: a synthetic test PDF can't round-trip fancy Unicode (em dash
# etc.) through its font, though real LinkedIn PDF exports encode Unicode fine.
_MARKER = "HirePath LinkedIn PDF import test - Senior ML Engineer"


def _client():
    from fastapi.testclient import TestClient
    from app.api.server import app
    return TestClient(app)


def _cleanup():
    with get_session() as s:
        for row in s.exec(
            select(UserPersonalMemory).where(UserPersonalMemory.source == "linkedin")
        ).all():
            if _MARKER in (row.raw_content or ""):
                s.delete(row)
        s.commit()


def _pypdf_ready() -> bool:
    """pypdf's crypto backend can be broken in minimal envs — check a real import."""
    try:
        from pypdf import PdfReader  # noqa: F401
        return True
    except Exception:
        return False


def _make_pdf(text: str) -> bytes:
    """Build a minimal one-page PDF containing `text` (no external deps)."""
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
    ]
    stream = b"BT /F1 18 Tf 72 700 Td (" + text.encode() + b") Tj ET"
    objs.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    pdf = b"%PDF-1.4\n"
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(pdf))
        pdf += str(i).encode() + b" 0 obj\n" + o + b"\nendobj\n"
    xref_pos = len(pdf)
    pdf += b"xref\n0 " + str(len(objs) + 1).encode() + b"\n0000000000 65535 f \n"
    for off in offsets:
        pdf += ("%010d 00000 n \n" % off).encode()
    pdf += (b"trailer\n<< /Size " + str(len(objs) + 1).encode()
            + b" /Root 1 0 R >>\nstartxref\n" + str(xref_pos).encode() + b"\n%%EOF")
    return pdf


@pytest.mark.skipif(not _pypdf_ready(), reason="pypdf not available")
def test_linkedin_pdf_import_stores_memory():
    _cleanup()
    try:
        pdf = _make_pdf(_MARKER + " Python FastAPI RAG pipelines")
        files = {"file": ("profile.pdf", pdf, "application/pdf")}
        r = _client().post("/api/profile/memory/linkedin/pdf", files=files)
        assert r.status_code == 200, r.text
        body = r.json()
        assert "id" in body
        with get_session() as s:
            row = s.get(UserPersonalMemory, body["id"])
            assert row is not None
            assert row.source == "linkedin"
            assert _MARKER in row.raw_content
    finally:
        _cleanup()


def test_linkedin_txt_import_stores_memory():
    """The endpoint also accepts a plain-text export (same ingest path, no pypdf)."""
    _cleanup()
    try:
        body = (_MARKER + "\nSkills: Python, PyTorch, FastAPI.").encode()
        files = {"file": ("profile.txt", body, "text/plain")}
        r = _client().post("/api/profile/memory/linkedin/pdf", files=files)
        assert r.status_code == 200, r.text
        with get_session() as s:
            rows = [
                row for row in s.exec(
                    select(UserPersonalMemory).where(UserPersonalMemory.source == "linkedin")
                ).all()
                if _MARKER in (row.raw_content or "")
            ]
            assert rows, "expected a stored linkedin memory row"
    finally:
        _cleanup()


def test_linkedin_pdf_rejects_bad_extension():
    files = {"file": ("resume.exe", b"MZ\x90\x00 not a document", "application/octet-stream")}
    r = _client().post("/api/profile/memory/linkedin/pdf", files=files)
    assert r.status_code == 400


def test_linkedin_pdf_rejects_empty_file():
    files = {"file": ("profile.txt", b"", "text/plain")}
    r = _client().post("/api/profile/memory/linkedin/pdf", files=files)
    assert r.status_code == 400
