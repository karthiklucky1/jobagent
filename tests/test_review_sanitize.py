"""Regression: stored-XSS in user reviews.

Review content was stored raw and interpolated into innerHTML on the public
landing page and the admin dashboard, so a review body like
<img src=x onerror=...> executed in every visitor's (and the admin's) browser.
_sanitize_review_text strips markup at the source so no HTML element can form.
"""
from app.api.server import _sanitize_review_text, _REVIEW_MAX_CHARS


def test_strips_img_onerror_payload():
    out = _sanitize_review_text('<img src=x onerror="fetch(evil)">Great tool')
    assert "<" not in out and ">" not in out
    assert "Great tool" in out


def test_strips_script_tag():
    out = _sanitize_review_text("<script>alert(1)</script>Loved it")
    assert "<" not in out and ">" not in out
    assert "Loved it" in out


def test_neutralizes_unclosed_bracket():
    out = _sanitize_review_text("Unclosed <img src=x onerror=1 and more")
    assert "<" not in out and ">" not in out


def test_plain_text_unchanged():
    assert _sanitize_review_text("  Best app ever  ") == "Best app ever"


def test_length_capped():
    assert len(_sanitize_review_text("a" * 5000)) == _REVIEW_MAX_CHARS
