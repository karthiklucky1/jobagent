"""Tests for the HN 'Who is hiring?' source — parsing + filtering logic."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.discovery.sources.hn_whoishiring import (
    _strip_html,
    _first_url,
    _parse_comment,
    _matches_keywords,
    _is_us_or_remote,
    _looks_like_posting,
    HNWhoIsHiringSource,
)


class TestStripHtml:
    def test_extracts_link_url(self):
        html = 'We are hiring! <a href="https://acme.com/jobs" rel="nofollow">apply here</a>'
        out = _strip_html(html)
        assert "https://acme.com/jobs" in out

    def test_unescapes_entities(self):
        assert "R&D" in _strip_html("R&amp;D team")


class TestFirstUrl:
    def test_finds_url(self):
        assert _first_url("apply at https://x.com/jobs today") == "https://x.com/jobs"

    def test_strips_trailing_punct(self):
        assert _first_url("see https://x.com/jobs.") == "https://x.com/jobs"

    def test_none_when_absent(self):
        assert _first_url("no link here") is None


class TestParseComment:
    def test_pipe_header(self):
        text = "Acme AI | Senior ML Engineer | San Francisco, CA | REMOTE\nWe build LLM tooling.\nhttps://acme.ai/jobs"
        company, role, location, remote = _parse_comment(text)
        assert company == "Acme AI"
        assert role == "Senior ML Engineer"
        assert "San Francisco" in location
        assert remote is True

    def test_dash_header(self):
        text = "Globex - Backend Engineer\nPython shop, fully remote."
        company, role, location, remote = _parse_comment(text)
        assert company == "Globex"
        assert "Backend Engineer" in role
        assert remote is True

    def test_no_structure_fallback(self):
        text = "Looking for engineers to join our onsite team in Austin."
        company, role, location, remote = _parse_comment(text)
        assert company  # never empty
        assert role
        assert remote is False


class TestFilters:
    def test_keyword_match_in_role(self):
        assert _matches_keywords("Machine Learning Engineer", "", ["Machine Learning Engineer"])

    def test_keyword_match_in_text(self):
        assert _matches_keywords("Engineer", "We use Python heavily", ["Python Developer", "Python"])

    def test_keyword_no_match(self):
        assert not _matches_keywords("Chef", "kitchen work", ["AI Engineer"])

    def test_us_or_remote_true(self):
        assert _is_us_or_remote("San Francisco, CA", "")
        assert _is_us_or_remote("", "This is a remote role")

    def test_us_or_remote_false(self):
        assert not _is_us_or_remote("Berlin, Germany", "onsite in Munich only")

    def test_looks_like_posting(self):
        assert _looks_like_posting("Senior Engineer, full-time. Apply: https://x.com")
        assert _looks_like_posting("We are hiring a backend developer")
        assert not _looks_like_posting("Great thread, thanks for organizing this every month!")


class TestFetchJobs:
    def _run(self, hits, thread_item):
        src = HNWhoIsHiringSource(keywords=["ML Engineer", "Python", "Backend"])

        search_resp = MagicMock(status_code=200)
        search_resp.json.return_value = {"hits": hits}
        item_resp = MagicMock(status_code=200)
        item_resp.json.return_value = thread_item

        async def _get(url, params=None):
            return search_resp if "search_by_date" in url else item_resp

        fake_client = MagicMock()
        fake_client.get = AsyncMock(side_effect=_get)
        fake_cm = MagicMock()
        fake_cm.__aenter__ = AsyncMock(return_value=fake_client)
        fake_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("app.discovery.sources.hn_whoishiring.httpx.AsyncClient", return_value=fake_cm):
            return asyncio.run(src.fetch_jobs())

    def test_parses_matching_postings(self):
        hits = [{"objectID": "999", "title": "Ask HN: Who is hiring? (June 2026)"}]
        thread = {
            "children": [
                {
                    "id": 1, "author": "alice", "created_at_i": 1700000000,
                    "text": 'Acme AI | ML Engineer | Remote (US)\nWe build with Python. <a href="https://acme.ai/apply">apply</a>',
                },
                {
                    "id": 2, "author": "bob", "created_at_i": 1700000001,
                    "text": "Berlin Co | Backend Engineer | Berlin, Germany\nOnsite only in Germany.",
                },
                {  # deleted comment
                    "id": 3, "author": None, "text": None,
                },
                {  # non-job chatter
                    "id": 4, "author": "carol", "created_at_i": 1700000002,
                    "text": "Thanks for running this thread!",
                },
            ]
        }
        jobs = self._run(hits, thread)
        # Only the US/remote keyword-matching posting should survive
        assert len(jobs) == 1
        assert jobs[0].company == "Acme AI"
        assert jobs[0].url == "https://acme.ai/apply"
        assert jobs[0].source == "indeed"
        assert jobs[0].remote is True

    def test_no_thread_returns_empty(self):
        hits = [{"objectID": "1", "title": "Some unrelated story"}]
        jobs = self._run(hits, {"children": []})
        assert jobs == []
