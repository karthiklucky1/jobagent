"""Tests for the autofill post-fill verification loop."""
from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub heavy ML deps so importing the agent doesn't require torch/faiss.
# The agent pulls in matcher (sentence_transformers) transitively; the
# verification helpers under test don't touch any of it.
for _name in ("sentence_transformers", "faiss", "rank_bm25"):
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        if _name == "sentence_transformers":
            _m.SentenceTransformer = object
            _m.CrossEncoder = object
        if _name == "rank_bm25":
            _m.BM25Okapi = object
        sys.modules[_name] = _m

from app.autofill.agent import (
    _values_match,
    _verify_filled_fields,
    _read_field_value,
    VerificationReport,
    FieldVerification,
)


class TestValuesMatch:
    def test_email_exact(self):
        assert _values_match("email", "a@b.com", "A@B.com")
        assert not _values_match("email", "a@b.com", "x@b.com")

    def test_phone_last_ten_digits(self):
        # country code differences tolerated, last 10 digits must match
        assert _values_match("phone", "513-555-1234", "+1 (513) 555-1234")
        assert not _values_match("phone", "513-555-1234", "513-555-9999")

    def test_phone_too_short_fails(self):
        assert not _values_match("phone", "5135551234", "555")

    def test_name_substring(self):
        assert _values_match("first_name", "Karthik", "Karthik ")
        assert not _values_match("first_name", "Karthik", "John")

    def test_empty_expected_is_ok(self):
        assert _values_match("first_name", "", "anything")

    def test_empty_actual_fails(self):
        assert not _values_match("first_name", "Karthik", "")


class TestVerificationReport:
    def test_all_ok(self):
        r = VerificationReport(checks=[
            FieldVerification("email", "a@b.com", "a@b.com", True),
            FieldVerification("first_name", "Kar", "Kar", True),
        ])
        assert r.all_ok
        assert r.mismatches == []

    def test_mismatches(self):
        r = VerificationReport(checks=[
            FieldVerification("email", "a@b.com", "", False),
            FieldVerification("first_name", "Kar", "Kar", True),
        ])
        assert not r.all_ok
        assert len(r.mismatches) == 1
        assert r.mismatches[0].field == "email"

    def test_summary_no_fields(self):
        assert "no verifiable fields" in VerificationReport(checks=[]).summary()


def _fake_input(value: str, visible: bool = True):
    el = MagicMock()
    el.is_visible = AsyncMock(return_value=visible)
    el.input_value = AsyncMock(return_value=value)
    el.focus = AsyncMock()
    el.fill = AsyncMock()
    el.type = AsyncMock()
    el.evaluate = AsyncMock()
    return el


def _fake_target(value_by_selector: dict):
    """A fake page/frame whose query_selector returns inputs by selector match."""
    target = MagicMock()

    async def _qs(sel):
        return value_by_selector.get(sel)

    target.query_selector = AsyncMock(side_effect=_qs)
    return target


class TestVerifyFilledFields:
    def test_all_fields_match(self):
        expected = {"first_name": "Karthik", "email": "k@x.com"}
        target = _fake_target({
            "input[name='first_name']": _fake_input("Karthik"),
            "input[type='email']": _fake_input("k@x.com"),
        })
        report = asyncio.run(_verify_filled_fields(target, expected, retry_fill=False))
        assert report.all_ok
        # only fields with both an expected value and a present input are checked
        assert {c.field for c in report.checks} == {"first_name", "email"}

    def test_absent_field_not_penalized(self):
        # phone expected but no phone input on the form → not a failure
        expected = {"first_name": "Karthik", "phone": "5135551234"}
        target = _fake_target({
            "input[name='first_name']": _fake_input("Karthik"),
        })
        report = asyncio.run(_verify_filled_fields(target, expected, retry_fill=False))
        assert report.all_ok
        assert {c.field for c in report.checks} == {"first_name"}

    def test_mismatch_detected_without_retry(self):
        expected = {"email": "k@x.com"}
        target = _fake_target({"input[type='email']": _fake_input("wrong@x.com")})
        report = asyncio.run(_verify_filled_fields(target, expected, retry_fill=False))
        assert not report.all_ok
        assert report.mismatches[0].field == "email"

    def test_retry_refills_and_passes(self):
        # First read returns empty, after re-fill the input reports the correct value.
        el = MagicMock()
        el.is_visible = AsyncMock(return_value=True)
        el.focus = AsyncMock()
        el.fill = AsyncMock()
        el.type = AsyncMock()
        el.evaluate = AsyncMock()
        # input_value: 1st call empty (initial read), then "Karthik" after refill reads
        el.input_value = AsyncMock(side_effect=["", "Karthik", "Karthik"])
        target = _fake_target({"input[name='first_name']": el})
        report = asyncio.run(_verify_filled_fields(target, {"first_name": "Karthik"}, retry_fill=True))
        assert report.all_ok
        el.type.assert_awaited()  # a re-fill happened
