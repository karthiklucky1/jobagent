"""Regression: altered employer names / employment dates must HARD-FAIL the
Doctor, not merely lose <=10 points and ship as TAILORED (app/tailoring/doctor.py).
"""
import pytest

from app.tailoring.doctor import ResumeDoctor

MASTER = """# JOHN DOE
Austin, TX | john@example.com

## EXPERIENCE
**Backend Engineer** | Acme Corp | Jun 2019 - Mar 2021 | Austin, TX
- Optimized queries, cutting API latency 43% for 2,500 users.
- Built ingestion handling 10,000 records per hour.
- Led a team of 4 engineers delivering 30% faster releases.

## EDUCATION
**Bachelor of Science** | 2019
University of Texas
"""

JD = "Backend engineer with API, database, and pipeline experience."


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    # Doctor only calls the LLM verdict on PASS; stub it so a passing case
    # doesn't hit the network in tests.
    monkeypatch.setattr(ResumeDoctor, "_llm_verdict", lambda self, *a, **k: None)


def test_faithful_tailor_can_pass():
    doc = ResumeDoctor()
    report = doc.check(MASTER, MASTER, JD)
    assert not report.integrity_issues
    assert report.passed is True


def test_altered_employer_and_dates_hard_fail():
    tailored = MASTER.replace("Acme Corp | Jun 2019 - Mar 2021", "Acme | 2018 - 2021")
    doc = ResumeDoctor()
    report = doc.check(tailored, MASTER, JD)
    assert report.integrity_issues            # employer + date anchors missing
    assert report.passed is False             # must NOT ship as TAILORED
