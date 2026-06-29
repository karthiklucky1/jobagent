"""Tailoring end-to-end test with mocked LLM.

Verifies that tailor_for_application() completes all phases — reads job+app,
calls tailor + cover letter, writes .docx + .txt, updates app status — without
requiring a real API key.
"""
from __future__ import annotations

import pathlib
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import select

from app.db.init_db import get_session
from app.db.models import Application, ApplicationStatus, Job, JobSource

FAKE_RESUME_MD = """# Karthik Test
## Summary
- Python engineer with 5 years experience
- Builds APIs and data pipelines

## Experience
### Senior Engineer — AcmeCorp (2021-2024)
- Built REST APIs with **FastAPI**, reduced latency by 30%
- Automated CI/CD with **GitHub Actions**

## Skills
Python, FastAPI, PostgreSQL, Docker
"""

FAKE_JD = """We're hiring a Senior Python Engineer to build scalable APIs.
Requirements: Python, FastAPI, PostgreSQL, REST APIs, CI/CD experience.
Nice to have: Kubernetes, Redis.
"""

FAKE_TAILORED_RESUME = """# Karthik Test — Tailored
## Summary
- Senior Python engineer, FastAPI specialist

## Experience
### Senior Engineer — AcmeCorp (2021-2024)
- Built high-scale REST APIs with **FastAPI** serving 50k RPM

## Skills
Python, FastAPI, PostgreSQL, Docker, Kubernetes
"""

FAKE_COVER = """You're building out scalable API infrastructure. At AcmeCorp I built FastAPI services that reduced latency by 30% at 50k RPM. This maps directly to your senior role."""


def _seed_job_and_app(session):
    j = Job(
        source=JobSource.REMOTEOK,
        external_id="tailor-test-001",
        company="TailorCo",
        title="Senior Python Engineer",
        url="https://example.com/job/1",
        description=FAKE_JD,
    )
    session.add(j)
    session.commit()
    session.refresh(j)
    a = Application(
        job_id=j.id,
        status=ApplicationStatus.SHORTLISTED,
        apply_track="manual",
    )
    session.add(a)
    session.commit()
    session.refresh(a)
    return j.id, a.id


def test_tailor_for_application_mocked(tmp_path):
    """Full pipeline: seeds job+app, mocks LLM, runs tailor_for_application, checks outputs."""
    with get_session() as s:
        job_id, app_id = _seed_job_and_app(s)

    try:
        # Write a fake master resume so the function can read it
        resume_path = tmp_path / "resume_master.md"
        resume_path.write_text(FAKE_RESUME_MD, encoding="utf-8")

        # Build a mock Anthropic client that returns fake LLM responses
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=FAKE_TAILORED_RESUME)]
        mock_cover_resp = MagicMock()
        mock_cover_resp.content = [MagicMock(text=FAKE_COVER)]
        mock_client.messages.create.side_effect = [mock_resp, mock_cover_resp]

        with patch("app.tailoring.tailor.settings") as mock_settings:
            mock_settings.anthropic_api_key = "sk-fake-key"
            mock_settings.openai_api_key = ""
            mock_settings.tailoring_model = "claude-sonnet-4-6"
            mock_settings.resume_path = resume_path
            mock_settings.profiles_dir = tmp_path / "profiles"
            mock_settings.data_dir = tmp_path
            mock_settings.jobs_keywords_list = ["python", "engineer"]

            from app.tailoring.tailor import tailor_for_application, Tailor
            from app.tailoring.grounding import GroundingResult
            with patch("app.tailoring.grounding.GroundingChecker") as MockGrounding:
                mock_g = MockGrounding.return_value
                mock_g.check.return_value = GroundingResult(passed=True, flagged_bullets=[], confidence_map={})
                with patch.object(Tailor, "__init__", lambda self: None):
                    # Manually set the internals that __init__ would set
                    with patch("app.tailoring.tailor.Tailor") as MockTailor:
                        instance = MockTailor.return_value
                        instance._active_backend = "anthropic"
                        instance._anthropic_client = mock_client
                        instance._openai_client = None
                        instance.tailor_resume.return_value = FAKE_TAILORED_RESUME
                        instance.write_cover_letter.return_value = FAKE_COVER

                        resume_out, cover_out = tailor_for_application(app_id)

        # Outputs should exist
        assert resume_out.exists(), f"Resume docx not written: {resume_out}"
        assert cover_out.exists(), f"Cover letter not written: {cover_out}"

        # Cover letter should contain the job info header
        cover_text = cover_out.read_text(encoding="utf-8")
        assert "TailorCo" in cover_text
        assert "Senior Python Engineer" in cover_text

        # App status should be TAILORED
        with get_session() as s:
            app = s.get(Application, app_id)
            assert app.status == ApplicationStatus.TAILORED, f"Expected TAILORED, got {app.status}"
            assert app.tailored_resume_path is not None
            assert app.cover_letter_path is not None

    finally:
        # Cleanup
        with get_session() as s:
            for a in s.exec(select(Application).where(Application.job_id == job_id)).all():
                s.delete(a)
            j = s.get(Job, job_id)
            if j:
                s.delete(j)
            s.commit()


def test_tailor_raises_without_llm():
    """Tailor raises RuntimeError when no API keys are configured."""
    from app.tailoring.tailor import Tailor

    with patch("app.tailoring.tailor.settings") as mock_settings:
        mock_settings.anthropic_api_key = ""
        mock_settings.openai_api_key = ""
        t = Tailor()

    with pytest.raises(RuntimeError, match="No LLM backend"):
        mock_job = MagicMock()
        mock_job.title = "Engineer"
        mock_job.company = "Co"
        mock_job.description = "desc"
        t.tailor_resume("# resume", mock_job)
