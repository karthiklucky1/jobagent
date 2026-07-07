"""Unit tests for generate_referral_drafts extensions."""
import pytest
from unittest.mock import patch, MagicMock
from sqlmodel import select
from app.db.init_db import get_session
from app.db.models import Job, JobSource, Application, ApplicationStatus, UserProfile
from app.intelligence.referral import generate_referral_drafts


def _seed():
    with get_session() as s:
        # Clean old test runs
        for j in s.exec(select(Job).where(Job.external_id.like("referral-test-%"))).all():
            for a in s.exec(select(Application).where(Application.job_id == j.id)).all():
                s.delete(a)
            s.delete(j)
        # Create user profile if none exists
        prof = s.exec(select(UserProfile).where(UserProfile.user_id == "local")).first()
        if not prof:
            prof = UserProfile(user_id="local", first_name="Karthik", university="University of Cincinnati")
            s.add(prof)
        else:
            prof.university = "University of Cincinnati"
            s.add(prof)
        
        j = Job(source=JobSource.MANUAL, external_id="referral-test-1",
                company="GitHub", title="Software Engineer",
                url="http://x", description="React, Python")
        s.add(j); s.commit(); s.refresh(j)
        a = Application(job_id=j.id, status=ApplicationStatus.SHORTLISTED, user_id="local")
        s.add(a); s.commit(); s.refresh(a)
        return a.id, j.id


def _cleanup():
    with get_session() as s:
        for j in s.exec(select(Job).where(Job.external_id.like("referral-test-%"))).all():
            for a in s.exec(select(Application).where(Application.job_id == j.id)).all():
                s.delete(a)
            s.delete(j)
        s.commit()


def test_generate_referral_drafts_extensions():
    app_id, _ = _seed()
    try:
        mock_people = [
            {"name": "Alice Smith", "headline": "Alum of University of Cincinnati | Ex-Apple", "url": "https://linkedin.com/in/alice"}
        ]
        mock_github_repos = ["cli", "desktop"]

        with patch("app.intelligence.linkedin_xray.find_champions", return_value={"ok": True, "people": mock_people}), \
             patch("app.intelligence.referral.get_company_github_repos", return_value=mock_github_repos), \
             patch("app.config.settings.anthropic_api_key", "dummy_key"), \
             patch("anthropic.Anthropic") as mock_anthropic:
             
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_message = MagicMock()
            
            mock_message.content = [MagicMock(text='''[
                {"type": "referral_request", "label": "Referral request", "channel": "LinkedIn", "body": "Referral draft"},
                {"type": "hiring_manager", "label": "Hiring-manager note", "channel": "LinkedIn", "body": "Hiring manager draft"},
                {"type": "university_alumni", "label": "University Alumni connection", "channel": "LinkedIn", "body": "Go Cincinnati!"},
                {"type": "github_outreach", "label": "GitHub outreach note", "channel": "LinkedIn", "body": "GitHub draft"}
            ]''')]
            mock_client.messages.create.return_value = mock_message

            res = generate_referral_drafts(app_id, user_id="local")
            
            draft_types = {d["type"] for d in res["drafts"]}
            assert "university_alumni" in draft_types
            assert "github_outreach" in draft_types
            
            univ_draft = next(d for d in res["drafts"] if d["type"] == "university_alumni")
            assert "Cincinnati" in univ_draft["body"]
    finally:
        _cleanup()
