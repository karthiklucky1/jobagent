"""Unit tests for pipeline door filter integration."""
import pytest
from unittest.mock import patch, MagicMock
from sqlmodel import select
from app.db.init_db import get_session
from app.db.models import Job, JobSource, Application, ApplicationStatus, UserProfile
from app.matching.pipeline import run_matching


def _seed():
    with get_session() as s:
        # Clean old test runs
        for j in s.exec(select(Job).where(Job.external_id.like("pipe-test-%"))).all():
            for a in s.exec(select(Application).where(Application.job_id == j.id)).all():
                s.delete(a)
            s.delete(j)
        # Create user profile if none exists
        prof = s.exec(select(UserProfile).where(UserProfile.user_id == "local")).first()
        if not prof:
            prof = UserProfile(user_id="local", first_name="Karthik", remote_ok=True, open_to_relocation=False, location="Cincinnati, OH")
            s.add(prof)
        else:
            prof.remote_ok = True
            prof.open_to_relocation = False
            prof.location = "Cincinnati, OH"
            s.add(prof)
        
        # This job is strictly on-site in Miami, FL (mismatch since candidate is remote-only in Cincinnati, OH)
        j = Job(source=JobSource.MANUAL, external_id="pipe-test-1",
                company="Miami Corp", title="AI Engineer",
                url="http://x", description="On-site Miami, FL. 3+ years experience.")
        s.add(j); s.commit(); s.refresh(j)
        return j.id


def _cleanup():
    with get_session() as s:
        for j in s.exec(select(Job).where(Job.external_id.like("pipe-test-%"))).all():
            for a in s.exec(select(Application).where(Application.job_id == j.id)).all():
                s.delete(a)
            s.delete(j)
        s.commit()


def test_pipeline_door_filter_skips_wrong_door():
    job_id = _seed()
    try:
        # Mock Matcher.search_for_resume to return our seeded job with a similarity score of 0.8
        with patch("app.matching.matcher.Matcher.search_for_resume", return_value=[(job_id, 0.8)]), \
             patch("app.matching.matcher.Matcher.rebuild") as mock_rebuild, \
             patch("app.matching.pipeline._load_resume", return_value="Resume text"):
             
            shortlisted = run_matching(user_id="local")
            
            # Since the job is strictly on-site in Miami, FL, it should be filtered out by the door match check
            # (Wrong Door) and not shortlisted!
            assert job_id not in shortlisted
            
            # Check that the job's score in the DB is set to 20.0 and reasoning contains "Wrong Door"
            with get_session() as s:
                job = s.get(Job, job_id)
                assert job.rerank_score == 20.0
                assert "Wrong Door" in job.rerank_reasoning
    finally:
        _cleanup()
