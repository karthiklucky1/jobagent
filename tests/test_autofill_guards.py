"""Autofill guards: profile-fact answers never hit the LLM, submit endpoint
is idempotent and never resurrects terminal applications."""
from datetime import datetime

from sqlmodel import select

from app.autofill.answer_pack import _profile_fact_answer, answer_question
from app.db.init_db import get_session
from app.db.models import Application, ApplicationStatus, Job, JobSource


class _Prof:
    first_name = "Jane"; last_name = "Smith"; email = "jane@x.com"
    phone = "555"; linkedin_url = ""; github_url = ""; portfolio_url = ""
    current_title = "Engineer"; location = "Austin, TX"
    university = "UT"; graduation_year = 2020; years_experience = 4


def test_identity_questions_resolved_from_profile():
    assert _profile_fact_answer("What is your first name?", _Prof) == "Jane"
    assert _profile_fact_answer("Please enter your Full Name", _Prof) == "Jane Smith"
    assert _profile_fact_answer("Phone number", _Prof) == "555"
    # Empty profile field -> "" (surface to user), never None (LLM path)
    assert _profile_fact_answer("LinkedIn profile", _Prof) == ""
    # Genuine essay question -> None (goes to cache/LLM)
    assert _profile_fact_answer("Why are you interested in this role?", _Prof) is None


def _mk_app(status):
    with get_session() as s:
        j = Job(source=JobSource.GREENHOUSE, external_id=f"guard-{status.value}",
                company="GuardCo", title="Role", url="http://x/g", description="d")
        s.add(j); s.commit(); s.refresh(j)
        a = Application(job_id=j.id, status=status, apply_track="manual")
        s.add(a); s.commit(); s.refresh(a)
        return a.id


def _cleanup():
    with get_session() as s:
        for j in s.exec(select(Job).where(Job.external_id.like("guard-%"))).all():
            for a in s.exec(select(Application).where(Application.job_id == j.id)).all():
                s.delete(a)
            s.delete(j)
        s.commit()


def test_submit_endpoint_guards_status():
    from fastapi.testclient import TestClient
    from app.api.server import app as fastapp
    client = TestClient(fastapp)
    try:
        # Pre-submit state -> transitions
        aid = _mk_app(ApplicationStatus.SHORTLISTED)
        r = client.post(f"/application/{aid}/submit")
        assert r.status_code == 200 and r.json()["success"] is True
        with get_session() as s:
            app_obj = s.get(Application, aid)
            assert app_obj.status == ApplicationStatus.SUBMITTED
            first_stamp = app_obj.submitted_at

        # Duplicate submit -> idempotent, submitted_at unchanged
        r = client.post(f"/application/{aid}/submit")
        assert r.json().get("already") is True
        with get_session() as s:
            assert s.get(Application, aid).submitted_at == first_stamp

        # A REJECTED application must never come back as SUBMITTED
        rid = _mk_app(ApplicationStatus.REJECTED)
        r = client.post(f"/application/{rid}/submit")
        assert r.json()["success"] is False
        with get_session() as s:
            assert s.get(Application, rid).status == ApplicationStatus.REJECTED
    finally:
        _cleanup()


def test_answer_question_uses_profile_before_llm(monkeypatch):
    # Even with an application context, an identity question returns the
    # profile value without touching memory or the LLM.
    aid = _mk_app(ApplicationStatus.SHORTLISTED)
    try:
        import app.autofill.answer_pack as ap
        monkeypatch.setattr(ap, "_get_or_create_profile", lambda user_id=None: _Prof)
        monkeypatch.setattr(ap, "_llm_essay_answer",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called for identity Q")))
        assert answer_question("What is your first name?", aid) == "Jane"
    finally:
        _cleanup()
