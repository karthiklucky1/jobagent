"""Per-user role gate in _upsert: a board dump can't flood one user's pool with
off-role postings, while the shared pool and un-gated calls keep everything."""
from sqlmodel import delete, select

from app.db.init_db import get_session
from app.db.models import Job
from app.discovery.base import RawJob
from app.discovery.pipeline import SHARED_POOL_USER, _upsert


def _clean():
    with get_session() as s:
        s.exec(delete(Job))
        s.commit()


def _raw(ext, title, company="Acme"):
    return RawJob(source="greenhouse", external_id=str(ext), company=company,
                  title=title, location="Remote", remote=True,
                  url=f"https://boards.greenhouse.io/acme/jobs/{ext}",
                  description="desc", posted_at=None)


BATCH = [
    _raw(1, "Senior Machine Learning Engineer"),
    _raw(2, "AI/ML Engineer"),
    _raw(3, "Mechanical Engineer III - Battery Systems"),
    _raw(4, "Contact Center Systems Analyst"),
    _raw(5, "Systems Engineer"),
]


def test_role_gate_keeps_on_role_drops_noise():
    _clean()
    _upsert(BATCH, user_id="u_ml", role_gate_terms=["machine learning engineer"])
    with get_session() as s:
        titles = {j.title for j in s.exec(select(Job).where(Job.user_id == "u_ml")).all()}
    assert "Senior Machine Learning Engineer" in titles
    assert "AI/ML Engineer" in titles
    assert "Mechanical Engineer III - Battery Systems" not in titles
    assert "Contact Center Systems Analyst" not in titles
    assert "Systems Engineer" not in titles


def test_no_gate_keeps_everything():
    _clean()
    _upsert(BATCH, user_id="u_all")  # no role_gate_terms
    with get_session() as s:
        n = len(s.exec(select(Job).where(Job.user_id == "u_all")).all())
    assert n == len(BATCH)


def test_shared_pool_is_never_role_gated():
    _clean()
    _upsert(BATCH, user_id=SHARED_POOL_USER)  # run_discovery passes role_gate_terms=None here
    with get_session() as s:
        n = len(s.exec(select(Job).where(Job.user_id == SHARED_POOL_USER)).all())
    assert n == len(BATCH)
