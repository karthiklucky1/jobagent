"""Adoption: instant per-user feeds copied from the shared job pool (no HTTP)."""
from __future__ import annotations

from datetime import datetime, timedelta

from sqlmodel import delete, select

from app.db.init_db import get_session
from app.db.models import Job, JobSource, UserProfile
from app.discovery.pipeline import SHARED_POOL_USER


def _clean():
    with get_session() as s:
        s.exec(delete(Job))
        s.exec(delete(UserProfile))
        s.commit()


def _shared_job(ext_id, title, days_old=1, closed=False):
    now = datetime.utcnow()
    return Job(user_id=SHARED_POOL_USER, source=JobSource.GREENHOUSE,
               external_id=ext_id, company=f"Co{ext_id}", title=title,
               location="Remote", remote=True, url=f"https://x/{ext_id}",
               description="Python ML role at a great company.",
               posted_at=now - timedelta(days=days_old),
               first_seen=now - timedelta(days=days_old), is_closed=closed)


def test_adopt_copies_role_matching_recent_jobs_only():
    _clean()
    with get_session() as s:
        s.add(UserProfile(user_id="u_adopt", target_roles="Machine Learning Engineer"))
        s.add(_shared_job("s1", "Senior ML Engineer", days_old=2))       # match
        s.add(_shared_job("s2", "Product Designer", days_old=2))         # wrong role
        s.add(_shared_job("s3", "Machine Learning Engineer", days_old=40))  # too old
        s.add(_shared_job("s4", "MLOps Engineer", days_old=5, closed=True))  # closed
        s.commit()

    from app.strategy.adoption import adopt_shared_jobs
    inserted = adopt_shared_jobs("u_adopt")
    assert inserted == 1

    with get_session() as s:
        mine = s.exec(select(Job).where(Job.user_id == "u_adopt")).all()
    assert [j.title for j in mine] == ["Senior ML Engineer"]
    # The copy keeps the original posting date (freshness stays honest).
    assert mine[0].posted_at is not None

    # Second pass is a no-op — dedupe by (source, external_id) per user.
    assert adopt_shared_jobs("u_adopt") == 0


def test_adopt_backfills_after_role_edit():
    """Editing roles must surface already-collected shared jobs for the NEW
    role — the reason adoption is triggered from the target-roles endpoint."""
    _clean()
    with get_session() as s:
        s.add(UserProfile(user_id="u_pivot", target_roles="Machine Learning Engineer"))
        s.add(_shared_job("p1", "Senior ML Engineer", days_old=1))
        s.add(_shared_job("p2", "Data Engineer", days_old=1))
        s.commit()

    from app.strategy.adoption import adopt_shared_jobs
    assert adopt_shared_jobs("u_pivot") == 1  # only the ML job

    # User pivots to data engineering — adoption now pulls the data job too.
    with get_session() as s:
        p = s.exec(select(UserProfile).where(UserProfile.user_id == "u_pivot")).first()
        p.target_roles = "Data Engineer"
        s.add(p)
        s.commit()
    assert adopt_shared_jobs("u_pivot") == 1

    with get_session() as s:
        titles = sorted(j.title for j in
                        s.exec(select(Job).where(Job.user_id == "u_pivot")).all())
    assert titles == ["Data Engineer", "Senior ML Engineer"]
