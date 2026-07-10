"""Skill-gap classification: résumé beats GitHub beats nothing."""
from __future__ import annotations

import json

from sqlmodel import delete, select

from app.db.init_db import get_session
from app.db.models import Job, JobSource, UserPersonalMemory, UserProfile


JD_A = """
Senior Backend Engineer. Requirements: strong Python, production Kafka
experience, and AWS infrastructure. You will build streaming pipelines with
Kafka and deploy services on AWS using Python and FastAPI.
"""
JD_B = """
Data Platform Engineer. Must have Python and Kafka. Experience with AWS and
Terraform is required. Kafka streaming and Python services are the core stack.
"""


def _seed(session):
    session.exec(delete(Job))
    session.exec(delete(UserPersonalMemory))
    for i, jd in enumerate((JD_A, JD_B)):
        session.add(Job(
            user_id=None, source=JobSource.GREENHOUSE, external_id=f"sg-{i}",
            company=f"Co{i}", title="Backend Engineer", url=f"https://x/{i}",
            description=jd, rerank_score=80.0 - i, blended_score=80.0 - i,
        ))
    # GitHub harvest cache: proof of kafka in a repo, nothing about aws
    session.add(UserPersonalMemory(
        user_id=None, source="github",
        raw_content=json.dumps({"github": {
            "ok": True, "username": "testuser",
            "repos": [{"name": "event-pipeline", "description": "Kafka consumer demo",
                       "language": "Python", "topics": ["kafka"], "stars": 1}],
            "events": [],
        }}),
    ))
    session.commit()


def test_skill_gap_classification(monkeypatch):
    import app.intelligence.skill_gap as sg

    with get_session() as session:
        _seed(session)

    # Résumé mentions python but not kafka/aws
    monkeypatch.setattr(
        "app.matching.pipeline._load_resume",
        lambda user_id=None: "Experienced engineer. Python, FastAPI, PostgreSQL.",
    )

    out = sg.compute_skill_gap("local", top_n_jobs=10)

    assert out["scanned_jobs"] == 2
    assert out["resume_loaded"] is True
    assert out["github"]["connected"] is True

    matched = {i["skill"] for i in out["matched"]}
    vis = {i["skill"] for i in out["add_visibility"]}
    learn = {i["skill"] for i in out["learn"]}

    assert "python" in matched
    assert "kafka" in vis, f"kafka should be add_visibility, got matched={matched} vis={vis} learn={learn}"
    kafka = next(i for i in out["add_visibility"] if i["skill"] == "kafka")
    assert kafka["evidence"]["repo"] == "event-pipeline"
    assert "add it to your résumé" in kafka["advice"]
    assert "aws" in learn
    aws = next(i for i in out["learn"] if i["skill"] == "aws")
    assert "GitHub" in aws["advice"] and "LinkedIn" in aws["advice"]
    # Demand counts: both JDs want kafka & python
    assert kafka["demand"] == 2


def test_skill_gap_empty_pool():
    from app.intelligence.skill_gap import compute_skill_gap
    with get_session() as session:
        session.exec(delete(Job))
        session.commit()
    out = compute_skill_gap("local")
    assert out["scanned_jobs"] == 0
    assert out["matched"] == [] and out["learn"] == []
