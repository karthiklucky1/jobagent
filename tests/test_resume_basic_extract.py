"""Free (no-LLM) resume extraction fallback — the parser that keeps profile
autofill working when Claude/OpenAI are unusable (no credits, outage)."""
from app.intelligence.resume_basic_extract import (
    basic_extract_profile,
    to_answer_pack_shape,
)

SAMPLE = """# Priya Sharma
Senior Backend Engineer
Cincinnati, OH | priya.sharma@example.com | +1 (513) 555-0142
linkedin.com/in/priya-sharma | github.com/priyasharma | https://priya.dev

## Summary
Backend engineer with a focus on distributed systems and LLM applications.
Shipped high-throughput data pipelines at two startups.

## Experience
Senior Backend Engineer @ Speechify    Jun 2024 - Present
- Built ingestion pipeline processing 2M docs/day on FastAPI + Postgres
- Cut LLM scoring cost 60% with a two-tier cascade

Software Engineer — DataWeave    Aug 2021 - May 2024
- Owned the retail-analytics crawler fleet (Python, Kafka)

## Education
University of Cincinnati    2019 - 2021
Master of Science in Computer Science, GPA: 3.8/4.0

## Skills
Languages: Python, Go, SQL
Infra: Docker, Kubernetes, AWS, Postgres, Kafka
"""


def test_contact_fields():
    out = basic_extract_profile(SAMPLE)
    assert out["first_name"] == "Priya"
    assert out["last_name"] == "Sharma"
    assert out["email"] == "priya.sharma@example.com"
    assert "513" in out["phone"]
    assert out["linkedin_url"] == "https://linkedin.com/in/priya-sharma"
    assert out["github_url"] == "https://github.com/priyasharma"
    assert out["portfolio_url"] == "https://priya.dev"
    assert out["location"] == "Cincinnati, OH"


def test_experience_entries():
    out = basic_extract_profile(SAMPLE)
    exp = out["experience"]
    assert len(exp) == 2
    assert exp[0]["title"] == "Senior Backend Engineer"
    assert exp[0]["company"] == "Speechify"
    assert exp[0]["start"].lower().startswith("jun")
    assert exp[0]["end"].lower() == "present"
    assert "ingestion pipeline" in exp[0]["summary"]
    assert exp[1]["company"] == "DataWeave"
    assert out["current_title"] == "Senior Backend Engineer"
    # Aug 2021 → present spans ≥ 4 years
    assert out["years_experience"] >= 4


def test_education_and_skills():
    out = basic_extract_profile(SAMPLE)
    edu = out["education"]
    assert len(edu) >= 1
    assert "University of Cincinnati" in edu[0]["university"]
    assert edu[0]["end_year"] == 2021
    assert edu[0]["gpa"].startswith("3.8")
    assert out["degree"] and "Master" in (out["degree"] or "") or edu[0]["degree"]
    assert out["graduation_year"] == 2021
    assert "Python" in out["key_skills"]
    assert "Kubernetes" in out["key_skills"]
    # Category labels are stripped, not kept as skills
    assert "Languages" not in out["key_skills"].split(", ")


def test_summary_and_roles_shape():
    out = basic_extract_profile(SAMPLE)
    assert "distributed systems" in (out["professional_summary"] or "")
    assert out["suggested_target_roles"] == []  # seeding heuristic runs upstream


def test_answer_pack_shape():
    pack = to_answer_pack_shape(basic_extract_profile(SAMPLE))
    assert pack["work_experience"][0]["company"] == "Speechify"
    assert pack["work_experience"][0]["start_date"].lower().startswith("jun")
    assert pack["education"][0]["school"].startswith("University of Cincinnati")
    assert pack["education"][0]["end_date"] == "2021"


def test_empty_and_garbage_input():
    out = basic_extract_profile("")
    assert out["email"] is None and out["experience"] == []
    out2 = basic_extract_profile("just some words with no structure at all")
    assert out2["education"] == [] and out2["first_name"] is None
