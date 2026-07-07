"""Unit tests for the public resume matching demo endpoint."""
import pytest
from fastapi.testclient import TestClient
from app.api.server import app
from app.api.demo import parse_experience_years, parse_university

client = TestClient(app)

def test_resume_parsers():
    # Test years of experience extraction
    resume_1 = "I have 5+ years of experience as a Software Engineer.\nStudied CS."
    assert parse_experience_years(resume_1) == 5
    
    resume_2 = "Graduate student with 2 yoe in Python."
    assert parse_experience_years(resume_2) == 2
    
    resume_3 = "Just graduated. No experience mentioned."
    assert parse_experience_years(resume_3) == 3  # fallback default
    
    # Test university extraction
    resume_uni_1 = "Education:\nUniversity of Cincinnati - BS in Computer Science\nGPA 3.8"
    assert parse_university(resume_uni_1) == "University of Cincinnati"
    
    resume_uni_2 = "MS in Data Science\nGeorgia Institute of Technology\nGraduated 2024"
    assert parse_university(resume_uni_2) == "Georgia Institute of Technology"

def test_public_demo_endpoint():
    payload = {
        "resume_text": "Experienced developer with 4 years of experience. Graduated from University of Cincinnati.",
        "target_role": "AI Engineer"
    }
    
    response = client.post("/api/public/demo-match", data=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "candidate" in data
    assert data["candidate"]["years"] == 4
    assert data["candidate"]["university"] == "University of Cincinnati"
    
    assert "jobs" in data
    assert len(data["jobs"]) == 3
    
    # Assert fields in returned jobs
    for job in data["jobs"]:
        assert "company" in job
        assert "title" in job
        assert "match_score" in job
        assert "wrong_door" in job
        assert "drafts" in job
        assert len(job["drafts"]) >= 2
        
        # Verify drafts contain university alumni link since university was parsed
        has_uni_draft = any(d["type"] == "university_alumni" for d in job["drafts"])
        assert has_uni_draft is True

def test_public_demo_endpoint_file():
    import io
    file_data = io.BytesIO(b"Resume content: 6 years experience. Graduated from Georgia Institute of Technology.")
    response = client.post(
        "/api/public/demo-match",
        data={"target_role": "Backend Developer"},
        files={"file": ("resume.txt", file_data, "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["candidate"]["years"] == 6
    assert data["candidate"]["university"] == "Georgia Institute of Technology"
    assert "resume_text" in data
    assert "Georgia Institute of Technology" in data["resume_text"]
