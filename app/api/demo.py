"""Public demo endpoints for anonymous resume matching."""
import logging
import re
from typing import List, Optional
from pydantic import BaseModel
from sqlmodel import select
from app.db.init_db import get_session
from app.db.models import Job, JobSource
from app.intelligence.door_match import CandidateProfile, classify_door
from app.intelligence.role_bar import build_role_bar

log = logging.getLogger(__name__)

class PublicDemoRequest(BaseModel):
    resume_text: str
    target_role: str

def extract_text_from_file(file) -> str:
    """Read file content and extract plain text depending on file extension."""
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    content = file.file.read()
    
    if ext == "pdf":
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    elif ext == "docx":
        import io
        from docx import Document
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    else:
        # Fallback to plain text/markdown
        return content.decode("utf-8", errors="ignore")

def parse_experience_years(text: str) -> int:
    """Tolerant regex parser to extract years of experience from resume text."""
    matches = re.findall(r"(\d+)\+?\s*(?:years?|yrs?|yoe)\b", text, re.IGNORECASE)
    if matches:
        try:
            return max(int(m) for m in matches)
        except ValueError:
            pass
    return 3  # reasonable default if not found

def parse_university(text: str) -> str:
    """Tolerant parser to find the university name in resume text."""
    # Step 1: Scan line-by-line (normal resume format)
    for line in text.split("\n"):
        line_clean = line.strip()
        if not line_clean:
            continue
        
        # Check if line contains university keywords
        if any(kw in line_clean.lower() for kw in ["university", "college", "institute of technology", "polytechnic"]):
            # Split line on common delimiters to extract the clean university name
            parts = re.split(r"\s{2,}|,|\(|\b(?:and|with|graduated|gpa)\b|-|–|—|:|\.", line_clean, flags=re.IGNORECASE)
            for part in parts:
                part_clean = part.strip()
                # Clean up bullets/special chars at the start
                part_clean = re.sub(r"^[\s•\-*+]+", "", part_clean)
                part_clean = re.sub(r"^(?:from|at|studied|graduated|education|degree)\s+", "", part_clean, flags=re.IGNORECASE)
                part_clean = part_clean.strip()
                if any(kw in part_clean.lower() for kw in ["university", "college", "institute of technology", "polytechnic"]):
                    if len(part_clean) > 5 and len(part_clean) < 80:
                        return part_clean
                        
    # Step 2: Fallback to paragraph-style search if line-by-line scan missed it
    match = re.search(
        r"\b(University\s+of\s+[A-Za-z\s]+?)\b(?:\s|-|–|—|,|\.|\(|:|$)",
        text,
        re.IGNORECASE
    )
    if match:
        uni = match.group(1).strip()
        uni = re.split(r"\s{2,}|,|\(|\b(?:and|with|graduated|gpa)\b|-|–|—|:|\n|\.", uni, flags=re.IGNORECASE)[0].strip()
        if len(uni) > 5 and len(uni) < 80:
            return uni
            
    match = re.search(
        r"\b([A-Za-z\s]+?University)\b",
        text,
        re.IGNORECASE
    )
    if match:
        uni = match.group(1).strip()
        uni = re.split(r"\s{2,}|,|\(|\b(?:and|with|graduated|gpa)\b|-|–|—|:|\n|\.", uni, flags=re.IGNORECASE)[0].strip()
        if len(uni) > 5 and len(uni) < 80:
            return uni
            
    return ""

def get_demo_jobs(target_role: str) -> List[dict]:
    """Fetch 3 jobs matching the target role or use fallback mock jobs."""
    jobs_list = []
    
    # Try to find real jobs in the local database matching the target role
    try:
        with get_session() as session:
            # Query jobs matching the target role case-insensitively
            q = select(Job).where(
                Job.title.like(f"%{target_role}%") | Job.description.like(f"%{target_role}%")
            ).limit(10)
            real_jobs = session.exec(q).all()
            
            for j in real_jobs:
                jobs_list.append({
                    "id": j.id,
                    "company": j.company or "Unknown Company",
                    "title": j.title or "Software Engineer",
                    "description": j.description or "",
                    "url": j.url or "#",
                    "location": j.location or "Remote",
                    "source": j.source.value if j.source else "discovered",
                })
    except Exception as e:
        log.warning("Could not load real jobs for demo matching: %s", e)

    # If database is clean or doesn't have enough matching jobs, supplement with mock jobs
    if len(jobs_list) < 3:
        mock_jobs = [
            {
                "id": -1,
                "company": "Stripe",
                "title": f"Staff {target_role or 'AI Engineer'} (Platform)",
                "description": (
                    "We are looking for a Staff Engineer to lead our Core Platform team. "
                    "You must have 10+ years of production experience scaling machine learning infrastructure "
                    "or backend systems at an enterprise scale. PhD or Master's in CS preferred. "
                    "Strictly on-site in San Francisco, CA. Visa sponsorship is not available for this role."
                ),
                "url": "https://stripe.com/jobs",
                "location": "San Francisco, CA (On-site)",
                "source": "stripe",
            },
            {
                "id": -2,
                "company": "Google",
                "title": f"Senior {target_role or 'AI Engineer'} - Google Cloud AI",
                "description": (
                    "Google Cloud AI is hiring a Senior Engineer to build next-generation enterprise API solutions. "
                    "Requirements: 5+ years of software development experience, proficiency in Python, Go, and PyTorch. "
                    "Experience with LLMs, RAG, and APIs is highly desired. Hybrid role (3 days in office in Mountain View, CA)."
                ),
                "url": "https://careers.google.com",
                "location": "Mountain View, CA (Hybrid)",
                "source": "google",
            },
            {
                "id": -3,
                "company": "FuzeRx",
                "title": f"{target_role or 'AI Engineer'} (Short Term Assignment)",
                "description": (
                    "FuzeRx is a fast-growing startup looking for an engineer to help us deploy our "
                    "initial recommendation models. This is a 6-month contract role. "
                    "Requirements: 2+ years of experience with Python, FastAPI, and PostgreSQL. "
                    "100% remote. F-1 OPT candidates are welcome to apply."
                ),
                "url": "https://fuzerx.com/careers",
                "location": "Remote",
                "source": "fuzerx",
            }
        ]
        # Merge or override to make sure we show 3 diverse jobs (including a wrong door!)
        jobs_list = mock_jobs

    return jobs_list[:3]

def run_demo_match(req: PublicDemoRequest) -> dict:
    """Run the matching cascade (JD-only) and outreach generation for the demo."""
    years = parse_experience_years(req.resume_text)
    uni = parse_university(req.resume_text)
    
    # Construct a temporary CandidateProfile
    # We default to remote_ok=True, open_to_relocation=True, and work_auth="OPT" (highly interactive)
    candidate = CandidateProfile(
        years=years,
        axis="applied",
        domains=["llm", "rag", "ml", "backend"],
        remote_ok=True,
        open_to_relocation=True,
        work_auth="OPT",
        home_metro="Cincinnati, OH",  # standard baseline
    )
    
    jobs = get_demo_jobs(req.target_role)
    results = []
    
    for job in jobs:
        # Build RoleBar and Door Match Verdict
        bar = build_role_bar(job["title"], job["description"])
        verdict = classify_door(candidate, bar, winners_n=0, data_quality="thin")
        
        # Determine a mock match score based on findings and alignment
        base_score = 85.0
        if verdict.wrong_door:
            base_score -= 30.0
        # Add years alignment factor
        if bar.years and years < bar.years:
            base_score -= min(30.0, (bar.years - years) * 5)
        
        # Clamp score between 10 and 98
        match_score = int(max(10.0, min(98.0, base_score)))
        
        # Generate custom mock outreach drafts
        drafts = [
            {
                "type": "referral_request",
                "label": "Referral request",
                "channel": "LinkedIn / email to a connection",
                "body": (
                    f"Hi {{name}}, I saw you work at {job['company']} as a {job['title']}. "
                    f"I'm an AI/ML developer with experience in Python and PyTorch, and I'd love to explore "
                    f"a referral for this role. I am work-authorized on OPT. Happy to share my resume!"
                )
            },
            {
                "type": "hiring_manager",
                "label": "Hiring-manager note",
                "channel": "LinkedIn DM to the hiring manager",
                "body": (
                    f"Hi {{name}}, I'm excited about the {job['title']} opening at {job['company']}. "
                    f"I bring practical ML skills including LLMs, RAG, and FastAPI deployment. "
                    f"I'm work-authorized on OPT with no sponsorship overhead upfront. Let's chat!"
                )
            }
        ]
        
        # Add university alumni card if uni is parsed
        if uni:
            import urllib.parse
            q_str = f'site:linkedin.com/in/ "{job["company"]}" "{uni}"'
            search_url = f"https://www.google.com/search?q={urllib.parse.quote(q_str)}"
            drafts.append({
                "type": "university_alumni",
                "label": "University Alumni connection",
                "channel": f"LinkedIn connection request to a fellow alum. Find them via: {search_url}",
                "body": (
                    f"Hi {{name}}, fellow {uni} grad here! I noticed you work at {job['company']} "
                    f"as a {job['title']}. I'm exploring the team and would love to connect to hear "
                    f"about your experience. Go {uni.split(' ')[-1]}!"
                )
            })
            
        # Add GitHub outreach card
        import urllib.parse
        search_url = f"https://github.com/search?q={urllib.parse.quote(job['company'])}&type=users"
        drafts.append({
            "type": "github_outreach",
            "label": "GitHub outreach note",
            "channel": f"LinkedIn message targeting open-source contributions. Search org: {search_url}",
            "body": (
                f"Hi {{name}}, I saw {job['company']}'s open-source contributions on GitHub. "
                f"As a developer working with similar tech, I'd love to connect and follow your work."
            )
        })
        
        # Map findings to simple JSON structures
        findings_json = []
        for f in verdict.findings:
            findings_json.append({
                "dim": f.dim,
                "status": f.status,
                "note": f.note
            })
            
        results.append({
            "company": job["company"],
            "title": job["title"],
            "location": job["location"],
            "description": job["description"],
            "match_score": match_score,
            "wrong_door": verdict.wrong_door,
            "top_reason": verdict.top_reason,
            "right_door": verdict.right_door,
            "findings": findings_json,
            "drafts": drafts,
        })
        
    return {
        "candidate": {
            "years": years,
            "university": uni,
        },
        "resume_text": req.resume_text,
        "jobs": results,
    }
