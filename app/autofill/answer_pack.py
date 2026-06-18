"""Answer Pack generator.

For every application on the "manual" track (Workday, iCIMS, custom portals),
this module pre-generates a structured set of answers the user can copy-paste
into the real application form. No browser automation involved — legally safe,
works on any ATS.

For "autofill" track (Greenhouse / Lever / Ashby) the form is filled
programmatically via their APIs; this pack serves as a human-readable backup.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from sqlmodel import select

from app.config import settings
from app.db.init_db import get_session
from app.db.models import AnswerMemory, Application, Job, UserProfile

log = logging.getLogger(__name__)

# Standard fields present on virtually every application form.
_STANDARD_FIELDS = [
    ("First Name", "first_name"),
    ("Last Name", "last_name"),
    ("Email", "email"),
    ("Phone", "phone"),
    ("Location / City", "location"),
    ("LinkedIn URL", "linkedin_url"),
    ("GitHub URL", "github_url"),
    ("Portfolio / Website", "portfolio_url"),
    ("Work Authorization", "work_authorization"),
    ("Requires Visa Sponsorship?", "requires_sponsorship"),
    ("Current / Most Recent Title", "current_title"),
    ("Years of Experience", "years_experience"),
    ("Expected Salary", "_salary"),          # synthesized
    ("Degree", "degree"),
    ("University", "university"),
    ("Graduation Year", "graduation_year"),
]

# Essay questions that appear on most ATS forms.
_ESSAY_QUESTIONS = [
    "Why do you want to work at {company}?",
    "Why are you interested in this role?",
    "Describe a challenging technical problem you solved.",
    "What is your greatest professional accomplishment?",
    "Where do you see yourself in 3-5 years?",
    "Tell us about yourself.",
]

_ESSAY_SYSTEM = """You write concise, honest, professional answers to job application essay questions.
Rules:
- 80-120 words per answer.
- First person ("I").
- Specific to the candidate's actual background — never fabricate experience.
- Warm but professional tone.
- No filler phrases like "I am passionate about" or "I am excited to".
Return only the answer text, nothing else."""


def _get_or_create_profile(user_id: str | None = None) -> UserProfile:
    """Return the UserProfile row for this user, seeding from config if absent."""
    with get_session() as session:
        q = select(UserProfile)
        if user_id:
            q = q.where(UserProfile.user_id == user_id)
        profile = session.exec(q).first()
        if not profile:
            profile = UserProfile(
                user_id=user_id,
                first_name=settings.applicant_first_name,
                last_name=settings.applicant_last_name,
                email=settings.applicant_email,
                phone=settings.applicant_phone,
                location=settings.applicant_location,
                linkedin_url=settings.applicant_linkedin,
                github_url=settings.applicant_github,
                work_authorization=settings.applicant_work_auth,
            )
            session.add(profile)
            session.commit()
            session.refresh(profile)
        return profile


def _profile_to_dict(profile: UserProfile) -> dict:
    salary = ""
    if profile.salary_min and profile.salary_max:
        salary = f"{profile.salary_currency} {profile.salary_min:,} – {profile.salary_max:,}"
    elif profile.salary_min:
        salary = f"{profile.salary_currency} {profile.salary_min:,}+"

    return {
        "first_name": profile.first_name,
        "last_name": profile.last_name,
        "email": profile.email,
        "phone": profile.phone,
        "location": profile.location,
        "linkedin_url": profile.linkedin_url,
        "github_url": profile.github_url,
        "portfolio_url": profile.portfolio_url,
        "work_authorization": profile.work_authorization,
        "requires_sponsorship": "Yes" if profile.requires_sponsorship else "No",
        "current_title": profile.current_title,
        "years_experience": str(profile.years_experience) if profile.years_experience else "",
        "_salary": salary,
        "degree": profile.degree,
        "university": profile.university,
        "graduation_year": str(profile.graduation_year) if profile.graduation_year else "",
    }


def _llm_essay_answer(question: str, job: Job, profile: UserProfile, resume_text: str) -> str:
    prompt = f"""Candidate profile:
- Name: {profile.first_name} {profile.last_name}
- Title: {profile.current_title}
- Experience: {profile.years_experience} years
- Summary: {profile.professional_summary[:800] if profile.professional_summary else ""}
- Key skills: {profile.key_skills[:400] if profile.key_skills else ""}

Job:
- Company: {job.company}
- Title: {job.title}
- Description (excerpt): {job.description[:2000] if job.description else ""}

Tailored resume excerpt:
{resume_text[:2000] if resume_text else ""}

Essay question: "{question}"

Write a concise, honest answer."""

    if settings.anthropic_api_key:
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=settings.anthropic_api_key)
            resp = client.messages.create(
                model=settings.cover_letter_model,   # Haiku — cheap
                max_tokens=250,
                system=_ESSAY_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as e:
            log.warning("Essay LLM failed: %s", e)
    return ""


def _load_resume_text(application: Application) -> str:
    if not application.tailored_resume_path:
        return ""
    try:
        from docx import Document
        doc = Document(application.tailored_resume_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception:
        pass
    try:
        from pathlib import Path
        p = Path(application.tailored_resume_path)
        if p.suffix == ".md":
            return p.read_text(encoding="utf-8")
    except Exception:
        pass
    return ""


import re

# Company-name stripping regex — used to normalize question cache keys.
# "Why do you want to join Figma?" → "why do you want to join [company]?"
_COMPANY_RE = re.compile(r'\b(at|join|with|for|to)\s+([A-Z][A-Za-z0-9\.\-& ]{1,40}?)(\?|$|\s)', re.UNICODE)


def _normalize_question(question: str, company: str | None = None) -> str:
    """Return a company-agnostic cache key for a question.

    Strips company names so the same answer is reused across companies.
    "Why do you want to work at Figma?" and "Why work at Stripe?" both
    map to a single cached answer template with {company} as placeholder.
    """
    q = question.lower().strip().rstrip("?").strip()
    if company:
        q = q.replace(company.lower(), "{company}")
    # Also strip any remaining title-case company reference
    q = _COMPANY_RE.sub(lambda m: f"{m.group(1)} {{company}}", q)
    return q


def _save_memory(label_normalized: str, label_original: str, answer: str, user_id: str | None = None) -> None:
    from datetime import datetime as _dt
    with get_session() as session:
        existing = session.exec(
            select(AnswerMemory).where(
                AnswerMemory.label_normalized == label_normalized,
                AnswerMemory.user_id == user_id,
            )
        ).first()
        if existing:
            existing.answer = answer
            existing.use_count += 1
            existing.last_used_at = _dt.utcnow()
            session.add(existing)
        else:
            session.add(AnswerMemory(
                user_id=user_id,
                label_normalized=label_normalized,
                label_original=label_original,
                answer=answer,
            ))
        session.commit()


def answer_question(question: str, application_id: int, user_id: str | None = None) -> str:
    """Return an answer for a single essay question.

    Cost strategy (hybrid):
    1. Check per-user AnswerMemory with company-agnostic key → free.
    2. If miss, call Haiku with ~1.5k tokens → ~$0.002.
    3. Save result to memory → all future occurrences of this question type are free.

    This means the user pays ~$0.002 the FIRST time a question type appears,
    then $0 forever after, regardless of how many companies use the same question.
    """
    with get_session() as session:
        application = session.get(Application, application_id)
        if not application:
            return ""
        job = session.get(Job, application.job_id)
    if not job:
        return ""

    norm_key = _normalize_question(question, company=job.company)

    # ── 1. Check cache ──
    cached = _lookup_memory(norm_key, user_id=user_id)
    if cached:
        # Inject actual company name back into templated answer
        return cached.replace("{company}", job.company)

    # ── 2. Call AI ──
    if not settings.anthropic_api_key:
        return ""

    profile = _get_or_create_profile(user_id=user_id)
    resume_text = _load_resume_text(application)
    answer = _llm_essay_answer(question, job, profile, resume_text)
    if not answer:
        return ""

    # ── 3. Save with company replaced by {company} placeholder ──
    template = answer.replace(job.company, "{company}") if job.company else answer
    _save_memory(norm_key, question, template, user_id=user_id)
    log.info("answer_question: generated + cached for key=%r user=%s (~$0.002 cost)", norm_key, user_id)
    return answer


    norm = label.lower().strip()
    with get_session() as session:
        q = select(AnswerMemory).where(AnswerMemory.label_normalized == norm)
        if user_id:
            q = q.where(AnswerMemory.user_id == user_id)
        mem = session.exec(q).first()
        return mem.answer if mem else None


def get_essay_answers(application_id: int, user_id: str | None = None) -> dict:
    """Return ONLY already-cached answers (no AI calls).

    The fill-pack includes only what's in memory — essentially free.
    For questions not yet in memory, the extension calls /api/answer-question
    per-question on-demand (only when that question actually appears on the form).
    """
    with get_session() as session:
        application = session.get(Application, application_id)
        if not application:
            return {}
        job = session.get(Job, application.job_id)
    if not job:
        return {}

    result = {}
    for question_template in _ESSAY_QUESTIONS:
        question = question_template.format(company=job.company)
        norm_key = _normalize_question(question, company=job.company)
        cached = _lookup_memory(norm_key, user_id=user_id)
        if cached:
            result[question.lower().strip()] = cached.replace("{company}", job.company)
    return result


def generate_answer_pack(application_id: int, user_id: str | None = None) -> dict:
    """Generate a complete answer pack for one application.

    Returns a dict with:
      - standard_fields: list of {label, value} for form fields
      - essay_answers: list of {question, answer} for open-ended questions
      - resume_path: path to tailored resume DOCX for upload
      - cover_letter: text of cover letter
    """
    with get_session() as session:
        application = session.get(Application, application_id)
        if not application:
            raise ValueError(f"Application {application_id} not found")
        job = session.get(Job, application.job_id)

    profile = _get_or_create_profile(user_id=user_id)
    profile_dict = _profile_to_dict(profile)
    resume_text = _load_resume_text(application)

    # Cover letter text
    cover_letter = ""
    if application.cover_letter_path:
        try:
            from pathlib import Path
            cover_letter = Path(application.cover_letter_path).read_text(encoding="utf-8")
        except Exception:
            pass

    # --- Standard fields ---
    standard_fields = []
    for label, key in _STANDARD_FIELDS:
        value = profile_dict.get(key, "")
        if not value:
            value = _lookup_memory(label, user_id=user_id) or ""
        standard_fields.append({"label": label, "value": str(value)})

    # --- Essay answers ---
    essay_answers = []
    for question_template in _ESSAY_QUESTIONS:
        question = question_template.format(company=job.company)
        # Check memory first
        answer = _lookup_memory(question, user_id=user_id)
        if not answer:
            answer = _llm_essay_answer(question, job, profile, resume_text)
        if answer:
            essay_answers.append({"question": question, "answer": answer})

    return {
        "application_id": application_id,
        "company": job.company,
        "title": job.title,
        "apply_url": application.apply_url or job.url,
        "ats": job.source.value,
        "standard_fields": standard_fields,
        "essay_answers": essay_answers,
        "resume_path": application.tailored_resume_path or "",
        "cover_letter": cover_letter,
    }
