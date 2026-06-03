"""Resume tailoring: given JD + master resume, generate a tailored version.

Strategy:
1. Claude reads JD and identifies the top 5-7 keywords/skills the ATS will scan for.
2. Claude rewrites bullets to surface relevant experience first, swap synonyms
   to match JD vocabulary, and reorder sections by relevance.
3. Generate a 200-word cover letter referencing specific JD signals.
4. Render to .docx using the master template (preserves formatting/ATS friendliness).

This is intentionally a thin skeleton — the quality lives in the prompt, which
you should iterate on against your actual resume. Run a few tailoring jobs
manually, eyeball the outputs, refine.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Tuple

from anthropic import Anthropic
from docx import Document
from sqlmodel import select

from app.config import settings
from app.db.init_db import get_session
from app.db.models import Application, ApplicationStatus, Job

log = logging.getLogger(__name__)

TAILOR_SYSTEM = """You are a senior career coach + ATS optimization expert. You rewrite resumes for specific jobs.

Rules:
- Output VALID markdown only. Same section structure as the input resume.
- Never invent experience. Only reword, reorder, and re-emphasize what's there.
- Use EXACT job description phrasing rather than synonyms (e.g., if the JD states "cross-functional collaboration", use that exact phrase instead of "interdepartmental teamwork").
- Spell out acronyms at least once: e.g. "Search Engine Optimization (SEO)" then use "SEO" thereafter.
- Front-load most-relevant keywords in the professional summary and in the very FIRST bullet point under each role, as these locations are heavily weighted by modern ATS parsing engines.
- Target a density of 15-25 key terms matching the JD. Do not exceed this to avoid keyword stuffing flags.
- Do not include or attempt to inject any hidden formatting tricks (such as white text, zero-opacity strings, or tiny font hacks), as modern ATS platforms (Workday, Greenhouse, Lever) actively flag these for auto-rejection.
- Keep length within 5% of original. ATS truncates long resumes."""

COVER_SYSTEM = """You write tight cover letters (180-220 words) that don't sound like cover letters.

Rules:
- No "I am writing to apply for…" openers. Start with a concrete hook tied to the company.
- One paragraph: the specific thing in the JD that maps to a specific thing on the resume.
- One paragraph: why this company, evidenced by one detail you'd only know if you cared.
- One sentence closer. No "I look forward to hearing from you."
- Plain prose, no markdown."""


class Tailor:
    def __init__(self):
        self._anthropic_client = None
        self._openai_client = None
        self._active_backend = None
        
        if settings.anthropic_api_key:
            try:
                from anthropic import Anthropic
                self._anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
                self._active_backend = "anthropic"
            except Exception:
                pass
        if settings.openai_api_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=settings.openai_api_key)
                if not self._active_backend:
                    self._active_backend = "openai"
            except Exception:
                pass

    def tailor_resume(self, master_resume_md: str, job: Job) -> str:
        prompt = f"""Job description:
---
Title: {job.title}
Company: {job.company}
{job.description[:5000]}
---

Return the tailored resume in markdown. No commentary."""

        if self._active_backend == "anthropic" and self._anthropic_client:
            try:
                system = [
                    {"type": "text", "text": TAILOR_SYSTEM, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": f"Master resume (markdown):\n---\n{master_resume_md}\n---", "cache_control": {"type": "ephemeral"}},
                ]
                resp = self._anthropic_client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=4000,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text
            except Exception as e:
                log.warning("Tailor: Anthropic failed, falling back to OpenAI: %s", e)

        if self._openai_client:
            resp = self._openai_client.chat.completions.create(
                model="gpt-4o",
                max_tokens=4000,
                messages=[
                    {"role": "system", "content": f"{TAILOR_SYSTEM}\n\nMaster resume (markdown):\n---\n{master_resume_md}\n---"},
                    {"role": "user", "content": prompt},
                ]
            )
            return resp.choices[0].message.content

        raise RuntimeError("No LLM backend available for tailoring resume")

    def write_cover_letter(self, master_resume_md: str, job: Job) -> str:
        prompt = f"""Job:
Title: {job.title}
Company: {job.company}
{job.description[:4000]}

Write the cover letter."""

        if self._active_backend == "anthropic" and self._anthropic_client:
            try:
                system = [
                    {"type": "text", "text": COVER_SYSTEM, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": f"Resume (markdown):\n{master_resume_md}", "cache_control": {"type": "ephemeral"}},
                ]
                resp = self._anthropic_client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=600,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text
            except Exception as e:
                log.warning("Tailor: Anthropic failed for cover letter, falling back to OpenAI: %s", e)

        if self._openai_client:
            resp = self._openai_client.chat.completions.create(
                model="gpt-4o",
                max_tokens=600,
                messages=[
                    {"role": "system", "content": f"{COVER_SYSTEM}\n\nResume (markdown):\n{master_resume_md}"},
                    {"role": "user", "content": prompt},
                ]
            )
            return resp.choices[0].message.content

        raise RuntimeError("No LLM backend available for cover letter")


def _add_formatted_run(para, text: str) -> None:
    """Add text to a paragraph with **bold** and *italic* markers rendered properly."""
    parts = re.split(r"(\*\*[^*\n]+\*\*|\*[^*\n]+\*)", text)
    for part in parts:
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            para.add_run(part[2:-2]).bold = True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            para.add_run(part[1:-1]).italic = True
        else:
            para.add_run(part)


def _md_to_docx(md_text: str, out_path: Path) -> None:
    doc = Document()
    for line in md_text.splitlines():
        stripped = line.strip()
        if not stripped:
            doc.add_paragraph("")
        elif stripped.startswith("# "):
            doc.add_heading(stripped[2:], level=1)
        elif stripped.startswith("## "):
            doc.add_heading(stripped[3:], level=2)
        elif stripped.startswith("### "):
            doc.add_heading(stripped[4:], level=3)
        elif stripped.startswith("- ") or stripped.startswith("* "):
            _add_formatted_run(doc.add_paragraph(style="List Bullet"), stripped[2:])
        else:
            _add_formatted_run(doc.add_paragraph(), stripped)
    doc.save(out_path)


def tailor_for_application(application_id: int) -> Tuple[Path, Path]:
    """Generate tailored resume + cover letter for one application."""
    with get_session() as session:
        app = session.get(Application, application_id)
        if not app:
            raise ValueError(f"Application {application_id} not found")
        job = session.get(Job, app.job_id)

        master = settings.resume_path.read_text(encoding="utf-8")
        tailor = Tailor()

        resume_md = tailor.tailor_resume(master, job)
        cover = tailor.write_cover_letter(master, job)

        out_dir = settings.data_dir / "tailored" / f"app_{application_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        resume_path = out_dir / "Karthik_Amruthaluri_Resume.docx"
        cover_path = out_dir / "Karthik_Amruthaluri_Cover_Letter.txt"
        _md_to_docx(resume_md, resume_path)
        cover_path.write_text(cover, encoding="utf-8")

        app.tailored_resume_path = str(resume_path)
        app.cover_letter_path = str(cover_path)
        app.status = ApplicationStatus.TAILORED
        session.add(app)
        session.commit()

        return resume_path, cover_path


def tailor_all_shortlisted() -> int:
    """Process every SHORTLISTED application."""
    with get_session() as session:
        apps = session.exec(
            select(Application).where(Application.status == ApplicationStatus.SHORTLISTED)
        ).all()
        ids = [a.id for a in apps]
    for aid in ids:
        try:
            tailor_for_application(aid)
            log.info("Tailored application %d", aid)
        except Exception as e:
            log.exception("Tailoring failed for app %d: %s", aid, e)
    return len(ids)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    n = tailor_all_shortlisted()
    print(f"Tailored {n} applications.")
