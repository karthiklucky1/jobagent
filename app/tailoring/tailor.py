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
from app.qa_store.resolver import QAResolver

log = logging.getLogger(__name__)

# Initialize canonical QA Resolver
qa_resolver = QAResolver()

TAILOR_SYSTEM = """You are a senior systems career coach, tech recruiter, and ATS optimization expert. You rewrite resumes to maximize match for a specific engineering role.

Rules:
1. FORMATTING & STRUCTURE:
   - Output VALID markdown only. Preserve the exact same section structure as the input resume.
   - BOLD KEY TECHNOLOGIES: Place core tools, libraries, and frameworks in bold (e.g., **Kubernetes**, **vLLM**, **FastAPI**, **Prometheus**) to catch the eye of the hiring manager.

2. HONEST & ATS-FRIENDLY KEYWORD BRIDGING:
   - We cannot pretend the candidate knows every technology. If the JD requires a skill/tool NOT present in the master resume, do NOT claim direct production experience with it.
   - Instead, bridge it honestly. Frame it as adjacent experience, a planned migration, or an active study area.
   - Examples of honest bridging:
     - "Exposed model inference endpoints via **FastAPI** (design aligned with high-performance runtimes like **vLLM** or **Triton**)."
     - "Engineered production MLOps pipelines using **Vertex AI**, with adjacent study of **Kubernetes Operators** and GPU memory management."
     - "Managed microservices deployment; currently studying **CUDA** and GPU acceleration to optimize inference workloads."
   - Do NOT invent entirely new jobs, degrees, or metrics. Keep all claims grounded.

3. BULLET STRUCTURE & ACTION-IMPACT:
   - Every experience bullet must start with a strong action verb (e.g., *Architected*, *Optimized*, *Scaled*, *Automated*, *Streamlined*).
   - Use the formula: `[Action Verb] + [Specific Technical Implementation using bolded tools] + [Quantifiable Performance/Business Impact]` (e.g., *Reduced latency by 24%*, *Decreased release cycles by 65%*).

4. STRICT FLUFF / JARGON BAN:
   - BAN generic AI filler and corporate buzzwords: do not use "leveraged", "synergized", "cutting-edge", "harnessing", "kernel-based systems", "orchestrated seamless integrations", "state-of-the-art", "spearheaded", "drove efficiency", "revolutionized", "demonstrated expertise in".
   - Use direct, concrete engineering terms. If you mean GPU scheduling, write **GPU node scheduling**; if you mean API integration, write **FastAPI routes**.

5. PROFESSIONAL SUMMARY:
   - Rewrite the summary (4 bullets) to highlight the candidate's core expertise matching the JD, and explicitly mention their adaptiveness to pick up adjacent tools/frameworks.

6. TECHNICAL SKILLS SECTION OPTIMIZATION:
   - Reorder skills to place JD-matching skills first.
   - For critical technologies requested by the JD that are NOT present in the candidate's master resume, add them under a dedicated subcategory:
     - `**Familiar / Actively Adopting / Transitioning to**` or `**Adjacent Tools Under Study**` (e.g., if the candidate knows PyTorch but the JD asks for Triton/vLLM, list: "**Familiar / Transitioning to**: Triton Inference Server, vLLM").
   - This ensures the resume remains ATS-friendly (contains keywords) while being completely honest and credible to a human reviewer."""

COVER_SYSTEM = """You write tight cover letters (180-220 words) using a Problem→Solution→Proof structure. They must NOT sound like generic cover letters.

The core structure (this is mandatory):
1. PROBLEM (opening, 1-2 sentences): Name the specific business/technical problem THIS company is hiring to solve, inferred from the JD. Frame it as "You're looking to <do X / solve Y>." Be concrete — pull the actual challenge from the responsibilities, not a generic statement.
2. SOLUTION (middle, 2-4 sentences): Map that problem to ONE specific thing the candidate has actually built or done. Use the formula: "I built/did <specific thing> at <company/project>." Name the real technologies and the real context from the resume.
3. PROOF (within the solution): Attach a concrete, quantified result from the resume — "which <reduced latency 24% / scaled to N users / cut release time 65%>." Only use metrics that appear in the resume. Never invent numbers.
4. WHY-THIS-COMPANY (1 sentence): One detail you'd only know if you actually cared about this company/role.
5. CLOSER (1 sentence): Forward-looking and specific. No "I look forward to hearing from you."

Hard rules:
- No "I am writing to apply for…" or "I am excited to…" openers. Open on THEIR problem, not your enthusiasm.
- Every claim must be grounded in the resume. Do NOT invent jobs, metrics, or technologies.
- Plain prose, no markdown, no bullet points. 180-220 words.
- Match the JD's vocabulary for key technologies (exact terms an ATS would scan)."""


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

    def tailor_resume(self, master_resume_md: str, job: Job, variant: str = "variant_a") -> str:
        # ── ATS exact-phrase targeting ──────────────────────────────────────
        # Find the JD phrases an ATS will scan for that are NOT already verbatim
        # in the master resume, so the tailor can incorporate them honestly.
        ats_block = ""
        try:
            from app.tailoring.ats_keywords import analyze as ats_analyze
            ats = ats_analyze(job.description or "", master_resume_md)
            if ats.missing:
                ats_block = (
                    "\n\nATS PRIORITY PHRASES — these exact terms appear in the JD but are "
                    "MISSING verbatim from the resume. Where the candidate's real experience "
                    "supports it, incorporate the EXACT phrasing below (do not paraphrase, "
                    "do not invent experience):\n"
                    + "\n".join(f"  - {p}" for p in ats.missing)
                    + f"\n(Already covered: {', '.join(ats.matched[:8])})"
                )
        except Exception as e:
            log.warning("ATS keyword analysis failed (continuing without it): %s", e)

        prompt = f"""Job description:
---
Title: {job.title}
Company: {job.company}
{job.description[:5000]}
---{ats_block}

Return the tailored resume in markdown. No commentary."""

        # A/B Testing routing
        use_openai = False
        if variant == "variant_b" and self._openai_client:
            use_openai = True
        elif not self._anthropic_client and self._openai_client:
            use_openai = True

        if not use_openai and self._active_backend == "anthropic" and self._anthropic_client:
            try:
                system = [
                    {"type": "text", "text": TAILOR_SYSTEM, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": f"Master resume (markdown):\n---\n{master_resume_md}\n---", "cache_control": {"type": "ephemeral"}},
                ]
                resp = self._anthropic_client.messages.create(
                    model=settings.tailoring_model,
                    max_tokens=4000,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text
            except Exception as e:
                log.warning("Tailor: Anthropic failed, falling back to OpenAI: %s", e)

        if self._openai_client:
            resp = self._openai_client.chat.completions.create(
                model="gpt-4o-mini",
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

Write the cover letter using the Problem→Solution→Proof structure:
1. Open by naming the specific problem {job.company} is hiring to solve (infer it from the responsibilities above).
2. Map it to ONE thing the candidate actually built or did, with the real company/project and technologies.
3. Attach a concrete quantified result that appears in the resume.
4. One sentence on why {job.company} specifically.
5. One forward-looking closer.

Ground every claim in the resume. Invent nothing."""

        if self._active_backend == "anthropic" and self._anthropic_client:
            try:
                system = [
                    {"type": "text", "text": COVER_SYSTEM, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": f"Resume (markdown):\n{master_resume_md}", "cache_control": {"type": "ephemeral"}},
                ]
                resp = self._anthropic_client.messages.create(
                    model=settings.tailoring_model,
                    max_tokens=600,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text
            except Exception as e:
                log.warning("Tailor: Anthropic failed for cover letter, falling back to OpenAI: %s", e)

        if self._openai_client:
            resp = self._openai_client.chat.completions.create(
                model="gpt-4o-mini",
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
    import random
    from datetime import datetime

    # --- Phase 1: read data in a short session, then close it ---
    with get_session() as session:
        app = session.get(Application, application_id)
        if not app:
            raise ValueError(f"Application {application_id} not found")
        job = session.get(Job, app.job_id)
        # snapshot all fields needed for LLM calls (detach-safe primitives)
        job_title = job.title
        job_company = job.company
        job_description = job.description
        job_url = job.url
        job_posted_at = job.posted_at
        profile_variant = app.profile_variant  # "backend" | "ai_agents" | "fullstack" | None
        custom_highlight_block = app.custom_highlight_block  # optional extra bullets from SeniorReviewer

    # --- Phase 2: all LLM work outside any session (no lock held) ---
    variant = random.choice(["variant_a", "variant_b"])

    # Use the profile-specific resume when SeniorReviewer recommended one; fall back to master
    profile_path = (
        settings.profiles_dir / f"{profile_variant}.md"
        if profile_variant and (settings.profiles_dir / f"{profile_variant}.md").exists()
        else settings.resume_path
    )
    master = profile_path.read_text(encoding="utf-8")

    # Append the custom highlight block so the tailor LLM sees the senior reviewer's framing
    if custom_highlight_block:
        master = master + f"\n\n## CUSTOM HIGHLIGHTS (Senior Reviewer — prioritize these)\n{custom_highlight_block}"

    log.info(
        "Tailoring app %d using profile=%s path=%s",
        application_id, profile_variant or "master", profile_path.name,
    )
    tailor = Tailor()

    # Build a lightweight Job-like object so Tailor methods still work
    class _JobSnapshot:
        title = job_title
        company = job_company
        description = job_description

    job_snap = _JobSnapshot()
    resume_md = tailor.tailor_resume(master, job_snap, variant=variant)
    cover = tailor.write_cover_letter(master, job_snap)

    # ── Quality checks ────────────────────────────────────────────────────────

    # 1. Grounding check — no hallucinated bullets
    grounding_failed = False
    grounding_notes = None
    try:
        from app.tailoring.grounding import GroundingChecker
        checker = GroundingChecker()
        log.info("Grounding check: app %d (variant: %s)...", application_id, variant)
        g_result = checker.check(master, resume_md)
        if not g_result.passed:
            grounding_failed = True
            grounding_notes = "Grounding check failed. Flagged bullets:\n" + "\n".join(
                [f"- {fb['bullet']}" for fb in g_result.flagged_bullets]
            )
            log.warning("Grounding FAILED for app %d: %d bullets flagged",
                        application_id, len(g_result.flagged_bullets))
        else:
            log.info("Grounding PASSED for app %d", application_id)
    except Exception as e:
        log.warning("Failed to run grounding check: %s", e)

    # 2. Resume Doctor — quality, ATS coverage, banned words, integrity
    doctor_failed = False
    doctor_notes = None
    try:
        from app.tailoring.doctor import ResumeDoctor
        doc = ResumeDoctor()
        d_result = doc.check(resume_md, master, job_description)
        log.info("Doctor report app %d: %s", application_id, d_result.summary())
        if not d_result.passed:
            doctor_failed = True
            doctor_notes = f"Doctor score={d_result.score}/100.\n" + "\n".join(d_result.issues)
            log.warning("Doctor FAILED for app %d (score=%d)", application_id, d_result.score)
    except Exception as e:
        log.warning("Failed to run resume doctor: %s", e)

    # Build output paths
    identity = qa_resolver.data.get("identity", {})
    first = identity.get("first_name", "Karthik")
    last = identity.get("last_name", "Amruthaluri")
    resume_filename = f"{first}_{last}_Resume.docx"
    cover_filename = f"{first}_{last}_Cover_Letter.txt"

    out_dir = settings.data_dir / "tailored" / f"app_{application_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_path = out_dir / resume_filename
    cover_path = out_dir / cover_filename
    _md_to_docx(resume_md, resume_path)

    posted_str = job_posted_at.strftime("%B %d, %Y") if job_posted_at else "unknown"
    cover_header = (
        f"Company:   {job_company}\n"
        f"Role:      {job_title}\n"
        f"Posted:    {posted_str}\n"
        f"URL:       {job_url}\n"
        f"\n---COVER---\n\n"
    )
    cover_path.write_text(cover_header + cover, encoding="utf-8")

    # --- Phase 3: write results in a short session ---
    with get_session() as session:
        app = session.get(Application, application_id)
        if not app:
            raise ValueError(f"Application {application_id} not found after LLM work")
        app.resume_variant = variant
        app.tailored_resume_path = str(resume_path)
        app.cover_letter_path = str(cover_path)
        app.updated_at = datetime.utcnow()

        if grounding_failed:
            app.status = ApplicationStatus.ERROR
            app.notes = grounding_notes
            log.warning("Application %d blocked at ERROR: grounding failure", application_id)
        elif doctor_failed:
            app.status = ApplicationStatus.ERROR
            app.notes = doctor_notes
            log.warning("Application %d blocked at ERROR: doctor quality failure", application_id)
        else:
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
