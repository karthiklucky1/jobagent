"""Senior Technical Reviewer and Profile Router.

Acts as a cynical Engineering Manager: first instinct is to REJECT,
protecting the candidate from irrelevant applications. Only approves when
the overlap is structurally genuine.

Pipeline position: runs after FAISS + RuleFilter + EmbeddingFilter + Reranker.
Called once per shortlisted application (rerank_score >= 60).

Outputs a strict JSON payload controlling which profile variant is used for
tailoring and whether a Telegram cockpit alert fires (fit_score >= 75).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import httpx

from app.config import settings
from app.db.init_db import get_session
from app.db.models import Application, ApplicationStatus, Job

log = logging.getLogger(__name__)

SENIOR_REVIEWER_SYSTEM = """You are the Senior Technical Reviewer and Profile Router for "JobAgent" — a private, hyper-selective career cockpit. Your core directive is to act like a cynical, highly protective Engineering Manager and Technical Recruiter. Instead of trying to "force" a match, your initial instinct must be to find structural reasons to REJECT the application to protect the candidate from automated company rejections or looking desperate/fictional.

### Input Data:
1. TARGET_JD: The raw, unstructured job description text block.
2. STATIC_PROFILES: Three unalterable, honest variations of the candidate's core resume:
   - "backend": Heavy on distributed systems, databases, infrastructure, and API scale.
   - "ai_agents": Focused on LLM orchestration, vector search, multi-agent pipelines, and cognitive architectures.
   - "fullstack": Focused on end-to-end feature delivery, modern API design, state management, and developer tools.
3. HISTORICAL_LEDGER: A context list of recently shortlisted/approved JDs and the profile variant + custom highlights used for them. Use this to maintain professional consistency and avoid generating conflicting engineering personas.

### Evaluation Protocol (follow sequentially):

#### STEP 1: Harsh Rejection & Risk Screening
Analyze the TARGET_JD for hard stop indicators: mismatch in core language proficiencies, unmentioned infra expectations, hidden location/timezone requirements, experience-level mismatch, explicit "no visa sponsorship" with no OPT exception, or a domain completely outside the candidate's history. Actively list the "Possibilities for Rejection." If rejection risk is high, flag `is_genuine_match` as false and set fit_score <= 30.

#### STEP 2: Historical Alignment & Consistency Check
Cross-reference the TARGET_JD with the HISTORICAL_LEDGER. If a similar role or company style exists, mirror that structural approach. If this company appeared in a prior application, the chosen profile_variant and project narrative MUST remain consistent.

#### STEP 3: Structural Missing-Piece Analysis & Custom Highlight
If the role survives Steps 1 and 2, compare the selected STATIC_PROFILE against the TARGET_JD. Identify what crucial technical or framing element is completely absent that the JD highly values.
- If nothing critical is missing: note it as a rock-solid match; leave custom_highlight_block empty.
- If a critical angle is missing: draft exactly 3 bullet points pulling REAL, buried technical elements from the candidate's master history and re-framing them directly against the core JD challenges. NEVER invent fake experience, new employers, or fabricated metrics.

### Hard Constraints:
- NEVER alter core employment dates, job titles, or company names.
- fit_score must be highly conservative. Any score above 75 requires undeniable overlap in technical stack AND architecture style.
- Output ONLY a raw, unescaped JSON object. Do NOT wrap it in markdown code fences or add any surrounding text.

### Output JSON Schema:
{
  "is_genuine_match": boolean,
  "rejection_risks": ["Detailed list of structural or stack reasons this could fail screening"],
  "recommended_resume_variant": "backend" | "ai_agents" | "fullstack",
  "fit_score": integer,
  "senior_reviewer_verdict": "Clear, blunt 2-sentence assessment of tailoring quality and alignment safety.",
  "custom_highlight_block": "Markdown string with exactly 3 bullet points framing existing projects to fill JD gaps. Empty string if match is already flawless."
}"""


@dataclass
class ReviewResult:
    is_genuine_match: bool
    rejection_risks: List[str]
    recommended_resume_variant: str
    fit_score: int
    senior_reviewer_verdict: str
    custom_highlight_block: str
    raw_json: dict = field(default_factory=dict)


class SeniorReviewer:
    def __init__(self):
        self._anthropic = None
        self._openai = None

        if settings.anthropic_api_key:
            try:
                from anthropic import Anthropic
                self._anthropic = Anthropic(api_key=settings.anthropic_api_key)
            except Exception:
                pass
        if settings.openai_api_key:
            try:
                from openai import OpenAI
                self._openai = OpenAI(api_key=settings.openai_api_key)
            except Exception:
                pass

        self._profiles: dict[str, str] = self._load_profiles()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def review(self, job: Job, user_id: str | None = None) -> Optional[ReviewResult]:
        """Run the full 3-step evaluation on a job. Returns None if both LLM backends fail.

        ``user_id`` scopes the historical ledger to the requesting user so one
        tenant's company names and private highlight blocks are never fed into
        another tenant's review prompt.
        """
        if not self._profiles:
            log.warning("SeniorReviewer: no profiles loaded from %s — skipping", settings.profiles_dir)
            return None

        ledger = self._build_historical_ledger(exclude_job_id=job.id, user_id=user_id)
        user_content = self._build_prompt(job, ledger)

        result = self._call_anthropic(user_content) or self._call_openai(user_content)
        if result is None:
            log.error("SeniorReviewer: both LLM backends failed for job %d (%s @ %s)", job.id, job.title, job.company)
            return None

        if result.fit_score >= 75 and result.is_genuine_match:
            self._push_telegram(job, result)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_profiles(self) -> dict[str, str]:
        d = settings.profiles_dir
        profiles: dict[str, str] = {}
        for name in ("backend", "ai_agents", "fullstack"):
            p = d / f"{name}.md"
            if p.exists():
                profiles[name] = p.read_text(encoding="utf-8")
            else:
                log.warning("SeniorReviewer: profile not found at %s", p)
        return profiles

    def _build_historical_ledger(self, exclude_job_id: int, user_id: str | None = None,
                                 limit: int = 6) -> list[dict]:
        """Pull recent shortlisted/tailored apps to give the LLM consistency context.

        Scoped to ``user_id`` (when provided) so the ledger never serializes
        another tenant's company names / custom_highlight_block into the prompt.
        """
        ledger: list[dict] = []
        try:
            with get_session() as session:
                from sqlmodel import select as sqlselect
                q = (
                    sqlselect(Application)
                    .where(Application.status.in_([
                        ApplicationStatus.SHORTLISTED,
                        ApplicationStatus.TAILORED,
                        ApplicationStatus.SUBMITTED,
                    ]))
                    .where(Application.profile_variant.isnot(None))
                )
                if user_id and user_id != "local":
                    q = q.where(Application.user_id == user_id)
                apps = session.exec(
                    q.order_by(Application.created_at.desc()).limit(limit)
                ).all()
                for app in apps:
                    job = session.get(Job, app.job_id)
                    if job and job.id != exclude_job_id:
                        ledger.append({
                            "company": job.company,
                            "title": job.title,
                            "profile_variant_used": app.profile_variant,
                            "fit_score": app.senior_fit_score,
                            "custom_highlight_block": app.custom_highlight_block or "",
                        })
        except Exception as e:
            log.warning("SeniorReviewer: could not load historical ledger: %s", e)
        return ledger

    def _profiles_block(self) -> str:
        """The static résumé variants — identical on every review, so they ride in
        a cached system block (see _call_anthropic) instead of being re-sent (and
        re-billed) inside each job's user message."""
        return "\n\n".join(
            f"--- PROFILE: {name} ---\n{text}"
            for name, text in self._profiles.items()
        )

    def _build_prompt(self, job: Job, ledger: list[dict]) -> str:
        # Only the per-job varying half here (JD + ledger). The static profiles are
        # sent once as a cached system block — putting them here (after the JD)
        # made the whole prompt uncacheable, since the cacheable prefix must be
        # stable and the JD changes every call.
        ledger_block = json.dumps(ledger, indent=2) if ledger else "[]"

        return f"""### TARGET_JD
Title: {job.title}
Company: {job.company}
Location: {job.location}

{(job.description or '')[:6000]}

### HISTORICAL_LEDGER
{ledger_block}

Now run the 3-step evaluation protocol and return the JSON object."""

    def _parse_response(self, raw: str) -> Optional[ReviewResult]:
        try:
            # Strip any accidental markdown fences the LLM snuck in
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text.strip())
            return ReviewResult(
                is_genuine_match=bool(data.get("is_genuine_match", False)),
                rejection_risks=data.get("rejection_risks", []),
                recommended_resume_variant=data.get("recommended_resume_variant", "ai_agents"),
                fit_score=int(data.get("fit_score", 0)),
                senior_reviewer_verdict=data.get("senior_reviewer_verdict", ""),
                custom_highlight_block=data.get("custom_highlight_block", ""),
                raw_json=data,
            )
        except Exception as e:
            log.error("SeniorReviewer: failed to parse LLM JSON: %s | raw=%r", e, raw[:300])
            return None

    def _call_anthropic(self, user_content: str) -> Optional[ReviewResult]:
        if not self._anthropic:
            return None
        try:
            resp = self._anthropic.messages.create(
                model=settings.scoring_model,
                max_tokens=1200,
                system=[
                    {"type": "text", "text": SENIOR_REVIEWER_SYSTEM, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": self._profiles_block(), "cache_control": {"type": "ephemeral"}},
                ],
                messages=[{"role": "user", "content": user_content}],
            )
            return self._parse_response(resp.content[0].text)
        except Exception as e:
            log.warning("SeniorReviewer: Anthropic call failed: %s", e)
            return None

    def _call_openai(self, user_content: str) -> Optional[ReviewResult]:
        if not self._openai:
            return None
        try:
            resp = self._openai.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=1200,
                messages=[
                    {"role": "system", "content": SENIOR_REVIEWER_SYSTEM + "\n\n" + self._profiles_block()},
                    {"role": "user", "content": user_content},
                ],
            )
            return self._parse_response(resp.choices[0].message.content)
        except Exception as e:
            log.warning("SeniorReviewer: OpenAI call failed: %s", e)
            return None

    def _push_telegram(self, job: Job, result: ReviewResult) -> None:
        """Fire a Telegram cockpit alert for high-confidence matches (fit_score >= 75)."""
        if not (settings.telegram_bot_token and settings.telegram_chat_id):
            return
        risks_text = ""
        if result.rejection_risks:
            risks_text = "\n⚠️ *Risks*: " + "; ".join(result.rejection_risks[:2])

        highlight_text = ""
        if result.custom_highlight_block:
            highlight_text = f"\n\n📝 *Custom Highlight*:\n{result.custom_highlight_block}"

        msg = (
            f"🎯 *Senior Review: APPROVED — {result.fit_score}/100*\n\n"
            f"🏢 *{job.company}*\n"
            f"💼 {job.title}\n"
            f"📄 Profile: `{result.recommended_resume_variant}`\n"
            f"🔗 {job.url}\n\n"
            f"_{result.senior_reviewer_verdict}_"
            f"{risks_text}"
            f"{highlight_text}"
        )
        try:
            httpx.post(
                f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
                json={"chat_id": settings.telegram_chat_id, "text": msg, "parse_mode": "Markdown"},
                timeout=10,
            )
            log.info("SeniorReviewer: Telegram alert pushed for %s @ %s (score=%d)", job.title, job.company, result.fit_score)
        except Exception as e:
            log.warning("SeniorReviewer: Telegram push failed: %s", e)
