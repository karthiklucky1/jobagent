"""Stage-2 reranker: LLM scores top-K from FAISS with reasoning.

Tries Claude first (Anthropic), falls back to GPT-4o (OpenAI) if Claude
is unavailable (e.g. credits depleted). Both use the same system prompt
and expect the same JSON output format.
"""
from __future__ import annotations

import json
import logging
from typing import List, Optional, Tuple

from app.config import settings
from app.db.models import Job

log = logging.getLogger(__name__)

SYSTEM = """You evaluate job-applicant fit. Given a candidate's resume and a job description, return a single JSON object:
{
  "score": <0-100 integer>,
  "reason": "<one sentence, max 25 words>",
  "concerns": ["<concern 1>", "<concern 2>"]
}

Score rubric:
- 85-100: Strong match. Core skills, experience level, and availability all align well.
- 70-84: Good match with one minor gap (e.g. missing one secondary skill, slightly under-experienced).
- 60-69: Reasonable stretch. Core skills overlap significantly but there's a seniority or domain gap.
- 40-59: Weak. Notable gaps in skills or experience.
- 0-39: Wrong role or hard blocker (explicit no-sponsorship, unrelated field).

Candidate Context:
- 3+ years of professional AI/ML engineering experience, including roles as an AI/ML Engineer at Home Depot (Jun 2025 - Mar 2026) and NTT DATA (May 2022 - Aug 2024). Currently completing a Master of Engineering at the University of Cincinnati (graduating Aug 2026).
- Strong in: Python, PyTorch, TensorFlow, Scikit-learn, XGBoost, LLMs, RAG pipelines, multi-agent systems, FAISS, FastAPI, Spark, Kubeflow, Vertex AI, Docker, Kubernetes, AWS/GCP, and CI/CD.
- Best fit: AI/ML Engineer, NLP Engineer, MLOps/Platform Engineer, or Backend Python Developer roles (Junior, Mid-level, or New Grad).
- Can stretch for roles titled "Senior" if the actual requirements match the candidate's skills.
- Work authorization: F-1 visa, eligible for OPT. Requires H-1B sponsorship long-term. ONLY score 0-10 if the posting EXPLICITLY states "No sponsorship" or "US Citizens/Permanent Residents only" or requires active security clearance. If the posting is silent on sponsorship, assume it is possible.
- Location & Country: ONLY consider jobs located within the United States (USA) or fully remote roles from US-based companies. If the job is located outside the USA (e.g., Canada, Europe, UK, India, etc.), score it 0-10 immediately as a hard blocker. For US-based roles: Cincinnati, OH or remote is preferred; on-site roles outside Cincinnati should be scored ≤50.
- Startups and growth-stage companies (Series A-D, <1000 employees) are a great fit for this candidate — give a +5 bonus for startup/growth-stage companies.

Return JSON only. No prose."""


def _build_prompt(resume_text: str, job: Job) -> str:
    return f"""<resume>
{resume_text[:6000]}
</resume>

<job>
Title: {job.title}
Company: {job.company}
Location: {job.location}
Remote: {job.remote}

Description:
{job.description[:5000]}
</job>

Return the JSON object."""


def _parse_response(text: str) -> Tuple[float, str, List[str]]:
    """Parse LLM JSON response, tolerating markdown fences."""
    text = text.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(text)
    return float(data["score"]), data.get("reason", ""), data.get("concerns", [])


class Reranker:
    def __init__(self):
        self._anthropic_client = None
        self._openai_client = None
        self._active_backend: Optional[str] = None  # "anthropic" or "openai"
        self._init_clients()

    def _init_clients(self):
        """Initialize available LLM clients."""
        # Try Anthropic first
        if settings.anthropic_api_key:
            try:
                from anthropic import Anthropic
                self._anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
                self._active_backend = "anthropic"
                log.info("Reranker: Anthropic (Claude) client initialized")
            except Exception as e:
                log.warning("Reranker: Failed to init Anthropic client: %s", e)

        # OpenAI fallback
        if settings.openai_api_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=settings.openai_api_key)
                if not self._active_backend:
                    self._active_backend = "openai"
                log.info("Reranker: OpenAI (GPT-4o) client initialized as %s",
                         "primary" if self._active_backend == "openai" else "fallback")
            except Exception as e:
                log.warning("Reranker: Failed to init OpenAI client: %s", e)

        if not self._active_backend:
            log.error("Reranker: No LLM backend available! Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")

    def _score_anthropic(self, prompt: str) -> str:
        """Call Claude for scoring."""
        resp = self._anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    def _score_openai(self, prompt: str) -> str:
        """Call GPT-4o for scoring."""
        resp = self._openai_client.chat.completions.create(
            model="gpt-4o",
            max_tokens=400,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    def score(self, resume_text: str, job: Job) -> Tuple[float, str, List[str]]:
        prompt = _build_prompt(resume_text, job)

        # Try primary backend
        for backend_name, call_fn in self._backends():
            try:
                text = call_fn(prompt)
                return _parse_response(text)
            except Exception as e:
                error_str = str(e).lower()
                is_credit_error = any(kw in error_str for kw in [
                    "credit", "insufficient", "rate_limit", "billing",
                    "quota", "payment", "overloaded"
                ])
                if is_credit_error:
                    log.warning("Reranker: %s failed (credits/rate-limit), trying fallback: %s",
                                backend_name, e)
                    continue  # try next backend
                else:
                    log.warning("Reranker: %s failed for job %s: %s", backend_name, job.id, e)
                    continue  # try next backend anyway

        log.error("Reranker: All backends failed for job %s", job.id)
        return 0.0, "rerank error: all backends failed", []

    def _backends(self):
        """Yield (name, callable) pairs in priority order."""
        if self._active_backend == "anthropic":
            if self._anthropic_client:
                yield "anthropic", self._score_anthropic
            if self._openai_client:
                yield "openai", self._score_openai
        else:
            if self._openai_client:
                yield "openai", self._score_openai
            if self._anthropic_client:
                yield "anthropic", self._score_anthropic
