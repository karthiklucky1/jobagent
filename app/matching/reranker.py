"""Stage-2 reranker: LLM scores top-K from FAISS with reasoning.

Tries Claude first (Anthropic), falls back to GPT-4o (OpenAI) if Claude
is unavailable (e.g. credits depleted). Both use the same system prompt
and expect the same JSON output format.
"""
from __future__ import annotations

import json
import logging
import random
import time
from typing import List, Optional, Tuple

from app.config import settings
from app.db.models import Job
from app.qa_store.resolver import QAResolver
from app.matching.filters.rule_filter import RuleFilter

log = logging.getLogger(__name__)

# Initialize canonical QA Resolver
qa_resolver = QAResolver()

# The JSON contract every backend must return — shared by both the per-user and
# the legacy rubric so the parser can rely on it.
_JSON_CONTRACT = """Return a single JSON object — no prose, no markdown:
{
  "score": <0-100 integer overall fit>,
  "reason": "<one sentence, max 25 words, plain English>",
  "concerns": ["<concern 1>", "<concern 2>"],
  "breakdown": {
    "skills":     {"score": <0-100>, "note": "<short why>"},
    "experience": {"score": <0-100>, "note": "<short why>"},
    "location":   {"score": <0-100>, "note": "<short why>"},
    "work_auth":  {"score": <0-100>, "note": "<short why>"}
  }
}
The overall "score" should roughly reflect the four breakdown factors, but a hard
blocker (wrong country, explicit no-sponsorship, impossible seniority gap) caps the
overall score low regardless of the other factors."""

_SCORE_BANDS = """Score bands:
- 85-100: Strong match — core skills, experience level, location, and authorization all align.
- 70-84: Good match with one minor gap.
- 60-69: Reasonable stretch — core skills overlap but there's a seniority or domain gap.
- 40-59: Weak — notable gaps in skills or experience.
- 0-39: Wrong role or a hard blocker (different country, explicit no-sponsorship, unrelated field)."""


def _profile_has_signal(profile) -> bool:
    """True when the user's profile carries enough info to drive a tailored rubric."""
    if profile is None:
        return False
    try:
        return bool(
            (getattr(profile, "key_skills", "") or "").strip()
            or (getattr(profile, "target_roles", "") or "").strip()
            or int(getattr(profile, "years_experience", 0) or 0) > 0
            or (getattr(profile, "current_title", "") or "").strip()
        )
    except Exception:
        return False


def _profile_system_prompt(profile) -> str:
    """Per-user scoring rubric built from the signed-in user's own profile."""
    yoe = int(getattr(profile, "years_experience", 0) or 0)
    skills = (getattr(profile, "key_skills", "") or "").strip() or "not specified"
    roles = (getattr(profile, "target_roles", "") or "").strip() \
        or (getattr(profile, "current_title", "") or "").strip() or "not specified"
    summary = (getattr(profile, "professional_summary", "") or "").strip()
    country = (getattr(profile, "preferred_country", "") or "United States").strip()
    remote_ok = bool(getattr(profile, "remote_ok", True))
    needs_sponsor = bool(getattr(profile, "requires_sponsorship", False))
    work_auth = (getattr(profile, "work_authorization", "")
                 or getattr(profile, "work_auth_status", "")
                 or getattr(profile, "visa_status", "")).strip() or "not specified"

    # Experience guidance is RELATIVE to this candidate's actual YoE.
    exp_rules = f"""- EXPERIENCE (candidate has ~{yoe} years):
  * JD requires roughly within {yoe}±1 years: score experience high (75-100).
  * JD requires up to ~{yoe + 2} years: moderate stretch (50-70).
  * JD requires more than ~{yoe + 3} years (or Staff/Principal/Distinguished with senior reqs): hard gap, experience ≤ 25.
  * JD asks for less experience than the candidate, or is silent on years: score experience normally (not a penalty)."""

    if needs_sponsor:
        auth_rule = (f"- WORK AUTHORIZATION: candidate is '{work_auth}' and WILL need visa sponsorship. "
                     f"Set work_auth low (0-15) ONLY if the posting explicitly says 'no sponsorship', "
                     f"'US citizens/permanent residents only', or requires an active security clearance. "
                     f"If the posting is silent on sponsorship, assume it is possible and score work_auth high.")
    else:
        auth_rule = (f"- WORK AUTHORIZATION: candidate is '{work_auth}' and does NOT need sponsorship. "
                     f"work_auth should be high unless the role requires a clearance/citizenship the candidate lacks.")

    loc_rule = (f"- LOCATION & COUNTRY: the candidate wants jobs in {country}"
                f"{' plus fully-remote roles' if remote_ok else ''}. "
                f"If the job is located in a DIFFERENT country than {country}"
                f"{' and is not remote' if remote_ok else ''}, set location 0-15 (hard blocker). "
                f"In-country roles score location high; for remote roles location is high.")

    return f"""You evaluate how well a candidate fits a job. {_JSON_CONTRACT}

{_SCORE_BANDS}

Candidate profile:
- Target roles: {roles}
- Core skills: {skills}
- Experience: ~{yoe} years.{(' ' + summary) if summary else ''}
{exp_rules}
{auth_rule}
{loc_rule}
- Judge the SKILLS factor on overlap between the candidate's skills/target roles and the job's requirements.

Be fair and realistic — do not invent disqualifications. Return JSON only."""


def _legacy_system_prompt() -> str:
    """Generic, candidate-neutral fallback rubric — used only when a user has no
    profile signal yet. Judges fit purely from the résumé text (passed in the
    user prompt), with no hardcoded personal assumptions, so it is safe in a
    multi-tenant setting (no other user's defaults leak in)."""
    return f"""You evaluate how well a candidate's résumé fits a job posting. {_JSON_CONTRACT}

{_SCORE_BANDS}

Scoring guidance (judge everything from the résumé provided — do not assume facts not present in it):
- SKILLS: score on overlap between the résumé's skills/experience and the job's stated requirements.
- EXPERIENCE: estimate the candidate's years from the résumé. If the JD requires noticeably more
  years than the candidate appears to have (roughly 4+ years beyond), lower the experience score; if
  the JD is silent on years or asks for less, score normally. Do not invent a seniority gap.
- WORK AUTHORIZATION: score work_auth low (0-15) ONLY if the posting explicitly states "no sponsorship",
  "US citizens/permanent residents only", or requires an active security clearance. If the posting is
  silent on sponsorship, assume it is possible and score work_auth high.
- LOCATION: prefer US-based or fully-remote roles. Score location low only for clearly non-remote roles
  located outside the candidate's region as indicated by the résumé.

Be fair and realistic — do not invent disqualifications. Return JSON only. No prose."""


def _get_system_prompt(profile=None) -> str:
    """Build the scoring rubric. Prefers the signed-in user's own profile;
    falls back to the bundled QA-resolver defaults when no profile signal exists."""
    if _profile_has_signal(profile):
        return _profile_system_prompt(profile)
    return _legacy_system_prompt()


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


def _clean_breakdown(raw, overall: float) -> dict:
    """Normalize the per-factor breakdown; synthesize a minimal one if absent."""
    factors = ("skills", "experience", "location", "work_auth")
    out: dict = {}
    raw = raw if isinstance(raw, dict) else {}
    for f in factors:
        item = raw.get(f) or {}
        if isinstance(item, dict):
            try:
                s = max(0.0, min(100.0, float(item.get("score", overall))))
            except (TypeError, ValueError):
                s = overall
            note = str(item.get("note", "") or "")
        else:
            s, note = overall, ""
        out[f] = {"score": round(s), "note": note[:160]}
    return out


def _parse_response(text: str) -> Tuple[float, str, List[str], dict]:
    """Parse LLM JSON response, tolerating markdown fences.

    Returns (score, reason, concerns, breakdown)."""
    text = text.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Reranker LLM returned invalid JSON: {e}") from e
    score = max(0.0, min(100.0, float(data["score"])))
    breakdown = _clean_breakdown(data.get("breakdown"), score)
    return score, data.get("reason", ""), data.get("concerns", []), breakdown


class Reranker:
    def __init__(self, profile=None):
        self._profile = profile
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
            model=settings.scoring_model,
            max_tokens=600,
            system=[{"type": "text", "text": _get_system_prompt(self._profile), "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    def _score_openai(self, prompt: str) -> str:
        """Call GPT-4o-mini for scoring."""
        resp = self._openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=600,
            messages=[
                {"role": "system", "content": _get_system_prompt(self._profile)},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    def _pre_filter_job(self, job: Job) -> Optional[Tuple[float, str, List[str], dict]]:
        """Apply rule-based pre-filters to catch obvious misfits without calling the LLM."""
        res = RuleFilter(profile=self._profile).filter(job)
        if not res.passed:
            score = float(res.score_override or 10.0)
            return score, res.reason, [res.reason], _clean_breakdown(None, score)
        return None

    def score(self, resume_text: str, job: Job) -> Tuple[float, str, List[str], dict]:
        # Run pre-filters first to avoid LLM calls on misfits
        pre_filtered = self._pre_filter_job(job)
        if pre_filtered is not None:
            log.info("Reranker: Pre-filtered job %s - %s", job.title, pre_filtered[1])
            return pre_filtered

        prompt = _build_prompt(resume_text, job)

        # Try each backend; retry rate-limit/overloaded errors with exponential
        # backoff + jitter before falling through. CRITICAL: on total failure we
        # RAISE (not return 0.0) so the caller leaves the job unscored and retries
        # it on a later run — a 429 must never become a silent score-0 drop that
        # biases the shortlist.
        max_retries = max(1, settings.llm_rerank_max_retries)
        for backend_name, call_fn in self._backends():
            for attempt in range(max_retries):
                try:
                    text = call_fn(prompt)
                    return _parse_response(text)
                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit = any(kw in error_str for kw in [
                        "rate_limit", "overloaded", "429", "529", "timeout",
                    ])
                    is_credit_error = any(kw in error_str for kw in [
                        "credit", "insufficient", "billing", "quota", "payment",
                    ])
                    if is_rate_limit and attempt < max_retries - 1:
                        # Exponential backoff: 1s, 2s, 4s, 8s (±20% jitter)
                        delay = (2 ** attempt) * (0.8 + 0.4 * random.random())
                        log.warning("Reranker: %s rate-limited (attempt %d/%d), retrying in %.1fs: %s",
                                    backend_name, attempt + 1, max_retries, delay, e)
                        time.sleep(delay)
                        continue
                    if is_credit_error:
                        log.warning("Reranker: %s out of credits/quota — trying fallback backend: %s",
                                    backend_name, e)
                        break  # don't burn retries; move to next backend
                    log.warning("Reranker: %s failed for job %s: %s", backend_name, job.id, e)
                    break  # try next backend

        log.error("Reranker: All backends/retries exhausted for job %s — leaving unscored", job.id)
        raise RuntimeError(f"rerank failed for job {job.id}: all backends exhausted")

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
