"""Stage-2 reranker: LLM scores top-K from FAISS with reasoning.

Tries Claude first (Anthropic), falls back to gpt-4o-mini (OpenAI) if Claude
is unavailable (e.g. credits depleted). Both use the same system prompt
and expect the same JSON output format.
"""
from __future__ import annotations

import json
import logging
import random
import threading
import time
from datetime import datetime
from typing import List, Optional, Tuple

from app.config import settings
from app.db.models import Job
from app.qa_store.resolver import QAResolver
from app.matching.filters.rule_filter import RuleFilter

log = logging.getLogger(__name__)

# ── Provider circuit breaker ──────────────────────────────────────────────────
# When a provider starts returning credit/quota errors, every scoring lane used
# to keep re-hitting it 4x per job per 90s cycle, forever (the Jul 15 log storm).
# Instead: mark the provider DOWN for a cooldown and skip it — jobs stay Queued
# and cost nothing until a provider is back.
_provider_down_until: dict = {}
_breaker_lock = threading.Lock()


# Errors that mean "this provider has no capacity left for a while" — credit /
# billing exhaustion, AND daily-quota rate limits ("requests per day (RPD)"),
# which a retry cannot fix until the quota window resets. Plain per-minute 429s
# are NOT here — those are transient and handled by retry/fallback.
_EXHAUSTION_MARKERS = ("credit", "insufficient", "billing", "quota", "payment",
                       "per day", "rpd", "daily limit")


def _is_exhaustion_error(error_str: str) -> bool:
    return any(kw in error_str for kw in _EXHAUSTION_MARKERS)


def _mark_provider_down(name: str) -> None:
    mins = settings.llm_provider_cooldown_minutes
    if mins <= 0:
        return
    with _breaker_lock:
        _provider_down_until[name] = time.time() + mins * 60
    log.warning("Reranker: provider %s marked DOWN (credit/quota) — cooling off %d min", name, mins)


def provider_available(name: str) -> bool:
    with _breaker_lock:
        return time.time() >= _provider_down_until.get(name, 0.0)


def any_provider_available() -> bool:
    """For the lanes: is at least one final-score provider not cooling down?"""
    return provider_available("anthropic") or provider_available("openai")


# ── Spend guards (daily ceiling + hourly smoothing) ───────────────────────────
# Hard ceilings on Tier-2 (authoritative) LLM scores across every lane. The
# scoring queue is unbounded (discovery keeps producing); the DAILY cap bounds
# what a runaway queue can COST, and the HOURLY cap bounds how FAST it burns —
# without it a big backlog drained at ~2K finals/hour and ate a day's budget in
# under an hour (Jul 15 evening). Process-local: resets on restart, which only
# ever errs toward spending slightly more — acceptable for a safety net.
_daily_finals = {"day": "", "count": 0}
_hourly_finals = {"hour": "", "count": 0}
_budget_lock = threading.Lock()


def _register_final_call() -> None:
    now = datetime.utcnow()
    today, hour = now.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d %H")
    with _budget_lock:
        if _daily_finals["day"] != today:
            _daily_finals["day"] = today
            _daily_finals["count"] = 0
        _daily_finals["count"] += 1
        if _hourly_finals["hour"] != hour:
            _hourly_finals["hour"] = hour
            _hourly_finals["count"] = 0
        _hourly_finals["count"] += 1


def llm_budget_exhausted() -> bool:
    now = datetime.utcnow()
    today, hour = now.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d %H")
    day_cap, hour_cap = settings.llm_daily_final_cap, settings.llm_hourly_final_cap
    with _budget_lock:
        if day_cap > 0 and _daily_finals["day"] == today and _daily_finals["count"] >= day_cap:
            return True
        if hour_cap > 0 and _hourly_finals["hour"] == hour and _hourly_finals["count"] >= hour_cap:
            return True
    return False


# ── Cache telemetry ───────────────────────────────────────────────────────────
# Aggregated Claude usage logged every N finals, so prod logs answer "is prompt
# caching actually engaging?" without console access. cache_read ≈ 0.1x price;
# a healthy steady state shows read ≫ input.
_usage_totals = {"calls": 0, "input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
_USAGE_LOG_EVERY = 25


def _track_anthropic_usage(resp) -> None:
    try:
        u = resp.usage
        with _budget_lock:
            _usage_totals["calls"] += 1
            _usage_totals["input"] += int(getattr(u, "input_tokens", 0) or 0)
            _usage_totals["cache_read"] += int(getattr(u, "cache_read_input_tokens", 0) or 0)
            _usage_totals["cache_write"] += int(getattr(u, "cache_creation_input_tokens", 0) or 0)
            _usage_totals["output"] += int(getattr(u, "output_tokens", 0) or 0)
            if _usage_totals["calls"] % _USAGE_LOG_EVERY:
                return
            t = dict(_usage_totals)
        seen = t["input"] + t["cache_read"] + t["cache_write"]
        ratio = (100.0 * t["cache_read"] / seen) if seen else 0.0
        log.info("Claude usage (last %d finals cumulative): uncached_in=%d cache_read=%d "
                 "cache_write=%d out=%d — cache-read share %.0f%%",
                 t["calls"], t["input"], t["cache_read"], t["cache_write"], t["output"], ratio)
    except Exception:
        pass

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

_SCORE_BANDS = """Score bands (use the FULL 0-100 range — do not cluster scores in the middle):
- 90-100: Excellent match — the candidate should be a top applicant; core skills and experience clearly align with no blockers.
- 75-89: Strong match — solid alignment with at most one minor gap.
- 60-74: Good match — real skills overlap but a visible stretch (seniority or domain gap).
- 40-59: Weak — notable gaps in skills or experience.
- 0-39: Wrong role or a hard blocker (different country, explicit no-sponsorship, unrelated field).
Calibration: when the candidate's core skills cover the job's main requirements and there is no
hard blocker, the overall score should land at 75 or higher — reserve the 50s for genuine
stretches, not for good fits with ordinary uncertainty."""


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
                f"{' plus fully-remote roles open to ' + country + ' (or truly global remote)' if remote_ok else ''}. "
                f"If the job is located in a DIFFERENT country than {country} — including a REMOTE role "
                f"anchored to another country or region (e.g. 'Remote, EU only'), which still requires "
                f"work authorization there — set location 0-15 (hard blocker). "
                f"In-country roles score location high; same-country or global remote scores location high.")

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


# ── Tier-1 cheap prescore (cascade) ──────────────────────────────────────────
# A fast, cheap first pass that decides which candidates are worth the
# authoritative (Claude) score. It only needs a rough number + one-line reason,
# so the prompt and output are deliberately tiny (cheap + high-throughput). It is
# ROLE-AWARE: the rubric is built from THIS user's target roles / skills /
# country, so an off-role posting scores low and drains out of the backlog.
_PRESCORE_CONTRACT = (
    'Return ONLY a JSON object, no prose, no markdown: '
    '{"score": <0-100 integer overall fit>, "reason": "<max 15 words>"}'
)


def _prescore_system_prompt(profile=None) -> str:
    if _profile_has_signal(profile):
        yoe = int(getattr(profile, "years_experience", 0) or 0)
        skills = (getattr(profile, "key_skills", "") or "").strip() or "not specified"
        roles = (getattr(profile, "target_roles", "") or "").strip() \
            or (getattr(profile, "current_title", "") or "").strip() or "not specified"
        country = (getattr(profile, "preferred_country", "") or "United States").strip()
        needs_sponsor = bool(getattr(profile, "requires_sponsorship", False))
        sponsor = (" The candidate needs visa sponsorship — score low only if the posting "
                   "explicitly refuses sponsorship or requires citizenship/clearance."
                   if needs_sponsor else "")
        return (
            f"You are a fast first-pass job-fit filter. {_PRESCORE_CONTRACT}\n"
            f"Candidate targets: {roles}. Core skills: {skills}. ~{yoe} years. "
            f"Wants jobs in {country} (or fully-remote roles open to {country}).{sponsor}\n"
            "Score 0-100 how well THIS candidate fits the job. A hard blocker (onsite in a "
            "different country, explicit no-sponsorship when needed, or an unrelated field) "
            "scores 0-30. Genuine skill/role overlap with no blocker scores 60+. When unsure, "
            "lean HIGHER — a stronger model re-checks every promising job, so only clear "
            "misfits should score low."
        )
    return (
        f"You are a fast first-pass job-fit filter. {_PRESCORE_CONTRACT}\n"
        "Judge fit purely from the résumé provided (do not assume facts not in it). An "
        "unrelated field or a hard blocker scores 0-30; genuine skill overlap with no blocker "
        "scores 60+. When unsure, lean HIGHER — a stronger model re-checks promising jobs."
    )


def _build_prescore_prompt(resume_text: str, job: Job) -> str:
    """Compact prompt for the cheap tier — short résumé + short JD keep it fast.

    The résumé leads the user message, so system+résumé form a static prefix
    across every prescore for the same user. The slice is sized to push that
    prefix past OpenAI's 1,024-token automatic-caching minimum (~200 system +
    ~1,000 résumé) — the old 2,500-char slice left it at ~825 tokens, so no
    prescore ever cached and the résumé was re-billed at full price per job."""
    return f"""<resume>
{resume_text[:4000]}
</resume>
<job>
Title: {job.title}
Company: {job.company}
Location: {job.location}
Remote: {job.remote}
Description:
{(job.description or '')[:1800]}
</job>

Return the JSON object."""


def _parse_prescore(text: str) -> Tuple[float, str]:
    """Parse the tiny Tier-1 response into (score 0-100, reason)."""
    text = text.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(text)
    score = max(0.0, min(100.0, float(data["score"])))
    return score, str(data.get("reason", "") or "")[:160]


def _sponsor_note(job: Job, profile) -> str:
    """When the candidate needs sponsorship AND the employer has a strong public
    H-1B filing record, tell the scorer explicitly: OPT is not a blocker here."""
    try:
        if not bool(getattr(profile, "requires_sponsorship", False)):
            return ""
        from app.intelligence.h1b_data import lookup as _h1b_lookup
        rec = _h1b_lookup(job.company or "")
        if rec and (rec.get("approvals", 0) or 0) >= 50:
            return ("\nNOTE: This employer is a VERIFIED visa sponsor with a strong public "
                    "USCIS H-1B filing record. Do NOT penalize the candidate's sponsorship "
                    "need for this job — score work_auth high (75+) unless the posting "
                    "explicitly refuses sponsorship.")
    except Exception:
        pass
    return ""


# The scoring prompt is split into a USER-STABLE half (résumé + preference
# feedback) and a PER-JOB half (the posting). The stable half is sent as a cached
# block, so scoring the next job for the same user re-reads it from cache instead
# of re-billing the full résumé every time. A user's batch of hundreds of jobs
# shares one résumé — this is the single biggest token/cost lever in scoring.
#
# CRITICAL SIZE CONSTRAINT: Anthropic silently ignores cache_control when the
# cumulative prefix is under the model's minimum — 4,096 tokens on Haiku 4.5.
# The old resume[:6000] slice put the rubric+résumé prefix at ~2.5K tokens, so
# NOTHING ever cached and every final re-billed the full résumé at full price.
# Fix: use more of the résumé (it can only improve grounding) and, when the
# block is still short, pad it with a labeled VERBATIM repetition of the résumé
# — no new information enters the prompt; the pad's only job is to push the
# prefix over the cache minimum so re-reads bill at ~0.1x instead of 1x.
_RESUME_SLICE_CHARS = 16000       # was 6000
_CACHE_MIN_BLOCK_CHARS = 15500    # + rubric (~2.5-3.5K chars) ≈ comfortably >4096 tokens
_CACHE_PAD_MAX_REPEATS = 5        # a tiny résumé isn't worth padding — skip (no cache, status quo)


def _resume_context_block(resume_text: str, feedback: str = "") -> str:
    """The user-stable half — identical across every job we score for this user,
    so it's the cacheable prefix. Deterministic for a given résumé+feedback."""
    fb = f"\n<user_feedback>\n{feedback}\n</user_feedback>" if feedback else ""
    body = resume_text[:_RESUME_SLICE_CHARS]
    block = f"<resume>\n{body}\n</resume>{fb}"
    short_by = _CACHE_MIN_BLOCK_CHARS - len(block)
    if short_by > 0 and body:
        repeats = -(-short_by // (len(body) + 1))  # ceil
        if repeats <= _CACHE_PAD_MAX_REPEATS:
            pad = "\n".join([body] * repeats)
            block += (
                "\n<resume_repeat>\nThe following is a verbatim repetition of the résumé "
                "above, included only for prompt-cache alignment. It contains no new "
                "information — read the résumé once and ignore the repetition.\n"
                f"{pad}\n</resume_repeat>"
            )
    return block


def _job_context_block(job: Job, profile=None) -> str:
    """The per-job half — changes every call, so it is NOT cached."""
    return f"""<job>
Title: {job.title}
Company: {job.company}
Location: {job.location}
Remote: {job.remote}
{_sponsor_note(job, profile)}
Description:
{(job.description or '')[:5000]}
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
    def __init__(self, profile=None, feedback: str = ""):
        self._profile = profile
        # Revealed-preference note from preference_learning — lets the LLM
        # calibrate fit to what this user actually dismisses/engages with.
        self._feedback = feedback or ""
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
                # Bound each request: the SDK default is a 10-MINUTE timeout with
                # internal retries, so one slow/overloaded call could freeze a
                # matching pass (up to llm_rerank_cap jobs) for many minutes while
                # it holds the discovery/matching lock — stalling ALL matching.
                self._anthropic_client = Anthropic(
                    api_key=settings.anthropic_api_key,
                    timeout=settings.llm_request_timeout,
                    max_retries=0,  # we do our own retry/backoff in score()
                )
                self._active_backend = "anthropic"
                log.info("Reranker: Anthropic (Claude) client initialized")
            except Exception as e:
                log.warning("Reranker: Failed to init Anthropic client: %s", e)

        # OpenAI fallback
        if settings.openai_api_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(
                    api_key=settings.openai_api_key,
                    timeout=settings.llm_request_timeout,
                    max_retries=0,  # we do our own retry/backoff in score()
                )
                if not self._active_backend:
                    self._active_backend = "openai"
                log.info("Reranker: OpenAI (gpt-4o-mini) client initialized as %s",
                         "primary" if self._active_backend == "openai" else "fallback")
            except Exception as e:
                log.warning("Reranker: Failed to init OpenAI client: %s", e)

        if not self._active_backend:
            log.error("Reranker: No LLM backend available! Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")

    def _score_anthropic(self, resume_block: str, job_block: str) -> str:
        """Call Claude for scoring. The rubric AND the résumé are cached system
        blocks, so scoring the next job for this user reads both from cache
        instead of re-sending the whole résumé each time."""
        resp = self._anthropic_client.messages.create(
            model=settings.scoring_model,
            max_tokens=600,
            system=[
                {"type": "text", "text": _get_system_prompt(self._profile),
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": resume_block,
                 "cache_control": {"type": "ephemeral"}},
            ],
            messages=[{"role": "user", "content": job_block}],
        )
        _track_anthropic_usage(resp)
        return resp.content[0].text

    def _score_openai(self, resume_block: str, job_block: str) -> str:
        """Call GPT-4o-mini for scoring (single-provider fallback path). The
        rubric+résumé go in the system message so OpenAI's automatic prefix
        caching can reuse them across the user's jobs."""
        resp = self._openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=600,
            messages=[
                {"role": "system", "content": _get_system_prompt(self._profile) + "\n\n" + resume_block},
                {"role": "user", "content": job_block},
            ],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    def _score_openai_final(self, resume_block: str, job_block: str) -> str:
        """Call the full GPT model (default gpt-4o) for an AUTHORITATIVE final
        score in dual-provider mode — same rubric as Claude, so the two are
        comparable. Used when the 60/40 router sends a job to OpenAI."""
        resp = self._openai_client.chat.completions.create(
            model=settings.dual_score_openai_model,
            max_tokens=600,
            messages=[
                {"role": "system", "content": _get_system_prompt(self._profile) + "\n\n" + resume_block},
                {"role": "user", "content": job_block},
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

    # ── Tier-1 cheap prescore (cascade) ──────────────────────────────────────
    def _prescore_openai(self, prompt: str) -> str:
        model = settings.prescore_model if not settings.prescore_model.startswith("claude") else "gpt-4o-mini"
        resp = self._openai_client.chat.completions.create(
            model=model,
            max_tokens=120,
            messages=[
                {"role": "system", "content": _prescore_system_prompt(self._profile)},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    def _prescore_anthropic(self, prompt: str) -> str:
        # If prescore_model is an Anthropic model use it, else the cheap Haiku scorer.
        model = settings.prescore_model if settings.prescore_model.startswith("claude") else settings.scoring_model
        resp = self._anthropic_client.messages.create(
            model=model,
            max_tokens=120,
            system=[{"type": "text", "text": _prescore_system_prompt(self._profile),
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )
        # An Anthropic prescore is ~5x a gpt-4o-mini one and happens per queued
        # job, so it draws from the SAME hourly/daily budget as finals — the
        # caps bound total Anthropic spend, whatever the tier mix. (Jul 15
        # evening: OpenAI hit its daily quota, prescores silently fell to Haiku
        # uncapped, and Tier-1 quietly outspent the capped finals.)
        _register_final_call()
        return resp.content[0].text

    def has_prescore_backend(self) -> bool:
        """True when at least one LLM client exists to run the cheap Tier-1 pass."""
        return bool(self._openai_client or self._anthropic_client)

    def _prescore_backends(self):
        """Yield (name, callable) for Tier-1, preferring the configured provider.
        Providers cooling down after credit/quota errors are skipped."""
        prefer_openai = (settings.prescore_provider or "openai").lower() == "openai"
        openai_pair = ("openai", self._prescore_openai) if self._openai_client else None
        anthropic_pair = ("anthropic", self._prescore_anthropic) if self._anthropic_client else None
        order = [openai_pair, anthropic_pair] if prefer_openai else [anthropic_pair, openai_pair]
        for pair in order:
            if pair and provider_available(pair[0]):
                yield pair

    def prescore(self, resume_text: str, job: Job) -> Optional[Tuple[float, str]]:
        """Tier-1 cheap bulk score. Returns (score 0-100, reason), or None on
        failure. IMPORTANT: None means "couldn't decide" — the caller must NOT
        drop the job on None; it should advance it to Tier-2 (fail-open) so a
        cheap-model hiccup never silently buries a good match."""
        # A hard rule rejection is authoritative and saves even the cheap call.
        pre = self._pre_filter_job(job)
        if pre is not None:
            return pre[0], pre[1]
        prompt = _build_prescore_prompt(resume_text, job)
        for name, call_fn in self._prescore_backends():
            # Anthropic Tier-1 draws from the finals budget (see
            # _prescore_anthropic) — past the cap, don't call it. Fail-open:
            # returning None advances the job; score() enforces the same budget.
            if name == "anthropic" and llm_budget_exhausted():
                continue
            try:
                return _parse_prescore(call_fn(prompt))
            except Exception as e:
                if _is_exhaustion_error(str(e).lower()):
                    _mark_provider_down(name)
                log.debug("Prescore: %s failed for job %s: %s", name, job.id, e)
                continue
        return None

    def has_dual(self) -> bool:
        """True when BOTH providers are available, so the final score can be
        split across them (Option A). With one provider, routing is a no-op."""
        return bool(self._anthropic_client and self._openai_client)

    def _calibrate(self, backend_name: str, result):
        """In dual mode, nudge GPT's scale onto Claude's so the shortlist bar is
        fair across providers. No-op for Claude and when no offset is set."""
        off = settings.dual_score_openai_offset
        if settings.dual_score_enabled and backend_name == "openai" and off:
            score, reason, concerns, breakdown = result
            score = max(0.0, min(100.0, score + off))
            return score, reason, concerns, breakdown
        return result

    def score(self, resume_text: str, job: Job,
              provider: Optional[str] = None) -> Tuple[float, str, List[str], dict]:
        """Authoritative final score. ``provider`` ('anthropic'|'openai') routes
        the FIRST attempt to that backend (Option A's 60/40 split); the other
        provider stays as the fallback, so a rate-limited/errored primary still
        gets the job scored. None = default priority order."""
        # Run pre-filters first to avoid LLM calls on misfits
        pre_filtered = self._pre_filter_job(job)
        if pre_filtered is not None:
            log.info("Reranker: Pre-filtered job %s - %s", job.title, pre_filtered[1])
            return pre_filtered

        # Daily spend guard — past the cap, jobs stay Queued (raise = unscored),
        # they are NOT silently mis-scored. Checked before any API call.
        if llm_budget_exhausted():
            raise RuntimeError(
                f"daily LLM final-score budget reached ({settings.llm_daily_final_cap}) "
                f"— job {job.id} left unscored for tomorrow")

        resume_block = _resume_context_block(resume_text, self._feedback)
        job_block = _job_context_block(job, self._profile)

        # Try each backend; retry rate-limit/overloaded errors with exponential
        # backoff + jitter before falling through. CRITICAL: on total failure we
        # RAISE (not return 0.0) so the caller leaves the job unscored and retries
        # it on a later run — a 429 must never become a silent score-0 drop that
        # biases the shortlist. Providers cooling down after credit/quota errors
        # (circuit breaker) are skipped entirely — no API call, no retry storm.
        max_retries = max(1, settings.llm_rerank_max_retries)
        backends = [(n, fn) for n, fn in self._score_backends(provider) if provider_available(n)]
        if not backends:
            raise RuntimeError(f"rerank skipped for job {job.id}: all providers cooling down")
        for backend_name, call_fn in backends:
            for attempt in range(max_retries):
                try:
                    text = call_fn(resume_block, job_block)
                    _register_final_call()
                    return self._calibrate(backend_name, _parse_response(text))
                except Exception as e:
                    error_str = str(e).lower()
                    is_credit_error = _is_exhaustion_error(error_str)
                    is_rate_limit = not is_credit_error and any(kw in error_str for kw in [
                        "rate_limit", "overloaded", "429", "529",
                        "timeout", "timed out", "timedout",  # SDK APITimeoutError says "Request timed out"
                    ])
                    if is_rate_limit and attempt < max_retries - 1:
                        # Exponential backoff: 1s, 2s, 4s, 8s (±20% jitter)
                        delay = (2 ** attempt) * (0.8 + 0.4 * random.random())
                        log.warning("Reranker: %s rate-limited (attempt %d/%d), retrying in %.1fs: %s",
                                    backend_name, attempt + 1, max_retries, delay, e)
                        time.sleep(delay)
                        continue
                    if is_credit_error:
                        _mark_provider_down(backend_name)  # circuit breaker: skip it for a cooldown
                        log.warning("Reranker: %s out of credits/quota — trying fallback backend: %s",
                                    backend_name, e)
                        break  # don't burn retries; move to next backend
                    log.warning("Reranker: %s failed for job %s: %s", backend_name, job.id, e)
                    break  # try next backend

        log.error("Reranker: All backends/retries exhausted for job %s — leaving unscored", job.id)
        raise RuntimeError(f"rerank failed for job {job.id}: all backends exhausted")

    def _score_backends(self, provider: Optional[str] = None):
        """Ordered (name, callable) backends for the FINAL score.

        - ``provider`` ('anthropic'|'openai'), when set and available, is tried
          first; the other provider remains the fallback.
        - In dual mode the OpenAI final scorer is the FULL model
          (settings.dual_score_openai_model) so it's comparable to Claude;
          otherwise it's the cheap gpt-4o-mini fallback.
        - With no ``provider`` the historical priority order is preserved.
        """
        anth = ("anthropic", self._score_anthropic) if self._anthropic_client else None
        oai_fn = self._score_openai_final if settings.dual_score_enabled else self._score_openai
        oai = ("openai", oai_fn) if self._openai_client else None

        if provider == "openai":
            order = [oai, anth]
        elif provider == "anthropic":
            order = [anth, oai]
        elif self._active_backend == "anthropic":
            order = [anth, oai]
        else:
            order = [oai, anth]
        return [p for p in order if p]
