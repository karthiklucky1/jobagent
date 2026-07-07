"""Rejection Autopsy orchestrator — composes the pieces into one result.

Pipeline:  X-ray "who actually holds this role" (linkedin_xray.find_champions)
        -> build_role_bar (the hidden bar)
        -> classify_door (right vs wrong door + reason + right door)

`finder` is dependency-injected so this is fully unit-testable without SerpAPI /
network. In production it defaults to the real X-ray. This function is ADDITIVE:
it returns a diagnosis dict and never hides/filters jobs.
"""
import logging
from typing import Callable, Optional

from app.intelligence.door_match import CandidateProfile, classify_door
from app.intelligence.role_bar import build_role_bar

log = logging.getLogger(__name__)

AXIS_REFINER_SYSTEM = """You are an expert AI recruiter and career coach analyzing a job and its team composition.
Your task is to classify the "observed hidden bar" of a role into one of four axes:
1. "applied": Standard software engineering, development, or machine learning engineering roles focused on shipping production code and building features.
2. "research": Advanced research or science roles focusing on publications (NeurIPS, ICLR, CVPR), core model training, and academic/scientific foundations.
3. "enterprise": Heavy corporate hierarchy, financial services, banks, insurance, or highly bureaucratic/regulated environments with slow processes and specific corporate titles.
4. "elite-outlier": Extreme high-compensation, early founding engineer roles, or top 1% talent demands (e.g., Olympiad champions, elite research labs).

Analyze the job description and the matched LinkedIn professionals (winners) who hold this role or similar roles at the company. Decide which axis is the most accurate representation of the role's actual bar.

Output ONLY a JSON object with:
{
  "axis": "applied" | "research" | "enterprise" | "elite-outlier",
  "reasoning": "A concise 1-sentence explanation of why this axis was chosen."
}"""


def refine_axis_with_llm(role_title: str, jd_text: str, winners: list[dict], current_axis: str) -> str:
    """Refine the deterministic axis classification using an LLM (Claude or OpenAI).
    
    Degrades gracefully to current_axis if API keys are missing or the call fails.
    """
    from app.config import settings
    import json

    # Check if we have API keys
    anthropic_api_key = settings.anthropic_api_key
    openai_api_key = settings.openai_api_key
    if not (anthropic_api_key or openai_api_key):
        return current_axis

    # Format winner profiles for the prompt
    winners_text = ""
    if winners:
        winners_text = "\n".join([
            f"- {w.get('name', 'Unknown')}: {w.get('headline', 'No headline')}"
            for w in winners[:10]
        ])
    else:
        winners_text = "(No matched LinkedIn winner profiles found; analyze JD text alone)"

    prompt = f"""### JOB INFO
Title: {role_title}
Deterministic Axis Suggestion: {current_axis}

### JOB DESCRIPTION (Truncated)
{jd_text[:4000]}

### MATCHED PROFESSIONALS (WINNERS)
{winners_text}

Analyze the job info and matched professionals to determine if the axis should be refined/overridden.
Return the refined axis in the requested JSON format."""

    raw_response = None
    try:
        if anthropic_api_key:
            from anthropic import Anthropic
            client = Anthropic(api_key=anthropic_api_key)
            resp = client.messages.create(
                model=settings.scoring_model,
                max_tokens=250,
                system=[{"type": "text", "text": AXIS_REFINER_SYSTEM}],
                messages=[{"role": "user", "content": prompt}],
            )
            raw_response = resp.content[0].text
        elif openai_api_key:
            from openai import OpenAI
            client = OpenAI(api_key=openai_api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=250,
                messages=[
                    {"role": "system", "content": AXIS_REFINER_SYSTEM},
                    {"role": "user", "content": prompt}
                ],
            )
            raw_response = resp.choices[0].message.content
    except Exception as e:
        log.warning("LLM axis refinement call failed: %s", e)
        return current_axis

    if not raw_response:
        return current_axis

    try:
        text = raw_response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())
        refined_axis = data.get("axis", "").strip().lower()
        if refined_axis in ("applied", "research", "enterprise", "elite-outlier"):
            log.info("LLM refined axis from %s -> %s (Reason: %s)", current_axis, refined_axis, data.get("reasoning"))
            return refined_axis
    except Exception as e:
        log.warning("Failed to parse LLM axis refinement response: %s | response: %r", e, raw_response)

    return current_axis


def _default_finder(company: str, role: str) -> dict:
    from app.intelligence.linkedin_xray import find_champions
    return find_champions(company, role)


def run_autopsy(company: str, role: str, jd_text: str,
                candidate: CandidateProfile,
                finder: Optional[Callable[[str, str], dict]] = None) -> dict:
    finder = finder or _default_finder
    res = finder(company, role) or {}
    winners = res.get("people", []) if res.get("ok") else []
    n = len(winners)

    bar = build_role_bar(role, jd_text, winners)
    
    # LLM axis refinement
    refined = refine_axis_with_llm(role, jd_text, winners, bar.axis)
    bar.axis = refined

    verdict = classify_door(
        candidate, bar, winners_n=n,
        data_quality="rich" if n >= 3 else "thin",
    )
    return {
        "ok": True,
        "company": company,
        "role": role,
        "winners_n": n,
        "winners": winners[:5],
        "data_ok": bool(res.get("ok")),
        "data_note": res.get("note") or res.get("reason"),
        "bar": {
            "axis": bar.axis, "years": bar.years, "domain": bar.domain,
            "onsite": bar.onsite, "onsite_metro": bar.onsite_metro,
            "pedigree": bar.pedigree, "elite_outlier": bar.elite_outlier,
        },
        "wrong_door": verdict.wrong_door,
        "top_reason": verdict.top_reason,
        "right_door": verdict.right_door,
        "confidence": verdict.confidence,
        "findings": [{"dim": f.dim, "status": f.status, "note": f.note}
                     for f in verdict.findings],
    }
