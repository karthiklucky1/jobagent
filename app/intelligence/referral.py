"""Referral & outreach co-pilot — DRAFT ONLY.

Agencies get people hired by going through the back door (referrals + direct
outreach), not the ATS. This module DRAFTS those messages for the user to send
themselves from their own account. It never connects to LinkedIn, never scrapes
third parties, and never auto-sends — that keeps the user's accounts safe and
the whole thing within ToS / the law.

Three drafts per job:
  1. referral_request  — ask a connection at the company to refer you
  2. hiring_manager    — a concise value pitch to the hiring manager
  3. visa_alumni       — (sponsorship-needing users) a warm note to someone who
                         went through the visa process at that company
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def _fallback_drafts(name: str, title: str, company: str, role: str,
                     skills: str, selling: str, needs_sponsorship: bool) -> list[dict]:
    first = (name or "there").split(" ")[0] if name else ""
    me = name or "I"
    skills_short = ", ".join([s.strip() for s in (skills or "").split(",") if s.strip()][:3])
    skill_line = f" My background is in {skills_short}." if skills_short else ""
    drafts = [
        {
            "type": "referral_request",
            "label": "Referral request",
            "channel": "LinkedIn / email to a connection at the company",
            "body": (
                f"Hi {{name}}, hope you're well! I noticed {company} is hiring a "
                f"{role} and it looks like a strong fit for my background.{skill_line} "
                f"Would you be open to referring me through your employee referral "
                f"link? Happy to send my resume and a few bullet points to make it "
                f"easy. Thanks so much either way!"
            ),
        },
        {
            "type": "hiring_manager",
            "label": "Hiring-manager note",
            "channel": "LinkedIn DM / email to the hiring manager",
            "body": (
                f"Hi {{name}}, I'm reaching out about the {role} role at {company}. "
                f"{('As a ' + title + ', ') if title else ''}I think I'd ramp fast —"
                f"{(' ' + skills_short + ' are right in my wheelhouse.') if skills_short else ''} "
                f"I've applied through your site; I'd love 10 minutes to share why I'm "
                f"a strong fit. Would that be welcome?"
            ),
        },
    ]
    if needs_sponsorship:
        drafts.append({
            "type": "visa_alumni",
            "label": "Visa-alumni connection",
            "channel": "LinkedIn connection request to a fellow visa-process alum",
            "body": (
                f"Hi {{name}}, I came across your profile and noticed you navigated "
                f"the visa journey while building your career at {company}. I'm "
                f"exploring the {role} role there and would love to ask one quick "
                f"question about how {company} approaches work authorization. "
                f"{selling or ''} Thanks for considering!"
            ).strip(),
        })
    return drafts


def get_company_github_repos(company: str) -> list[str]:
    """Search for the company's GitHub organization and return up to 3 popular repos."""
    from app.config import settings
    import httpx
    import re

    token = settings.github_token
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    slug = None
    try:
        q = re.sub(r"[^a-z0-9 ]", "", company.lower()).strip()
        r = httpx.get(
            "https://api.github.com/search/users",
            params={"q": f"{q} type:org", "per_page": 3},
            headers=headers,
            timeout=5,
        )
        if r.status_code == 200:
            items = r.json().get("items", [])
            if items:
                slug = items[0]["login"]
    except Exception:
        pass

    if not slug:
        return []

    try:
        r = httpx.get(
            f"https://api.github.com/orgs/{slug}/repos",
            params={"sort": "stars", "per_page": 3},
            headers=headers,
            timeout=5,
        )
        if r.status_code == 200:
            return [repo["name"] for repo in r.json()]
    except Exception:
        pass
    return []


def generate_referral_drafts(application_id: int, user_id: str | None = None) -> dict:
    """Return draft outreach messages for one application (user must send them)."""
    from app.db.init_db import get_session
    from app.db.models import Application, Job
    from app.autofill.answer_pack import _get_or_create_profile

    with get_session() as session:
        application = session.get(Application, application_id)
        if not application:
            raise ValueError(f"Application {application_id} not found")
        job = session.get(Job, application.job_id)

    profile = _get_or_create_profile(user_id=user_id)
    name = f"{getattr(profile,'first_name','') or ''} {getattr(profile,'last_name','') or ''}".strip()
    title = getattr(profile, "current_title", "") or ""
    skills = getattr(profile, "key_skills", "") or ""
    role = job.title
    company = job.company or "the company"

    # Get winners from SerpAPI/X-Ray to check for university matches
    winners = []
    try:
        from app.intelligence.linkedin_xray import find_champions
        res = find_champions(company, role)
        if res.get("ok"):
            winners = res.get("people", [])
    except Exception:
        pass

    # Legal work-auth selling point for the visa-alumni draft.
    selling, needs_sponsorship = "", False
    try:
        from app.intelligence.work_auth import assess_profile
        fr = assess_profile(profile)
        selling = fr.selling_point or ""
        needs_sponsorship = bool(fr.needs_future_sponsorship)
    except Exception:
        pass

    # Build fallbacks
    drafts = _fallback_drafts(name, title, company, role, skills, selling, needs_sponsorship)

    # University Alumni check
    uni = getattr(profile, "university", "").strip()
    if uni:
        matched_alum = None
        for w in winners:
            if uni.lower() in w.get("headline", "").lower():
                matched_alum = w
                break
        target_name = matched_alum["name"] if matched_alum else "{Alumni Name}"
        drafts.append({
            "type": "university_alumni",
            "label": "University Alumni connection",
            "channel": "LinkedIn connection request to a fellow alum",
            "body": (
                f"Hi {target_name.split(' ')[0] if target_name else 'there'}, I noticed you also went to {uni} "
                f"and now work at {company} as a {role}. I'm exploring the team there "
                f"and would love to connect with a fellow alum to hear about your experience. Go "
                f"{uni.split(' ')[-1]}!"
            )
        })

    # GitHub check
    repos = []
    try:
        repos = get_company_github_repos(company)
    except Exception:
        pass

    repo_mention = f"the {repos[0]}" if repos else "open-source projects"
    drafts.append({
        "type": "github_outreach",
        "label": "GitHub outreach note",
        "channel": "LinkedIn message targeting open-source contributions",
        "body": (
            f"Hi {{name}}, I noticed your profile and also saw {company}'s active open-source contributions on GitHub"
            + (f", particularly the {repos[0]} repository." if repos else ".")
            + f" As a developer working with similar tech, I'd love to connect and follow your work."
        )
    })

    # Try to upgrade the drafts with the LLM (cheap Haiku). Non-fatal on failure.
    try:
        from app.config import settings
        from anthropic import Anthropic
        if settings.anthropic_api_key:
            client = Anthropic(api_key=settings.anthropic_api_key)
            prompt = (
                "You are a job-search outreach coach. Rewrite each draft below to be "
                "warm, specific, and under 90 words. Keep the placeholder {name} for "
                "the recipient. Return STRICT JSON: a list of objects with keys "
                "type,label,channel,body — same types/labels/channels as given.\n\n"
                f"Candidate: {name or 'the candidate'}, {title or 'applicant'}. "
                f"Skills: {skills or 'n/a'}. University: {uni or 'n/a'}. "
                f"Role: {role} at {company}. "
                f"Needs sponsorship: {needs_sponsorship}. Selling point: {selling or 'n/a'}.\n\n"
                f"Drafts: {drafts}"
            )
            resp = client.messages.create(
                model=settings.cover_letter_model, max_tokens=1200,
                messages=[{"role": "user", "content": prompt}],
            )
            import json, re
            raw = resp.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            if isinstance(parsed, list) and parsed and all("body" in d for d in parsed):
                drafts = parsed
    except Exception as e:
        log.debug("referral LLM enrichment skipped: %s", e)

    # Post-process drafts to add clickable helper links/instructions
    for d in drafts:
        if d.get("type") == "university_alumni" and uni:
            import urllib.parse
            q_str = f'site:linkedin.com/in/ "{company}" "{uni}"'
            search_url = f"https://www.google.com/search?q={urllib.parse.quote(q_str)}"
            d["channel"] = f"LinkedIn connection request to a fellow alum. Find them via: {search_url}"
        elif d.get("type") == "github_outreach":
            import urllib.parse
            search_url = f"https://github.com/search?q={urllib.parse.quote(company)}&type=users"
            d["channel"] = f"LinkedIn message targeting open-source contributions. Search org: {search_url}"

    return {
        "application_id": application_id,
        "company": company,
        "title": role,
        "note": "Drafts only — review, personalize the recipient, and send from your own account. JobAgent never sends these for you.",
        "drafts": drafts,
    }
