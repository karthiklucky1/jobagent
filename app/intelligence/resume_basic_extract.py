"""Free, deterministic resume → profile extraction — the $0 fallback.

Used when the Claude extraction in /api/resume/extract-profile (and the
answer-pack experience/education extraction) can't run: missing API key,
exhausted credits, provider outage, or an unparseable LLM response. Reuses the
Resume X-Ray's regex/section parsers so signup never produces a silently-empty
profile again (Jul 2026: a credits outage left new users with blank profiles
and broken autofill).

The LLM stays primary — this parser is line-oriented and heuristic, so fields
like `field` (major) and role summaries are best-effort or empty. Callers tag
results (method="basic") and must NOT cache them where a later LLM pass would
be blocked from upgrading them.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from app.intelligence.resume_xray import (
    _EMAIL_RE,
    _GITHUB_RE,
    _LINKEDIN_RE,
    _PHONE_RE,
    _RANGE_RE,
    _SECTION_HEADERS,
    _months_between,
    _parse_month,
)

_URL_RE = re.compile(r"https?://[^\s)>\]]+", re.I)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_DEGREE_RE = re.compile(
    r"\b(bachelor|master|ph\.?d|doctorate|mba|b\.?\s?s\b|m\.?\s?s\b|b\.?\s?a\b|"
    r"m\.?\s?a\b|b\.?\s?e\b|m\.?\s?e\b|b\.?\s?tech|m\.?\s?tech|associate)\b", re.I)
_SCHOOL_RE = re.compile(r"\b(university|college|institute|polytechnic|school of)\b", re.I)
_TITLE_TOKEN_RE = re.compile(
    r"\b(engineer|developer|scientist|analyst|manager|designer|architect|consultant|"
    r"lead|intern|administrator|specialist|researcher)\b", re.I)
_BULLET_PREFIX_RE = re.compile(r"^\s*[-*•·▪]\s*")
# "City, ST" or "City, Country" — used only on the resume's top lines.
_LOCATION_RE = re.compile(r"\b([A-Z][a-zA-Z.\- ]{2,25},\s*(?:[A-Z]{2}|[A-Z][a-zA-Z]{3,15}))\b")


def _split_sections(text: str) -> dict:
    """Map section name → body text, using the X-Ray's header patterns.
    Body runs from the end of a header line to the start of the next header."""
    marks: List[Tuple[int, str]] = []
    for name, rx in _SECTION_HEADERS.items():
        m = rx.search(text)
        if m:
            marks.append((m.start(), name))
    marks.sort()
    sections: dict = {}
    for i, (start, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[start:end]
        body = body.split("\n", 1)[1] if "\n" in body else ""  # drop the header line
        sections[name] = body.strip()
    return sections


def _extract_name(lines: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """First plausible name line near the top: 2-4 alphabetic words, no
    contact info, no section keywords."""
    for raw in lines[:6]:
        line = re.sub(r"^#+\s*", "", raw).strip()
        if not line or len(line) > 60:
            continue
        if _EMAIL_RE.search(line) or _URL_RE.search(line) or any(c.isdigit() for c in line):
            continue
        if _TITLE_TOKEN_RE.search(line) or _SECTION_HEADERS["summary"].match(line):
            continue
        words = [w for w in re.split(r"\s+", line) if w]
        if 2 <= len(words) <= 4 and all(re.fullmatch(r"[A-Za-z.'-]+", w) for w in words):
            return words[0], " ".join(words[1:])
    return None, None


def _extract_location(lines: List[str]) -> Optional[str]:
    for raw in lines[:8]:
        m = _LOCATION_RE.search(raw)
        if m and not _SCHOOL_RE.search(m.group(1)):
            return m.group(1).strip()
    return None


def _clean_skill_line(line: str) -> str:
    line = _BULLET_PREFIX_RE.sub("", line).strip()
    # "Languages: Python, Go" → keep the part after the category label
    if ":" in line and len(line.split(":", 1)[0]) < 30:
        line = line.split(":", 1)[1]
    return line.strip()


def _extract_skills(section: str) -> Optional[str]:
    if not section:
        return None
    tokens: List[str] = []
    seen = set()
    for line in section.splitlines():
        cleaned = _clean_skill_line(line)
        if not cleaned:
            continue
        for tok in re.split(r"[,|/•·]+", cleaned):
            tok = tok.strip(" .;")
            if tok and len(tok) < 40 and tok.lower() not in seen:
                seen.add(tok.lower())
                tokens.append(tok)
        if len(tokens) >= 40:
            break
    joined = ", ".join(tokens)
    return joined[:500] or None


def _extract_education(section: str) -> List[dict]:
    """One entry per school line; degree/years attached from nearby lines."""
    entries: List[dict] = []
    if not section:
        return entries
    lines = [ln.strip() for ln in section.splitlines() if ln.strip()]
    current: Optional[dict] = None
    for line in lines:
        line = _BULLET_PREFIX_RE.sub("", line)
        is_school = bool(_SCHOOL_RE.search(line))
        is_degree = bool(_DEGREE_RE.search(line))
        if is_school:
            # A school line that also names the degree stays one entry.
            current = {"degree": "", "field": "", "university": "", "start_year": None,
                       "end_year": None, "gpa": ""}
            entries.append(current)
            current["university"] = _YEAR_RE.sub("", line).strip(" ,|–—-")
            if is_degree:
                current["degree"] = current["university"]
        elif is_degree:
            if current is None:
                current = {"degree": "", "field": "", "university": "", "start_year": None,
                           "end_year": None, "gpa": ""}
                entries.append(current)
            if not current["degree"]:
                current["degree"] = _YEAR_RE.sub("", line).strip(" ,|–—-")
        if current is not None:
            years = [int(y.group(0)) for y in _YEAR_RE.finditer(line)]
            if years:
                current["start_year"] = current["start_year"] or (min(years) if len(years) > 1 else None)
                current["end_year"] = max(years)
            gpa = re.search(r"gpa[:\s]*([0-9.]{1,4}\s*(?:/\s*[0-9.]{1,4})?)", line, re.I)
            if gpa and not current["gpa"]:
                current["gpa"] = gpa.group(1).strip()
    return entries[:10]


def _extract_experience(section: str) -> List[dict]:
    """One entry per date-range line. Title/company come from the range line
    itself (minus the dates) plus the previous line when the range stands alone."""
    entries: List[dict] = []
    if not section:
        return entries
    lines = section.splitlines()
    for i, raw in enumerate(lines):
        m = _RANGE_RE.search(raw)
        if not m:
            continue
        header = (raw[: m.start()] + " " + raw[m.end():]).strip(" ,|–—-\t")
        header = _BULLET_PREFIX_RE.sub("", header)
        if not header:
            for back in range(i - 1, max(i - 3, -1), -1):
                prev = lines[back].strip()
                if prev and not _RANGE_RE.search(prev) and not _BULLET_PREFIX_RE.match(lines[back]):
                    header = prev
                    break
        title, company = header, ""
        # "Title @ Company" / "Title — Company" / "Title | Company" / "Title, Company"
        for sep in (" @ ", " at ", " — ", " – ", " | ", ", "):
            if sep in header:
                title, company = header.split(sep, 1)[0], header.split(sep, 1)[1]
                break
        summary = ""
        for j in range(i + 1, min(i + 4, len(lines))):
            nxt = lines[j].strip()
            if _BULLET_PREFIX_RE.match(lines[j]):
                summary = _BULLET_PREFIX_RE.sub("", nxt)
                break
            if _RANGE_RE.search(nxt) or _SCHOOL_RE.search(nxt):
                break
        entries.append({
            "title": title.strip()[:120],
            "company": company.strip(" ,|–—-")[:120],
            "location": "",
            "start": m.group(1).strip(),
            "end": m.group(2).strip(),
            "summary": summary[:300],
        })
    return entries[:15]


def _years_experience(experience: List[dict]) -> Optional[int]:
    periods = []
    for e in experience:
        start, end = _parse_month(e.get("start") or ""), _parse_month(e.get("end") or "")
        if start and end:
            periods.append((start, end))
    if not periods:
        return None
    earliest = min(p[0] for p in periods)
    latest = max(p[1] for p in periods)
    months = _months_between(earliest, latest)
    return max(months // 12, 0) if months > 0 else None


def basic_extract_profile(resume_text: str) -> dict:
    """Deterministic extraction returning the SAME shape as the Claude prompt in
    /api/resume/extract-profile (missing fields None/empty — the endpoint only
    writes non-empty values, so partial results never blank existing data)."""
    text = resume_text or ""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    sections = _split_sections(text)

    first_name, last_name = _extract_name(lines)
    email = _EMAIL_RE.search(text)
    phone = _PHONE_RE.search(text)
    linkedin = _LINKEDIN_RE.search(text)
    github = _GITHUB_RE.search(text)
    portfolio = next(
        (u.group(0).rstrip(".,") for u in _URL_RE.finditer(text)
         if "linkedin.com" not in u.group(0).lower() and "github.com" not in u.group(0).lower()),
        None)

    experience = _extract_experience(sections.get("experience", ""))
    education = _extract_education(sections.get("education", ""))

    current_title = experience[0]["title"] if experience else None
    if not current_title:
        for raw in lines[:6]:
            if _TITLE_TOKEN_RE.search(raw) and len(raw.strip()) < 80:
                current_title = re.sub(r"^#+\s*", "", raw).strip()
                break

    summary_text = sections.get("summary", "")
    professional_summary = re.sub(r"\s+", " ", summary_text).strip()[:400] or None

    top_edu = education[0] if education else {}
    grad_year = top_edu.get("end_year")

    return {
        "first_name": first_name,
        "last_name": last_name,
        "email": email.group(0) if email else None,
        "phone": phone.group(1).strip() if phone else None,
        "location": _extract_location(lines),
        "current_title": current_title,
        "years_experience": _years_experience(experience),
        "linkedin_url": f"https://{linkedin.group(0)}" if linkedin else None,
        "github_url": f"https://{github.group(0)}" if github else None,
        "portfolio_url": portfolio,
        "degree": top_edu.get("degree") or None,
        "university": top_edu.get("university") or None,
        "graduation_year": int(grad_year) if grad_year else None,
        "key_skills": _extract_skills(sections.get("skills", "")),
        "professional_summary": professional_summary,
        "suggested_target_roles": [],  # role seeding falls back to title/skills upstream
        "education": education,
        "experience": experience,
    }


def to_answer_pack_shape(extracted: dict) -> dict:
    """Convert basic_extract_profile output to the answer-pack extraction shape
    (work_experience / education with the autofill agent's key names)."""
    work = [{
        "company": e.get("company") or "",
        "title": e.get("title") or "",
        "location": e.get("location") or "",
        "start_date": e.get("start") or "",
        "end_date": e.get("end") or "",
        "description": e.get("summary") or "",
    } for e in extracted.get("experience") or []]
    edu = [{
        "school": e.get("university") or "",
        "degree": e.get("degree") or "",
        "field_of_study": e.get("field") or "",
        "start_date": str(e.get("start_year") or ""),
        "end_date": str(e.get("end_year") or ""),
    } for e in extracted.get("education") or []]
    return {"work_experience": work, "education": edu}
