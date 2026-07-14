"""Lock layer — protect immutable factual sections during tailoring.

Tailoring may reword the summary, skills, and bullet wording to match a JD, but
it must NEVER alter *facts*: your degree, school, and graduation dates. The LLM
sometimes paraphrases the Education section ("B.S. Computer Science" →
"Bachelor's in CS", or drops a specialization). The Resume Doctor catches that as
an integrity issue AFTER the fact — but an altered credential should be
impossible, not merely reported.

So instead of flagging it, we restore the Education section VERBATIM from the
master résumé right after generation (before grounding / the Doctor / the .docx
render). The title, summary, skills, and bullets are left fully tailored — only
the factual Education block is pinned.

This is deterministic string surgery (no LLM), so it's fast, testable, and can't
introduce a hallucination. If the master has no recognizable Education section,
it is a safe no-op.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

# A markdown ATX header line: up to 3 leading spaces, 1-6 '#', then the title.
_HEADER_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*\S)\s*$")

# Section titles whose CONTENT is purely factual and must survive tailoring
# verbatim. Matched as a substring of the (lowercased) header title, so
# "Education", "EDUCATION & CERTIFICATIONS", "Academic Background" all match.
_LOCKED_TITLE_TERMS = ("education", "academic")

Section = Tuple[Optional[str], List[str]]  # (header_line_or_None, body_lines)


def _split_sections(md: str) -> List[Section]:
    """Split markdown into sections. The preamble (everything before the first
    header — typically the name/contact block) is the first section with a None
    header. Every subsequent section is (header_line, body_lines)."""
    sections: List[Section] = []
    cur_header: Optional[str] = None
    cur_body: List[str] = []
    for line in md.splitlines():
        if _HEADER_RE.match(line):
            sections.append((cur_header, cur_body))
            cur_header, cur_body = line, []
        else:
            cur_body.append(line)
    sections.append((cur_header, cur_body))
    return sections


def _header_title(header_line: Optional[str]) -> str:
    if not header_line:
        return ""
    m = _HEADER_RE.match(header_line)
    return (m.group(2) if m else header_line).lower()


def _is_locked_section(header_line: Optional[str]) -> bool:
    title = _header_title(header_line)
    return bool(title) and any(term in title for term in _LOCKED_TITLE_TERMS)


def _normalize(s: str) -> str:
    """Whitespace/case-insensitive form for change detection."""
    return re.sub(r"\s+", " ", s).strip().lower()


def _block(section: Section) -> str:
    header, body = section
    lines = ([header] if header is not None else []) + body
    return "\n".join(lines)


def _rejoin(sections: List[Section]) -> str:
    out: List[str] = []
    for header, body in sections:
        if header is not None:
            out.append(header)
        out.extend(body)
    return "\n".join(out)


def restore_locked_fields(tailored_md: str, master_md: str) -> Tuple[str, List[str]]:
    """Pin immutable factual sections (Education) verbatim from the master résumé.

    Returns ``(fixed_md, restored)`` where ``restored`` names the sections that
    were re-locked (empty when nothing needed fixing). Only sections whose text
    actually drifted from the master are touched, so a clean tailoring pass is a
    pure no-op.
    """
    restored: List[str] = []

    # The master's locked sections, verbatim, keyed by a normalized title so a
    # renamed tailored header ("EDUCATION" vs "Education & Certs") still matches.
    master_locked: dict[str, Section] = {}
    for section in _split_sections(master_md):
        header, _ = section
        if _is_locked_section(header):
            master_locked.setdefault(_header_title(header).split()[0], section)
    if not master_locked:
        return tailored_md, restored  # nothing lockable in the master → no-op

    tail_sections = _split_sections(tailored_md)
    out_sections: List[Section] = []
    seen_keys: set[str] = set()

    for section in tail_sections:
        header, _ = section
        if _is_locked_section(header):
            key = _header_title(header).split()[0]
            master_section = master_locked.get(key) or next(iter(master_locked.values()))
            seen_keys.add(_header_title(master_section[0]).split()[0])
            if _normalize(_block(section)) != _normalize(_block(master_section)):
                out_sections.append(master_section)
                restored.append("education (degree, school, dates)")
            else:
                out_sections.append(section)
        else:
            out_sections.append(section)

    # A locked section the tailored résumé dropped entirely — re-append it so the
    # degree/school never silently vanish.
    for key, master_section in master_locked.items():
        canon = _header_title(master_section[0]).split()[0]
        if canon not in seen_keys:
            out_sections.append(master_section)
            restored.append("education (degree, school, dates)")

    # De-dupe the human-facing labels while preserving order.
    restored = list(dict.fromkeys(restored))
    return _rejoin(out_sections), restored
