"""Two-stage job title filter used by all discovery sources.

Stage 1 — Broad regex catch (free, instant):
  Match titles containing any root term associated with ML/AI/Python/backend.
  Deliberately wide — catches "Member of Technical Staff", "Founding Engineer",
  "Research Scientist, AI", "Software Engineer II, ML Platform", etc.

Stage 2 — Semantic kill filter (cheap, only for ambiguous stage-1 passes):
  Titles that passed stage 1 but look unrelated (DevOps, Sales, QA, etc.)
  are compared against a tiny anchor embedding set. Below threshold → rejected.
  Only runs when sentence_transformers is available; otherwise skips gracefully.

Usage in any source:
    from app.discovery.title_filter import matches_title

    if not matches_title(title):
        continue
"""
from __future__ import annotations

import re
import logging
from functools import lru_cache
from typing import List

log = logging.getLogger(__name__)

# ── Stage 1: broad root-term regex ───────────────────────────────────────────
# Catches titles that contain any of these root terms (word-boundary aware).
_INCLUDE_ROOTS = [
    r"machine\s+learn",
    r"\bml\b",
    r"\bai\b",
    r"\bllm\b",
    r"\bnlp\b",
    r"deep\s+learn",
    r"neural",
    r"generative",
    r"gen[\s\-]?ai",
    r"large\s+language",
    r"foundation\s+model",
    r"data\s+sci",
    r"data\s+engin",
    r"\bpython\b",
    r"backend",
    r"back[\s\-]end",
    r"mlops",
    r"platform\s+engin",
    r"inference",
    r"model\s+serv",
    r"research\s+sci",
    r"applied\s+sci",
    r"software\s+engin",   # broad — stage 2 will kill unrelated ones
    r"member\s+of\s+technical",
    r"founding\s+engin",
    r"staff\s+engin",
    r"principal\s+engin",
    r"computer\s+vision",
    r"\bcv\b.*engin",
    r"reinforcement",
    r"\brlhf\b",
    r"fine[\s\-]?tun",
    r"embeddings?",
    r"vector\s+search",
    r"rag\b",
]

# If any of these are in the title → reject immediately (stage 1 kill list).
_EXCLUDE_ROOTS = [
    r"sales\s+engin",
    r"solutions?\s+engin",
    r"customer\s+success",
    r"account\s+execut",
    r"recruiter",
    r"marketing",
    r"\bseo\b",
    r"qa\s+engin",
    r"quality\s+assur",
    r"test\s+engin",
    r"support\s+engin",
    r"field\s+engin",
    r"hardware\s+engin",
    r"mechanical\s+engin",
    r"electrical\s+engin",
    r"civil\s+engin",
    r"\bhr\b",
    r"human\s+resource",
    r"finance\b",
    r"accountant",
    r"legal\b",
    r"counsel\b",
    r"designer\b",
    r"product\s+design",
    r"ux\s+",
    r"ui\s+design",
]

_include_re = re.compile("|".join(_INCLUDE_ROOTS), re.IGNORECASE)
_exclude_re = re.compile("|".join(_EXCLUDE_ROOTS), re.IGNORECASE)

# ── Stage 2: semantic anchors ─────────────────────────────────────────────────
_ANCHORS = [
    "machine learning engineer",
    "AI engineer",
    "python developer",
    "LLM engineer",
    "MLOps engineer",
    "backend engineer python",
    "data scientist",
    "applied scientist",
    "NLP engineer",
    "GenAI engineer",
    "deep learning researcher",
    "software engineer machine learning",
]

_SEMANTIC_THRESHOLD = 0.30   # cosine similarity — below this = not a match
_semantic_available: bool | None = None   # None = not yet checked
_anchor_embeddings = None
_model = None


def _try_init_semantic():
    global _semantic_available, _anchor_embeddings, _model
    if _semantic_available is not None:
        return _semantic_available
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        import numpy as np
        _anchor_embeddings = _model.encode(_ANCHORS, convert_to_numpy=True, normalize_embeddings=True)
        _semantic_available = True
        log.info("TitleFilter: semantic stage initialized (all-MiniLM-L6-v2)")
    except Exception as e:
        _semantic_available = False
        log.debug("TitleFilter: semantic stage unavailable (%s) — using regex only", e)
    return _semantic_available


def _semantic_matches(title: str) -> bool:
    """Return True if title is semantically similar to any anchor."""
    if not _try_init_semantic():
        return True   # can't check → allow through
    try:
        import numpy as np
        emb = _model.encode([title], convert_to_numpy=True, normalize_embeddings=True)
        sims = (_anchor_embeddings @ emb.T).flatten()
        return float(sims.max()) >= _SEMANTIC_THRESHOLD
    except Exception:
        return True   # on error → allow through


# Titles that are ambiguous at stage 1 (contain "software engineer" but might
# be DevOps, QA, etc.) get routed to semantic stage 2.
_AMBIGUOUS_RE = re.compile(
    r"(software\s+engin|founding\s+engin|staff\s+engin|principal\s+engin|member\s+of\s+technical)",
    re.IGNORECASE,
)


def matches_title(title: str, extra_keywords: List[str] | None = None) -> bool:
    """Return True if this job title is relevant for ML/AI/Python roles.

    Args:
        title: Raw job title string.
        extra_keywords: Optional caller-supplied keywords (settings.jobs_keywords_list).
                        If any match verbatim → fast-pass, skip regex/semantic.
    """
    if not title:
        return False

    title_lower = title.lower()

    # Fast-pass: exact keyword match from caller's configured list
    if extra_keywords:
        if any(kw.lower() in title_lower for kw in extra_keywords):
            return True

    # Stage 1a: exclude obvious non-matches
    if _exclude_re.search(title):
        return False

    # Stage 1b: include root terms
    if not _include_re.search(title):
        return False

    # Stage 2: semantic check only for ambiguous titles
    if _AMBIGUOUS_RE.search(title):
        return _semantic_matches(title)

    return True
