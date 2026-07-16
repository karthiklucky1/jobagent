import logging
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from app.config import settings
from app.db.init_db import get_session
from app.tailoring.doctor import _METRIC_RE

log = logging.getLogger(__name__)


def _adds_unbacked_metric(tailored: str, source_bullet: str) -> bool:
    """True when the tailored bullet contains a metric (e.g. '43%', '2,500 req/min',
    '3x') that does NOT appear verbatim in its matched source bullet.

    This is the exact shape of a fabricated number grafted onto a near-copy of a
    real bullet — high cosine similarity to the source, but with an invented
    metric. Similarity alone waves it through; this forces an LLM fact-check.
    """
    src = (source_bullet or "").lower()
    for m in _METRIC_RE.finditer(tailored or ""):
        token = (m.group(0) or "").strip().lower()
        if token and token not in src:
            return True
    return False

@dataclass
class GroundingResult:
    passed: bool
    flagged_bullets: List[Dict[str, Any]]
    confidence_map: Dict[str, float]

class GroundingChecker:
    def __init__(self):
        import torch
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    _SECTION_KEYS = ("EXPERIENCE", "PROJECT", "WORK", "EMPLOYMENT")
    _BULLET_PREFIXES = ("- ", "* ", "• ", "·", "– ", "— ", "‣ ", "▪ ", "◦ ", "● ", "» ")

    def _extract_bullets(self, resume_md: str) -> List[str]:
        """Extract experience/project bullets.

        Handles both markdown résumés (## EXPERIENCE + "- " bullets) and the
        plain text produced by PDF/DOCX extraction (ALL-CAPS or title-case
        section headers, varied bullet glyphs, or no bullet glyph at all).
        Falls back to substantive content lines so the grounding check never
        silently no-ops on a real résumé.
        """
        bullets: List[str] = []
        current_section = ""

        _other_sections = ("EDUCATION", "SKILLS", "SUMMARY", "PROJECTS", "CERTIFICATION",
                           "AWARDS", "PUBLICATION", "CONTACT", "OBJECTIVE", "INTERESTS")

        def _is_header(s: str) -> bool:
            if s.startswith("#"):
                return True
            if len(s) > 40 or s.endswith((".", ",", ";")):
                return False
            up = s.upper()
            # All-caps short line, or a short line naming a known résumé section.
            # (Avoids misreading a Title-Case content line like "Managed Backend
            # Systems" as a header.)
            return up == s or any(k in up for k in self._SECTION_KEYS) or any(k in up for k in _other_sections)

        for line in resume_md.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if _is_header(stripped):
                current_section = stripped.upper()
                continue
            in_target = any(k in current_section for k in self._SECTION_KEYS)
            if not in_target:
                continue
            cleaned = stripped
            for pre in self._BULLET_PREFIXES:
                if cleaned.startswith(pre):
                    cleaned = cleaned[len(pre):]
                    break
            cleaned = cleaned.replace("**", "").replace("*", "").strip()
            # Keep substantive lines (skips dates/company headers/one-word lines).
            if len(cleaned) >= 12 and " " in cleaned:
                bullets.append(cleaned)

        # Safety fallback: if section detection found nothing (e.g. messy PDF
        # extraction with no recognizable headers), compare against ALL
        # substantive lines so grounding still runs instead of passing blindly.
        if not bullets:
            for line in resume_md.splitlines():
                s = line.strip().lstrip("-*•·–—‣▪◦●» ").replace("**", "").strip()
                if len(s) >= 25 and " " in s:
                    bullets.append(s)
        return bullets

    def verify_with_llm(self, bullet: str, source_resume_md: str) -> bool:
        """Use the LLM to verify if a flagged bullet is supported by the master resume."""
        prompt = f"""You are a Fact-Checking Assistant for job applications.
Your task is to determine whether the claim in the Tailored Bullet is supported by the Master Resume.

Master Resume:
---
{source_resume_md}
---

Tailored Bullet:
"{bullet}"

Analyze whether the Tailored Bullet represents a factual claim that is supported by or reasonably derived from the Master Resume.
Guidelines:
1. CORE CLAIMS & METRICS: The core metrics (e.g., "22% accuracy", "65% cycle reduction", "2,500+ requests per minute") and core professional experience responsibilities must match or be directly derived from the Master Resume.
2. HONEST BRIDGING: If the Tailored Bullet introduces new technologies or tools (e.g. Triton, vLLM, CUDA) but frames them honestly as adjacent, under study, planned transition, or similar learning/bridging frameworks (e.g., "designed with plans to transition to...", "with adjacent study of...", "familiar with..."), this is SUPPORTED and should pass.
3. FABRICATED CLAIMS: If the bullet claims direct, hands-on production experience, design, implementation, or deployment of a technology that the candidate does not have in their Master Resume (e.g., claiming they actively developed Triton services or built CUDA kernels if not in the Master Resume), it is FABRICATED.

Return exactly "SUPPORTED" if it is supported, or "FABRICATED" if it is not supported. No other text.
"""
        try:
            from app.tailoring.tailor import Tailor
            tailor = Tailor()
            answer = ""
            
            # Try Anthropic first if it is the active backend
            if tailor._active_backend == "anthropic" and tailor._anthropic_client:
                try:
                    resp = tailor._anthropic_client.messages.create(
                        model=settings.scoring_model,
                        max_tokens=10,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    answer = resp.content[0].text.strip()
                except Exception as ae:
                    log.warning("Grounding: Anthropic failed during verify_with_llm, falling back to OpenAI: %s", ae)
            
            # Fall back to OpenAI if Anthropic failed, was not run, or answer is empty
            if not answer and tailor._openai_client:
                resp = tailor._openai_client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=10,
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = resp.choices[0].message.content.strip()
                
            if not answer:
                return False
                
            return "SUPPORTED" in answer.upper()
        except Exception as e:
            log.warning("LLM verification of flagged bullet failed: %s", e)
            return False

    def check(self, source_resume_md: str, tailored_resume_md: str) -> GroundingResult:
        source_bullets = self._extract_bullets(source_resume_md)
        tailored_bullets = self._extract_bullets(tailored_resume_md)

        if not tailored_bullets:
            log.info("No tailored bullets found. Passing.")
            return GroundingResult(passed=True, flagged_bullets=[], confidence_map={})

        if not source_bullets:
            # Don't fail open: with no comparable source bullets, every tailored
            # bullet is LLM-verified against the FULL master resume text instead
            # of being waved through unchecked.
            log.warning("No source bullets extracted — LLM-verifying each tailored "
                        "bullet against the full master resume.")
            flagged_bullets = []
            confidence_map = {}
            for t_bullet in tailored_bullets:
                confidence_map[t_bullet] = 0.0
                if not self.verify_with_llm(t_bullet, source_resume_md):
                    flagged_bullets.append({
                        "bullet": t_bullet,
                        "best_match_bullet": "",
                        "best_match_score": 0.0,
                    })
            return GroundingResult(passed=not flagged_bullets,
                                   flagged_bullets=flagged_bullets,
                                   confidence_map=confidence_map)

        log.info("Computing embeddings for %d source bullets and %d tailored bullets...", len(source_bullets), len(tailored_bullets))
        
        source_embeddings = self.model.encode(source_bullets, convert_to_tensor=True)
        tailored_embeddings = self.model.encode(tailored_bullets, convert_to_tensor=True)
        
        similarity_matrix = cos_sim(tailored_embeddings, source_embeddings)
        
        flagged_bullets = []
        confidence_map = {}
        threshold = settings.grounding_similarity_threshold
        
        for i, t_bullet in enumerate(tailored_bullets):
            best_match_idx = similarity_matrix[i].argmax().item()
            best_match_score = similarity_matrix[i][best_match_idx].item()
            best_match_bullet = source_bullets[best_match_idx]
            
            confidence_map[t_bullet] = best_match_score

            # Verify when the bullet is dissimilar to any source bullet OR when it
            # is a near-copy that ADDS a metric not present in its matched source
            # bullet — otherwise a fabricated number on a real bullet (cosine ~0.9)
            # would sail past the similarity gate unchecked.
            adds_metric = _adds_unbacked_metric(t_bullet, best_match_bullet)
            if best_match_score < threshold or adds_metric:
                reason = "below threshold" if best_match_score < threshold else "adds unbacked metric"
                log.info("Grounding: verifying bullet (%s, sim=%.3f): %s", reason, best_match_score, t_bullet)
                is_supported = self.verify_with_llm(t_bullet, source_resume_md)
                if not is_supported:
                    flagged_bullets.append({
                        "bullet": t_bullet,
                        "best_match_bullet": best_match_bullet,
                        "best_match_score": best_match_score
                    })
                else:
                    log.info("Grounding: LLM verified bullet as SUPPORTED: %s", t_bullet)
                    
        passed = len(flagged_bullets) == 0
        return GroundingResult(passed=passed, flagged_bullets=flagged_bullets, confidence_map=confidence_map)
