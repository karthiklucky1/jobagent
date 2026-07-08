"""The grounding check must not fail open: when no source bullets can be
extracted from the master resume, tailored bullets are LLM-verified against
the full master text instead of being waved through."""
from app.tailoring.grounding import GroundingChecker

TAILORED = """## EXPERIENCE
- Built CUDA kernels powering a 10x inference speedup at Google DeepMind.
- Deployed production RAG pipelines with FastAPI and FAISS.
"""

UNPARSEABLE_MASTER = "short note"  # yields zero source bullets


def _checker_without_model() -> GroundingChecker:
    # Skip __init__ (would load a SentenceTransformer); the no-source-bullets
    # path never touches self.model.
    return object.__new__(GroundingChecker)


def test_no_source_bullets_verifies_instead_of_passing(monkeypatch):
    checker = _checker_without_model()
    seen = []

    def fake_verify(bullet, master):
        seen.append(bullet)
        return "RAG" in bullet  # first bullet fabricated, second supported

    monkeypatch.setattr(checker, "verify_with_llm", fake_verify)
    result = checker.check(UNPARSEABLE_MASTER, TAILORED)

    assert len(seen) == 2          # every tailored bullet was verified
    assert result.passed is False  # the fabricated one is flagged
    assert len(result.flagged_bullets) == 1
    assert "CUDA" in result.flagged_bullets[0]["bullet"]


def test_no_source_bullets_all_supported_passes(monkeypatch):
    checker = _checker_without_model()
    monkeypatch.setattr(checker, "verify_with_llm", lambda b, m: True)
    result = checker.check(UNPARSEABLE_MASTER, TAILORED)
    assert result.passed is True
    assert result.flagged_bullets == []


def test_no_tailored_bullets_still_passes():
    checker = _checker_without_model()
    result = checker.check(UNPARSEABLE_MASTER, "nothing here")
    assert result.passed is True
