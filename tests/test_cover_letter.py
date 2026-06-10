"""Structural tests for the Problem-Solution-Proof cover letter generation.

These don't call a real LLM — they verify the system prompt and the
per-job prompt carry the required structure and grounding constraints,
and that the captured prompt is well-formed.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from app.tailoring.tailor import Tailor, COVER_SYSTEM


class _JobSnap:
    title = "Senior ML Engineer"
    company = "Acme AI"
    description = (
        "We need someone to scale our LLM inference platform to millions of "
        "requests. You will design RESTful APIs with FastAPI and deploy on "
        "Kubernetes. Build RAG pipelines and optimize model latency."
    )


def test_cover_system_has_problem_solution_proof():
    s = COVER_SYSTEM.lower()
    assert "problem" in s
    assert "solution" in s
    assert "proof" in s
    # grounding guardrails
    assert "invent" in s
    assert "180-220" in COVER_SYSTEM


def test_cover_prompt_includes_structure_and_company():
    """Capture the prompt passed to the LLM and assert it carries the structure."""
    tailor = Tailor.__new__(Tailor)  # bypass __init__ (no API keys needed)
    tailor._active_backend = "anthropic"

    captured = {}

    class _FakeClient:
        class messages:
            @staticmethod
            def create(model, max_tokens, system, messages):
                captured["system"] = system
                captured["user"] = messages[0]["content"]
                resp = MagicMock()
                resp.content = [MagicMock(text="A grounded cover letter.")]
                return resp

    tailor._anthropic_client = _FakeClient()
    tailor._openai_client = None

    out = tailor.write_cover_letter("master resume text", _JobSnap())
    assert out == "A grounded cover letter."

    user_prompt = captured["user"].lower()
    # company name threaded into the prompt
    assert "acme ai" in user_prompt
    # structure cues present
    assert "problem" in user_prompt
    assert "solution" in user_prompt or "built or did" in user_prompt
    assert "quantified result" in user_prompt
    assert "invent nothing" in user_prompt

    # system message is the Problem-Solution-Proof system prompt (cached block)
    sys_text = " ".join(
        blk["text"] for blk in captured["system"] if isinstance(blk, dict)
    ).lower()
    assert "problem" in sys_text and "proof" in sys_text


def test_cover_letter_openai_fallback_path():
    """When only OpenAI is configured, the structure must still be in the prompt."""
    tailor = Tailor.__new__(Tailor)
    tailor._active_backend = "openai"
    tailor._anthropic_client = None

    captured = {}

    class _FakeOpenAI:
        class chat:
            class completions:
                @staticmethod
                def create(model, max_tokens, messages):
                    captured["system"] = messages[0]["content"]
                    captured["user"] = messages[1]["content"]
                    resp = MagicMock()
                    choice = MagicMock()
                    choice.message.content = "OpenAI cover letter."
                    resp.choices = [choice]
                    return resp

    tailor._openai_client = _FakeOpenAI()

    out = tailor.write_cover_letter("master resume text", _JobSnap())
    assert out == "OpenAI cover letter."
    assert "problem" in captured["user"].lower()
    assert "acme ai" in captured["user"].lower()
