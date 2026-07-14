"""Lock layer: immutable factual sections (Education) survive tailoring verbatim.

Pure string surgery — no LLM, no DB — so these run anywhere.
"""
from __future__ import annotations

from app.tailoring.lock import restore_locked_fields

MASTER = """# Karthik Amruthaluri
Senior ML Engineer
karthik@example.com | Cincinnati, OH

## Professional Summary
Original summary text.

## Education
Bachelor of Science in Computer Science, State University, 2020 - 2024

## Skills
Python, PyTorch
"""


def test_altered_degree_is_restored_verbatim():
    tailored = """# Karthik Amruthaluri
Senior ML Engineer — AI Solutions
karthik@example.com | Cincinnati, OH

## Professional Summary
Tailored summary aimed at the job description.

## Education
Bachelor's in CS, State University, 2020

## Skills
Python, PyTorch, RAG, FAISS, Docker
"""
    fixed, restored = restore_locked_fields(tailored, MASTER)

    # Degree restored to the master's exact wording…
    assert "Bachelor of Science in Computer Science" in fixed
    assert "2020 - 2024" in fixed
    assert "Bachelor's in CS" not in fixed
    assert restored  # something was re-locked

    # …but the tailored title, summary, and skills are untouched.
    assert "Senior ML Engineer — AI Solutions" in fixed
    assert "Tailored summary aimed at the job description." in fixed
    assert "RAG, FAISS, Docker" in fixed


def test_unchanged_education_is_a_noop():
    # Tailored keeps the exact master Education block → nothing to restore.
    tailored = MASTER.replace("Original summary text.", "A tailored summary.")
    fixed, restored = restore_locked_fields(tailored, MASTER)
    assert restored == []
    assert "A tailored summary." in fixed
    assert "Bachelor of Science in Computer Science" in fixed


def test_dropped_education_is_reappended():
    tailored = """# Karthik Amruthaluri
Senior ML Engineer

## Professional Summary
Tailored summary.

## Skills
Python, PyTorch
"""
    fixed, restored = restore_locked_fields(tailored, MASTER)
    assert restored  # re-locked
    assert "Bachelor of Science in Computer Science" in fixed


def test_case_insensitive_header_match():
    tailored = MASTER.replace("## Education", "## EDUCATION").replace(
        "Bachelor of Science in Computer Science, State University, 2020 - 2024",
        "BS Computer Science",
    )
    fixed, restored = restore_locked_fields(tailored, MASTER)
    assert restored
    assert "Bachelor of Science in Computer Science" in fixed
    assert "BS Computer Science" not in fixed


def test_master_without_education_is_a_noop():
    master_no_edu = """# Jane Doe
## Summary
Hi.
## Skills
Go, Rust
"""
    tailored = "# Jane Doe\n## Summary\nTailored.\n## Skills\nGo, Rust, K8s\n"
    fixed, restored = restore_locked_fields(tailored, master_no_edu)
    assert restored == []
    assert fixed == tailored
