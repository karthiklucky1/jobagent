"""Tests for role_bar.build_role_bar + the end-to-end bar->verdict chain.

Deterministic; no network / API keys. Uses the founder's real rejection cases.
"""
from app.intelligence.role_bar import build_role_bar
from app.intelligence.door_match import CandidateProfile, classify_door

APPLIED = CandidateProfile(years=3, axis="applied", domains=["genai", "llm"],
                           remote_ok=True, open_to_relocation=False,
                           work_auth="OPT", home_metro="Cincinnati, OH")


def test_cohere_research_axis():
    bar = build_role_bar(
        "Senior Research Engineer, Model Evaluation",
        "Research on LLM evaluation, reinforcement learning, model foundations.",
        winners=[{"headline": "Member of Technical Staff, PhD"},
                 {"headline": "MTS · RL post-training"},
                 {"headline": "Published NLP researcher"}])
    assert bar.axis == "research"
    assert bar.pedigree and "PhD" in bar.pedigree


def test_state_street_enterprise_axis_onsite_years():
    bar = build_role_bar(
        "Senior AI Infra Platform Engineer, AVP",
        "5+ years in ML engineering. Financial services enterprise. On-site Boston, MA.",
        winners=[{"headline": "ML Engineer ~5 years"}, {"headline": "Senior Associate"}])
    assert bar.axis == "enterprise"
    assert bar.years == 5
    assert bar.onsite is True and bar.onsite_metro == "Boston, MA"


def test_royal_caribbean_enterprise_onsite_miami():
    bar = build_role_bar(
        "AI Engineer",
        "7+ years. Enterprise Data & Analytics team. On-site Miami, FL.",
        winners=[{"headline": "SVP, Chief AI Officer, PhD"}, {"headline": "Sr Data Engineer 7+ yrs"}])
    assert bar.axis == "enterprise"
    assert bar.years == 7
    assert bar.onsite is True and bar.onsite_metro == "Miami, FL"
    # end-to-end: candidate should be WRONG DOOR here
    v = classify_door(APPLIED, bar, winners_n=7, data_quality="rich")
    assert v.wrong_door is True


def test_applied_startup_is_right_door():
    bar = build_role_bar(
        "Applied GenAI Engineer",
        "Fully remote. Ship LLM features fast. 2+ years experience. RAG, prompts.",
        winners=[{"headline": "Applied AI Engineer, shipped LLM apps"}])
    assert bar.axis == "applied"
    assert bar.onsite is False
    v = classify_door(APPLIED, bar, winners_n=6, data_quality="rich")
    assert v.wrong_door is False  # you fit -> right door


def test_elite_outlier_detected():
    bar = build_role_bar(
        "Junior Software Engineer",
        "$300k/year. Prove you're better than AI. San Francisco.",
        winners=[])
    assert bar.axis == "elite-outlier"
    assert bar.elite_outlier is True
