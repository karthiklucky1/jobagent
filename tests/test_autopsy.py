"""Tests for the rejection-autopsy orchestrator (mocked finder — no network)."""
from app.intelligence.autopsy import run_autopsy
from app.intelligence.door_match import CandidateProfile

APPLIED = CandidateProfile(years=3, axis="applied", domains=["genai", "llm"],
                           remote_ok=True, open_to_relocation=False,
                           work_auth="OPT", home_metro="Cincinnati, OH")


def _finder(people):
    return lambda company, role: {"ok": True, "people": people}


def test_autopsy_wrong_door_enterprise_onsite():
    res = run_autopsy(
        "Royal Caribbean Group", "AI Engineer",
        "7+ years. Enterprise Data & Analytics. On-site Miami, FL.",
        APPLIED,
        finder=_finder([{"headline": "SVP Chief AI Officer PhD"},
                        {"headline": "Sr Data Engineer 7 years"},
                        {"headline": "ML Engineer, on-site Miami"}]))
    assert res["ok"] and res["wrong_door"] is True
    assert res["bar"]["axis"] == "enterprise"
    assert res["bar"]["onsite"] is True
    assert res["confidence"].startswith("HIGH") or res["confidence"].startswith("MEDIUM")
    assert res["right_door"]


def test_autopsy_right_door_applied():
    res = run_autopsy(
        "Series-A startup", "Applied GenAI Engineer",
        "Fully remote. Ship LLM/RAG features. 2+ years.",
        APPLIED,
        finder=_finder([{"headline": "Applied AI Engineer shipped LLM apps"},
                        {"headline": "GenAI product engineer"},
                        {"headline": "ML engineer, remote"}]))
    assert res["wrong_door"] is False
    assert "visibility" in res["right_door"].lower()


def test_autopsy_thin_data_low_confidence():
    res = run_autopsy("Stealth Co", "Junior SWE",
                      "$300k. Prove you're better than AI.",
                      APPLIED, finder=_finder([]))
    assert res["confidence"].startswith("LOW")
    assert res["wrong_door"] is True  # elite-outlier bar


def test_autopsy_handles_finder_failure_gracefully():
    res = run_autopsy("X", "AI Engineer", "Remote. 2+ years.", APPLIED,
                      finder=lambda c, r: {"ok": False, "reason": "serpapi_key_not_set"})
    assert res["ok"] is True          # still returns a diagnosis
    assert res["winners_n"] == 0
    assert res["data_ok"] is False


def test_autopsy_llm_axis_refinement():
    from unittest.mock import patch, MagicMock
    
    with patch("app.config.settings.anthropic_api_key", "dummy_key"), \
         patch("anthropic.Anthropic") as mock_anthropic:
        
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text='{"axis": "research", "reasoning": "LLM override"}')]
        mock_client.messages.create.return_value = mock_message

        res = run_autopsy(
            "Acme Corp", "Applied AI Engineer",
            "Applied role.",
            APPLIED,
            finder=_finder([{"headline": "Applied AI Engineer"}])
        )
        
        assert res["bar"]["axis"] == "research"
        mock_client.messages.create.assert_called_once()
