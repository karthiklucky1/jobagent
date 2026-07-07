"""Tests for the door-match engine (app/intelligence/door_match.py).

Pure logic — no network, no API keys. Verifies the 'right door vs wrong door'
verdict and, critically, the user-preference-driven location weighting.
"""
from app.intelligence.door_match import (
    CandidateProfile, RoleBar, classify_door,
)


APPLIED = CandidateProfile(years=3, axis="applied", domains=["genai", "llm"],
                           remote_ok=True, open_to_relocation=False,
                           work_auth="OPT", home_metro="Cincinnati, OH")


def _dims(verdict):
    return {f.dim: f.status for f in verdict.findings}


def test_wrong_door_enterprise_vs_applied():
    bar = RoleBar(years=6, axis="enterprise", domain="financial services", onsite=True,
                  onsite_metro="Boston, MA", level="senior")
    v = classify_door(APPLIED, bar, winners_n=6, data_quality="rich")
    assert v.wrong_door is True
    dims = _dims(v)
    assert dims["AXIS"] == "BLOCKER"
    assert dims["SENIORITY"] == "BLOCKER"
    assert "Applied" in v.right_door or "startups" in v.right_door


def test_right_door_when_candidate_fits():
    bar = RoleBar(years=3, axis="applied", domain="genai / llm / rag", onsite=False)
    v = classify_door(APPLIED, bar, winners_n=7, data_quality="rich")
    assert v.wrong_door is False
    assert "visibility" in v.right_door.lower()
    # OPT is still surfaced as a silent leak even on a right-door role.
    assert _dims(v)["WORK-AUTH"] == "SILENT-LEAK"


def test_location_blocks_remote_only_candidate_in_other_metro():
    bar = RoleBar(axis="applied", onsite=True, onsite_metro="Miami, FL")
    v = classify_door(APPLIED, bar, winners_n=7)
    assert _dims(v)["LOCATION"] == "BLOCKER"
    assert v.wrong_door is True


def test_location_ok_when_candidate_open_to_relocation():
    cand = CandidateProfile(years=3, axis="applied", domains=["llm"],
                            remote_ok=True, open_to_relocation=True, home_metro="Cincinnati, OH")
    bar = RoleBar(axis="applied", onsite=True, onsite_metro="Miami, FL")
    v = classify_door(cand, bar, winners_n=7)
    assert _dims(v)["LOCATION"] == "MATCH"
    assert v.wrong_door is False  # no blockers -> right door


def test_location_same_metro_is_stretch_not_blocker():
    bar = RoleBar(axis="applied", onsite=True, onsite_metro="Cincinnati, OH")
    v = classify_door(APPLIED, bar, winners_n=7)
    assert _dims(v)["LOCATION"] == "STRETCH"


def test_elite_outlier_thin_data_low_confidence():
    bar = RoleBar(axis="elite-outlier", elite_outlier=True)
    v = classify_door(APPLIED, bar, winners_n=1, data_quality="thin")
    assert v.wrong_door is True
    assert v.confidence.startswith("LOW")


def test_from_user_profile_reads_prefs():
    class P:
        years_experience = 3
        key_skills = "Python, LLM, RAG, FastAPI"
        target_roles = "AI Engineer"
        remote_ok = True
        open_to_relocation = False
        work_authorization = "OPT"
        location = "Cincinnati, OH"
    cand = CandidateProfile.from_user_profile(P())
    assert cand.years == 3
    assert cand.remote_ok is True and cand.open_to_relocation is False
    assert "llm" in cand.domains
    assert cand.work_auth == "OPT"
