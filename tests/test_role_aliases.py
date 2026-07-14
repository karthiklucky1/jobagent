"""Role-alias regressions: title variants that routing previously missed."""
from app.discovery.title_filter import role_title_match


def test_sde_matches_software_engineer_user():
    assert role_title_match("SDE II", ["software engineer"])
    assert role_title_match("Software Development Engineer - 1", ["software engineer"])
    assert role_title_match("Senior Software Engineer", ["sde"])


def test_react_matches_frontend_user_and_reverse():
    assert role_title_match("React Developer", ["frontend engineer"])
    assert role_title_match("Senior Frontend Engineer", ["react developer"])
    assert role_title_match("JavaScript Engineer", ["frontend engineer"])


def test_sre_devops_bidirectional():
    assert role_title_match("Site Reliability Engineer", ["devops engineer"])
    assert role_title_match("DevOps Engineer", ["site reliability engineer"])


def test_nurse_family():
    assert role_title_match("RN - ICU Nights", ["registered nurse"])
    assert role_title_match("Registered Nurse (Med/Surg)", ["rn"])


def test_unrelated_titles_still_rejected():
    assert not role_title_match("Bakery Production Manager", ["software engineer"])
    assert not role_title_match("VP, Physical Production", ["frontend engineer"])
    assert not role_title_match("Registered Nurse", ["software engineer"])
