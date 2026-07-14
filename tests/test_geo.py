"""Tests for app.common.geo — the shared location→country detection."""
from app.common.geo import detect_country, location_allowed, norm_country


def test_norm_country_aliases():
    assert norm_country("US") == "united states"
    assert norm_country("USA") == "united states"
    assert norm_country("United States of America") == "united states"
    assert norm_country("UK") == "united kingdom"
    assert norm_country("England") == "united kingdom"
    assert norm_country("Germany") == "germany"
    assert norm_country("") == ""


def test_detect_country_us_signals():
    assert detect_country("Austin, TX") == "united states"
    assert detect_country("Remote - US") == "united states"
    assert detect_country("New York, NY, United States") == "united states"


def test_detect_country_foreign():
    assert detect_country("London, UK") == "united kingdom"
    assert detect_country("Berlin, Germany") == "germany"
    assert detect_country("Bengaluru") == "india"
    assert detect_country("Toronto, Ontario") == "canada"


def test_detect_country_unknown_is_empty():
    assert detect_country("") == ""
    assert detect_country("Anywhere") == ""
    # "Indiana" must not be mistaken for India, "Brooklyn" not for UK.
    assert detect_country("Indianapolis, Indiana") in ("", "united states")
    assert detect_country("Brooklyn") == ""


def test_location_allowed_respects_preferred_country():
    # A UK user keeps London and drops New York.
    assert location_allowed("London, UK", False, "United Kingdom", remote_ok=False)
    assert not location_allowed("New York, NY", False, "United Kingdom", remote_ok=False)
    # A US user keeps New York and drops London.
    assert location_allowed("New York, NY", False, "United States", remote_ok=False)
    assert not location_allowed("London, UK", False, "United States", remote_ok=False)


def test_location_allowed_keeps_remote_and_unknown():
    # Remote is country-scoped: a remote role anchored to another country still
    # needs work authorization there, so it is dropped.
    assert not location_allowed("Remote (Berlin)", True, "United States", remote_ok=True)
    assert location_allowed("Remote - United States", True, "United States", remote_ok=True)
    assert location_allowed("Anywhere", False, "India", remote_ok=True)
    # Ambiguous locations are kept rather than over-filtered.
    assert location_allowed("Main Office", False, "United States", remote_ok=False)


# ── Regression: audit fixes (state-code collision, regions, empty country) ────

def test_foreign_city_beats_colliding_state_code():
    from app.common.geo import detect_country
    # CA/IN/DE are both US state codes and country codes — the known foreign
    # city must win over the bare 2-letter heuristic.
    assert detect_country("Toronto, CA") == "canada"
    assert detect_country("Bengaluru, IN") == "india"
    assert detect_country("Berlin, DE") == "germany"
    # Genuine US "city, state" still detects as US.
    assert detect_country("Austin, TX") == "united states"
    assert detect_country("San Francisco, CA") == "united states"


def test_region_anchored_remote_gated_by_membership():
    from app.common.geo import location_allowed
    # EU-only remote: kept for a German user, dropped for a US user.
    assert location_allowed("Remote, EU only", True, "Germany", True) is True
    assert location_allowed("Remote, EU only", True, "United States", True) is False
    assert location_allowed("Remote (APAC)", True, "India", True) is True
    assert location_allowed("Remote (APAC)", True, "United States", True) is False


def test_empty_preferred_country_means_no_gate():
    from app.common.geo import location_allowed
    assert location_allowed("Berlin, Germany", False, "", True) is True
    assert location_allowed("Austin, TX", False, "", True) is True
    assert location_allowed("Remote, EU only", True, "", True) is True


def test_norm_country_aliases_expanded():
    from app.common.geo import norm_country
    assert norm_country("Deutschland") == "germany"
    assert norm_country("Holland") == "netherlands"
    assert norm_country("Brasil") == "brazil"
    assert norm_country("U.K.") == "united kingdom"
