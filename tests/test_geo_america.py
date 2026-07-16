"""Regression: bare 'america' must not classify LatAm/regional locations as US.

Bug: 'america' was a US signal token, so 'Latin America' / 'South America' /
'North America' resolved to 'united states', silently dropping valid LatAm jobs
for non-US users (app/common/geo.py).
"""
from app.common.geo import detect_country, location_allowed


def test_latin_america_is_not_united_states():
    assert detect_country("Remote - Latin America") != "united states"


def test_south_and_north_america_not_us():
    assert detect_country("South America") != "united states"
    # North America spans Canada/Mexico too — must not be pinned to the US.
    assert detect_country("Remote (North America)") != "united states"


def test_explicit_us_signals_still_detected():
    assert detect_country("San Francisco, CA, USA") == "united states"
    assert detect_country("United States of America") == "united states"
    assert detect_country("Remote US") == "united states"
    assert detect_country("Austin, TX") == "united states"


def test_latam_job_kept_for_latam_user():
    # Brazil user, remote OK: a 'Latin America' remote role must be KEPT.
    assert location_allowed("Remote - Latin America", remote=True,
                            preferred_country="brazil", remote_ok=True) is True


def test_us_user_not_shown_latam_only_role():
    # Region gate still drops a LatAm-anchored role for a US user.
    assert location_allowed("Remote - Latin America", remote=True,
                            preferred_country="united states", remote_ok=True) is False
