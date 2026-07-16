"""Country detection from free-text job locations — the single source of truth.

Used by discovery (drop postings outside the user's preferred country), the
rule filter, and retrieval so every stage of the pipeline agrees on what
country a posting belongs to. Detection is intentionally conservative: when a
location is ambiguous/unknown we KEEP it rather than risk dropping good jobs.
"""
from __future__ import annotations

import re

_US_STATE_CODES = {
    "al","ak","az","ar","ca","co","ct","de","fl","ga","hi","id","il","in","ia",
    "ks","ky","la","me","md","ma","mi","mn","ms","mo","mt","ne","nv","nh","nj",
    "nm","ny","nc","nd","oh","ok","or","pa","ri","sc","sd","tn","tx","ut","vt",
    "va","wa","wv","wi","wy","dc",
}

# country -> signal tokens (lowercase). US handled separately via state codes too.
# NOTE: the "Search jobs in country" select in app/templates/dashboard.html
# mirrors these keys — when adding a country here, add its <option> there too.
_COUNTRY_SIGNALS = {
    # NOTE: bare "america" is deliberately NOT a US signal — "Latin America",
    # "South America" and "North America" (which also spans Canada/Mexico) are
    # not the United States. "United States of America" still matches via
    # "united states"; "USA"/"U.S.A"/"US" cover the abbreviations.
    "united states": ["united states", "usa", "u.s.a", "u.s.", " us ", "remote us", "us remote"],
    "united kingdom": ["united kingdom", " uk", "u.k", "england", "scotland", "wales",
                        "london", "manchester", "birmingham", "edinburgh", "glasgow", "bristol", "leeds"],
    "canada": ["canada", "ontario", "toronto", "vancouver", "montreal", "québec", "quebec",
               "ottawa", "calgary", "alberta", "british columbia", "winnipeg", "edmonton"],
    "india": ["india", "bangalore", "bengaluru", "hyderabad", "mumbai", "new delhi", "delhi",
              "pune", "chennai", "gurgaon", "gurugram", "noida", "kolkata", "ahmedabad"],
    "germany": ["germany", "deutschland", "berlin", "munich", "münchen", "frankfurt", "hamburg", "cologne"],
    "france": ["france", "paris", "lyon", "marseille", "toulouse", "bordeaux"],
    "spain": ["spain", "madrid", "barcelona", "valencia", "seville"],
    "netherlands": ["netherlands", "amsterdam", "rotterdam", "the hague", "utrecht"],
    "ireland": ["ireland", "dublin", "cork", "galway"],
    "australia": ["australia", "sydney", "melbourne", "brisbane", "perth", "canberra"],
    "poland": ["poland", "warsaw", "krakow", "kraków", "wroclaw", "gdansk"],
    "portugal": ["portugal", "lisbon", "porto"],
    "brazil": ["brazil", "brasil", "são paulo", "sao paulo", "rio de janeiro"],
    "mexico": ["mexico", "méxico", "mexico city", "guadalajara", "monterrey"],
    "singapore": ["singapore"],
    "japan": ["japan", "tokyo", "osaka"],
    "philippines": ["philippines", "manila", "cebu", "makati"],
    "ukraine": ["ukraine", "kyiv", "kiev", "lviv"],
    "nigeria": ["nigeria", "lagos", "abuja"],
    "pakistan": ["pakistan", "karachi", "lahore", "islamabad"],
    "argentina": ["argentina", "buenos aires"],
}


# Signals are matched with letter boundaries so "india" can't match "Indiana",
# "us" can't match "status", and "uk" can't match inside another word. Built
# once at import; "u.s." style signals keep working because the boundary is
# letter-based, not \b-based (a trailing "." has no word boundary before space).
_SIGNAL_RES = {
    country: [
        re.compile(rf"(?<![a-z]){re.escape(sig.strip())}(?![a-z])")
        for sig in signals
    ]
    for country, signals in _COUNTRY_SIGNALS.items()
}


# Region anchors ("Remote — EU only", "EMEA", "APAC") → member countries we know.
# A region-locked posting is kept only for users whose country is in the region.
_REGION_MEMBERS = {
    "eu": {"germany", "france", "spain", "netherlands", "ireland", "poland", "portugal"},
    "emea": {"germany", "france", "spain", "netherlands", "ireland", "poland",
             "portugal", "united kingdom", "ukraine", "nigeria"},
    "apac": {"india", "australia", "singapore", "japan", "philippines", "pakistan"},
    "latam": {"brazil", "mexico", "argentina"},
}
_REGION_RES = {
    region: [re.compile(rf"(?<![a-z]){re.escape(t)}(?![a-z])") for t in tokens]
    for region, tokens in {
        "eu": ["eu only", "eu-only", "european union", "eea", "europe only", "within europe", "european residents"],
        "emea": ["emea"],
        "apac": ["apac", "asia-pacific", "asia pacific"],
        "latam": ["latam", "latin america"],
    }.items()
}


def norm_country(name: str) -> str:
    """Normalize a country name/alias to its canonical lowercase form."""
    n = (name or "").strip().lower().rstrip(".")
    aliases = {
        "us": "united states", "u.s": "united states", "usa": "united states",
        "u.s.a": "united states", "america": "united states", "united states of america": "united states",
        "uk": "united kingdom", "u.k": "united kingdom", "england": "united kingdom",
        "great britain": "united kingdom", "britain": "united kingdom",
        "deutschland": "germany", "holland": "netherlands", "the netherlands": "netherlands",
        "bharat": "india", "republic of india": "india", "brasil": "brazil",
        "méxico": "mexico", "españa": "spain", "republic of ireland": "ireland",
        "aus": "australia", "ca": "canada", "can": "canada",
    }
    return aliases.get(n, n)


def detect_region(location: str) -> str:
    """Region anchor ('eu', 'emea', 'apac', 'latam') in a location, '' if none."""
    loc = " " + (location or "").lower().strip() + " "
    for region, res in _REGION_RES.items():
        if any(r.search(loc) for r in res):
            return region
    return ""


def detect_country(location: str) -> str:
    """Best-effort country guess from a free-text location. '' when unknown."""
    loc = " " + (location or "").lower().strip() + " "
    if not loc.strip():
        return ""
    # Explicit US signals first ("USA", "Remote US", ...).
    if any(r.search(loc) for r in _SIGNAL_RES["united states"]):
        return "united states"
    # Foreign countries BEFORE the bare state-code heuristic: many ISO country/
    # province codes collide with US state codes ("Toronto, CA" / "Bengaluru, IN"
    # / "Berlin, DE" would otherwise read as California/Indiana/Delaware), and a
    # known foreign city is a much stronger signal than a trailing 2-letter code.
    for country, sig_res in _SIGNAL_RES.items():
        if country == "united states":
            continue
        if any(r.search(loc) for r in sig_res):
            return country
    # Only now: treat "city, XX" as a US state code.
    if re.search(r",\s*[a-z]{2}\b", loc) and any(t in _US_STATE_CODES for t in re.findall(r",\s*([a-z]{2})\b", loc)):
        return "united states"
    return ""


def location_allowed(location: str, remote: bool, preferred_country: str, remote_ok: bool) -> bool:
    """True if a posting should be kept for a user targeting `preferred_country`.

    Remote is NOT borderless: a remote role anchored to another country
    ("Remote — Berlin") or region ("Remote, EU only", "EMEA") still requires
    work authorization there, so it is treated like an on-site role in that
    country/region. Remote is kept only when it matches the user's own country,
    is truly global, or is unspecified.

    An EMPTY ``preferred_country`` means the user hasn't chosen one — no
    country gate is applied (better to show everything than to silently
    assume the wrong country).
    """
    preferred = norm_country(preferred_country)
    if not preferred:
        return True
    loc = (location or "").lower()
    region = detect_region(loc)
    if region and preferred not in _REGION_MEMBERS[region]:
        return False  # region-locked posting, user outside the region
    detected = detect_country(loc)
    if remote_ok and (remote or "remote" in loc or "anywhere" in loc or "worldwide" in loc):
        return (not detected) or detected == preferred
    if not detected:
        return True  # ambiguous/unknown — keep rather than over-filter
    return detected == preferred
