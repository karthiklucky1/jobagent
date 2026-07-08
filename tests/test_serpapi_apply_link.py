"""SerpAPI apply-link ranking: prefer the company's own ATS/careers form
over aggregator middlemen (Jobright, ZipRecruiter, LinkedIn, ...) and
Google redirects."""

from app.discovery.sources.serpapi import _best_apply_url, _rank_apply_link


def test_direct_ats_beats_aggregators():
    options = [
        {"title": "Jobright", "link": "https://jobright.ai/jobs/info/abc123"},
        {"title": "ZipRecruiter", "link": "https://www.ziprecruiter.com/c/Acme/job/xyz"},
        {"title": "Acme Careers", "link": "https://boards.greenhouse.io/acme/jobs/456"},
        {"title": "LinkedIn", "link": "https://www.linkedin.com/jobs/view/789"},
    ]
    assert _best_apply_url(options) == "https://boards.greenhouse.io/acme/jobs/456"


def test_company_site_beats_aggregators():
    options = [
        {"title": "Jobright", "link": "https://jobright.ai/jobs/info/abc123"},
        {"title": "Careers", "link": "https://careers.acme.com/openings/staff-engineer"},
    ]
    assert _best_apply_url(options) == "https://careers.acme.com/openings/staff-engineer"


def test_aggregator_beats_google_redirect():
    options = [
        {"title": "Google", "link": "https://www.google.com/search?q=acme+jobs"},
        {"title": "Jobright", "link": "https://jobright.ai/jobs/info/abc123"},
    ]
    assert _best_apply_url(options) == "https://jobright.ai/jobs/info/abc123"


def test_empty_options_returns_empty():
    assert _best_apply_url([]) == ""
    assert _best_apply_url([{"title": "x"}]) == ""


def test_rank_ordering():
    ats = _rank_apply_link("https://jobs.lever.co/acme/1234")
    company = _rank_apply_link("https://acme.com/careers/apply/1234")
    aggregator = _rank_apply_link("https://jobright.ai/jobs/info/abc")
    google = _rank_apply_link("https://www.google.com/search?q=jobs")
    assert ats > company > aggregator > google > _rank_apply_link("")
