"""Regression: grounding must fact-check a near-copy bullet that ADDS a metric.

Bug: verify_with_llm only ran when cosine similarity was BELOW threshold, so a
near-copy of a real bullet with an invented number (cosine ~0.9) shipped with no
fact-check (app/tailoring/grounding.py).
"""
from app.tailoring.grounding import _adds_unbacked_metric


def test_added_percentage_is_flagged():
    src = "Optimized database queries to speed up the API."
    tailored = "Optimized database queries, cutting API p99 latency 43% and serving 2,500 requests per minute."
    assert _adds_unbacked_metric(tailored, src) is True


def test_metric_present_in_source_is_not_flagged():
    src = "Cut API latency 43% via query optimization."
    tailored = "Optimized queries, cutting API latency 43%."
    assert _adds_unbacked_metric(tailored, src) is False


def test_no_metric_no_flag():
    src = "Built internal tooling for the data team."
    tailored = "Built internal tooling and dashboards for the data team."
    assert _adds_unbacked_metric(tailored, src) is False


def test_added_throughput_metric_flagged():
    src = "Maintained the ingestion service."
    tailored = "Maintained the ingestion service handling 10,000 records per hour."
    assert _adds_unbacked_metric(tailored, src) is True
