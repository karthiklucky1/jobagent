"""Fresh-first LLM-budget allocation in run_matching.

The reranker only scores the top ``llm_rerank_cap`` candidates per run. These
tests lock the rule that those scarce slots go to the freshest postings first
(so a day's new jobs get scored before older-but-similar ones), breaking ties
within a freshness tier by weighted fit priority.
"""
from datetime import datetime, timedelta

from app.db.models import Job, JobSource
from app.matching.fresh_budget import (
    _FRESHNESS_TIER_HOURS,
    freshness_tier as _freshness_tier,
    order_fresh_first as _order_fresh_first,
    order_fit_first as _order_fit_first,
    reserve_fresh_slice as _reserve_fresh_slice,
)


def _job(posted_hours_ago=None, first_seen_hours_ago=None) -> Job:
    now = datetime.utcnow()
    return Job(
        source=JobSource.MANUAL,
        external_id="x",
        company="C",
        title="Engineer",
        url="http://x",
        posted_at=(now - timedelta(hours=posted_hours_ago)) if posted_hours_ago is not None else None,
        first_seen=(now - timedelta(hours=first_seen_hours_ago)) if first_seen_hours_ago is not None else None,
    )


def test_freshness_tier_buckets():
    assert _freshness_tier(_job(posted_hours_ago=1)) == 0     # <24h
    assert _freshness_tier(_job(posted_hours_ago=48)) == 1    # <72h
    assert _freshness_tier(_job(posted_hours_ago=120)) == 2   # <168h
    assert _freshness_tier(_job(posted_hours_ago=400)) == len(_FRESHNESS_TIER_HOURS)  # older


def test_freshness_tier_falls_back_to_first_seen():
    # No posted_at, but first_seen is fresh → treated as fresh.
    assert _freshness_tier(_job(first_seen_hours_ago=2)) == 0


def test_freshness_tier_undated_is_oldest():
    # Neither timestamp → sorts to the oldest tier, never outranks a real posting.
    assert _freshness_tier(_job()) == len(_FRESHNESS_TIER_HOURS)


def test_fresh_posting_beats_higher_similarity_stale_one():
    # jid 1 = stale but very high similarity; jid 2 = fresh, lower similarity.
    to_rerank = [(1, 0.95), (2, 0.40)]
    tier_of = {1: 3, 2: 0}
    prio_of = {1: 0.95, 2: 0.40}
    ordered = _order_fresh_first(to_rerank, tier_of, prio_of)
    assert [jid for jid, _ in ordered] == [2, 1]  # fresh job first despite lower fit


def test_within_tier_higher_priority_wins():
    # Two equally-fresh jobs → the higher weighted priority is scored first.
    to_rerank = [(1, 0.3), (2, 0.8)]
    tier_of = {1: 0, 2: 0}
    prio_of = {1: 0.3, 2: 0.8}
    ordered = _order_fresh_first(to_rerank, tier_of, prio_of)
    assert [jid for jid, _ in ordered] == [2, 1]


def test_cap_keeps_fresh_and_drops_stale():
    # With a budget of 2, the two freshest survive; the stale high-sim one is cut.
    to_rerank = [(1, 0.99), (2, 0.5), (3, 0.6)]
    tier_of = {1: 3, 2: 0, 3: 0}       # jid1 stale, jid2/jid3 fresh
    prio_of = {1: 0.99, 2: 0.5, 3: 0.6}
    ordered = _order_fresh_first(to_rerank, tier_of, prio_of)
    kept = [jid for jid, _ in ordered[:2]]
    assert set(kept) == {2, 3}         # both fresh jobs kept
    assert 1 not in kept               # stale high-similarity job dropped


def test_missing_entries_default_to_oldest_tier():
    # A jid absent from tier_of must not float to the top.
    to_rerank = [(1, 0.9), (2, 0.2)]
    tier_of = {2: 0}                   # jid1 missing → treated as oldest
    prio_of = {1: 0.9, 2: 0.2}
    ordered = _order_fresh_first(to_rerank, tier_of, prio_of)
    assert [jid for jid, _ in ordered] == [2, 1]


# ── Phase 3 shortlist creation stays FIT-FIRST (regression guard) ─────────────

def test_fit_first_shortlist_order_is_by_score_not_freshness():
    # jid 1 = fresher but marginal fit; jid 2 = older but strong fit.
    # Shortlist creation must take the strong-fit one first so it never loses a
    # scarce daily/company-cap slot to a marginal fresher role.
    to_rerank = [(1, 0.8), (2, 0.3)]
    score_of = {1: 40.0, 2: 90.0}
    ordered = _order_fit_first(to_rerank, score_of)
    assert [jid for jid, _ in ordered] == [2, 1]


def test_fit_first_same_company_strong_role_wins_last_cap_slot():
    # Concrete regression from review: two same-company survivors, 1 cap slot.
    # The higher LLM fit score must be first regardless of freshness tier.
    fresh_marginal = (10, 0.9)   # posted 2h ago, fit 40
    older_strong = (11, 0.5)     # posted 4d ago, fit 90
    score_of = {10: 40.0, 11: 90.0}
    ordered = _order_fit_first([fresh_marginal, older_strong], score_of)
    assert ordered[0][0] == 11   # strong-fit older role claims the slot first


def test_fit_first_unscored_jobs_sort_last():
    to_rerank = [(1, 0.9), (2, 0.2), (3, 0.5)]
    score_of = {1: 50.0, 3: 70.0}    # jid2 unscored → 0.0 → last
    ordered = _order_fit_first(to_rerank, score_of)
    assert [jid for jid, _ in ordered] == [3, 1, 2]


# ── Cross-encoder freshness reserve (retrieval-stage budget) ──────────────────
# Guards the fix for "fresh jobs show in All Jobs but never reach the Shortlist":
# narrowing the unscored corpus to the top ce_cap by RELEVANCE alone starved
# brand-new postings out of the cross-encoder, so they were never LLM-scored.

def _idkey(pair):
    return pair[0]


def test_reserve_guarantees_freshest_a_cross_encoder_slot():
    # corpus newest-first (id 0 = freshest); relevance is INVERSELY correlated
    # with freshness (older jobs are the most resume-similar) — the exact
    # scenario that starved fresh postings before the fix.
    corpus = [(i, float(i)) for i in range(100)]              # newest-first order
    ranked = sorted(corpus, key=lambda p: p[1], reverse=True)  # relevance-first
    kept = {jid for jid, _ in _reserve_fresh_slice(corpus, ranked, 20, key=_idkey)}
    # OLD behavior (pure relevance) would keep ids 80-99 and ZERO of the freshest.
    assert {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.issubset(kept)      # freshest-10 guaranteed
    assert 99 in kept                                         # top-relevance still kept


def test_reserve_is_cost_neutral_and_deduped():
    corpus = [(i, float(i)) for i in range(100)]
    ranked = sorted(corpus, key=lambda p: p[1], reverse=True)
    out = _reserve_fresh_slice(corpus, ranked, 20, key=_idkey)
    assert len(out) == 20                                     # exactly ce_cap
    assert len({jid for jid, _ in out}) == 20                 # no duplicates


def test_reserve_noop_when_budget_covers_corpus():
    # ce_cap >= corpus → everyone is scored, order follows relevance ranking.
    corpus = [(i, float(i)) for i in range(10)]
    ranked = sorted(corpus, key=lambda p: p[1], reverse=True)
    out = _reserve_fresh_slice(corpus, ranked, 50, key=_idkey)
    assert out == ranked
