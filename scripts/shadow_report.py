"""Agreement report for the distilled scorer's shadow mode.

Reads the FunnelEvents that shadow mode records (stage="shadow_score") and
prints the numbers that decide the flip: MAE, within-10, correlation, and —
what actually matters for the product — shortlist-decision agreement at the
threshold (would the local model have shortlisted the same jobs?).

    python -m scripts.shadow_report [--days 7]
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta

log = logging.getLogger(__name__)


def report(days: int) -> dict:
    from sqlmodel import select

    from app.config import settings
    from app.db.init_db import get_session
    from app.db.models import FunnelEvent

    cutoff = datetime.utcnow() - timedelta(days=days)
    with get_session() as session:
        events = session.exec(
            select(FunnelEvent).where(
                FunnelEvent.stage == "shadow_score",
                FunnelEvent.created_at >= cutoff,
            )
        ).all()

    pairs = []
    for e in events:
        try:
            m = json.loads(e.metadata_json or "{}")
            pairs.append((float(m["llm"]), float(m["local"])))
        except Exception:
            continue
    if not pairs:
        return {"n": 0}

    n = len(pairs)
    errs = [abs(a - b) for a, b in pairs]
    mae = sum(errs) / n
    within10 = 100.0 * sum(1 for e in errs if e <= 10) / n

    # Pearson correlation
    ma = sum(a for a, _ in pairs) / n
    mb = sum(b for _, b in pairs) / n
    cov = sum((a - ma) * (b - mb) for a, b in pairs)
    va = sum((a - ma) ** 2 for a, _ in pairs) ** 0.5
    vb = sum((b - mb) ** 2 for _, b in pairs) ** 0.5
    corr = cov / (va * vb) if va and vb else 0.0

    # The decision that matters: same side of the shortlist threshold?
    thr = settings.shortlist_score_threshold
    agree = 100.0 * sum(1 for a, b in pairs if (a >= thr) == (b >= thr)) / n

    return {"n": n, "mae": round(mae, 1), "within10_pct": round(within10),
            "pearson": round(corr, 3), "shortlist_decision_agreement_pct": round(agree, 1),
            "threshold": thr}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    r = report(args.days)
    if not r.get("n"):
        log.info("No shadow_score events in the last %d day(s) — is the model "
                 "deployed at LOCAL_SCORER_PATH and LOCAL_SCORER_SHADOW on?", args.days)
        return
    log.info("Shadow agreement over last %d day(s), n=%d finals:", args.days, r["n"])
    log.info("  MAE vs LLM:                     %.1f points", r["mae"])
    log.info("  within 10 points:               %d%%", r["within10_pct"])
    log.info("  Pearson correlation:            %.3f", r["pearson"])
    log.info("  shortlist decision agreement:   %.1f%% (threshold=%d)",
             r["shortlist_decision_agreement_pct"], r["threshold"])
    log.info("Flip guidance: decision agreement >= 90%% sustained for a week is the "
             "signal that local-first scoring will not change what users see.")


if __name__ == "__main__":
    main()
