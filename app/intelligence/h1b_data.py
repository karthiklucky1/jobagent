"""USCIS H-1B Employer Data Hub / DOL LCA ingestion.

Loads the *public* employer sponsorship record into the H1BSponsor table and
exposes a fast in-memory lookup so sponsorship scoring can be data-backed
(exact approval rates) instead of curated guesses — without a DB hit per call.

Usage:
    python -m app.intelligence.h1b_data /path/to/h1b_datahubexport.csv

The CSV is public and free to download from:
    https://www.uscis.gov/tools/reports-and-studies/h-1b-employer-data-hub
Column names vary by year, so detection is fuzzy. Nothing here is tenant-scoped
— it's public reference data shared by every user.
"""
from __future__ import annotations

import csv
import logging
import re
from typing import Optional

log = logging.getLogger(__name__)

_CACHE: Optional[dict] = None   # {employer_key: {"approvals","denials","rate","year","name"}}

_SUFFIXES = re.compile(
    r"[,\.]?\s*\b(inc|inc\.|llc|l\.l\.c|ltd|corp|corporation|co|company|"
    r"incorporated|plc|lp|llp|the)\b\.?", re.IGNORECASE
)


def normalize(name: str) -> str:
    """Normalize an employer name for matching (lowercase, strip legal suffixes)."""
    n = (name or "").lower().strip()
    n = _SUFFIXES.sub("", n)
    n = re.sub(r"[^a-z0-9 ]+", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def _find_col(headers: list[str], *needles: str) -> Optional[str]:
    for h in headers:
        hl = h.lower()
        if all(n in hl for n in needles):
            return h
    return None


def ingest_csv(path: str) -> int:
    """Load a USCIS H-1B Employer Data Hub CSV into the H1BSponsor table.
    Returns the number of employer-year rows written. Idempotent (upsert)."""
    from app.db.init_db import get_session, init_db
    from app.db.models import H1BSponsor
    from sqlmodel import select
    init_db()

    written = 0
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []
        emp_col = _find_col(headers, "employer") or _find_col(headers, "petitioner")
        year_col = _find_col(headers, "fiscal") or _find_col(headers, "year")
        if not emp_col:
            raise ValueError(f"Could not find an employer column in {headers}")
        appr_cols = [h for h in headers if "approval" in h.lower()]
        deny_cols = [h for h in headers if "denial" in h.lower()]
        wage_col = _find_col(headers, "wage", "level")

        # Aggregate per (employer_key, year) since the hub has multiple rows.
        agg: dict = {}
        for row in reader:
            name = (row.get(emp_col) or "").strip()
            if not name:
                continue
            key = normalize(name)
            year = None
            if year_col:
                try:
                    year = int(re.sub(r"\D", "", row.get(year_col) or "") or 0) or None
                except ValueError:
                    year = None
            ap = sum(_to_int(row.get(c)) for c in appr_cols)
            dn = sum(_to_int(row.get(c)) for c in deny_cols)
            k = (key, year)
            cur = agg.setdefault(k, {"name": name, "ap": 0, "dn": 0, "wage": ""})
            cur["ap"] += ap
            cur["dn"] += dn
            if wage_col and not cur["wage"]:
                cur["wage"] = (row.get(wage_col) or "").strip()

    with get_session() as session:
        for (key, year), v in agg.items():
            total = v["ap"] + v["dn"]
            rate = (v["ap"] / total) if total else 0.0
            existing = session.exec(
                select(H1BSponsor).where(
                    H1BSponsor.employer_key == key, H1BSponsor.fiscal_year == year
                )
            ).first()
            if existing:
                existing.approvals = v["ap"]
                existing.denials = v["dn"]
                existing.approval_rate = rate
                existing.typical_wage_level = v["wage"]
                existing.employer_name = v["name"]
                session.add(existing)
            else:
                session.add(H1BSponsor(
                    employer_key=key, employer_name=v["name"], fiscal_year=year,
                    approvals=v["ap"], denials=v["dn"], approval_rate=rate,
                    typical_wage_level=v["wage"],
                ))
            written += 1
        session.commit()
    refresh_cache()
    log.info("Ingested %d H-1B employer-year rows from %s", written, path)
    return written


def _to_int(x) -> int:
    try:
        return int(re.sub(r"\D", "", str(x or "")) or 0)
    except ValueError:
        return 0


def refresh_cache() -> None:
    global _CACHE
    _CACHE = None


def _load_cache() -> dict:
    """Build {employer_key: best-record} once from the DB (latest year wins)."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    cache: dict = {}
    try:
        from app.db.init_db import get_session
        from app.db.models import H1BSponsor
        from sqlmodel import select
        with get_session() as session:
            for row in session.exec(select(H1BSponsor)).all():
                prev = cache.get(row.employer_key)
                if not prev or (row.fiscal_year or 0) >= prev["year"]:
                    cache[row.employer_key] = {
                        "approvals": row.approvals, "denials": row.denials,
                        "rate": row.approval_rate, "year": row.fiscal_year or 0,
                        "name": row.employer_name, "wage": row.typical_wage_level,
                    }
    except Exception as e:
        log.debug("H-1B cache load skipped: %s", e)
    _CACHE = cache
    return cache


def lookup(company: str) -> Optional[dict]:
    """O(1) lookup of an employer's public H-1B record (None if absent/empty)."""
    if not company:
        return None
    cache = _load_cache()
    if not cache:
        return None
    key = normalize(company)
    rec = cache.get(key)
    if rec:
        return rec
    # Lenient: match when the normalized name is a prefix of a known employer.
    for k, v in cache.items():
        if k.startswith(key) and len(key) >= 4:
            return v
    return None


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("usage: python -m app.intelligence.h1b_data <csv_path>")
        raise SystemExit(1)
    n = ingest_csv(sys.argv[1])
    print(f"Ingested {n} employer-year rows.")
