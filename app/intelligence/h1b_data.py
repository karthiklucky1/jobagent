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

# Last ingest result, surfaced to the admin upload page so background errors
# (bad columns, wrong file) are visible instead of silently writing 0 rows.
LAST_INGEST: dict = {"rows": 0, "error": "", "headers": [], "at": None}

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


def _open_reader(path: str):
    """Open the CSV, sniffing the delimiter (comma/tab/semicolon/pipe)."""
    fh = open(path, newline="", encoding="utf-8-sig", errors="replace")
    sample = fh.read(8192)
    fh.seek(0)
    delim = ","
    try:
        delim = csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except Exception:
        # Fall back: pick whichever common delimiter appears most in the header.
        first = sample.splitlines()[0] if sample else ""
        delim = max(",\t;|", key=lambda d: first.count(d)) if first else ","
    return fh, csv.DictReader(fh, delimiter=delim)


def ingest_csv(path: str) -> int:
    """Load a USCIS H-1B Employer Data Hub CSV into the H1BSponsor table.
    Idempotent per fiscal year (replaces that year's rows). Fast bulk insert."""
    import datetime as _dt
    from app.db.init_db import get_session, init_db
    from app.db.models import H1BSponsor
    from sqlmodel import delete
    init_db()
    LAST_INGEST.update(rows=0, error="", headers=[], at=_dt.datetime.utcnow().isoformat())

    fh, reader = _open_reader(path)
    try:
        headers = reader.fieldnames or []
        LAST_INGEST["headers"] = headers
        emp_col = _find_col(headers, "employer") or _find_col(headers, "petitioner")
        if not emp_col:
            # Last resort: a column literally named like a company field.
            emp_col = _find_col(headers, "company") or _find_col(headers, "name")
        if not emp_col:
            raise ValueError("Couldn't find an employer/company column. "
                             f"Columns seen: {headers}")
        year_col = _find_col(headers, "fiscal") or _find_col(headers, "year")
        appr_cols = [h for h in headers if "approval" in h.lower()]
        deny_cols = [h for h in headers if "denial" in h.lower()]
        if not appr_cols:
            appr_cols = [h for h in headers if h.lower().strip() in ("approved", "approvals", "certified")]
        if not deny_cols:
            deny_cols = [h for h in headers if h.lower().strip() in ("denied", "denials")]
        wage_col = _find_col(headers, "wage", "level")

        agg: dict = {}
        for row in reader:
            name = (row.get(emp_col) or "").strip()
            if not name:
                continue
            key = normalize(name)
            if not key:
                continue
            year = None
            if year_col:
                try:
                    year = int(re.sub(r"\D", "", row.get(year_col) or "") or 0) or None
                except ValueError:
                    year = None
            ap = sum(_to_int(row.get(c)) for c in appr_cols)
            dn = sum(_to_int(row.get(c)) for c in deny_cols)
            cur = agg.setdefault((key, year), {"name": name, "ap": 0, "dn": 0, "wage": ""})
            cur["ap"] += ap
            cur["dn"] += dn
            if wage_col and not cur["wage"]:
                cur["wage"] = (row.get(wage_col) or "").strip()
    finally:
        fh.close()

    if not agg:
        raise ValueError(f"Parsed 0 rows. Detected employer column '{emp_col}'. "
                         f"Columns seen: {headers}")

    now = _dt.datetime.utcnow()
    objs = []
    years_present = set()
    for (key, year), v in agg.items():
        years_present.add(year)
        total = v["ap"] + v["dn"]
        objs.append(H1BSponsor(
            employer_key=key, employer_name=v["name"][:300], fiscal_year=year,
            approvals=v["ap"], denials=v["dn"],
            approval_rate=(v["ap"] / total) if total else 0.0,
            typical_wage_level=v["wage"][:40], updated_at=now,
        ))

    written = 0
    with get_session() as session:
        # Idempotent: clear the fiscal years we're about to (re)load, then bulk add.
        for y in years_present:
            session.exec(delete(H1BSponsor).where(H1BSponsor.fiscal_year == y))
        session.commit()
        for i in range(0, len(objs), 1000):
            session.add_all(objs[i:i + 1000])
            session.commit()
            written += len(objs[i:i + 1000])
    refresh_cache()
    LAST_INGEST.update(rows=written)
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
    import sys, os
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("usage: python -m app.intelligence.h1b_data <csv_path>")
        raise SystemExit(1)
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"\n❌ File not found: {path}")
        print(f"   Current directory: {os.getcwd()}")
        print("   This shell is the SERVER — it can't see files on your laptop.")
        print("   Either download the CSV onto this box first, e.g.:")
        print('     curl -L -o h1b.csv "<direct USCIS CSV link>"')
        print("   then re-run. Or run this command on your laptop with")
        print("   DATABASE_URL set to your Supabase connection string.\n")
        raise SystemExit(2)
    n = ingest_csv(path)
    print(f"Ingested {n} employer-year rows.")
