"""Public sponsor-registry ingestion (multi-country).

Loads *public* employer sponsorship records into the H1BSponsor table and
exposes a fast in-memory lookup so sponsorship scoring can be data-backed
instead of curated guesses — without a DB hit per call.

Two kinds of source files:
- United States — USCIS H-1B Employer Data Hub CSV (approval/denial stats):
  https://www.uscis.gov/tools/reports-and-studies/h-1b-employer-data-hub
  → ``ingest_csv(path)``
- Everywhere else — official licensed-sponsor registers, where being listed
  means "authorised to hire foreign workers" (UK Register of Licensed
  Sponsors, Canada positive-LMIA employers, NL IND recognised sponsors,
  Ireland employment-permit lists, Australia accredited sponsors, ...):
  → ``ingest_register(path, country)``

Usage:
    python -m app.intelligence.h1b_data /path/to/h1b_datahubexport.csv
    python -m app.intelligence.h1b_data /path/to/uk_register.csv "united kingdom"

Column names vary by file/year, so detection is fuzzy. Nothing here is
tenant-scoped — it's public reference data shared by every user.
"""
from __future__ import annotations

import csv
import logging
import re
from typing import Optional

log = logging.getLogger(__name__)

_CACHE: Optional[dict] = None   # {country: {employer_key: record-dict}}
_CACHE_AT: float = 0.0          # when the cache was last built (epoch seconds)
# The sponsor table only changes when an admin uploads a new file, and the
# upload path invalidates this cache explicitly — so effectively "load once".
# The 24h TTL is only a safety net for multi-process deployments where another
# worker did the upload; refreshes happen in the background, never on-request.
_CACHE_TTL: float = 24 * 3600.0

# Last ingest result, surfaced to the admin upload page so background errors
# (bad columns, wrong file) are visible instead of silently writing 0 rows.
LAST_INGEST: dict = {"rows": 0, "error": "", "headers": [], "at": None}

_SUFFIXES = re.compile(
    r"[,\.]?\s*\b(inc|inc\.|llc|l\.l\.c|ltd|corp|corporation|co|company|"
    r"incorporated|plc|lp|llp|the|limited|gmbh|pty|b\.v|bv|s\.r\.l|sarl)\b\.?",
    re.IGNORECASE
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


def _read_text(path: str) -> str:
    """Read the CSV as text, auto-detecting encoding (USCIS ships UTF-16!)."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16", errors="replace")
    elif raw[:3] == b"\xef\xbb\xbf":
        text = raw.decode("utf-8-sig", errors="replace")
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="replace")
    # Strip stray BOM / null bytes that survive a mis-encoded export.
    return text.replace("\x00", "").replace("﻿", "")


def _clean_header(h: str) -> str:
    return (h or "").lstrip("﻿").replace("\x00", "").strip()


def _open_reader(path: str):
    """Return a DictReader over the decoded text, delimiter auto-sniffed,
    with cleaned header names. (USCIS files are UTF-16 + have a junk leading
    'Line by line' column and trailing spaces in some headers.)"""
    import io
    text = _read_text(path)
    sample = text[:8192]
    try:
        delim = csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except Exception:
        first = sample.splitlines()[0] if sample else ""
        delim = max(",\t;|", key=lambda d: first.count(d)) if first else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delim)
    orig = reader.fieldnames or []
    reader.fieldnames = [_clean_header(h) for h in orig]
    return reader


US = "united states"


def ingest_csv(path: str) -> int:
    """Load a USCIS H-1B Employer Data Hub CSV into the H1BSponsor table.
    Idempotent per fiscal year (replaces that year's US rows). Fast bulk insert."""
    import datetime as _dt
    from app.db.init_db import get_session, init_db
    from app.db.models import H1BSponsor
    from sqlmodel import delete
    init_db()
    LAST_INGEST.update(rows=0, error="", headers=[], at=_dt.datetime.utcnow().isoformat())

    reader = _open_reader(path)
    headers = reader.fieldnames or []
    LAST_INGEST["headers"] = headers
    emp_col = (_find_col(headers, "employer") or _find_col(headers, "petitioner")
               or _find_col(headers, "company") or _find_col(headers, "name"))
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
            country=US, record_type="stats",
        ))

    written = 0
    with get_session() as session:
        # Idempotent: clear the US fiscal years we're about to (re)load, then bulk add.
        for y in years_present:
            session.exec(delete(H1BSponsor)
                         .where(H1BSponsor.fiscal_year == y)
                         .where(H1BSponsor.country == US))
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


def ingest_register(path: str, country: str) -> int:
    """Load a licensed-sponsor register CSV for a non-US country.

    Understands the common official formats and falls back to "one employer
    name per row" for anything else:
    - United Kingdom — Register of Licensed Sponsors ("Organisation Name",
      "Route", "Type & Rating"): routes are aggregated into ``detail``.
    - Canada — positive-LMIA employer files ("Employer", "Program Stream",
      "Approved Positions"): positions land in ``approvals``.
    - Generic — any file with an organisation/employer/company/name column
      (or a single column of names).

    Being listed means "authorised to sponsor" — these are license records
    (record_type="license"), not approval statistics. Idempotent per country:
    each upload replaces that country's license rows.
    """
    import datetime as _dt
    from app.common.geo import norm_country
    from app.db.init_db import get_session, init_db
    from app.db.models import H1BSponsor
    from sqlmodel import delete
    init_db()
    cc = norm_country(country)
    if not cc or cc == US:
        raise ValueError("ingest_register is for non-US registers; "
                         "US stats go through ingest_csv.")
    LAST_INGEST.update(rows=0, error="", headers=[], at=_dt.datetime.utcnow().isoformat())

    reader = _open_reader(path)
    headers = reader.fieldnames or []
    LAST_INGEST["headers"] = headers
    emp_col = (_find_col(headers, "organisation", "name")
               or _find_col(headers, "organization", "name")
               or _find_col(headers, "employer")
               or _find_col(headers, "company")
               or _find_col(headers, "sponsor")
               or _find_col(headers, "name"))
    header_row_is_data = False
    if not emp_col:
        if len(headers) == 1:
            # Plain one-name-per-line list: the first name was eaten as the
            # header — treat that value as data too.
            emp_col = headers[0]
            header_row_is_data = True
        else:
            raise ValueError("Couldn't find an organisation/employer/company "
                             f"column. Columns seen: {headers}")
    # UK routes / Canada program stream / visa subclass → human-readable detail.
    detail_col = (_find_col(headers, "route") or _find_col(headers, "stream")
                  or _find_col(headers, "type", "rating")
                  or _find_col(headers, "scheme") or _find_col(headers, "permit", "type"))
    # Canada LMIA: approved positions/LMIAs count as evidence of volume.
    pos_col = (_find_col(headers, "position") or _find_col(headers, "approved", "lmia"))

    agg: dict = {}

    def _take(name: str, row: dict) -> None:
        name = (name or "").strip()
        key = normalize(name)
        if not key:
            return
        cur = agg.setdefault(key, {"name": name, "ap": 0, "details": []})
        if pos_col:
            cur["ap"] += _to_int(row.get(pos_col))
        if detail_col:
            d = (row.get(detail_col) or "").strip()
            if d and d not in cur["details"]:
                cur["details"].append(d)

    if header_row_is_data:
        _take(emp_col, {})
    for row in reader:
        _take(row.get(emp_col) or "", row)

    if not agg:
        raise ValueError(f"Parsed 0 rows. Detected employer column '{emp_col}'. "
                         f"Columns seen: {headers}")

    now = _dt.datetime.utcnow()
    objs = [
        H1BSponsor(
            employer_key=key, employer_name=v["name"][:300], fiscal_year=None,
            approvals=v["ap"], denials=0, approval_rate=0.0,
            country=cc, record_type="license",
            detail="; ".join(v["details"])[:300], updated_at=now,
        )
        for key, v in agg.items()
    ]

    written = 0
    with get_session() as session:
        # Idempotent: each upload replaces this country's license records.
        session.exec(delete(H1BSponsor)
                     .where(H1BSponsor.country == cc)
                     .where(H1BSponsor.record_type == "license"))
        session.commit()
        for i in range(0, len(objs), 1000):
            session.add_all(objs[i:i + 1000])
            session.commit()
            written += len(objs[i:i + 1000])
    refresh_cache()
    LAST_INGEST.update(rows=written)
    log.info("Ingested %d licensed sponsors for %s from %s", written, cc, path)
    return written


def refresh_cache() -> None:
    global _CACHE, _CACHE_AT
    _CACHE = None
    _CACHE_AT = 0.0


_KEYS_SORTED: dict | None = None   # {country: sorted list of employer_keys}
_REFRESH_LOCK = None  # created lazily (threading.Lock)


def _build_cache() -> dict:
    """{country: {employer_key: best-record}} across stats AND license rows."""
    cache: dict = {}
    try:
        from app.db.init_db import get_session
        from app.db.models import H1BSponsor
        from sqlmodel import select
        with get_session() as session:
            for row in session.exec(select(H1BSponsor)).all():
                cc = getattr(row, "country", None) or US
                by_key = cache.setdefault(cc, {})
                prev = by_key.get(row.employer_key)
                if not prev or (row.fiscal_year or 0) >= prev["year"]:
                    by_key[row.employer_key] = {
                        "approvals": row.approvals, "denials": row.denials,
                        "rate": row.approval_rate, "year": row.fiscal_year or 0,
                        "name": row.employer_name, "wage": row.typical_wage_level,
                        "country": cc,
                        "record_type": getattr(row, "record_type", None) or "stats",
                        "detail": getattr(row, "detail", None) or "",
                    }
    except Exception as e:
        log.debug("Sponsor-registry cache load skipped: %s", e)
    return cache


def _refresh_cache() -> None:
    global _CACHE, _CACHE_AT, _KEYS_SORTED
    import time as _time
    cache = _build_cache()
    _CACHE = cache
    _KEYS_SORTED = {cc: sorted(by_key.keys()) for cc, by_key in cache.items()}
    _CACHE_AT = _time.time()


def warm_cache_async() -> None:
    """Warm the cache off the request path (called at server startup)."""
    import threading
    threading.Thread(target=_refresh_cache, daemon=True).start()


def _load_cache() -> dict:
    """{employer_key: best-record}, cached in-process.

    Stale-while-revalidate: once loaded, an expired cache is returned
    IMMEDIATELY and refreshed on a background thread — reloading a
    50k-row table must never block a dashboard render (it used to add
    seconds to the first request after every TTL expiry)."""
    global _REFRESH_LOCK
    import time as _time
    if _CACHE is not None:
        if (_time.time() - _CACHE_AT) >= _CACHE_TTL:
            import threading
            if _REFRESH_LOCK is None:
                _REFRESH_LOCK = threading.Lock()
            if _REFRESH_LOCK.acquire(blocking=False):
                def _bg():
                    try:
                        _refresh_cache()
                    finally:
                        _REFRESH_LOCK.release()
                threading.Thread(target=_bg, daemon=True).start()
        return _CACHE
    # First call in this process (startup warm not done yet) — load once.
    _refresh_cache()
    return _CACHE or {}


def lookup(company: str, country: str = US) -> Optional[dict]:
    """O(log n) lookup of an employer's public sponsor record for a country
    (None if absent). Default country keeps existing US/H-1B call sites intact."""
    if not company:
        return None
    cache = _load_cache()
    if not cache:
        return None
    try:
        from app.common.geo import norm_country
        cc = norm_country(country) or US
    except Exception:
        cc = country or US
    by_key = cache.get(cc)
    if not by_key:
        return None
    key = normalize(company)
    rec = by_key.get(key)
    if rec:
        return rec
    # Lenient prefix match via binary search (a linear scan over ~50k keys per
    # job card made dashboard renders crawl).
    keys_sorted = (_KEYS_SORTED or {}).get(cc)
    if len(key) >= 4 and keys_sorted:
        import bisect
        i = bisect.bisect_left(keys_sorted, key)
        if i < len(keys_sorted) and keys_sorted[i].startswith(key):
            return by_key.get(keys_sorted[i])
    return None


def has_country_data(country: str) -> bool:
    """True if a sponsor register has been loaded for this country — lets the
    caller distinguish "employer not in the register" from "no register yet"."""
    cache = _load_cache()
    try:
        from app.common.geo import norm_country
        cc = norm_country(country) or ""
    except Exception:
        cc = country or ""
    return bool(cache.get(cc))


if __name__ == "__main__":
    import sys, os
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("usage: python -m app.intelligence.h1b_data <csv_path> [country]")
        print('       country defaults to the US (USCIS stats); pass e.g. "united kingdom"')
        print("       to load that country's licensed-sponsor register instead.")
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
    _country = sys.argv[2].strip().lower() if len(sys.argv) > 2 else US
    if _country and _country not in (US, "us", "usa"):
        n = ingest_register(path, _country)
        print(f"Ingested {n} licensed sponsors for {_country}.")
    else:
        n = ingest_csv(path)
        print(f"Ingested {n} employer-year rows.")
