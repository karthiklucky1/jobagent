"""DB engine, session, init.

Supports two backends:
- SQLite  (default, zero config)  — set nothing in .env
- Supabase PostgreSQL (production) — set DATABASE_URL + SUPABASE_URL in .env
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlmodel import Session, SQLModel, create_engine

from app.config import settings

if settings.use_supabase:
    # PostgreSQL — no check_same_thread, use connection pooling.
    # Resilience against Supabase/pgbouncer dropping connections ("SSL
    # connection has been closed unexpectedly"):
    #   pool_pre_ping   — validate a connection before handing it out
    #   pool_recycle    — proactively drop connections older than the pooler's
    #                     idle timeout so a stale one is never used
    #   TCP keepalives  — detect a dead peer instead of blocking on a zombie
    #   connect_timeout — fail fast (and retry) instead of hanging on connect
    engine = create_engine(
        settings.sqlite_url,   # returns database_url when use_supabase=True
        echo=False,
        # Env-tunable (DB_POOL_SIZE / DB_MAX_OVERFLOW). The old hardcoded 5+10
        # starved funnel/registry/web whenever the lanes overlapped.
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,
        pool_recycle=280,      # under Supabase's ~5-min pooler idle timeout
        connect_args={
            "connect_timeout": 10,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        },
    )
else:
    engine = create_engine(
        settings.sqlite_url,
        echo=False,
        connect_args={"timeout": 30, "check_same_thread": False},
)


def _missing_enum_labels(existing: set[str], enum_cls) -> list[str]:
    """Labels to ADD to a pg enum type for members not yet represented.

    SQLAlchemy's Enum column type stores/compares the member NAME, so the
    NAME must exist as a label — a lowercase VALUE label (added by an older,
    buggy migration) does NOT make the member usable and must not suppress
    adding the real one."""
    return [member.name for member in enum_cls if member.name not in existing]


def _create_all_with_retry(attempts: int = 5) -> None:
    """create_all(), retried with backoff. Supabase can drop the first SSL
    connection ("SSL connection has been closed unexpectedly"), especially when
    the DB is briefly overloaded — a single transient failure must NOT crash the
    whole boot (which crash-loops the container and hides /health). In prod the
    schema already exists, so create_all is a cheap idempotent check."""
    import time as _t
    from sqlalchemy.exc import OperationalError, DBAPIError
    for i in range(attempts):
        try:
            SQLModel.metadata.create_all(engine)
            return
        except (OperationalError, DBAPIError) as e:
            if i == attempts - 1:
                # Don't abort startup: log loudly and continue. Tables almost
                # certainly already exist; pool_pre_ping reconnects per request
                # once the DB recovers, and /health can serve meanwhile.
                print(f"create_all failed after {attempts} attempts "
                      f"(continuing so the app can still serve): {e}")
                return
            delay = min(2 ** i, 10)
            print(f"create_all attempt {i + 1}/{attempts} failed ({e}); retrying in {delay}s")
            try:
                engine.dispose()  # drop any poisoned pooled connections
            except Exception:
                pass
            _t.sleep(delay)


def init_db() -> None:
    """Create tables if they don't exist."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    # Importing models registers them with SQLModel.metadata
    from app.db import models  # noqa: F401
    from sqlalchemy import text, inspect
    _create_all_with_retry()

    # Migrations: Add missing pg enum values if using Supabase.
    # IMPORTANT: SQLAlchemy persists/compares Enum columns by the member NAME
    # (e.g. 'RECRUITEE'), not the value ('recruitee') — so the label we add
    # must be the NAME, or queries against new members keep failing.
    if settings.use_supabase:
        for enum_type, enum_cls_name in (("applicationstatus", "ApplicationStatus"),
                                          ("jobsource", "JobSource")):
            try:
                import app.db.models as _models
                enum_cls = getattr(_models, enum_cls_name)
                with engine.connect() as conn:
                    res = conn.execute(text(
                        f"SELECT enumlabel FROM pg_enum WHERE enumtypid = '{enum_type}'::regtype"
                    )).all()
                    existing_enums = {r[0] for r in res}
                for label in _missing_enum_labels(existing_enums, enum_cls):
                    autocommit_conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
                    with autocommit_conn:
                        autocommit_conn.execute(text(f"ALTER TYPE {enum_type} ADD VALUE '{label}'"))
                    print(f"Added enum value '{label}' to {enum_type} type")
            except Exception as e:
                print(f"Failed to migrate {enum_type} enum type: {e}")

    # Migrations: Add new columns if they don't exist
    # Use inspector to check columns per table
    inspector = inspect(engine)
    
    def add_column_if_missing(table_name: str, col: str, col_type: str):
        try:
            if not inspector.has_table(table_name):
                return
            existing_cols = {c["name"].lower() for c in inspector.get_columns(table_name)}
            if col.lower() not in existing_cols:
                db_type = col_type
                if settings.use_supabase:
                    if "DATETIME" in db_type.upper():
                        db_type = db_type.upper().replace("DATETIME", "TIMESTAMP")
                    if "BOOLEAN DEFAULT 0" in db_type.upper():
                        db_type = db_type.upper().replace("BOOLEAN DEFAULT 0", "BOOLEAN DEFAULT FALSE")
                    if "BOOLEAN DEFAULT 1" in db_type.upper():
                        db_type = db_type.upper().replace("BOOLEAN DEFAULT 1", "BOOLEAN DEFAULT TRUE")
                    if "FLOAT" in db_type.upper():
                        db_type = db_type.upper().replace("FLOAT", "DOUBLE PRECISION")
                
                import time
                retries = 3
                for attempt in range(retries):
                    try:
                        with engine.begin() as conn:
                            if settings.use_supabase:
                                conn.execute(text("SET statement_timeout = 60000"))
                                conn.execute(text("SET lock_timeout = 30000"))
                            conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col} {db_type}"))
                        print(f"Added column {col} ({db_type}) to {table_name}")
                        break
                    except Exception as ex:
                        if attempt < retries - 1:
                            print(f"Retry {attempt+1}/{retries} adding column {col} to {table_name}: {ex}")
                            time.sleep(2.0)
                        else:
                            raise ex
        except Exception as e:
            print(f"Failed to add column {col} to {table_name}: {e}")

    # Migrations for application table
    for col, col_type in [
        ("resume_variant", "VARCHAR"),
        ("response_type", "VARCHAR DEFAULT 'none'"),
        ("apply_track", "VARCHAR NOT NULL DEFAULT 'autofill'"),
        ("profile_variant", "VARCHAR"),
        ("senior_fit_score", "FLOAT"),
        ("senior_verdict", "TEXT"),
        ("custom_highlight_block", "TEXT"),
        ("rejection_analysis", "TEXT"),
    ]:
        add_column_if_missing("application", col, col_type)

    # Migrations for job table
    add_column_if_missing("job", "cross_source_slug", "VARCHAR")
    for col, col_type in [
        ("ghost_score", "FLOAT DEFAULT 0.0"),
        ("ghost_flags", "TEXT"),
        ("hire_probability_score", "FLOAT"),
        ("hire_probability_signals", "TEXT"),
        ("blended_score", "FLOAT"),
        ("first_seen", "DATETIME"),
        ("last_seen", "DATETIME"),
        ("is_closed", "BOOLEAN DEFAULT FALSE"),
        ("closed_reason", "TEXT"),
        ("content_hash", "VARCHAR"),
        ("job_type", "VARCHAR DEFAULT 'full_time'"),
        ("is_cap_exempt", "BOOLEAN DEFAULT FALSE"),
        ("urgency_score", "FLOAT DEFAULT 0.0"),
        ("rerank_breakdown", "TEXT"),
        ("corporate_insights", "TEXT"),
    ]:
        add_column_if_missing("job", col, col_type)
        
    # Migrations for companyregistry table
    for col, col_type in [
        ("company_name", "VARCHAR"),
        ("career_url", "VARCHAR"),
        ("confidence_score", "INTEGER DEFAULT 100"),
        ("target_fit_score", "FLOAT DEFAULT 0.0"),
        ("last_validated_at", "DATETIME"),
        ("failure_count", "INTEGER DEFAULT 0"),
        ("sponsorship_signal", "VARCHAR"),
        ("last_error", "VARCHAR"),
        ("inactive_reason", "VARCHAR"),
        ("next_retry_at", "DATETIME"),
        ("new_jobs_last_poll", "INTEGER DEFAULT 0"),
        ("last_new_job_at", "DATETIME"),
        ("next_poll_at", "DATETIME"),
        ("poll_hash", "VARCHAR"),
    ]:
        add_column_if_missing("companyregistry", col, col_type)

    # Multi-tenant user_id columns
    for tbl in ["job", "application", "userprofile", "answermemory"]:
        add_column_if_missing(tbl, "user_id", "VARCHAR")

    # Migrations for userprofile table
    for col, col_type in [
        ("portfolio_url", "VARCHAR DEFAULT ''"),
        ("visa_status", "VARCHAR DEFAULT ''"),
        ("current_title", "VARCHAR DEFAULT ''"),
        ("years_experience", "INTEGER DEFAULT 0"),
        ("salary_min", "INTEGER DEFAULT 0"),
        ("salary_max", "INTEGER DEFAULT 0"),
        ("salary_currency", "VARCHAR DEFAULT 'USD'"),
        ("degree", "VARCHAR DEFAULT ''"),
        ("university", "VARCHAR DEFAULT ''"),
        ("graduation_year", "INTEGER"),
        ("gender", "VARCHAR DEFAULT 'Decline to self-identify'"),
        ("ethnicity", "VARCHAR DEFAULT 'Decline to self-identify'"),
        ("veteran_status", "VARCHAR DEFAULT 'I am not a protected veteran'"),
        ("disability_status", "VARCHAR DEFAULT 'No, I do not have a disability, or history/record of having a disability'"),
        ("professional_summary", "TEXT DEFAULT ''"),
        ("key_skills", "TEXT DEFAULT ''"),
        ("target_roles", "TEXT DEFAULT ''"),
        ("job_type_preference", "VARCHAR DEFAULT 'full_time'"),
        ("work_auth_status", "VARCHAR DEFAULT ''"),
        ("include_internships_in_discovery", "BOOLEAN DEFAULT FALSE"),
        ("industry", "VARCHAR DEFAULT ''"),
        ("preferred_country", "VARCHAR DEFAULT 'United States'"),
        ("remote_ok", "BOOLEAN DEFAULT TRUE"),
        ("referral_code", "VARCHAR"),
        ("referred_by_id", "VARCHAR"),
        ("email_verified", "BOOLEAN DEFAULT FALSE"),
        ("phone_verified", "BOOLEAN DEFAULT FALSE"),
        ("public_handle", "VARCHAR"),
        ("account_type", "VARCHAR DEFAULT 'candidate'"),
        ("availability", "VARCHAR DEFAULT ''"),
        ("open_to_relocation", "BOOLEAN DEFAULT FALSE"),
        ("articulation_video_url", "VARCHAR DEFAULT ''"),
        ("articulation_pr", "VARCHAR DEFAULT ''"),
        ("trust_identity_score", "INTEGER DEFAULT 0"),
        ("trust_technical_score", "INTEGER DEFAULT 0"),
        ("trust_consistency_score", "INTEGER DEFAULT 0"),
        ("trust_activity_score", "INTEGER DEFAULT 0"),
        ("trust_completeness_score", "INTEGER DEFAULT 0"),
        ("trust_tier", "VARCHAR DEFAULT ''"),
        ("trust_evidence", "TEXT"),
        ("resume_grounded_ratio", "FLOAT"),
        ("trust_computed_at", "DATETIME"),
        ("updated_at", "DATETIME"),
        ("target_companies", "VARCHAR DEFAULT ''"),
    ]:
        add_column_if_missing("userprofile", col, col_type)

    # Migrations for discoveryrun table
    for col, col_type in [
        ("total_shortlisted", "INTEGER DEFAULT 0"),
        ("error", "VARCHAR"),
    ]:
        add_column_if_missing("discoveryrun", col, col_type)

    # Migrations for h1bsponsor table (multi-country sponsor registry)
    for col, col_type in [
        ("country", "VARCHAR DEFAULT 'united states'"),
        ("record_type", "VARCHAR DEFAULT 'stats'"),
        ("detail", "VARCHAR DEFAULT ''"),
    ]:
        add_column_if_missing("h1bsponsor", col, col_type)
    # The unique key must include country now (same employer can appear in the
    # US stats AND the UK register). Postgres can swap the constraint in place;
    # old SQLite dev DBs keep the 2-column constraint, which is harmless because
    # license rows carry fiscal_year=NULL and NULLs never collide in SQLite.
    if settings.use_supabase:
        try:
            with engine.begin() as conn:
                conn.execute(text(
                    "ALTER TABLE h1bsponsor DROP CONSTRAINT IF EXISTS uq_h1b_employer_year"
                ))
                conn.execute(text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_h1b_employer_year_country "
                    "ON h1bsponsor (employer_key, COALESCE(fiscal_year, 0), country)"
                ))
        except Exception as e:
            print(f"Failed to migrate h1bsponsor unique key: {e}")

    # Performance indexes are built by ensure_performance_indexes(), scheduled
    # as a background task at startup so a slow index build on a big table can't
    # block app startup / the Railway health check. (SQLite/tests build inline.)
    if not settings.use_supabase:
        ensure_performance_indexes()


# Columns declared index=True only get their index when create_all() builds the
# table fresh. In prod these were ADDED via bare ALTER TABLE (add_column_if_missing),
# so NO index existed — every `WHERE user_id = ?` was a full-table scan. Fine
# while small; once the shared pool grew the job table this caused statement-
# timeout QueryCanceled errors on matching/discovery. The composite
# (user_id, is_closed) covers the matcher's main query.
_PERF_INDEXES = [
    ("ix_job_user_id", "job", "(user_id)"),
    ("ix_job_user_closed", "job", "(user_id, is_closed)"),
    ("ix_job_user_slug", "job", "(user_id, cross_source_slug)"),
    ("ix_app_user_id", "application", "(user_id)"),
    ("ix_app_job_id", "application", "(job_id)"),
    ("ix_funnel_stage_created", "funnel_events", "(stage, created_at)"),
    # Scoring lane scans `rerank_score IS NULL` across ALL users every cycle and
    # per user fetches the freshest unscored jobs — a partial index keeps that
    # off a full-table scan as the shared pool grows (same statement-timeout risk
    # the user_id indexes above were added to fix).
    ("ix_job_unscored", "job", "(user_id, first_seen) WHERE rerank_score IS NULL"),
    # Pulse lane selects due boards by next_poll_at every tick. The column is
    # declared index=True but was added by bare ALTER in prod, so no index exists.
    ("ix_registry_next_poll", "companyregistry", "(next_poll_at)"),
]


def ensure_performance_indexes(indexes: list | None = None) -> None:
    """Create each (name, table, columns) index if missing. On Postgres uses
    CONCURRENTLY (no table lock) with autocommit + a long timeout so building
    an index on a big table can't block writes or get killed by the statement
    timeout. SQLite (local/tests) uses a plain CREATE INDEX IF NOT EXISTS."""
    from sqlalchemy import text, inspect
    insp = inspect(engine)
    for name, table, cols in (indexes or _PERF_INDEXES):
        try:
            if not insp.has_table(table):
                continue
            existing = {ix["name"] for ix in insp.get_indexes(table)}
            if name in existing:
                continue
            if settings.use_supabase:
                conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
                with conn:
                    conn.execute(text("SET statement_timeout = 600000"))  # 10 min for the build
                    conn.execute(text(
                        f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} ON {table} {cols}"
                    ))
            else:
                with engine.begin() as conn:
                    conn.execute(text(f"CREATE INDEX IF NOT EXISTS {name} ON {table} {cols}"))
            print(f"Ensured index {name} on {table}{cols}")
        except Exception as e:
            print(f"Index {name} on {table} not created (non-fatal): {e}")


def reconcile_job_owners() -> int:
    """Adopt legacy ownerless Job rows into the tenant whose Application
    references them.

    Jobs created in the single-user era carry ``user_id IS NULL`` while the
    Applications pointing at them were later written with a real user_id —
    so the owner's dashboards count 0 jobs even though their applications
    render. Only NULL-owned rows are touched; rows that belong to another
    tenant are never reassigned. Idempotent — safe to run on every boot.
    Returns the number of adopted rows.
    """
    from sqlalchemy import text
    sql = text(
        "UPDATE job SET user_id = ("
        "  SELECT a.user_id FROM application a"
        "  WHERE a.job_id = job.id AND a.user_id IS NOT NULL LIMIT 1"
        ") WHERE job.user_id IS NULL AND EXISTS ("
        "  SELECT 1 FROM application a"
        "  WHERE a.job_id = job.id AND a.user_id IS NOT NULL)"
    )
    try:
        with engine.begin() as conn:
            result = conn.execute(sql)
            adopted = result.rowcount or 0
        if adopted:
            print(f"Reconciled {adopted} ownerless job(s) to their application owners")
        return adopted
    except Exception as e:
        print(f"Job owner reconciliation failed: {e}")
        return 0


@contextmanager
def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session


if __name__ == "__main__":
    init_db()
    print(f"DB initialized at {settings.sqlite_path}")
