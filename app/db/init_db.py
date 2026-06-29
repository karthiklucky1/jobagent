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
    # PostgreSQL — no check_same_thread, use connection pooling
    engine = create_engine(
        settings.sqlite_url,   # returns database_url when use_supabase=True
        echo=False,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,    # reconnect after Supabase idle timeout
    )
else:
    engine = create_engine(
        settings.sqlite_url,
        echo=False,
        connect_args={"timeout": 30, "check_same_thread": False},
)


def init_db() -> None:
    """Create tables if they don't exist."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    # Importing models registers them with SQLModel.metadata
    from app.db import models  # noqa: F401
    SQLModel.metadata.create_all(engine)

    # Migrations: Add new columns if they don't exist
    from sqlalchemy import text, inspect
    
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
                
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col} {db_type}"))
                print(f"Added column {col} ({db_type}) to {table_name}")
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
        ("content_hash", "VARCHAR"),
        ("job_type", "VARCHAR DEFAULT 'full_time'"),
        ("is_cap_exempt", "BOOLEAN DEFAULT FALSE"),
        ("urgency_score", "FLOAT DEFAULT 0.0"),
        ("rerank_breakdown", "TEXT"),
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
        ("next_retry_at", "DATETIME")
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
        ("trust_computed_at", "DATETIME"),
        ("updated_at", "DATETIME"),
    ]:
        add_column_if_missing("userprofile", col, col_type)

    # Migrations for discoveryrun table
    for col, col_type in [
        ("total_shortlisted", "INTEGER DEFAULT 0"),
        ("error", "VARCHAR"),
    ]:
        add_column_if_missing("discoveryrun", col, col_type)


@contextmanager
def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session


if __name__ == "__main__":
    init_db()
    print(f"DB initialized at {settings.sqlite_path}")
