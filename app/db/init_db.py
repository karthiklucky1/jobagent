"""DB engine, session, init."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlmodel import Session, SQLModel, create_engine

from app.config import settings

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

    # Migrations: Add new columns to existing application table if they don't exist
    from sqlalchemy import text
    with engine.begin() as conn:
        for col, col_type in [
            ("resume_variant", "VARCHAR"),
            ("response_type", "VARCHAR DEFAULT 'none'"),
            ("apply_track", "VARCHAR NOT NULL DEFAULT 'autofill'"),
            ("profile_variant", "VARCHAR"),
            ("senior_fit_score", "FLOAT"),
            ("senior_verdict", "TEXT"),
            ("custom_highlight_block", "TEXT"),
        ]:
            try:
                conn.execute(text(f"ALTER TABLE application ADD COLUMN {col} {col_type}"))
            except Exception:
                pass
        try:
            conn.execute(text("ALTER TABLE job ADD COLUMN cross_source_slug VARCHAR"))
        except Exception:
            pass
        for col, col_type in [
            ("ghost_score", "FLOAT DEFAULT 0.0"),
            ("ghost_flags", "TEXT"),
            ("hire_probability_score", "FLOAT"),
            ("hire_probability_signals", "TEXT"),
            ("blended_score", "FLOAT"),
        ]:
            try:
                conn.execute(text(f"ALTER TABLE job ADD COLUMN {col} {col_type}"))
            except Exception:
                pass
            
        # Migrations for job lifecycle tracking columns
        for col, col_type in [
            ("first_seen", "DATETIME"),
            ("last_seen", "DATETIME"),
            ("is_closed", "BOOLEAN DEFAULT 0"),
            ("content_hash", "VARCHAR"),
        ]:
            try:
                conn.execute(text(f"ALTER TABLE job ADD COLUMN {col} {col_type}"))
            except Exception:
                pass
        
        # Migrations for CompanyRegistry graph metadata columns
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
            try:
                conn.execute(text(f"ALTER TABLE companyregistry ADD COLUMN {col} {col_type}"))
            except Exception:
                pass


@contextmanager
def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session


if __name__ == "__main__":
    init_db()
    print(f"DB initialized at {settings.sqlite_path}")
