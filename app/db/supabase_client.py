"""Supabase client helpers.

Provides two clients:
- anon_client()        — uses SUPABASE_ANON_KEY, respects Row Level Security
- service_client()     — uses SUPABASE_SERVICE_ROLE_KEY, bypasses RLS (server-only)

Both are lazy singletons. If Supabase is not configured (no SUPABASE_URL),
calling these raises RuntimeError so callers know to fall back to SQLite.

Usage:
    from app.db.supabase_client import service_client
    sb = service_client()
    sb.storage.from_("resumes").upload(...)
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

log = logging.getLogger(__name__)

try:
    from supabase import Client, create_client
    _SUPABASE_AVAILABLE = True
except ImportError:
    _SUPABASE_AVAILABLE = False


@lru_cache(maxsize=1)
def anon_client() -> "Client":
    """Public client — safe for use with user JWT tokens (respects RLS)."""
    from app.config import settings
    if not _SUPABASE_AVAILABLE:
        raise RuntimeError("supabase package not installed. Run: pip install supabase")
    if not settings.supabase_url or not settings.supabase_anon_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")
    return create_client(settings.supabase_url, settings.supabase_anon_key)


@lru_cache(maxsize=1)
def service_client() -> "Client":
    """Service role client — server-side only, bypasses Row Level Security."""
    from app.config import settings
    if not _SUPABASE_AVAILABLE:
        raise RuntimeError("supabase package not installed. Run: pip install supabase")
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env")
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


def verify_jwt(token: str) -> Optional[dict]:
    """Verify a Supabase JWT by calling the Supabase auth API (get_user).

    This is the only reliable approach — avoids needing the raw JWT secret.
    Returns the user payload dict or None on failure.
    """
    from app.config import settings
    if not settings.supabase_url:
        return None
    try:
        sb = service_client()
        result = sb.auth.get_user(token)
        if result and result.user:
            u = result.user
            return {"sub": u.id, "email": getattr(u, "email", None)}
    except Exception as e:
        log.debug("JWT verification failed: %s", e)
    return None


def get_user_id_from_token(token: str) -> Optional[str]:
    """Extract Supabase user UUID from a JWT bearer token."""
    payload = verify_jwt(token)
    if payload:
        return payload.get("sub")
    return None
