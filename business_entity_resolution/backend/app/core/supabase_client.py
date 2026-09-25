"""supabase_client.py - Supabase client singleton."""
from __future__ import annotations
from functools import lru_cache
from typing import Optional

try:
    from supabase import create_client, Client
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False

from .config import settings


@lru_cache(maxsize=1)
def get_supabase() -> Optional[object]:
    """Return the Supabase client, or None if not configured."""
    if not HAS_SUPABASE:
        return None
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_KEY:
        return None
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
