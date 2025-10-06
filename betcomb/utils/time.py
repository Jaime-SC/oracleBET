from __future__ import annotations
from datetime import datetime, timedelta, timezone

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def within_days(dt: datetime, days: int) -> bool:
    return (dt - now_utc()) <= timedelta(days=days)
