from __future__ import annotations
from typing import List, Dict
from datetime import datetime
from abc import ABC, abstractmethod
from ..domain.schemas import League, Match, MarketQuote

class ProviderStats(ABC):
    @abstractmethod
    def list_leagues(self, region: str | None = None) -> List[League]:
        ...

    @abstractmethod
    def fixtures(self, leagues: List[str], days: int) -> List[Match]:
        ...

class ProviderOdds(ABC):
    @abstractmethod
    def odds_for_matches(self, match_ids: List[str]) -> Dict[str, Dict[str, List[MarketQuote]]]:
        ...

def parse_iso(dt_str: str) -> datetime:
    # '2025-10-01T19:30:00Z'
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
