from __future__ import annotations
import logging
from typing import List, Dict
from ..config import SETTINGS
from .base import ProviderOdds
from ..domain.schemas import MarketQuote
logger = logging.getLogger(__name__)

class TheOddsAPI(ProviderOdds):
    """
    Placeholder seguro. Sin key => retorna vacío y recomienda mock.
    """
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or SETTINGS.odds_api_key

    def odds_for_matches(self, match_ids: List[str]) -> Dict[str, Dict[str, List[MarketQuote]]]:
        if not self.api_key:
            logger.warning("Sin ODDS_API_KEY. Usa provider-odds=mock para cuotas.")
            return {}
        # Implementación real omitida por ToS. Retornar vacío evita scraping.
        return {}
