from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    # Prob thresholds
    p_cards_min: float = float(os.getenv("P_CARDS_MIN", 0.62))
    p_fhgoal_min: float = float(os.getenv("P_FHGOAL_MIN", 0.60))
    # Sourcing / providers
    default_stats_provider: str = os.getenv("DEFAULT_STATS_PROVIDER", "mock")
    default_odds_provider: str = os.getenv("DEFAULT_ODDS_PROVIDER", "mock")
    # API keys
    api_football_key: str | None = os.getenv("API_FOOTBALL_KEY") or None
    sportmonks_key: str | None = os.getenv("SPORTMONKS_KEY") or None
    odds_api_key: str | None = os.getenv("ODDS_API_KEY") or None
    betfair_app_key: str | None = os.getenv("BETFAIR_APP_KEY") or None
    # Behavior
    days_default: int = int(os.getenv("DAYS_DEFAULT", 7))
    data_dir: str = os.getenv("DATA_DIR", "betcomb/data")
    samples_dir: str = os.path.join(data_dir, "samples")
    cache_dir: str = os.getenv("CACHE_DIR", ".betcomb_cache")

SETTINGS = Settings()
