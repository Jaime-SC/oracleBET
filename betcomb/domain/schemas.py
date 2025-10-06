from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict

@dataclass
class ScoreResult:
    score: float
    prob: float
    reasons: List[str] = field(default_factory=list)
    penalties: float = 0.0
    rationale: Optional[str] = None  # <— agregado para compatibilidad con heuristics

    # alias de compatibilidad (por si en algún lugar se accede como .explain)
    @property
    def explain(self) -> List[str]:
        return self.reasons

@dataclass(frozen=True)
class League:
    code: str
    name: str
    region: str  # "eu" | "conmebol" | "uefa"

@dataclass(frozen=True)
class Team:
    id: str
    name: str
    city: Optional[str] = None

@dataclass
class Match:
    id: str
    league: League
    date_utc: datetime
    home: Team
    away: Team
    # Minimal stats used by heuristics (averages on last N)
    home_cards_pg: float
    away_cards_pg: float
    referee_cards_pg: float | None = None
    derby: bool = False
    knockout: bool = False
    intl_comp_bonus: float = 0.0  # Libertadores/Sudamericana bonus if any
    # First half signals
    home_fh_goals_pg: float = 0.0
    away_fh_goals_pg: float = 0.0
    home_shots_on_target_pg: float = 0.0
    away_shots_on_target_pg: float = 0.0
    ranking_diff: float = 0.0  # + favors home, - favors away
    # Optional
    weather_penalty: float = 0.0
    congestion_penalty: float = 0.0

@dataclass(frozen=True)
class MarketQuote:
    market: str  # "CARDS_BTTS" | "FH_OVER_0_5"
    provider: str
    odds: float  # decimal odds
    fetched_at: datetime

@dataclass
class Pick:
    match_id: str
    market: str
    provider: str
    odds: float
    prob: float
    rationale: str

@dataclass
class Slip:
    picks: List[Pick] = field(default_factory=list)

    def total_odds(self) -> float:
        total = 1.0
        for p in self.picks:
            total *= p.odds
        return round(total, 3)

    def joint_prob(self) -> float:
        prod = 1.0
        for p in self.picks:
            prod *= p.prob
        return round(prod, 4)

    def to_dict(self) -> Dict:
        return {
            "total_odds": self.total_odds(),
            "joint_prob": self.joint_prob(),
            "picks": [asdict(p) for p in self.picks],
        }
