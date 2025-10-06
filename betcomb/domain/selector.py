# betcomb/domain/selector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .markets import CARDS_BTTS, FH_OVER_0_5
from .schemas import Match, Pick, MarketQuote
from .heuristics import score_market


@dataclass
class Candidate:
    match: Match
    picks: List[Pick]
    joint_prob: float
    total_odds: float


def choose_lowest_odds(quotes: List[MarketQuote]) -> Optional[MarketQuote]:
    """Devuelve la cuota más baja disponible para un mercado."""
    return min(quotes, key=lambda q: q.odds) if quotes else None


def _product(vals: List[float]) -> float:
    x = 1.0
    for v in vals:
        x *= float(v)
    return x


def build_picks_for_match(
    match: Match,
    quotes_by_market: Dict[str, List[MarketQuote]],
    *,
    p_cards_min: float,
    p_fhgoal_min: float,
    value_check: bool = False,
    mode: str = "both",  # "both" | "fh-only"
) -> Tuple[List[Pick], float]:
    """
    Construye las selecciones para un partido según el modo.
    Retorna (picks, joint_prob). Si alguna falta o no pasa umbral -> ([], 0.0)
    """
    picks: List[Pick] = []
    probs: List[float] = []

    mode = (mode or "both").lower()

    # --- CARDS ---
    if mode == "both":
        res_cards = score_market(match, CARDS_BTTS)
        q_cards = choose_lowest_odds(quotes_by_market.get(CARDS_BTTS, []))
        if (not q_cards) or (res_cards.prob < p_cards_min):
            return [], 0.0
        # opcional value check: implied vs modelo (omitimos si value_check=False)
        picks.append(
            Pick(
                match_id=match.id,
                market=CARDS_BTTS,
                odds=float(q_cards.odds),
                provider=q_cards.provider,
                prob=float(res_cards.prob),
                rationale="; ".join(res_cards.reasons) if getattr(res_cards, "reasons", None) else "",
            )
        )
        probs.append(float(res_cards.prob))

    # --- FH OVER 0.5 ---
    res_fh = score_market(match, FH_OVER_0_5)
    q_fh = choose_lowest_odds(quotes_by_market.get(FH_OVER_0_5, []))
    if (not q_fh) or (res_fh.prob < p_fhgoal_min):
        return [], 0.0

    picks.append(
        Pick(
            match_id=match.id,
            market=FH_OVER_0_5,
            odds=float(q_fh.odds),
            provider=q_fh.provider,
            prob=float(res_fh.prob),
            rationale="; ".join(res_fh.reasons) if getattr(res_fh, "reasons", None) else "",
        )
    )
    probs.append(float(res_fh.prob))

    return picks, _product(probs)
