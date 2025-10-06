# betcomb/domain/combiner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .schemas import Match, MarketQuote, Pick, Slip
from .selector import build_picks_for_match

@dataclass
class _Cand:
    match: Match
    picks: List[Pick]
    joint_prob: float
    total_odds: float


def _product(vals: List[float]) -> float:
    x = 1.0
    for v in vals:
        x *= float(v)
    return x


def _picks_total_odds(picks: List[Pick]) -> float:
    return _product([p.odds for p in picks])


def best_double(
    *,
    matches: List[Match],
    quotes: Dict[str, Dict[str, List[MarketQuote]]],
    p_cards_min: float,
    p_fhgoal_min: float,
    target_total_odds: float = 2.0,
    value_check: bool = False,
    avoid_same_time: bool = False,  # no implementado por ahora (placeholder)
    mode: str = "both",  # "both" | "fh-only"
) -> Optional[Slip]:
    """
    Elige la mejor combinada de 2 partidos (double) cumpliendo umbrales por pick
    y objetivo de cuota total. Si no hay con 'both', usa 'fh-only' si así se pidió.
    """
    # 1) Construimos candidatos por partido
    cands: List[_Cand] = []
    for m in matches:
        picks, joint = build_picks_for_match(
            m,
            quotes.get(m.id, {}),
            p_cards_min=p_cards_min,
            p_fhgoal_min=p_fhgoal_min,
            value_check=value_check,
            mode=mode,
        )
        if not picks:
            continue
        cands.append(_Cand(match=m, picks=picks, joint_prob=joint, total_odds=_picks_total_odds(picks)))

    # orden por calidad (prob conjunta por partido y luego por total_odds)
    cands.sort(key=lambda c: (c.joint_prob, c.total_odds), reverse=True)

    # 2) Buscamos el mejor par de partidos (greedy)
    best_slip: Optional[Slip] = None
    best_score: float = -1.0

    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]

            # (opcional) evitar horarios iguales/similares
            if avoid_same_time and a.match.date_utc == b.match.date_utc:
                continue

            all_picks = a.picks + b.picks
            total_odds = _picks_total_odds(all_picks)
            if total_odds < target_total_odds:
                continue

            joint_prob = a.joint_prob * b.joint_prob
            score = joint_prob  # puedes ponderar con total_odds si quieres

            if score > best_score:
                best_score = score
                best_slip = Slip(picks=all_picks)

    return best_slip

def list_singles(
    matches: List[Match],
    quotes: dict,
    p_cards_min: float,
    p_fhgoal_min: float,
    mode: str = "auto",           # "both" | "fh-only" | "auto"
    value_check: bool = False,
    min_odds: Optional[float] = None,
    max_odds: Optional[float] = None,
    top_n: int = 20,
) -> List[Pick]:
    singles: List[Pick] = []
    for m in matches:
        picks, _ = build_picks_for_match(
            m,
            quotes.get(m.id, {}),
            p_cards_min=p_cards_min,
            p_fhgoal_min=p_fhgoal_min,
            mode=mode,
            value_check=value_check,
        )
        for p in picks:
            if (min_odds is not None) and (p.odds < min_odds):
                continue
            if (max_odds is not None) and (p.odds > max_odds):
                continue
            singles.append(p)

    if not singles:
        return []

    def implied_prob(odds: float) -> float:
        return 1.0 / odds if odds > 0 else 1.0

    def edge(p: Pick) -> float:
        return float(p.prob) - implied_prob(float(p.odds))

    singles.sort(key=lambda p: (edge(p), float(p.prob), float(p.odds)), reverse=True)
    return singles[:top_n] if top_n else singles
