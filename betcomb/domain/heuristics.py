from __future__ import annotations
import math
from dataclasses import dataclass
from .schemas import Match, ScoreResult
from .markets import CARDS_BTTS, FH_OVER_0_5

def _sigmoid(x: float, k: float = 1.0, x0: float = 0.0) -> float:
    # 1 / (1 + e^{-k(x-x0)})
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))

def cards_score(match: Match) -> ScoreResult:
    """
    Estima prob. de 'Ambos equipos reciben ≥1 tarjeta'.
    Señales:
      - Tarjetas por partido de ambos equipos.
      - Árbitro (si existe) normalizado.
      - KO/derbi.
      - Bono internacional (Libertadores/Sudamericana/UCL/UEL).
    """
    # Base: tarjetas medias de los equipos (promedio)
    team_cards = (match.home_cards_pg + match.away_cards_pg) / 2.0

    # Árbitro: si hay dato, normalizamos alrededor de ~4.5–5.5 (promedios comunes)
    # para evitar over/under-shoot; si no hay, damos 0.
    ref_component = 0.0
    if match.referee_cards_pg is not None:
        # centro 4.8, escala 1.2 → valores entre ~3.5 y 6.5 aportan +/- 1.5 aprox
        ref_component = (match.referee_cards_pg - 4.8) / 1.2

    # Tensiones: KO y derbi. En copas/KO es donde más se ve señal para ambas tarjetas.
    tension = 0.0
    if match.knockout:
        tension += 0.4
    if match.derby:
        tension += 0.3

    # Bono internacional: si viene del adaptador, úsalo directo como refuerzo (0.0–0.4 típico)
    intl = match.intl_comp_bonus or 0.0

    # Score lineal sencillo (expresivo y robusto a faltantes):
    # - Partimos de team_cards (2.0–3.5 suele ser rango sano)
    # - Sumamos árbitro normalizado, tensiones e internacional.
    score = (team_cards) + (0.35 * ref_component) + tension + intl

    # Sigmoide: bajamos el centro (x0) para no subestimar cuando hay info parcial.
    #   k más suave que goles para que el aumento sea gradual.
    prob = _sigmoid(score, k=1.15, x0=1.35)

    # Texto explicativo
    reasons = [
        "Ambos equipos reciben al menos 1 tarjeta",
        f"Tarjetas medias equipos: {team_cards:.2f}",
    ]
    if match.referee_cards_pg is not None:
        reasons.append(f"Árbitro media tarjetas: {match.referee_cards_pg:.2f}")
    if match.derby:
        reasons.append("Derbi/clásico: +0.30")
    if match.knockout:
        reasons.append("Eliminación directa/KO: +0.40")
    if intl > 0:
        reasons.append(f"Bono torneo internacional: +{intl:.2f}")

    return ScoreResult(score=score, prob=prob, reasons=reasons)


def fh_over_0_5_score(match: Match) -> ScoreResult:
    """
    Señal de gol temprano: 1er tiempo.
    """
    reasons: list[str] = []
    fh_g = (match.home_fh_goals_pg + match.away_fh_goals_pg) / 2.0
    sot = (match.home_shots_on_target_pg + match.away_shots_on_target_pg) / 2.0

    reasons.append(f"Goles 1T medios: {fh_g:.2f}")
    reasons.append(f"Shots on target medios: {sot:.2f}")

    strength = 0.25 * abs(match.ranking_diff)  # desbalance favorece gol temprano
    if strength:
        reasons.append(f"Dif. fuerzas (abs): +{strength:.2f}")

    penalties = match.weather_penalty + match.congestion_penalty
    if penalties:
        reasons.append(f"Penalizaciones clima/rotación: -{penalties:.2f}")

    score = fh_g * 0.9 + (sot / 3.0) * 0.6 + strength - penalties
    # mapeo a prob: centro cerca de 0.85
    prob = _sigmoid(score, k=1.35, x0=0.80)

    return ScoreResult(score=round(score, 3), prob=round(prob, 4), rationale=reasons)

def score_market(match: Match, market: str) -> ScoreResult:
    if market == CARDS_BTTS:
        return cards_score(match)
    elif market == FH_OVER_0_5:
        return fh_over_0_5_score(match)
    else:
        raise ValueError(f"Mercado no soportado: {market}")