from __future__ import annotations
import json
import os
from typing import List, Dict
from datetime import datetime, timezone
import hashlib
import random
from .base import ProviderStats, ProviderOdds, parse_iso
from ..domain.schemas import League, Team, Match, MarketQuote
from ..domain.markets import CARDS_BTTS, FH_OVER_0_5
from ..config import SETTINGS

SAMPLES_DIR = SETTINGS.samples_dir

class MockStats(ProviderStats):
    def __init__(self, samples_dir: str = SAMPLES_DIR):
        self.samples_dir = samples_dir

    def list_leagues(self, region: str | None = None) -> List[League]:
        path = os.path.join(self.samples_dir, "leagues.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        leagues = [League(**x) for x in data["leagues"]]
        if region:
            leagues = [l for l in leagues if l.region == region]
        return leagues

    def fixtures(self, leagues: List[str], days: int) -> List[Match]:
        path = os.path.join(self.samples_dir, "fixtures.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out: List[Match] = []
        for m in data["fixtures"]:
            if leagues and m["league"]["code"] not in leagues:
                continue
            dt = parse_iso(m["date_utc"])
            # simple days filter (hacia adelante)
            if (dt - datetime.now(timezone.utc)).days > days:
                continue
            lg = League(**m["league"])
            home = Team(**m["home"])
            away = Team(**m["away"])
            out.append(Match(
                id=m["id"], league=lg, date_utc=dt, home=home, away=away,
                home_cards_pg=m["home_cards_pg"], away_cards_pg=m["away_cards_pg"],
                referee_cards_pg=m.get("referee_cards_pg"),
                derby=m.get("derby", False), knockout=m.get("knockout", False),
                intl_comp_bonus=m.get("intl_comp_bonus", 0.0),
                home_fh_goals_pg=m.get("home_fh_goals_pg", 0.0),
                away_fh_goals_pg=m.get("away_fh_goals_pg", 0.0),
                home_shots_on_target_pg=m.get("home_shots_on_target_pg", 0.0),
                away_shots_on_target_pg=m.get("away_shots_on_target_pg", 0.0),
                ranking_diff=m.get("ranking_diff", 0.0),
                weather_penalty=m.get("weather_penalty", 0.0),
                congestion_penalty=m.get("congestion_penalty", 0.0),
            ))
        return out

class MockOdds(ProviderOdds):
    def __init__(self, samples_dir: str = SAMPLES_DIR, provider_name: str = "MockBook"):
        self.samples_dir = samples_dir
        self.provider_name = provider_name

    def odds_for_matches(self, match_ids: List[str]) -> Dict[str, Dict[str, List[MarketQuote]]]:
        path = os.path.join(self.samples_dir, "odds.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out: Dict[str, Dict[str, List[MarketQuote]]] = {}
        for m in data["odds"]:
            if match_ids and m["match_id"] not in match_ids:
                continue
            quotes = {
                CARDS_BTTS: [],
                FH_OVER_0_5: [],
            }
            dt = parse_iso(m["fetched_at"])
            # agregar la "cuota más baja" y alternativas para demostrar filtro
            for q in m["quotes"]:
                quotes[q["market"]].append(
                    MarketQuote(
                        market=q["market"],
                        provider=q.get("provider", self.provider_name),
                        odds=float(q["odds"]),
                        fetched_at=dt,
                    )
                )
            out[m["match_id"]] = quotes

        # === NUEVO: generar cuotas sintéticas para IDs no presentes en odds.json ===
        for mid in match_ids:
            if mid in out:
                continue
            # Semilla determinística por match_id
            seed = int(hashlib.md5(mid.encode("utf-8")).hexdigest(), 16) % (2**32)
            rng = random.Random(seed)
            # Rango de baja cuota (alta probabilidad) según nuestros mercados
            cards_odds = round(rng.uniform(1.45, 1.70), 2)   # CARDS_BTTS
            fh_odds    = round(rng.uniform(1.35, 1.55), 2)   # FH_OVER_0_5
            dt = datetime.now(timezone.utc)

            out[mid] = {
                CARDS_BTTS: [
                    MarketQuote(
                        market=CARDS_BTTS,
                        provider=f"{self.provider_name}Synth",
                        odds=cards_odds,
                        fetched_at=dt,
                    )
                ],
                FH_OVER_0_5: [
                    MarketQuote(
                        market=FH_OVER_0_5,
                        provider=f"{self.provider_name}Synth",
                        odds=fh_odds,
                        fetched_at=dt,
                    )
                ],
            }

        return out
    
    def national_fixtures(self, confeds: list[str], days: int):
        # Si quieres, puedes generar 2-3 fixtures sintéticos de selecciones
        # para probar end-to-end. Si no, deja vacío:
        return []

