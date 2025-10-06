from __future__ import annotations
from ..config import SETTINGS
from .base import ProviderStats, parse_iso
from ..domain.schemas import League, Team, Match
from typing import List, Set, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import os
import logging
import requests
import datetime as dt

LOG = logging.getLogger(__name__)

API_BASE = "https://v3.football.api-sports.io"


# Mapea confederación -> "country" que usa API-Football
_CONFED_COUNTRIES: Dict[str, Set[str]] = {
    "uefa": {"Europe", "World"},       # Nations League, Euro, Friendlies, etc.
    "conmebol": {"South America", "World"},  # WC Qual CONMEBOL, Copa América, Amistosos
}

# Palabras clave MUY amplias para captar torneos de selecciones
_INTL_NAME_PATTERNS = (
    "world cup", "fifa", "wc qualification", "world cup qualification",
    "international", "friendlies", "friendly",
    "uefa", "euro", "nations league",
    "conmebol", "copa america", "qualifier", "qualification", "qualifiers",
)

def _name_matches(name: str) -> bool:
    n = (name or "").lower()
    return any(p in n for p in _INTL_NAME_PATTERNS)


# Nombre “amigable” por código interno
LEAGUE_NAME_BY_CODE = {
    "EPL": "Premier League",
    "LALIGA": "La Liga",
    "SERIE_A": "Serie A",
    "LIGUE_1": "Ligue 1",
    "BUNDES": "Bundesliga",
    "UCL": "UEFA Champions League",
    "UEL": "UEFA Europa League",
    "LIBERTADORES": "Copa Libertadores",
    "SUDAMERICANA": "Copa Sudamericana",
}

# Alias/estrategias de búsqueda
LEAGUE_ALIASES = {
    "EPL": {
        "queries": ["Premier League", "England Premier League", "EPL"],
        "country": "England",
        "type": "league",
    },
    "LALIGA": {
        "queries": ["La Liga", "LaLiga", "Spain La Liga", "Primera Division"],
        "country": "Spain",
        "type": "league",
    },
    "SERIE_A": {
        "queries": ["Serie A", "Italy Serie A"],
        "country": "Italy",
        "type": "league",
    },
    "LIGUE_1": {
        "queries": ["Ligue 1", "France Ligue 1"],
        "country": "France",
        "type": "league",
    },
    "BUNDES": {
        "queries": ["Bundesliga", "Germany Bundesliga"],
        "country": "Germany",
        "type": "league",
    },
    "UCL": {
        "queries": ["UEFA Champions League", "Champions League"],
        "country": None,
        "type": "cup",
    },
    "UEL": {
        "queries": ["UEFA Europa League", "Europa League"],
        "country": None,
        "type": "cup",
    },
    "LIBERTADORES": {
        "queries": ["Copa Libertadores", "CONMEBOL Libertadores", "Libertadores"],
        "country": None,
        "type": "cup",
    },
    "SUDAMERICANA": {
        "queries": ["Copa Sudamericana", "CONMEBOL Sudamericana", "Sudamericana"],
        "country": None,
        "type": "cup",
    },
}

# Fallback estático (IDs comunes en API-Football; ajustables si tu cuenta usa otros)
STATIC_LEAGUE_ID_MAP = {
    "EPL": 39,
    "LALIGA": 140,
    "SERIE_A": 135,
    "LIGUE_1": 61,
    "BUNDES": 78,
    "UCL": 2,
    "UEL": 3,
    # CONMEBOL (pueden variar por temporada/cobertura):
    "LIBERTADORES": 13,      # Ajusta si tu cuenta usa otro
    "SUDAMERICANA": 267,     # Ajusta si tu cuenta usa otro
}

LEAGUE_NAME_POOL = {
    "uefa": [
        ("UEFA Nations League", "Europe"),
        ("Euro", "Europe"),  # Euro, Euro Qualification
        ("Friendlies", "World"),  # Friendlies a veces vienen como "World"
    ],
    "conmebol": [
        ("World Cup", "South America"),  # CONMEBOL WC Qualifiers
        ("Copa America", "South America"),
        ("Friendlies", "World"),
    ],
}

# Mapear confederaciones -> países (según API-Football)
_CONFED_COUNTRIES: Dict[str, Set[str]] = {
    "uefa": {"Europe"},
    "conmebol": {"South America"},
    # puedes añadir otras si te interesan:
    # "concacaf": {"North America"},
    # "caf": {"Africa"},
    # "afc": {"Asia"},
    # "ofc": {"Oceania"},
}

# Patrones de nombres de torneos internacionales
_INTL_NAME_PATTERNS = (
    "World Cup",
    "WC Qualification",
    "World Cup Qualification",
    "UEFA EURO",
    "Euro",
    "UEFA Nations League",
    "Nations League",
    "Copa America",
    "CONMEBOL",
    "International Friendlies",
    "Friendlies",
    "Qualification",
    "Qualifiers",
)


# Copas internacionales a las que daremos un bonus para tarjetas
INTL_CUPS = {"UCL", "UEL", "LIBERTADORES", "SUDAMERICANA"}

def _headers(api_key: str) -> Dict[str, str]:
    return {"x-apisports-key": api_key}

def _iso_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

def _safe_get(d: dict, path: List[str], default=None):
    cur = d
    try:
        for k in path:
            cur = cur[k]
        return cur
    except Exception:
        return default

def _current_season_year() -> int:
    # Para Europa: temporada suele iniciar la segunda mitad del año.
    today = datetime.now(timezone.utc)
    y = today.year
    if today.month < 7:
        return y - 1
    return y

class APIFootballStats(ProviderStats):
    """
    Implementación mínima segura:
    - Resuelve league_id con varias estrategias (aliases, country/type, fallback estático).
    - Trae fixtures por ventana de fechas.
    - Deriva estadísticas simples por equipo desde /teams/statistics con fallbacks.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or SETTINGS.api_football_key
        if not self.api_key:
            logger.warning("API_FOOTBALL_KEY ausente. Usa --provider-stats mock o configura .env.")

    # ---------- Interfaz pública ----------
    def list_leagues(self, region: str | None = None) -> List[League]:
        # Para el listado “bonito” usamos el catálogo mock (estable)
        from ..providers.mock import MockStats
        return MockStats().list_leagues(region)

    def fixtures(self, leagues: List[str], days: int) -> List[Match]:
        if not self.api_key:
            logger.warning("Sin API key, retorna []. Usa provider-stats=mock para fixtures.")
            return []

        # Resolver league_id y season por cada código
        uniq_league_info: Dict[str, Tuple[int, int]] = {}
        for code in leagues:
            try:
                lid, season = self._resolve_league_id_and_season(code)
                if lid:
                    uniq_league_info[code] = (lid, season)
                else:
                    logger.warning(f"No se pudo resolver league_id para {code}. Se omite.")
            except Exception as e:
                logger.error(f"Error resolviendo liga {code}: {e}", exc_info=True)

        if not uniq_league_info:
            return []

        # Ventana de fechas
        today = datetime.now(timezone.utc)
        date_from = _iso_date(today)
        date_to = _iso_date(today + timedelta(days=days))

        matches: List[Match] = []
        for code, (league_id, season) in uniq_league_info.items():
            fx = self._get_fixtures_window(league_id, season, date_from, date_to)
            if not fx:
                continue

            # Cache para stats por equipo durante esta llamada
            team_stats_cache: Dict[int, dict] = {}

            for item in fx:
                try:
                    fid = item["fixture"]["id"]
                    dt_iso = item["fixture"]["date"]  # "YYYY-MM-DDTHH:MM:SS+00:00"
                    dt = parse_iso(dt_iso)

                    # Construir objetos
                    lg = item["league"]
                    lg_obj = League(code=code, name=lg.get("name", code), region=self._region_for_code(code))
                    home_team = item["teams"]["home"]
                    away_team = item["teams"]["away"]
                    home = Team(id=str(home_team["id"]), name=home_team["name"])
                    away = Team(id=str(away_team["id"]), name=away_team["name"])

                    # Stats por equipo (con fallback)
                    h_stats = self._team_stats(team_stats_cache, league_id, season, int(home.id))
                    a_stats = self._team_stats(team_stats_cache, league_id, season, int(away.id))

                    home_cards_pg = self._cards_per_game(h_stats)
                    away_cards_pg = self._cards_per_game(a_stats)
                    home_fh_g_pg = self._fh_goals_per_game(h_stats)
                    away_fh_g_pg = self._fh_goals_per_game(a_stats)
                    home_sot_pg = self._shots_on_target_pg(h_stats)
                    away_sot_pg = self._shots_on_target_pg(a_stats)

                    intl_bonus = 0.35 if code in INTL_CUPS else 0.0

                    m = Match(
                        id=str(fid),
                        league=lg_obj,
                        date_utc=dt,
                        home=home,
                        away=away,
                        home_cards_pg=home_cards_pg,
                        away_cards_pg=away_cards_pg,
                        referee_cards_pg=None,  # enriquecer si quisieras con otros endpoints
                        derby=False,
                        knockout=self._is_knockout(item),
                        intl_comp_bonus=intl_bonus,
                        home_fh_goals_pg=home_fh_g_pg,
                        away_fh_goals_pg=away_fh_g_pg,
                        home_shots_on_target_pg=home_sot_pg,
                        away_shots_on_target_pg=away_sot_pg,
                        ranking_diff=self._rank_diff_proxy(item),
                        weather_penalty=0.0,
                        congestion_penalty=0.0,
                    )
                    matches.append(m)
                except Exception as e:
                    logger.error(f"Error parseando fixture: {e}", exc_info=True)
                    continue

        return matches

    # ---------- Internos ----------
    def _region_for_code(self, code: str) -> str:
        if code in {"EPL", "LALIGA", "SERIE_A", "LIGUE_1", "BUNDES"}:
            return "eu"
        if code in {"LIBERTADORES", "SUDAMERICANA"}:
            return "conmebol"
        if code in {"UCL", "UEL"}:
            return "uefa"
        return "eu"

    def _resolve_league_id_and_season(self, code: str) -> Tuple[int | None, int]:
        """
        Estrategia:
        1) /leagues?search=<alias>&season=<season> para varios alias
        2) /leagues?country=<country>&type=<league/cup>&season=<season>
        3) Fallback estático a IDs conocidos
        """
        season = _current_season_year()
        aliases = LEAGUE_ALIASES.get(
            code,
            {"queries": [LEAGUE_NAME_BY_CODE.get(code, code)], "country": None, "type": None},
        )
        # 1) Buscar por alias de texto
        for q in aliases.get("queries", []):
            try:
                url = f"{API_BASE}/leagues"
                params = {"search": q, "season": season}
                r = requests.get(url, headers=_headers(self.api_key), params=params, timeout=20)
                r.raise_for_status()
                resp = r.json().get("response", [])
                league_id = self._pick_league_id_from_response(resp, q)
                if league_id:
                    return league_id, season
            except Exception as e:
                logger.debug(f"/leagues?search={q} falló: {e}")

        # 2) Buscar por país + tipo
        country = aliases.get("country")
        ltype = aliases.get("type")
        if country or ltype:
            try:
                url = f"{API_BASE}/leagues"
                params = {"season": season}
                if country:
                    params["country"] = country
                if ltype:
                    params["type"] = ltype  # "league" o "cup"
                r = requests.get(url, headers=_headers(self.api_key), params=params, timeout=20)
                r.raise_for_status()
                resp = r.json().get("response", [])
                for alias in aliases.get("queries", []):
                    lid = self._pick_league_id_from_response(resp, alias)
                    if lid:
                        return lid, season
                if resp:
                    lid = resp[0].get("league", {}).get("id")
                    if lid:
                        return lid, season
            except Exception as e:
                logger.debug(f"/leagues country/type fallback falló: {e}")

        # 3) Fallback estático
        static_id = STATIC_LEAGUE_ID_MAP.get(code)
        if static_id:
            logger.warning(f"Usando fallback estático para {code}: league_id={static_id}")
            return static_id, season

        return None, season

    def _pick_league_id_from_response(self, resp: List[dict], alias_query: str) -> int | None:
        alias_low = alias_query.lower()
        best_id = None
        for it in resp:
            lg = it.get("league", {})
            name = str(lg.get("name", "")).lower()
            if alias_low in name or name in alias_low:
                best_id = lg.get("id")
                break
        if not best_id and resp:
            best_id = resp[0].get("league", {}).get("id")
        return best_id

    def _get_fixtures_window(self, league_id: int, season: int, date_from: str, date_to: str) -> List[dict]:
        url = f"{API_BASE}/fixtures"
        params = {"league": league_id, "season": season, "from": date_from, "to": date_to}
        try:
            r = requests.get(url, headers=_headers(self.api_key), params=params, timeout=30)
            r.raise_for_status()
            return r.json().get("response", [])
        except Exception as e:
            logger.exception(f"Error consultando fixtures league={league_id} season={season}: {e}")
            return []

    def _team_stats(self, cache: Dict[int, dict], league_id: int, season: int, team_id: int) -> dict:
        if team_id in cache:
            return cache[team_id]
        url = f"{API_BASE}/teams/statistics"
        params = {"league": league_id, "season": season, "team": team_id}
        try:
            r = requests.get(url, headers=_headers(self.api_key), params=params, timeout=20)
            r.raise_for_status()
            stats = r.json().get("response", {})
            cache[team_id] = stats or {}
            return cache[team_id]
        except Exception as e:
            logger.warning(f"Stats no disponibles team={team_id} league={league_id}: {e}")
            cache[team_id] = {}
            return cache[team_id]

    # ---- Derivaciones numéricas (con fallback) ----
    def _played_matches(self, stats: dict) -> float:
        return float(_safe_get(stats, ["fixtures", "played", "total"], 10) or 10)

    def _cards_per_game(self, stats: dict) -> float:
        yell = float(_safe_get(stats, ["cards", "yellow", "total"], 0) or 0)
        red = float(_safe_get(stats, ["cards", "red", "total"], 0) or 0)
        played = self._played_matches(stats)
        val = (yell + red) / played if played > 0 else 2.0
        return round(max(0.5, min(val, 4.5)), 3)

    def _fh_goals_per_game(self, stats: dict) -> float:
        played = self._played_matches(stats)
        minutes_paths = [
            ["goals", "for", "minute", "0-15", "total"],
            ["goals", "for", "minute", "16-30", "total"],
            ["goals", "for", "minute", "31-45", "total"],
            ["goals", "for", "minute", "45-60", "total"],
            ["goals", "for", "minute", "46-60", "total"],
        ]
        tot = 0.0
        for p in minutes_paths:
            v = _safe_get(stats, p, None)
            if isinstance(v, (int, float)):
                tot += float(v)
        if tot == 0.0:
            gf = float(_safe_get(stats, ["goals", "for", "total", "total"], 0) or 0)
            approx = (gf / played * 0.62) if played > 0 else 0.5
            return round(max(0.1, min(approx, 1.5)), 3)
        return round(max(0.1, min(tot / played, 1.5)), 3)

    def _shots_on_target_pg(self, stats: dict) -> float:
        sot = _safe_get(stats, ["shots", "on", "total"], None)
        if sot is None:
            sot = _safe_get(stats, ["shots", "on", "total", "total"], None)
        played = self._played_matches(stats)
        if isinstance(sot, (int, float)) and played > 0:
            return round(max(1.0, min(float(sot) / played, 8.0)), 3)
        return 4.0

    def _rank_diff_proxy(self, fixture_item: dict) -> float:
        home_form = _safe_get(fixture_item, ["teams", "home", "form"], "")
        away_form = _safe_get(fixture_item, ["teams", "away", "form"], "")
        if home_form and away_form:
            h = home_form.count("W")
            a = away_form.count("W")
            if h - a >= 2:
                return 0.6
            if a - h >= 2:
                return -0.6
            if h > a:
                return 0.4
            if a > h:
                return -0.4
        return 0.0

    def _is_knockout(self, fixture_item: dict) -> bool:
        stage = _safe_get(fixture_item, ["league", "round"], "") or _safe_get(fixture_item, ["league", "stage"], "")
        if not stage:
            return False
        stage_s = str(stage).lower()
        return any(k in stage_s for k in ["knock", "round of", "quarter", "semi", "final", "play-off"])
    
    def _search_league_ids(self, wanted: list[tuple[str, str]]):
        """Devuelve lista de (league_id, best_season_year) para nombres aproximados."""
        out = []
        for name_substr, area in wanted:
            try:
                data = self._get("leagues", params={"search": name_substr})
            except Exception:
                continue
            for row in (data or []):
                lg = row.get("league", {})
                country = row.get("country", {}) or {}
                cname = country.get("name") or ""
                lname = (lg.get("name") or "")
                # filtro por región (Europe / South America / World) y substring
                if area and area.lower() not in (cname or "").lower():
                    # caso "Friendlies" suele salir como "World", permitimos si el name matchea
                    if name_substr.lower() not in lname.lower():
                        continue
                elif name_substr.lower() not in lname.lower():
                    continue

                # elegir la temporada activa o la más reciente
                seasons = row.get("seasons") or []
                season_year = None
                for s in seasons:
                    if s.get("current"):
                        season_year = s.get("year")
                        break
                if season_year is None and seasons:
                    season_year = seasons[-1].get("year")

                if lg.get("id") and season_year:
                    out.append((lg["id"], season_year))
        # dedupe por league_id
        seen = set()
        uniq = []
        for lid, sy in out:
            if lid in seen:
                continue
            seen.add(lid)
            uniq.append((lid, sy))
        return uniq

    # --- dentro de la clase APIFootballStats ---

    def _af_http(self, endpoint: str, params: dict) -> dict:
        """
        Intenta usar el método HTTP interno de la clase (si existe) y,
        si no, hace fallback a requests con la API key.
        """
        # 1) Intenta métodos internos con distintas convenciones de nombre
        for name in ("_get", "get", "_request", "request", "_api_get", "api_get", "_api", "api"):
            fn = getattr(self, name, None)
            if callable(fn):
                try:
                    # Firma típica: fn("fixtures", params={...})
                    return fn(endpoint, params=params)
                except TypeError:
                    # Variación: fn("fixtures", **{"params": {...}})
                    return fn(endpoint, **{"params": params})
                except Exception as e:
                    LOG.debug("Método HTTP interno '%s' falló: %s", name, e)

        # 2) Fallback directo con requests
        base = getattr(self, "base_url", None) or "https://v3.football.api-sports.io"
        if not base.endswith("/"):
            base += "/"

        key = (
            getattr(self, "api_key", None)
            or getattr(self, "token", None)
            or os.getenv("APIFOOTBALL_KEY")
            or os.getenv("API_FOOTBALL_KEY")
        )
        headers = {"x-apisports-key": key} if key else {}
        try:
            resp = requests.get(base + endpoint, params=params, headers=headers, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            LOG.error("Fallback HTTP a API-Football falló: %s", e)
            return {"response": []}


    def national_fixtures(self, confeds: List[str], days: int) -> List[Match]:
        """
        Devuelve fixtures internacionales (UEFA/CONMEBOL…) en los próximos 'days' días.
        Descubre ligas por 'country' (Europe / South America / World) y por nombre de torneo.
        """
        try:
            # 1) Países relevantes para las confederaciones pedidas
            countries: Set[str] = set()
            for c in confeds or []:
                countries |= _CONFED_COUNTRIES.get(c.lower(), set())

            if not countries:
                LOG.info("national_fixtures: sin 'countries' para confeds=%s", confeds)
                return []

            # 2) Descubrir ligas candidatas por país
            league_ids: Set[int] = set()
            for country in sorted(countries):
                try:
                    leagues_resp = self._af_http("leagues", params={"country": country})
                except Exception as e:
                    LOG.warning("national_fixtures: fallo leagues country=%s err=%s", country, e)
                    continue

                for item in (leagues_resp or {}).get("response", []):
                    lg = item.get("league") or {}
                    name = lg.get("name") or ""
                    # Hay torneos con type="League" y otros "Cup" -> filtra solo por nombre
                    if not _name_matches(name):
                        continue
                    try:
                        league_ids.add(int(lg["id"]))
                    except Exception:
                        continue

            if not league_ids:
                LOG.info("national_fixtures: sin ligas internacionales (confeds=%s)", confeds)
                return []

            # 3) Ventana de fechas
            date_from = dt.date.today()
            date_to = date_from + dt.timedelta(days=days)

            # 4) Pedir fixtures por liga y normalizar a Match
            out: List[Match] = []
            normalizer: Callable[[dict], Match] | None = getattr(self, "_to_match", None) or getattr(self, "to_match", None)
            if not callable(normalizer):
                LOG.error("national_fixtures: no encuentro normalizador (_to_match/to_match).")
                return []

            for lid in sorted(league_ids):
                try:
                    fx_resp = self._af_http(
                        "fixtures",
                        params={
                            "league": lid,
                            "from": date_from.isoformat(),
                            "to": date_to.isoformat(),
                            "timezone": "UTC",
                        },
                    )
                except Exception as e:
                    LOG.warning("national_fixtures: fallo fixtures lid=%s err=%s", lid, e)
                    continue

                for fx in (fx_resp or {}).get("response", []):
                    try:
                        m = normalizer(fx)
                        out.append(m)
                    except Exception as e:
                        LOG.debug("national_fixtures: no pude mapear fixture lid=%s err=%s", lid, e)

            # 5) Dedup por id y orden por fecha
            unique = {m.id: m for m in out}.values()
            return sorted(unique, key=lambda m: getattr(m, "date_utc", dt.datetime.max))

        except Exception as e:
            LOG.exception("national_fixtures falló: %s", e)
            return []

    
    def fixtures_internationals(self, regions: list[str], days: int):
        """
        Trae partidos de selecciones para las regiones indicadas dentro de la ventana 'days'.
        Devuelve List[Match] como fixtures() normal.
        """
        from datetime import datetime, timedelta, timezone
        start = datetime.now(timezone.utc)
        end = start + timedelta(days=days)

        league_ids: list[int] = []
        for r in regions:
            league_ids += self.FIFA_COMP_IDS.get(r, [])

        out: list[Match] = []
        for lid in sorted(set(league_ids)):
            # Si tu wrapper ya tiene un método _get o similar, úsalo:
            # resp = self._get("fixtures", params={
            #     "league": lid,
            #     "from": start.strftime("%Y-%m-%d"),
            #     "to": end.strftime("%Y-%m-%d"),
            #     "timezone": "UTC",
            # })
            resp = self._fixtures_by_league(lid, start, end)  # crea un helper si te queda más cómodo

            for fx in resp:
                # mapea a tu dominio Match(...)
                m = self._to_match(fx)  # reusa el mismo adaptador que uses en fixtures()
                # marca región para distinguir
                if hasattr(m.league, "region") and not m.league.region:
                    # asigna 'eu' para uefa, 'conmebol' para conmebol, si quieres:
                    m.league.region = "eu" if lid in self.FIFA_COMP_IDS.get("uefa", []) else "conmebol"
                out.append(m)

        return out
    
    def _http_get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback si tu clase no define self._get."""
        _get = getattr(self, "_get", None)
        if callable(_get):
            return _get(path, params=params)

        base_url = getattr(self, "BASE_URL", "https://v3.football.api-sports.io")
        api_key = getattr(self, "api_key", None) or os.getenv("APIFOOTBALL_KEY") or os.getenv("API_FOOTBALL_KEY")
        if not api_key:
            raise RuntimeError("API-FOOTBALL api_key no configurada (APIFOOTBALL_KEY).")

        url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
        headers = {"x-apisports-key": api_key}
        r = requests.get(url, headers=headers, params=params, timeout=25)
        r.raise_for_status()
        return r.json()

    def _to_match_fallback(self, fx: Dict[str, Any]) -> Match:
        """Normalizador mínimo desde v3/fixtures -> Match (por si no tienes self._to_match)."""
        f = fx.get("fixture", {}) or {}
        lg = fx.get("league", {}) or {}
        t  = fx.get("teams", {}) or {}

        fid = str(f.get("id"))
        date_iso = f.get("date") or ""
        try:
            date_utc = dt.datetime.fromisoformat(date_iso.replace("Z", "+00:00"))
        except Exception:
            date_utc = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

        league_name = (lg.get("name") or "").strip() or str(lg.get("id"))
        league_country = (lg.get("country") or "").strip() or "International"

        league = League(
            code=league_name,  # que en tus MD salga el nombre del torneo
            name=league_name,
            region=("eu" if league_country == "Europe" else
                    "sa" if league_country == "South America" else
                    "world"),
        )
        home = Team(id=str(t.get("home", {}).get("id")), name=t.get("home", {}).get("name") or "Home")
        away = Team(id=str(t.get("away", {}).get("id")), name=t.get("away", {}).get("name") or "Away")

        return Match(
            id=fid,
            league=league,
            date_utc=date_utc,
            home=home,
            away=away,
            # defaults razonables para que tu scoring funcione
            home_cards_pg=0.5, away_cards_pg=0.5, referee_cards_pg=None,
            derby=False, knockout=False, intl_comp_bonus=0.3,
            home_fh_goals_pg=0.7, away_fh_goals_pg=0.7,
            home_shots_on_target_pg=4.0, away_shots_on_target_pg=4.0,
            ranking_diff=0.0, weather_penalty=0.0, congestion_penalty=0.0,
        )

    def _name_matches_international(self, name: str, confeds: Set[str]) -> bool:
        if not name:
            return False
        n = name.lower()
        keys = {"friendlies", "world cup", "wc qualification", "nations league"}
        if "uefa" in confeds:
            keys |= {
                "uefa nations league",
                "european championship",  # EURO
                "euro qualification",
                "wc qualification europe",
            }
        if "conmebol" in confeds:
            keys |= {
                "copa america",
                "wc qualification south america",
                "conmebol",
            }
        return any(k in n for k in keys)
    
    def _current_season_for_league(self, league_id: int, fallback_year: int) -> int:
        """Lee /leagues?id=... y devuelve la season current; si no, la más cercana al fallback."""
        try:
            resp = self._http_get("leagues", params={"id": league_id})
            for item in resp.get("response", []):
                for s in item.get("seasons", []) or []:
                    if s.get("current"):
                        return int(s.get("year"))
                years = [int(s.get("year")) for s in item.get("seasons", []) if s.get("year")]
                if years:
                    # elige la más cercana al año solicitado
                    return min(years, key=lambda y: abs(y - fallback_year))
        except Exception as e:
            LOG.warning("No pude determinar season para league=%s: %s", league_id, e)
        return fallback_year

    def national_fixtures(self, confeds: List[str], days: int) -> List[Match]:
        """
        Devuelve fixtures internacionales (UEFA/CONMEBOL) en los próximos 'days' días.
        """
        try:
            confeds_set = {c.lower().strip() for c in (confeds or [])}
            # 1) países/ámbitos válidos
            countries: Set[str] = set()
            for c in confeds_set:
                countries |= _CONFED_COUNTRIES.get(c, set())
            countries.add("World")  # por las dudas

            # 2) descubrir ligas de selecciones
            league_ids: Set[int] = set()
            for country in sorted(countries):
                resp = self._http_get("leagues", params={"country": country, "type": "Cup"})
                for item in resp.get("response", []):
                    lg = (item or {}).get("league", {}) or {}
                    name = lg.get("name", "")
                    if not self._name_matches_international(name, confeds_set):
                        continue
                    try:
                        league_ids.add(int(lg["id"]))
                    except Exception:
                        continue

            if not league_ids:
                LOG.info("national_fixtures: no hay ligas para confeds=%s", sorted(confeds_set))
                return []

            # 3) ventana y season por liga
            date_from = dt.date.today()
            date_to = date_from + dt.timedelta(days=days)

            out: List[Match] = []
            to_match = getattr(self, "_to_match", None)

            for lid in sorted(league_ids):
                season = self._current_season_for_league(lid, fallback_year=date_from.year)
                fx = self._http_get(
                    "fixtures",
                    params={
                        "league": lid,
                        "season": season,
                        "from": date_from.isoformat(),
                        "to": date_to.isoformat(),
                        "timezone": "UTC",
                        "status": "NS",  # sólo próximos; quita si quieres en juego
                    },
                )
                for row in fx.get("response", []):
                    try:
                        m = to_match(row) if callable(to_match) else self._to_match_fallback(row)
                        out.append(m)
                    except Exception as e:
                        LOG.warning("national_fixtures: map fail lid=%s: %s", lid, e)

            # 4) dedup
            dedup = list({m.id: m for m in out}.values())
            LOG.info("national_fixtures: %d ligas -> %d fixtures en %s..%s",
                     len(league_ids), len(dedup), date_from, date_to)
            return dedup

        except Exception as e:
            LOG.exception("national_fixtures falló: %s", e)
            return []


