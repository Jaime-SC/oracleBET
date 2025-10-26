# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, hashlib
from datetime import datetime, timedelta, timezone, date
from typing import List, Dict, Any, Optional
from types import SimpleNamespace as NS

from .providers.apifootball import APIFootballStats
from .providers.mock import MockOdds
from .providers.theodds import TheOddsAPI
from .domain.heuristics import fh_over_0_5_score
from .domain.markets import FH_OVER_0_5

CACHE_DIR = ".betcomb_cache"
META_FILE = os.path.join(CACHE_DIR, "_meta.json")

DEFAULT_COMP_NAMES = [
    "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1",
    "UEFA Champions League", "UEFA Europa League",
    "Copa Libertadores", "Copa Sudamericana",
]

NAME2CODE = {
    "premier league": "EPL",
    "la liga": "LALIGA",
    "serie a": "SERIE_A",
    "bundesliga": "BUNDES",
    "ligue 1": "LIGUE_1",
    "uefa champions league": "UCL",
    "champions league": "UCL",
    "uefa europa league": "UEL",
    "europa league": "UEL",
    "copa libertadores": "LIBERTADORES",
    "conmebol libertadores": "LIBERTADORES",
    "copa sudamericana": "SUDAMERICANA",
    "conmebol sudamericana": "SUDAMERICANA",
}

# -------------------- util de caché --------------------

def _ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)

def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def _today() -> date:
    return datetime.now(timezone.utc).date()

def _window_for_days(days: int) -> tuple[str, str]:
    if days < 1:
        days = 1
    start = _today()
    return start.isoformat(), (start + timedelta(days=days)).isoformat()

def _codes_from_names(names: List[str]) -> List[str]:
    out = []
    for n in names or []:
        code = NAME2CODE.get(n.strip().lower())
        if code and code not in out:
            out.append(code)
    return out

def _comp_key(competitions: List[str]) -> str:
    base = "|".join(sorted([c.strip().lower() for c in competitions or []]))
    return hashlib.md5(base.encode("utf-8")).hexdigest()[:8]

def _fixtures_path(date_from: str, date_to: str, compkey: str) -> str:
    return os.path.join(CACHE_DIR, f"fixtures_{date_from}_{date_to}_{compkey}.json")

def _singles_path(date_from: str, date_to: str, compkey: str) -> str:
    return os.path.join(CACHE_DIR, f"singles_fh_{date_from}_{date_to}_{compkey}.json")

def _write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------- serialización mínima de fixtures --------------------

def _to_cache_match(m) -> Dict[str, Any]:
    return {
        "id": m.id,
        "league": {"code": m.league.code, "name": m.league.name, "region": getattr(m.league, "region", "")},
        "date_utc": m.date_utc.isoformat(),
        "home": {"id": m.home.id, "name": m.home.name},
        "away": {"id": m.away.id, "name": m.away.name},
        # features que puedan usar heurísticas
        "home_cards_pg": getattr(m, "home_cards_pg", None),
        "away_cards_pg": getattr(m, "away_cards_pg", None),
        "referee_cards_pg": getattr(m, "referee_cards_pg", None),
        "derby": getattr(m, "derby", False),
        "knockout": getattr(m, "knockout", False),
        "intl_comp_bonus": getattr(m, "intl_comp_bonus", 0.0),
        "home_fh_goals_pg": getattr(m, "home_fh_goals_pg", None),
        "away_fh_goals_pg": getattr(m, "away_fh_goals_pg", None),
        "home_shots_on_target_pg": getattr(m, "home_shots_on_target_pg", None),
        "away_shots_on_target_pg": getattr(m, "away_shots_on_target_pg", None),
        "ranking_diff": getattr(m, "ranking_diff", None),
        "weather_penalty": getattr(m, "weather_penalty", 0.0),
        "congestion_penalty": getattr(m, "congestion_penalty", 0.0),
    }

def _ns_from_cache(d: Dict[str, Any]) -> Any:
    league = NS(**(d.get("league") or {}))
    home = NS(**(d.get("home") or {}))
    away = NS(**(d.get("away") or {}))
    dt = datetime.fromisoformat(d.get("date_utc").replace("Z", "+00:00"))
    return NS(
        id=d.get("id"), league=league, home=home, away=away, date_utc=dt,
        home_cards_pg=d.get("home_cards_pg"), away_cards_pg=d.get("away_cards_pg"),
        referee_cards_pg=d.get("referee_cards_pg"), derby=d.get("derby"),
        knockout=d.get("knockout"), intl_comp_bonus=d.get("intl_comp_bonus"),
        home_fh_goals_pg=d.get("home_fh_goals_pg"), away_fh_goals_pg=d.get("away_fh_goals_pg"),
        home_shots_on_target_pg=d.get("home_shots_on_target_pg"),
        away_shots_on_target_pg=d.get("away_shots_on_target_pg"),
        ranking_diff=d.get("ranking_diff"),
        weather_penalty=d.get("weather_penalty"),
        congestion_penalty=d.get("congestion_penalty"),
    )

# -------------------- gestión de ventanas (superset-aware) --------------------

def _list_fixture_windows(compkey: str) -> List[Dict[str, Any]]:
    """
    Escanea .betcomb_cache y devuelve ventanas disponibles para un compkey.
    Cada item: {"from": "YYYY-MM-DD", "to":"YYYY-MM-DD", "path":"...", "entries": int}
    """
    out: List[Dict[str, Any]] = []
    if not os.path.isdir(CACHE_DIR):
        return out
    for fname in os.listdir(CACHE_DIR):
        if not (fname.startswith("fixtures_") and fname.endswith(f"_{compkey}.json")):
            continue
        fpath = os.path.join(CACHE_DIR, fname)
        try:
            data = _read_json(fpath)
            meta = data.get("meta", {})
            out.append({
                "from": meta.get("from"),
                "to": meta.get("to"),
                "path": fpath,
                "entries": len(data.get("matches", [])),
            })
        except Exception:
            continue
    # ordenar por tamaño de ventana asc (el superset más pequeño primero)
    def _span(x):
        try:
            a = datetime.fromisoformat((x["from"] or "") + "T00:00:00+00:00")
            b = datetime.fromisoformat((x["to"] or "") + "T00:00:00+00:00")
            return (b - a).days
        except Exception:
            return 10**9
    out.sort(key=_span)
    return out

def _covering(window_from: str, window_to: str, target_from: str, target_to: str) -> bool:
    return (window_from <= target_from) and (window_to >= target_to)

def _load_covering_fixtures(date_from: str, date_to: str, compkey: str) -> Optional[Dict[str, Any]]:
    """
    Busca el archivo de fixtures cuyo rango cubra [date_from, date_to].
    Devuelve {"path":..., "from":..., "to":..., "matches":[...]} o None si no hay cobertura.
    """
    for w in _list_fixture_windows(compkey):
        wf, wt = w.get("from"), w.get("to")
        if isinstance(wf, str) and isinstance(wt, str) and _covering(wf, wt, date_from, date_to):
            data = _read_json(w["path"])
            return {
                "path": w["path"],
                "from": wf,
                "to": wt,
                "matches": data.get("matches", []),
            }
    return None

def _merge_meta_windows(meta: Dict[str, Any], compkey: str, date_from: str, date_to: str, fpath: str, entries: int) -> Dict[str, Any]:
    """
    Mantiene en _meta.json una lista 'windows' con las ventanas disponibles por compkey.
    """
    windows: List[Dict[str, Any]] = meta.get("windows", [])
    # eliminar duplicados exactos
    windows = [w for w in windows if not (w.get("compkey")==compkey and w.get("from")==date_from and w.get("to")==date_to)]
    windows.append({
        "compkey": compkey,
        "from": date_from,
        "to": date_to,
        "path": os.path.basename(fpath),
        "entries": entries,
    })
    meta["windows"] = windows
    return meta

# -------------------- API: refresco de caché --------------------

def refresh_cache_api(days: int = 20, competitions: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Descarga fixtures (hoy → hoy+days) para las competiciones dadas y cachea UN archivo por ventana y compkey.
    Si luego pides 7 o 10 días, se reutiliza este archivo de 20 (si cubre el rango).
    """
    _ensure_cache_dir()
    comp_names = competitions or DEFAULT_COMP_NAMES
    codes = _codes_from_names(comp_names)
    if not codes:
        return {"ok": False, "msg": "Sin competiciones válidas para refrescar."}

    date_from, date_to = _window_for_days(days)
    compkey = _comp_key(comp_names)
    fpath = _fixtures_path(date_from, date_to, compkey)

    stats = APIFootballStats()
    matches = stats.fixtures(codes, days=days) or []

    payload = {
        "meta": {
            "from": date_from,
            "to": date_to,
            "competitions": comp_names,
            "codes": codes,
            "last_refresh": _iso(datetime.now(timezone.utc)),
        },
        "matches": [_to_cache_match(m) for m in matches],
    }
    _write_json(fpath, payload)

    # actualizar META
    meta = {}
    if os.path.isfile(META_FILE):
        try:
            meta = _read_json(META_FILE) or {}
        except Exception:
            meta = {}
    meta.update({
        "ok": True,
        "last_refresh": payload["meta"]["last_refresh"],
        "storage_path": CACHE_DIR,
    })
    meta = _merge_meta_windows(meta, compkey, date_from, date_to, fpath, len(matches))
    meta["last_window"] = {"from": date_from, "to": date_to, "compkey": compkey}
    meta["entries"] = len(matches)
    meta["files"] = list({*(meta.get("files", []) or []), os.path.basename(fpath)})

    _write_json(META_FILE, meta)
    return meta

# -------------------- API: estado de caché --------------------

def cache_status_api() -> Dict[str, Any]:
    _ensure_cache_dir()
    if os.path.isfile(META_FILE):
        try:
            meta = _read_json(META_FILE)
            ts = meta.get("last_refresh")
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if isinstance(ts, str) else None
            meta["age_seconds"] = (datetime.now(timezone.utc) - dt).total_seconds() if dt else None
            meta["ok"] = True
            return meta
        except Exception as e:
            return {"ok": False, "msg": f"Error leyendo meta: {e}", "storage_path": CACHE_DIR}

    # fallback mínimo si no hay _meta.json
    try:
        mtime = os.path.getmtime(CACHE_DIR)
        last_dt = datetime.fromtimestamp(mtime).astimezone().isoformat(timespec="seconds")
        total = 0
        for _root, _dirs, files in os.walk(CACHE_DIR):
            total += len(files)
        return {"ok": True, "last_refresh": last_dt, "entries": total, "storage_path": CACHE_DIR}
    except Exception as e:
        return {"ok": False, "msg": f"No existe caché o no se pudo leer: {e}", "storage_path": CACHE_DIR}

# -------------------- API: singles desde caché (superset-aware) --------------------

def _build_rows_from_cached(fixtures: List[Dict[str, Any]], use_odds: bool) -> List[Dict[str, Any]]:
    odds_provider = TheOddsAPI() if (use_odds and os.getenv("ODDS_API_KEY")) else None
    if not odds_provider:
        odds_provider = MockOdds()
    ids = [fx.get("id") for fx in fixtures if fx.get("id") is not None]
    all_quotes = odds_provider.odds_for_matches(ids) if odds_provider and ids else {}

    rows: List[Dict[str, Any]] = []
    for fx in fixtures:
        m = _ns_from_cache(fx)
        res = fh_over_0_5_score(m)
        p = float(res.prob)
        quotes = (all_quotes or {}).get(m.id, {})
        qlist = quotes.get(FH_OVER_0_5, [])
        if qlist:
            odd = float(min(qlist, key=lambda q: q.odds).odds)
        else:
            odd = round(1.0 / p, 2) if p > 0 else None
        rows.append({
            "date": m.date_utc.strftime("%Y-%m-%d"),
            "league": getattr(m.league, "name", ""),
            "match": f"{getattr(m.home, 'name', 'Home')} vs {getattr(m.away, 'name', 'Away')}",
            "market": "1erT +0.5",
            "odd": odd if odd is not None else "—",
            "p": round(p, 4),
        })
    rows.sort(key=lambda r: (r["date"], -r["p"], r["league"], r["match"]))
    return rows

def singles_fh_over05(
    days: int,
    competitions: List[str],
    persist: bool = True,
    use_odds: bool = False,
    threshold: float = 0.80,   # <-- nuevo: mínimo de probabilidad (0..1)
) -> List[Dict[str, Any]]:
    """
    Genera singles FH +0.5 para 'days' días **usando fixtures en caché**.
    Si no existe el archivo exacto, intenta encontrar un **superset** que cubra el rango
    (p. ej., si hay 20 días y pides 7, usa el de 20).
    Luego filtra a [today, today+days) y aplica umbral de probabilidad (p >= threshold).

    Si persist=True, guarda singles_fh_{from}_{to}_{compkey}.json con el rango pedido.
    """
    _ensure_cache_dir()
    if days < 7:
        days = 7
    try:
        threshold = float(threshold)
    except Exception:
        threshold = 0.80
    threshold = max(0.0, min(1.0, threshold))

    comp_names = competitions or DEFAULT_COMP_NAMES
    date_from, date_to = _window_for_days(days)
    compkey = _comp_key(comp_names)

    # 1) Buscar cobertura: exacta o superset
    cov = _load_covering_fixtures(date_from, date_to, compkey)
    if not cov:
        return []

    fixtures_all = cov["matches"]

    # 2) Filtrar al rango pedido (hoy → hoy+days)
    def _inside_window(fx: Dict[str, Any]) -> bool:
        try:
            d = fx.get("date_utc")
            if not isinstance(d, str):
                return False
            dt = datetime.fromisoformat(d.replace("Z", "+00:00")).date()
            return (date_from <= dt.isoformat() <= date_to)
        except Exception:
            return False

    fixtures = [fx for fx in fixtures_all if _inside_window(fx)]

    # 3) Armar rows desde caché (pero aplicando umbral antes de añadir)
    odds_provider = TheOddsAPI() if (use_odds and os.getenv("ODDS_API_KEY")) else None
    if not odds_provider:
        odds_provider = MockOdds()
    ids = [fx.get("id") for fx in fixtures if fx.get("id") is not None]
    all_quotes = odds_provider.odds_for_matches(ids) if odds_provider and ids else {}

    rows: List[Dict[str, Any]] = []
    for fx in fixtures:
        m = _ns_from_cache(fx)
        res = fh_over_0_5_score(m)
        p = float(res.prob)
        if p < threshold:
            continue  # <-- filtra por p mínima

        quotes = (all_quotes or {}).get(m.id, {})
        qlist = quotes.get(FH_OVER_0_5, [])
        if qlist:
            odd = float(min(qlist, key=lambda q: q.odds).odds)
        else:
            odd = round(1.0 / p, 2) if p > 0 else None

        rows.append({
            "date": m.date_utc.strftime("%Y-%m-%d"),
            "league": getattr(m.league, "name", ""),
            "match": f"{getattr(m.home, 'name', 'Home')} vs {getattr(m.away, 'name', 'Away')}",
            "market": "1erT +0.5",
            "odd": odd if odd is not None else "—",
            "p": round(p, 4),
        })

    rows.sort(key=lambda r: (r["date"], -r["p"], r["league"], r["match"]))

    # 4) (Opcional) Persistir singles específicas del rango pedido (incluye threshold en meta)
    if persist:
        singles_path = _singles_path(date_from, date_to, compkey)
        payload = {
            "meta": {
                "from": date_from, "to": date_to,
                "competitions": comp_names,
                "source_fixtures": os.path.basename(cov["path"]),
                "threshold": threshold,                        # <-- guardamos el umbral usado
                "generated_at": _iso(datetime.now(timezone.utc)),
            },
            "rows": rows,
        }
        _write_json(singles_path, payload)

    return rows
