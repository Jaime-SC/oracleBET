"""cards_bt_aggregator
========================

Herramientas para construir los datasets necesarios para el mercado
"Ambos equipos reciben ≥1 tarjeta" (BT Card).

El objetivo del módulo es seguir la filosofía del proyecto:
- Mantener comandos Typer que se integren al CLI principal.
- Trabajar sobre una carpeta ``cache/`` local y reutilizar los cachés
  existentes (``.betcomb_cache``) siempre que sea posible.
- Proveer *fallbacks* reproducibles con datasets en memoria para poder
  demostrar el flujo end-to-end aunque falten datos reales.

El dataset de entrada esperado contiene **una fila por equipo y por
partido** con, al menos, las siguientes columnas::

    fixture_id, date, league_id, season, team_id, opponent_id,
    is_home (bool), home_team_id,
    cards_for, cards_against, fouls_for, fouls_against,
    yellow_for, red_for, minutes_played,
    referee_id, referee_name, is_derby, is_knockout, rest_days

Si alguna columna falta, se intenta deducirla o completar valores
razonables. El dataset puede llegar desde ``cache/`` (CSV/Parquet/JSON),
algún caché producido por ``hooks_api`` o, en última instancia, un
*fixture* de demostración.

Comandos disponibles (se registran como ``oraclebet agg-cards``):

``build-historicals``
    Genera ``cache/historicos_cards_bt.parquet`` con las features y el
    target ``bt_card``.

``build-rollings``
    Exporta ``cache/rollings.parquet`` con rolling features por equipo.

``build-leagues``
    Calcula promedios por liga/temporada en ``cache/leagues.parquet``.

``build-referees``
    Resume el desempeño de árbitros en ``cache/referees.parquet``.

``build-all``
    Ejecuta los cuatro pasos anteriores.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import typer

from ..config import SETTINGS
from ..utils.logging import setup_logging

LOG = logging.getLogger(__name__)

app = typer.Typer(name="agg-cards", help="Datasets para el mercado BT Card.")

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------

ROLLING_WINDOWS: Tuple[int, ...] = (5, 10, 20)
MIN_MATCHES_ROLLING = 3
CACHE_DIR = Path(os.getenv("ORACLEBET_CARDS_CACHE", "cache"))
DEFAULT_SOURCE_CANDIDATES: Tuple[str, ...] = (
    # Usuario puede ubicar sus dumps en cache/
    "cache/cards_bt_source.parquet",
    "cache/cards_bt_source.csv",
    "cache/cards_bt_source.json",
    # Caché histórico del proyecto (.betcomb_cache)
    os.path.join(getattr(SETTINGS, "cache_dir", ".betcomb_cache"), "cards_bt_source.parquet"),
)

COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "fixture_id": ("fixture_id", "fixtureId", "match_id", "matchId", "game_id"),
    "date": ("date", "match_date", "fixture_date", "kickoff"),
    "league_id": ("league_id", "leagueId", "competition_id", "comp_id"),
    "season": ("season", "season_id", "year"),
    "team_id": ("team_id", "teamId", "teamID", "home_id", "away_id"),
    "opponent_id": ("opponent_id", "opponentId", "opp_id", "rival_id"),
    "is_home": ("is_home", "home", "local"),
    "home_team_id": ("home_team_id", "homeId", "home_team"),
    "cards_for": ("cards_for", "cards", "team_cards", "cards_received"),
    "cards_against": ("cards_against", "opp_cards", "cards_conceded"),
    "fouls_for": ("fouls_for", "fouls", "team_fouls"),
    "fouls_against": ("fouls_against", "opp_fouls", "fouls_conceded"),
    "yellow_for": ("yellow_for", "yellow_cards", "team_yellow"),
    "red_for": ("red_for", "red_cards", "team_red"),
    "minutes_played": ("minutes_played", "mins", "minutes"),
    "referee_id": ("referee_id", "ref_id", "official_id"),
    "referee_name": ("referee_name", "ref_name", "official"),
    "is_derby": ("is_derby", "derby", "is_classic"),
    "is_knockout": ("is_knockout", "knockout", "is_cup"),
    "rest_days": ("rest_days", "days_rest", "rest"),
}


@dataclass
class AggregationResult:
    """Representa el resultado de una etapa de agregación."""

    dataframe: pd.DataFrame
    path: Path


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def _ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def _find_column(df: pd.DataFrame, logical_name: str) -> Optional[str]:
    aliases = COLUMN_ALIASES.get(logical_name, (logical_name,))
    for alias in aliases:
        if alias in df.columns:
            return alias
        # allow case-insensitive match
        for col in df.columns:
            if col.lower() == alias.lower():
                return col
    return None


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map: Dict[str, str] = {}
    for logical, aliases in COLUMN_ALIASES.items():
        existing = _find_column(df, logical)
        if existing and existing != logical:
            rename_map[existing] = logical
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _read_source(path: Path) -> pd.DataFrame:
    LOG.info("Leyendo dataset base: %s", path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)
    if path.suffix.lower() in {".json"}:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return pd.json_normalize(raw)
    raise typer.BadParameter(f"Formato no soportado: {path.suffix}")


def _sample_dataset() -> pd.DataFrame:
    LOG.warning("Usando dataset de demostración en memoria para BT Card.")
    data = [
        {
            "fixture_id": "FX1",
            "date": "2023-03-10",
            "league_id": "EPL",
            "season": "2022/23",
            "team_id": "ARS",
            "opponent_id": "CHE",
            "is_home": True,
            "home_team_id": "ARS",
            "cards_for": 2,
            "cards_against": 3,
            "fouls_for": 10,
            "fouls_against": 12,
            "yellow_for": 2,
            "red_for": 0,
            "minutes_played": 90,
            "referee_id": "REF1",
            "referee_name": "Howard Webb",
            "is_derby": True,
            "is_knockout": False,
            "rest_days": 7,
        },
        {
            "fixture_id": "FX1",
            "date": "2023-03-10",
            "league_id": "EPL",
            "season": "2022/23",
            "team_id": "CHE",
            "opponent_id": "ARS",
            "is_home": False,
            "home_team_id": "ARS",
            "cards_for": 3,
            "cards_against": 2,
            "fouls_for": 14,
            "fouls_against": 9,
            "yellow_for": 3,
            "red_for": 1,
            "minutes_played": 90,
            "referee_id": "REF1",
            "referee_name": "Howard Webb",
            "is_derby": True,
            "is_knockout": False,
            "rest_days": 6,
        },
        {
            "fixture_id": "FX2",
            "date": "2023-03-18",
            "league_id": "EPL",
            "season": "2022/23",
            "team_id": "CHE",
            "opponent_id": "MCI",
            "is_home": True,
            "home_team_id": "CHE",
            "cards_for": 0,
            "cards_against": 2,
            "fouls_for": 11,
            "fouls_against": 13,
            "yellow_for": 0,
            "red_for": 0,
            "minutes_played": 90,
            "referee_id": "REF2",
            "referee_name": "Mike Dean",
            "is_derby": False,
            "is_knockout": False,
            "rest_days": 7,
        },
        {
            "fixture_id": "FX2",
            "date": "2023-03-18",
            "league_id": "EPL",
            "season": "2022/23",
            "team_id": "MCI",
            "opponent_id": "CHE",
            "is_home": False,
            "home_team_id": "CHE",
            "cards_for": 2,
            "cards_against": 0,
            "fouls_for": 9,
            "fouls_against": 15,
            "yellow_for": 2,
            "red_for": 0,
            "minutes_played": 90,
            "referee_id": "REF2",
            "referee_name": "Mike Dean",
            "is_derby": False,
            "is_knockout": False,
            "rest_days": 6,
        },
        {
            "fixture_id": "FX3",
            "date": "2023-04-02",
            "league_id": "UEL",
            "season": "2022/23",
            "team_id": "SEV",
            "opponent_id": "ARS",
            "is_home": True,
            "home_team_id": "SEV",
            "cards_for": 4,
            "cards_against": 5,
            "fouls_for": 17,
            "fouls_against": 18,
            "yellow_for": 4,
            "red_for": 0,
            "minutes_played": 90,
            "referee_id": "REF3",
            "referee_name": "Clément Turpin",
            "is_derby": False,
            "is_knockout": True,
            "rest_days": 8,
        },
        {
            "fixture_id": "FX3",
            "date": "2023-04-02",
            "league_id": "UEL",
            "season": "2022/23",
            "team_id": "ARS",
            "opponent_id": "SEV",
            "is_home": False,
            "home_team_id": "SEV",
            "cards_for": 5,
            "cards_against": 4,
            "fouls_for": 19,
            "fouls_against": 17,
            "yellow_for": 5,
            "red_for": 1,
            "minutes_played": 90,
            "referee_id": "REF3",
            "referee_name": "Clément Turpin",
            "is_derby": False,
            "is_knockout": True,
            "rest_days": 6,
        },
        {
            "fixture_id": "FX4",
            "date": "2024-02-11",
            "league_id": "EPL",
            "season": "2023/24",
            "team_id": "ARS",
            "opponent_id": "MCI",
            "is_home": True,
            "home_team_id": "ARS",
            "cards_for": 1,
            "cards_against": 0,
            "fouls_for": 12,
            "fouls_against": 8,
            "yellow_for": 1,
            "red_for": 0,
            "minutes_played": 90,
            "referee_id": "REF2",
            "referee_name": "Mike Dean",
            "is_derby": False,
            "is_knockout": False,
            "rest_days": 9,
        },
        {
            "fixture_id": "FX4",
            "date": "2024-02-11",
            "league_id": "EPL",
            "season": "2023/24",
            "team_id": "MCI",
            "opponent_id": "ARS",
            "is_home": False,
            "home_team_id": "ARS",
            "cards_for": 0,
            "cards_against": 1,
            "fouls_for": 7,
            "fouls_against": 10,
            "yellow_for": 0,
            "red_for": 0,
            "minutes_played": 90,
            "referee_id": "REF2",
            "referee_name": "Mike Dean",
            "is_derby": False,
            "is_knockout": False,
            "rest_days": 7,
        },
    ]
    return pd.DataFrame(data)


def _resolve_source_path(source: Optional[str]) -> Optional[Path]:
    if source:
        p = Path(source)
        if p.exists():
            return p
        raise typer.BadParameter(f"No se encontró el archivo fuente: {source}")
    for candidate in DEFAULT_SOURCE_CANDIDATES:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


def load_match_dataset(source: Optional[str] = None) -> pd.DataFrame:
    path = _resolve_source_path(source)
    if path is None:
        df = _sample_dataset()
    else:
        df = _read_source(path)
    df = _standardise_columns(df)

    # Normalizar tipos básicos
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise typer.BadParameter("El dataset debe incluir una columna de fecha (date/kickoff).")

    defaults = {
        "cards_for": 0.0,
        "cards_against": 0.0,
        "fouls_for": 0.0,
        "fouls_against": 0.0,
        "yellow_for": 0.0,
        "red_for": 0.0,
        "minutes_played": 90.0,
        "is_derby": False,
        "is_knockout": False,
        "rest_days": np.nan,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)

    for col in ("fixture_id", "league_id", "season", "team_id", "opponent_id"):
        if col not in df.columns:
            raise typer.BadParameter(f"Falta la columna obligatoria: {col}")

    if "is_home" not in df.columns:
        LOG.info("Columna 'is_home' ausente; se asume True si team_id == home_team_id.")
        if "home_team_id" not in df.columns:
            raise typer.BadParameter("Faltan columnas 'is_home' y 'home_team_id'.")
        df["is_home"] = df["team_id"] == df["home_team_id"]
    else:
        df["is_home"] = df["is_home"].astype(bool)

    if "home_team_id" not in df.columns:
        LOG.info("Deduciendo 'home_team_id' a partir de filas locales.")
        home_map = df[df["is_home"]].set_index("fixture_id")["team_id"].to_dict()
        df["home_team_id"] = df["fixture_id"].map(home_map)

    if df["home_team_id"].isna().any():
        raise typer.BadParameter("No se pudo deducir 'home_team_id' para todos los fixtures.")

    # Rest days: si vienen como NaN se calculan en compute_team_features
    df["referee_id"] = df.get("referee_id")
    df["referee_name"] = df.get("referee_name")
    df["is_derby"] = df["is_derby"].astype(bool)
    df["is_knockout"] = df["is_knockout"].astype(bool)

    return df


# ---------------------------------------------------------------------------
# Cálculo de features
# ---------------------------------------------------------------------------


TEAM_STATS_BASE = ("cards_for", "cards_against", "fouls_for", "fouls_against")


def compute_team_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["team_id", "date", "fixture_id"])

    # Minutes normalizados
    df["minutes_played"] = df["minutes_played"].replace(0, np.nan).fillna(90.0)

    for stat in TEAM_STATS_BASE + ("yellow_for", "red_for"):
        per90_col = f"{stat}_90"
        df[per90_col] = df[stat].astype(float) * 90.0 / df["minutes_played"]

    # Rest days si faltan
    if df["rest_days"].isna().any():
        df["rest_days"] = (
            df.groupby("team_id")["date"].diff().dt.total_seconds().div(86400)
        )
    df["rest_days"] = df["rest_days"].fillna(7.0)

    grouped = df.groupby(["team_id", "is_home"], sort=False)
    for stat in TEAM_STATS_BASE:
        per90_col = f"{stat}_90"
        for window in ROLLING_WINDOWS:
            new_col = f"{stat}_90_l{window}"
            df[new_col] = grouped[per90_col].transform(
                lambda s: s.shift(1).rolling(window, min_periods=MIN_MATCHES_ROLLING).mean()
            )

    for window in ROLLING_WINDOWS:
        missing = df[f"cards_for_90_l{window}"].isna().sum()
        if missing:
            LOG.warning(
                "Ventana l%s: %s filas sin historial suficiente (min=%s partidos).",
                window,
                missing,
                MIN_MATCHES_ROLLING,
            )

    df["games_played"] = grouped["fixture_id"].cumcount()

    return df


def compute_league_baselines(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["date"].notna()]
    df["season"] = df["season"].astype(str)
    df["league_id"] = df["league_id"].astype(str)

    agg = df.groupby(["league_id", "season"]).agg(
        matches=("fixture_id", "nunique"),
        teams=("team_id", "nunique"),
        league_cards_avg_90=("cards_for_90", "mean"),
        league_fouls_avg_90=("fouls_for_90", "mean"),
        league_yellow_avg_90=("yellow_for_90", "mean"),
        league_red_avg_90=("red_for_90", "mean"),
    )
    agg = agg.reset_index()
    return agg


def compute_referee_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["referee_id"].notna() & df["referee_id"].astype(str).str.len() > 0]
    if df.empty:
        LOG.warning("Sin árbitros confirmados en el dataset base.")
        return pd.DataFrame(
            columns=[
                "referee_id",
                "season",
                "ref_cards_90",
                "ref_yellow_90",
                "ref_red_90",
                "ref_cards_sd",
                "ref_form_l10",
                "matches",
            ]
        )

    fixture_totals = (
        df.groupby(["fixture_id", "referee_id", "season", "date"], as_index=False)
        .agg(
            cards_total=("cards_for", "sum"),
            yellow_total=("yellow_for", "sum"),
            red_total=("red_for", "sum"),
            minutes_total=("minutes_played", "sum"),
        )
    )
    fixture_totals["minutes_match"] = fixture_totals["minutes_total"].replace(0, np.nan).fillna(180.0) / 2.0
    fixture_totals["cards_90"] = fixture_totals["cards_total"] * 90.0 / fixture_totals["minutes_match"]
    fixture_totals["yellow_90"] = fixture_totals["yellow_total"] * 90.0 / fixture_totals["minutes_match"]
    fixture_totals["red_90"] = fixture_totals["red_total"] * 90.0 / fixture_totals["minutes_match"]

    fixture_totals = fixture_totals.sort_values(["referee_id", "season", "date", "fixture_id"])

    def _form(series: pd.Series) -> float:
        tail = series.dropna().tail(10)
        if tail.empty:
            return float("nan")
        weights = np.exp(np.linspace(-2.0, 0.0, num=len(tail)))
        return float(np.average(tail, weights=weights))

    summary_rows: List[Dict[str, float]] = []
    for (ref_id, season), group in fixture_totals.groupby(["referee_id", "season"], sort=False):
        cards_series = group["cards_90"].astype(float)
        row = {
            "referee_id": ref_id,
            "season": str(season),
            "matches": int(group.shape[0]),
            "ref_cards_90": float(cards_series.mean()),
            "ref_yellow_90": float(group["yellow_90"].astype(float).mean()),
            "ref_red_90": float(group["red_90"].astype(float).mean()),
            "ref_cards_sd": float(cards_series.std(ddof=0)) if group.shape[0] > 1 else 0.0,
            "ref_form_l10": _form(cards_series),
        }
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Construcción de datasets finales
# ---------------------------------------------------------------------------


TARGET_COLUMNS = [
    "fixture_id",
    "date",
    "league_id",
    "season",
    "team_a_id",
    "team_b_id",
    "home_team_id",
    "bt_card",
    "a_cards_for_90_l10",
    "a_cards_against_90_l10",
    "a_fouls_for_90_l10",
    "a_fouls_against_90_l10",
    "b_cards_for_90_l10",
    "b_cards_against_90_l10",
    "b_fouls_for_90_l10",
    "b_fouls_against_90_l10",
    "league_cards_avg_90",
    "league_fouls_avg_90",
    "ref_cards_90",
    "ref_yellow_90",
    "ref_red_90",
    "ref_cards_sd",
    "ref_form_l10",
    "rest_days_a",
    "rest_days_b",
    "league_yellow_avg_90",
    "league_red_avg_90",
    "a_home",
    "is_derby",
    "is_knockout",
    "ref_missing",
]


def build_historicals_df(df: pd.DataFrame, team_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    team_features = team_features if team_features is not None else compute_team_features(df)
    leagues = compute_league_baselines(team_features)
    referees = compute_referee_summary(team_features)

    home = team_features[team_features["is_home"]].copy()
    away = team_features[~team_features["is_home"]].copy()

    if home.empty or away.empty:
        raise typer.BadParameter("El dataset base debe incluir filas de local y visita por fixture.")

    rename_map_home = {
        "team_id": "team_a_id",
        "opponent_id": "team_b_id",
        "cards_for_90_l10": "a_cards_for_90_l10",
        "cards_against_90_l10": "a_cards_against_90_l10",
        "fouls_for_90_l10": "a_fouls_for_90_l10",
        "fouls_against_90_l10": "a_fouls_against_90_l10",
        "rest_days": "rest_days_a",
    }
    rename_map_away = {
        "team_id": "team_b_id",
        "opponent_id": "team_a_id",
        "cards_for_90_l10": "b_cards_for_90_l10",
        "cards_against_90_l10": "b_cards_against_90_l10",
        "fouls_for_90_l10": "b_fouls_for_90_l10",
        "fouls_against_90_l10": "b_fouls_against_90_l10",
        "rest_days": "rest_days_b",
    }

    home = home.rename(columns=rename_map_home)
    away = away.rename(columns=rename_map_away)

    merged = pd.merge(
        home,
        away[
            [
                "fixture_id",
                "team_b_id",
                "team_a_id",
                "b_cards_for_90_l10",
                "b_cards_against_90_l10",
                "b_fouls_for_90_l10",
                "b_fouls_against_90_l10",
                "rest_days_b",
            ]
        ],
        on=["fixture_id", "team_b_id"],
        how="inner",
        validate="one_to_one",
    )

    merged["bt_card"] = (
        (merged["cards_for"] >= 1) &
        (merged["cards_against"] >= 1)
    ).astype(int)

    if {"team_a_id_x", "team_a_id_y"}.issubset(merged.columns):
        if not merged["team_a_id_x"].equals(merged["team_a_id_y"]):
            LOG.warning("team_a_id inconsistente entre local y visita; se prioriza la fila local.")
        merged["team_a_id"] = merged["team_a_id_x"]
        merged = merged.drop(columns=["team_a_id_x", "team_a_id_y"], errors="ignore")
    elif "team_a_id_x" in merged.columns:
        merged = merged.rename(columns={"team_a_id_x": "team_a_id"})
    elif "team_a_id_y" in merged.columns:
        merged = merged.rename(columns={"team_a_id_y": "team_a_id"})

    merged["a_home"] = True

    league_cols = [
        "league_id",
        "season",
        "league_cards_avg_90",
        "league_fouls_avg_90",
        "league_yellow_avg_90",
        "league_red_avg_90",
    ]
    merged = merged.merge(leagues[league_cols], on=["league_id", "season"], how="left")

    merged["referee_id"] = merged.get("referee_id")
    merged = merged.merge(
        referees,
        left_on=["referee_id", "season"],
        right_on=["referee_id", "season"],
        how="left",
    )

    numeric_ref_cols = ["ref_cards_90", "ref_yellow_90", "ref_red_90", "ref_cards_sd", "ref_form_l10"]
    for ref_col in numeric_ref_cols:
        if ref_col in merged.columns:
            merged[ref_col] = pd.to_numeric(merged[ref_col], errors="coerce")

    merged["ref_missing"] = merged[["ref_cards_90", "ref_yellow_90"]].isna().any(axis=1).astype(int)

    fill_map = {
        "ref_cards_90": "league_cards_avg_90",
        "ref_yellow_90": "league_yellow_avg_90",
        "ref_red_90": "league_red_avg_90",
    }
    for ref_col, league_col in fill_map.items():
        if league_col in merged.columns:
            merged[ref_col] = merged[ref_col].fillna(merged[league_col])
    merged["ref_cards_sd"] = merged["ref_cards_sd"].fillna(0.0)
    merged["ref_form_l10"] = merged["ref_form_l10"].fillna(merged["ref_cards_90"])

    merged["a_home"] = merged["a_home"].astype(int)
    merged["is_derby"] = merged["is_derby"].astype(int)
    merged["is_knockout"] = merged["is_knockout"].astype(int)

    merged = merged.assign(
        rest_days_a=merged["rest_days_a"].astype(float),
        rest_days_b=merged["rest_days_b"].astype(float),
    )

    merged = merged[TARGET_COLUMNS].copy()
    merged = merged.sort_values(["date", "league_id", "fixture_id"])
    return merged


# ---------------------------------------------------------------------------
# Comandos Typer
# ---------------------------------------------------------------------------


def _export(df: pd.DataFrame, filename: str) -> AggregationResult:
    path = _ensure_cache_dir() / filename
    df.to_parquet(path, index=False)
    typer.echo(f"✓ {filename} → {len(df):,} filas")
    return AggregationResult(dataframe=df, path=path)


@app.command("build-rollings")
def build_rollings(source: Optional[str] = typer.Option(None, "--source", help="Ruta CSV/Parquet base.")) -> None:
    """Construye rolling features por equipo."""
    _ = setup_logging()
    df = load_match_dataset(source)
    features = compute_team_features(df)
    export_cols = [
        "fixture_id",
        "team_id",
        "opponent_id",
        "is_home",
        "date",
        "league_id",
        "season",
        "rest_days",
    ]
    for stat in TEAM_STATS_BASE:
        for window in ROLLING_WINDOWS:
            export_cols.append(f"{stat}_90_l{window}")
    result = features[export_cols].copy()
    _export(result, "rollings.parquet")


@app.command("build-leagues")
def build_leagues(source: Optional[str] = typer.Option(None, "--source", help="Ruta CSV/Parquet base.")) -> None:
    """Calcula promedios por liga/temporada."""
    _ = setup_logging()
    df = load_match_dataset(source)
    features = compute_team_features(df)
    leagues = compute_league_baselines(features)
    _export(leagues, "leagues.parquet")


@app.command("build-referees")
def build_referees(source: Optional[str] = typer.Option(None, "--source", help="Ruta CSV/Parquet base.")) -> None:
    """Genera métricas de árbitros por temporada."""
    _ = setup_logging()
    df = load_match_dataset(source)
    features = compute_team_features(df)
    referees = compute_referee_summary(features)
    _export(referees, "referees.parquet")


@app.command("build-historicals")
def build_historicals(source: Optional[str] = typer.Option(None, "--source", help="Ruta CSV/Parquet base.")) -> None:
    """Construye el dataset histórico con label ``bt_card``."""
    _ = setup_logging()
    df = load_match_dataset(source)
    historicals = build_historicals_df(df)
    _export(historicals, "historicos_cards_bt.parquet")


@app.command("build-all")
def build_all(source: Optional[str] = typer.Option(None, "--source", help="Ruta CSV/Parquet base.")) -> None:
    """Ejecuta el pipeline completo (historicals + rollings + leagues + referees)."""
    _ = setup_logging()
    df = load_match_dataset(source)
    features = compute_team_features(df)
    historicals = build_historicals_df(df, team_features=features)
    _export(features[[
        "fixture_id",
        "team_id",
        "opponent_id",
        "is_home",
        "date",
        "league_id",
        "season",
        "rest_days",
        *[f"{stat}_90_l{window}" for stat in TEAM_STATS_BASE for window in ROLLING_WINDOWS],
    ]], "rollings.parquet")
    _export(compute_league_baselines(features), "leagues.parquet")
    _export(compute_referee_summary(features), "referees.parquet")
    _export(historicals, "historicos_cards_bt.parquet")


__all__ = [
    "app",
    "load_match_dataset",
    "compute_team_features",
    "compute_league_baselines",
    "compute_referee_summary",
    "build_historicals_df",
]
