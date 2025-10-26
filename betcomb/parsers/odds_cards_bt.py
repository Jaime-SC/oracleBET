"""Parser de cuotas para el mercado "Both Teams To Receive a Card"."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import typer

from ..utils.logging import setup_logging

LOG = logging.getLogger(__name__)

app = typer.Typer(name="odds-cards", help="Normaliza cuotas BT Card en CSV/JSON.")

FIXTURE_ID_CANDIDATES: tuple[str, ...] = (
    "fixture_id",
    "fixtureId",
    "match_id",
    "matchId",
    "event_id",
    "id",
)
ODDS_CANDIDATES: tuple[str, ...] = (
    "odds",
    "odd",
    "price",
    "decimal_odds",
    "value",
    "book_odds",
)
BOOKMAKER_CANDIDATES: tuple[str, ...] = (
    "bookmaker",
    "book",
    "provider",
    "source",
)
MARKET_CANDIDATES: tuple[str, ...] = (
    "market",
    "selection",
    "bet_type",
)
MARKET_KEYWORDS: tuple[str, ...] = (
    "both teams to receive a card",
    "both teams booked",
    "both teams to get",
    "ambos equipos",
    "bt card",
)


def _sample_odds() -> pd.DataFrame:
    LOG.warning("Usando cuotas de demostración.")
    data = [
        {"fixture_id": "FX1", "bookmaker": "DemoBook", "book_odds": 1.85},
        {"fixture_id": "FX2", "bookmaker": "DemoBook", "book_odds": 2.10},
        {"fixture_id": "FX3", "bookmaker": "DemoBook", "book_odds": 1.65},
        {"fixture_id": "FX4", "bookmaker": "DemoBook", "book_odds": 2.40},
    ]
    return pd.DataFrame(data)


def _read_input(path: Path, decimal: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path, decimal=decimal)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".json"}:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, dict) and "data" in raw:
            raw = raw["data"]
        return pd.json_normalize(raw)
    raise typer.BadParameter(f"Formato no soportado: {path.suffix}")


def _resolve_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in df.columns:
        low = col.lower()
        for candidate in candidates:
            if low == candidate.lower():
                return col
    return None


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map: Dict[str, str] = {}
    fixture_col = _resolve_column(df, FIXTURE_ID_CANDIDATES)
    odds_col = _resolve_column(df, ODDS_CANDIDATES)
    book_col = _resolve_column(df, BOOKMAKER_CANDIDATES)
    market_col = _resolve_column(df, MARKET_CANDIDATES)

    if not fixture_col or not odds_col:
        raise typer.BadParameter("El archivo debe contener fixture_id y odds.")

    rename_map.update({fixture_col: "fixture_id", odds_col: "book_odds"})
    if book_col:
        rename_map[book_col] = "bookmaker"
    if market_col:
        rename_map[market_col] = "market"

    df = df.rename(columns=rename_map)
    return df


def _filter_market(df: pd.DataFrame) -> pd.DataFrame:
    if "market" not in df.columns:
        return df
    mask = df["market"].astype(str).str.lower().apply(
        lambda x: any(keyword in x for keyword in MARKET_KEYWORDS)
    )
    filtered = df[mask]
    if filtered.empty:
        LOG.warning(
            "No se encontraron filas que coincidan con el mercado BT Card; se mantiene dataset original."
        )
        return df
    return filtered


def _aggregate(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if df.empty:
        return df
    df["book_odds"] = pd.to_numeric(df["book_odds"], errors="coerce")
    df = df[df["book_odds"].notna()].copy()
    if df.empty:
        raise typer.BadParameter("No hay cuotas válidas tras la limpieza.")

    mode = mode.lower()
    if mode not in {"best", "mean"}:
        raise typer.BadParameter("--mode debe ser 'best' o 'mean'.")

    agg = df.groupby("fixture_id")
    if mode == "best":
        result = agg["book_odds"].max().reset_index()
    else:
        result = agg["book_odds"].mean().reset_index()

    return result


@app.command("parse")
def parse_odds(
    in_path: Path = typer.Option(..., "--in", help="Archivo CSV/JSON/Parquet a normalizar."),
    out_path: Path = typer.Option(Path("cache/odds_cards_bt.csv"), "--out", help="Destino CSV."),
    mode: str = typer.Option("best", "--mode", help="best|mean para combinar cuotas."),
    decimal: str = typer.Option(".", "--decimal", help="Separador decimal del CSV de origen."),
) -> None:
    """Lee cuotas crudas y exporta un CSV homogéneo con columnas ``fixture_id`` y ``book_odds``."""
    _ = setup_logging()

    try:
        if str(in_path).lower() == "demo" or (not in_path.exists() and in_path.name.lower() == "demo"):
            df = _sample_odds()
        else:
            df = _read_input(in_path, decimal)
    except FileNotFoundError:
        typer.echo(f"Archivo no encontrado: {in_path}")
        raise typer.Exit(code=2)

    df = _normalise_columns(df)
    df = _filter_market(df)

    result = _aggregate(df, mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    typer.echo(f"✓ {out_path} → {len(result)} fixtures")


__all__ = ["app", "parse_odds"]
