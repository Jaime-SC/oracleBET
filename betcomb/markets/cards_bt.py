"""Modelo logístico para el mercado "Both Teams Receive ≥1 Card"."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import typer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .. import __version__
from ..aggregators.cards_bt_aggregator import (
    TEAM_STATS_BASE,
    build_historicals_df,
    compute_league_baselines,
    compute_referee_summary,
    compute_team_features,
    load_match_dataset,
)
from ..utils.logging import setup_logging

LOG = logging.getLogger(__name__)

app = typer.Typer(name="cards-bt", help="Entrena y puntúa el modelo BT Card.")

TARGET_COLUMN = "bt_card"
CONTINUOUS_FEATURES: Tuple[str, ...] = (
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
)
FLAG_FEATURES: Tuple[str, ...] = (
    "a_home",
    "is_derby",
    "is_knockout",
    "ref_missing",
)
CATEGORICAL_FEATURES: Tuple[str, ...] = ("league_id", "season")
NUMERIC_FEATURES: Tuple[str, ...] = CONTINUOUS_FEATURES + FLAG_FEATURES
MODEL_FEATURES: Tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def _read_parquet(path: Path) -> Optional[pd.DataFrame]:
    if path and path.exists():
        return pd.read_parquet(path)
    return None


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if path and path.exists():
        return pd.read_csv(path)
    return None


def _load_training_frame(data_path: Path) -> pd.DataFrame:
    if data_path.exists():
        LOG.info("Cargando dataset de entrenamiento desde %s", data_path)
        df = pd.read_parquet(data_path)
    else:
        LOG.warning("%s no existe; se generará dataset de demostración.", data_path)
        base = load_match_dataset(None)
        team_features = compute_team_features(base)
        df = build_historicals_df(base, team_features=team_features)
    required = set(MODEL_FEATURES) | {TARGET_COLUMN}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise typer.BadParameter(f"Dataset de entrenamiento incompleto. Faltan columnas: {missing}")
    return df


def _build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(NUMERIC_FEATURES)),
            ("cat", categorical_transformer, list(CATEGORICAL_FEATURES)),
        ]
    )
    model = LogisticRegression(max_iter=600, class_weight="balanced")
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def _load_model(model_path: Path) -> Tuple[Pipeline, Dict[str, object]]:
    if not model_path.exists():
        raise typer.BadParameter(f"Modelo no encontrado en {model_path}")
    package = joblib.load(model_path)
    if isinstance(package, Pipeline):
        return package, {}
    if isinstance(package, dict):
        model = package.get("model")
        if model is None:
            raise typer.BadParameter("El archivo de modelo no contiene la clave 'model'.")
        metadata = package.get("metadata", {})
        return model, metadata
    raise typer.BadParameter("Formato de modelo desconocido.")


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df


def _prepare_rollings_export(df: pd.DataFrame) -> pd.DataFrame:
    export_cols = [
        "fixture_id",
        "team_id",
        "is_home",
        "rest_days",
        *[f"{stat}_90_l10" for stat in TEAM_STATS_BASE],
    ]
    available = [col for col in export_cols if col in df.columns]
    return df[available].copy()


def _generate_demo_inputs() -> Dict[str, pd.DataFrame]:
    base = load_match_dataset(None)
    team_features = compute_team_features(base)
    leagues = compute_league_baselines(team_features)
    referees = compute_referee_summary(team_features)
    historicals = build_historicals_df(base, team_features=team_features)
    odds = pd.DataFrame(
        {
            "fixture_id": historicals["fixture_id"],
            "book_odds": np.linspace(1.7, 2.5, num=len(historicals)),
        }
    )
    fixtures = historicals.drop(columns=[TARGET_COLUMN], errors="ignore").copy()
    rollings = _prepare_rollings_export(team_features)
    return {
        "fixtures": fixtures,
        "rollings": rollings,
        "leagues": leagues,
        "referees": referees,
        "odds": odds,
    }


def _merge_features(
    fixtures: pd.DataFrame,
    rollings: Optional[pd.DataFrame],
    leagues: Optional[pd.DataFrame],
    referees: Optional[pd.DataFrame],
) -> pd.DataFrame:
    fixtures = fixtures.copy()
    fixtures["a_home"] = fixtures.get("a_home", 1).fillna(1).astype(int)
    fixtures["is_derby"] = fixtures.get("is_derby", 0).fillna(0).astype(int)
    fixtures["is_knockout"] = fixtures.get("is_knockout", 0).fillna(0).astype(int)

    features_present = all(col in fixtures.columns for col in CONTINUOUS_FEATURES)
    if not features_present:
        if rollings is None or leagues is None:
            raise typer.BadParameter(
                "Se requieren rollings y leagues para completar las features de BT Card."
            )
        if "team_a_id" not in fixtures.columns or "team_b_id" not in fixtures.columns:
            raise typer.BadParameter("Fixtures debe incluir 'team_a_id' y 'team_b_id'.")
        home_rollings = rollings[rollings.get("is_home", False)].copy()
        away_rollings = rollings[~rollings.get("is_home", False)].copy()

        home_map = {
            "team_id": "team_a_id",
            "rest_days": "rest_days_a",
            "cards_for_90_l10": "a_cards_for_90_l10",
            "cards_against_90_l10": "a_cards_against_90_l10",
            "fouls_for_90_l10": "a_fouls_for_90_l10",
            "fouls_against_90_l10": "a_fouls_against_90_l10",
        }
        away_map = {
            "team_id": "team_b_id",
            "rest_days": "rest_days_b",
            "cards_for_90_l10": "b_cards_for_90_l10",
            "cards_against_90_l10": "b_cards_against_90_l10",
            "fouls_for_90_l10": "b_fouls_for_90_l10",
            "fouls_against_90_l10": "b_fouls_against_90_l10",
        }

        home_rollings = home_rollings.rename(columns=home_map)
        away_rollings = away_rollings.rename(columns=away_map)

        merge_home_cols = [
            "fixture_id",
            "team_a_id",
            "rest_days_a",
            "a_cards_for_90_l10",
            "a_cards_against_90_l10",
            "a_fouls_for_90_l10",
            "a_fouls_against_90_l10",
        ]
        merge_away_cols = [
            "fixture_id",
            "team_b_id",
            "rest_days_b",
            "b_cards_for_90_l10",
            "b_cards_against_90_l10",
            "b_fouls_for_90_l10",
            "b_fouls_against_90_l10",
        ]

        fixtures = fixtures.merge(
            home_rollings[merge_home_cols],
            on=["fixture_id", "team_a_id"],
            how="left",
        )
        fixtures = fixtures.merge(
            away_rollings[merge_away_cols],
            on=["fixture_id", "team_b_id"],
            how="left",
        )

        fixtures["rest_days_a"] = fixtures.get("rest_days_a", fixtures.get("rest_days", 7)).fillna(7.0)
        fixtures["rest_days_b"] = fixtures.get("rest_days_b", 7.0).fillna(7.0)

    if leagues is not None and "league_cards_avg_90" not in fixtures.columns:
        league_cols = [
            "league_id",
            "season",
            "league_cards_avg_90",
            "league_fouls_avg_90",
            "league_yellow_avg_90",
            "league_red_avg_90",
        ]
        available = [col for col in league_cols if col in leagues.columns]
        fixtures = fixtures.merge(
            leagues[available],
            on=["league_id", "season"],
            how="left",
        )

    if referees is not None and "ref_cards_90" not in fixtures.columns:
        fixtures = fixtures.merge(
            referees,
            left_on=["referee_id", "season"],
            right_on=["referee_id", "season"],
            how="left",
        )

    fixtures["ref_missing"] = fixtures[["ref_cards_90", "ref_yellow_90"]].isna().any(axis=1).astype(int)
    fill_map = {
        "ref_cards_90": "league_cards_avg_90",
        "ref_yellow_90": "league_yellow_avg_90",
        "ref_red_90": "league_red_avg_90",
    }
    for ref_col, league_col in fill_map.items():
        if ref_col in fixtures.columns and league_col in fixtures.columns:
            fixtures[ref_col] = fixtures[ref_col].fillna(fixtures[league_col])
    if "ref_cards_sd" in fixtures.columns:
        fixtures["ref_cards_sd"] = fixtures["ref_cards_sd"].fillna(0.0)
    else:
        fixtures["ref_cards_sd"] = 0.0
    if "ref_form_l10" in fixtures.columns:
        fixtures["ref_form_l10"] = fixtures["ref_form_l10"].fillna(fixtures.get("ref_cards_90"))
    else:
        fixtures["ref_form_l10"] = fixtures.get("ref_cards_90")

    fixtures = _ensure_columns(fixtures, MODEL_FEATURES)
    return fixtures


def _prepare_prediction_dataset(
    fixtures_path: Path,
    rollings_path: Path,
    referees_path: Path,
    leagues_path: Path,
    odds_path: Path,
) -> pd.DataFrame:
    fixtures = _read_parquet(fixtures_path)
    rollings = _read_parquet(rollings_path)
    referees = _read_parquet(referees_path)
    leagues = _read_parquet(leagues_path)
    odds = _read_csv(odds_path)

    if fixtures is None or rollings is None or leagues is None:
        LOG.warning("Entradas incompletas detectadas. Se usarán datos de demostración.")
        inputs = _generate_demo_inputs()
        fixtures = inputs["fixtures"]
        rollings = inputs["rollings"]
        leagues = inputs["leagues"]
        referees = inputs["referees"]
        odds = inputs["odds"]
    else:
        fixtures = fixtures.copy()

    fixtures = _merge_features(fixtures, rollings, leagues, referees)

    if odds is None:
        LOG.warning("Archivo de cuotas no disponible; se asignan odds=NaN.")
        fixtures["book_odds"] = np.nan
    else:
        odds = odds.rename(columns={"odd": "book_odds"})
        fixtures = fixtures.merge(odds[["fixture_id", "book_odds"]], on="fixture_id", how="left")

    fixtures = fixtures.drop(columns=[TARGET_COLUMN], errors="ignore")
    fixtures = fixtures.sort_values(["date", "league_id", "fixture_id"], ignore_index=True)
    return fixtures


# ---------------------------------------------------------------------------
# Comandos Typer
# ---------------------------------------------------------------------------


@app.command("train")
def train_cards_bt(
    data_path: Path = typer.Option(Path("cache/historicos_cards_bt.parquet"), "--data-path"),
    model_out: Path = typer.Option(Path("models/cards_bt.joblib"), "--model-out"),
    test_size: float = typer.Option(0.2, "--test-size", min=0.1, max=0.5),
    random_state: int = typer.Option(42, "--random-state"),
) -> None:
    """Entrena un modelo logístico calibrado para BT Card."""
    _ = setup_logging()
    df = _load_training_frame(data_path)
    df = df.dropna(subset=[TARGET_COLUMN])
    X = df[list(MODEL_FEATURES)]
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "log_loss": float(log_loss(y_test, np.clip(y_proba, 1e-6, 1 - 1e-6))),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "samples_train": int(len(y_train)),
        "samples_test": int(len(y_test)),
    }

    metadata = {
        "version": __version__,
        "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "features": list(MODEL_FEATURES),
        "metrics": metrics,
    }

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": pipeline, "metadata": metadata}, model_out)

    typer.echo(f"Modelo guardado en {model_out}")
    typer.echo(json.dumps(metrics, indent=2, ensure_ascii=False))


@app.command("predict")
def predict_cards_bt(
    fixtures_parquet: Path = typer.Option(Path("cache/fixtures_next.parquet"), "--fixtures-parquet"),
    rollings_parquet: Path = typer.Option(Path("cache/rollings.parquet"), "--rollings-parquet"),
    referees_parquet: Path = typer.Option(Path("cache/referees.parquet"), "--refs-parquet"),
    leagues_parquet: Path = typer.Option(Path("cache/leagues.parquet"), "--league-parquet"),
    odds_csv: Path = typer.Option(Path("cache/odds_cards_bt.csv"), "--odds-csv"),
    model_path: Path = typer.Option(Path("models/cards_bt.joblib"), "--model-path"),
    ev_min: float = typer.Option(0.05, "--ev-min"),
    top_n: Optional[int] = typer.Option(None, "--top-n", help="Limita la salida a N picks."),
    out_path: Optional[Path] = typer.Option(None, "--out", help="Exporta resultados a CSV."),
) -> None:
    """Genera predicciones y EV para el mercado BT Card."""
    _ = setup_logging()

    model, metadata = _load_model(model_path)
    df = _prepare_prediction_dataset(
        fixtures_path=fixtures_parquet,
        rollings_path=rollings_parquet,
        referees_path=referees_parquet,
        leagues_path=leagues_parquet,
        odds_path=odds_csv,
    )

    X = df[list(MODEL_FEATURES)]
    probs = model.predict_proba(X)[:, 1]
    df["prob"] = probs
    df["ev"] = df["prob"] * df["book_odds"] - 1.0

    if ev_min is not None:
        df = df[df["ev"].fillna(-1) >= ev_min]

    df = df.sort_values("ev", ascending=False)
    if top_n is not None and top_n > 0:
        df = df.head(top_n)

    columns_to_show = [
        "fixture_id",
        "date",
        "league_id",
        "team_a_id",
        "team_b_id",
        "book_odds",
        "prob",
        "ev",
    ]
    display_cols = [col for col in columns_to_show if col in df.columns]
    typer.echo(df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        typer.echo(f"Predicciones exportadas a {out_path}")

    if metadata:
        typer.echo(f"Modelo entrenado en: {metadata.get('trained_at')} (v{metadata.get('version')})")


__all__ = ["app", "train_cards_bt", "predict_cards_bt"]
