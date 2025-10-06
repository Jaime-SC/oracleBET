# ¡DEBE SER LA PRIMERA INSTRUCCIÓN!
from __future__ import annotations

# betcomb/cli.py
import json
import os
import datetime
from typing import Optional, List, Dict, Tuple

import pandas as pd
import typer

from .config import SETTINGS
from .utils.logging import setup_logging
from .providers.mock import MockStats, MockOdds
from .providers.apifootball import APIFootballStats
from .providers.theodds import TheOddsAPI
from .domain.schemas import Slip, Match
from .domain.combiner import best_double, list_singles
from .domain.markets import MARKET_LABELS, FH_OVER_0_5
from .cache import make_key, load_pickle, save_pickle


app = typer.Typer(name="betcomb", help="Combinador de apuestas (doubles) con heurísticas.")


def _stats_provider(name: str):
    if name == "mock":
        return MockStats()
    elif name == "apifootball":
        return APIFootballStats()
    else:
        raise typer.BadParameter(f"Proveedor stats no soportado: {name}")


def _odds_provider(name: str):
    if name == "mock":
        return MockOdds()
    elif name == "theodds":
        return TheOddsAPI()
    else:
        raise typer.BadParameter(f"Proveedor odds no soportado: {name}")


@app.command()
def leagues(region: Optional[str] = typer.Option(None, "--region", help="eu|conmebol|uefa")):
    """
    Lista ligas disponibles y su código.
    """
    _ = setup_logging()
    stats = MockStats()  # Mock tiene el catálogo en samples
    leagues = stats.list_leagues(region)
    if not leagues:
        typer.echo("No hay ligas disponibles (¿region mal escrita?).")
        raise typer.Exit(code=1)
    df = pd.DataFrame([{"code": l.code, "name": l.name, "region": l.region} for l in leagues])
    typer.echo(df.to_string(index=False))


@app.command()
def fetch(
    leagues: str = typer.Option(..., "--leagues", help="Códigos separados por coma"),
    days: int = typer.Option(SETTINGS.days_default, "--days"),
    use_odds: str = typer.Option("yes", "--use-odds", help="yes|no"),
    provider_stats: str = typer.Option(SETTINGS.default_stats_provider, "--provider-stats"),
    provider_odds: str = typer.Option(SETTINGS.default_odds_provider, "--provider-odds"),
):
    """
    Descarga/lee fixtures y (opcional) cuotas.
    """
    _ = setup_logging()
    leagues_list = [x.strip() for x in leagues.split(",") if x.strip()]
    stats = _stats_provider(provider_stats)
    odds = _odds_provider(provider_odds)
    matches = stats.fixtures(leagues_list, days=days)
    typer.echo(f"Fixtures: {len(matches)} en {leagues_list} (≤{days} días).")
    if use_odds.lower() == "yes":
        q = odds.odds_for_matches([m.id for m in matches])
        typer.echo(f"Cuotas recuperadas para {len(q)} partidos.")
    else:
        typer.echo("Saltando cuotas (--use-odds=no).")


@app.command()
def suggest(
    leagues: str = typer.Option(...),
    days: int = typer.Option(7),
    min_total_odds: float = typer.Option(2.0),
    max_legs: int = typer.Option(2),
    explain: str = typer.Option("no"),
    provider_stats: str = typer.Option("mock"),
    provider_odds: str = typer.Option("mock"),
    export_format: Optional[str] = typer.Option(None, "--export-format", case_sensitive=False),
    markets: str = typer.Option("auto", help="both | fh-only | auto"),
    also_singles: int = typer.Option(0, help="Muestra además N mejores singles."),
    cache: str = typer.Option("readwrite", help="off|read|write|readwrite|refresh"),
    cache_ttl: int = typer.Option(6, help="TTL cache en horas."),
    fifa_regions: Optional[str] = typer.Option(None, "--fifa-regions", help="uefa,conmebol"),
):
    """
    Genera una combinada de 2 partidos. Modos de mercados:
      both   -> 2 selecciones por partido (CARDS_BTTS y 1T+0.5)
      fh-only-> solo 1T+0.5 por partido (1 selección por partido)
      auto   -> intenta both; si no hay, cae a fh-only

    Además, con --fifa-regions puedes sumar partidos de selecciones (fecha FIFA):
      uefa, conmebol (separados por coma)
    """
    if max_legs != 2:
        raise typer.BadParameter("Esta versión arma doubles (max_legs debe ser 2).")

    _ = setup_logging()
    leagues_list = [x.strip() for x in leagues.split(",") if x.strip()]

    stats = _stats_provider(provider_stats)
    odds = _odds_provider(provider_odds)

    # 1) Clubes
    matches: List[Match] = stats.fixtures(leagues_list, days=days)

    # 2) Selecciones (opcional)
    if fifa_regions:
        confeds = [c.strip().lower() for c in fifa_regions.split(",") if c.strip()]
        if hasattr(stats, "national_fixtures"):
            nat_matches: List[Match] = stats.national_fixtures(confeds=confeds, days=days)
            m_by_id = {m.id: m for m in matches}
            for nm in nat_matches:
                m_by_id[nm.id] = nm
            matches = list(m_by_id.values())
            typer.echo(f"+ Partidos de selecciones añadidos: {len(nat_matches)} (confeds={confeds})")
        else:
            typer.echo("Aviso: el proveedor de stats no implementa 'national_fixtures'; se ignora --fifa-regions.")

    if not matches:
        typer.echo("No hay fixtures en la ventana solicitada.")
        raise typer.Exit(code=2)

    by_id = {m.id: m for m in matches}
    quotes = odds.odds_for_matches([m.id for m in matches])

    mode = markets.lower()
    slip: Slip | None = None

    if mode in ("both", "fh-only"):
        slip = best_double(
            matches=matches,
            quotes=quotes,
            p_cards_min=SETTINGS.p_cards_min,
            p_fhgoal_min=SETTINGS.p_fhgoal_min,
            target_total_odds=min_total_odds,
            value_check=False,
            mode=mode,
            avoid_same_time=False,
        )
    else:
        slip = best_double(
            matches=matches,
            quotes=quotes,
            p_cards_min=SETTINGS.p_cards_min,
            p_fhgoal_min=SETTINGS.p_fhgoal_min,
            target_total_odds=min_total_odds,
            value_check=False,
            mode="both",
            avoid_same_time=False,
        ) or best_double(
            matches=matches,
            quotes=quotes,
            p_cards_min=SETTINGS.p_cards_min,
            p_fhgoal_min=SETTINGS.p_fhgoal_min,
            target_total_odds=min_total_odds,
            value_check=False,
            mode="fh-only",
            avoid_same_time=False,
        )

    if not slip:
        typer.echo("No se encontró combinada que cumpla los umbrales.")
        raise typer.Exit(code=2)

    # Helpers compat método/atributo
    def _get_total_odds(s: Slip) -> float:
        v = getattr(s, "total_odds", None)
        return v() if callable(v) else (v if v is not None else 0.0)

    def _get_joint_prob(s: Slip) -> float:
        v = getattr(s, "joint_prob", None)
        return v() if callable(v) else (v if v is not None else 0.0)

    fh_only = all(getattr(p, "market", "") == FH_OVER_0_5 for p in slip.picks)
    header = "=== SUGERENCIA DE COMBINADA (DOUBLE) ==="
    if fh_only:
        header += " [FH-only]"
    typer.echo(header)

    for i, p in enumerate(slip.picks, 1):
        label = MARKET_LABELS.get(p.market, p.market)
        m = by_id.get(p.match_id)
        mline = (f"{m.league.code} | {m.home.name} vs {m.away.name} | {m.date_utc:%Y-%m-%d %H:%M} UTC"
                 if m else f"Partido {p.match_id}")
        typer.echo(f"{i}. {label} | {mline} | cuota {p.odds:.2f} [{p.provider}]")
        if explain.lower() == "yes":
            typer.echo(f"   - Prob. estimada: {p.prob:.2%}")
            if getattr(p, "rationale", None):
                typer.echo(f"   - Razones: {p.rationale}")

    typer.echo(f"\nCuota total: {_get_total_odds(slip):.3f}")
    typer.echo(f"Probabilidad conjunta (modelo): {_get_joint_prob(slip):.2%}")

    # Export opcional
    if export_format:
        os.makedirs(SETTINGS.cache_dir, exist_ok=True)
        if export_format.lower() == "json":
            path = os.path.join(SETTINGS.cache_dir, "last_slip.json")
            with open(path, "w", encoding="utf-8") as f:
                if hasattr(slip, "to_dict"):
                    json.dump(slip.to_dict(), f, ensure_ascii=False, indent=2)
                else:
                    payload = {
                        "picks": [p.__dict__ for p in slip.picks],
                        "total_odds": _get_total_odds(slip),
                        "joint_prob": _get_joint_prob(slip),
                    }
                    json.dump(payload, f, ensure_ascii=False, indent=2)
            typer.echo(f"Exportado a JSON: {path}")
        elif export_format.lower() == "md":
            path = os.path.join(SETTINGS.cache_dir, "last_slip.md")
            with open(path, "w", encoding="utf-8") as f:
                title = "# Combinada sugerida (double)"
                if fh_only:
                    title += " [FH-only]"
                f.write(title + "\n\n")
                for p in slip.picks:
                    m = by_id.get(p.match_id)
                    if m:
                        f.write(
                            f"- **{MARKET_LABELS.get(p.market,p.market)}** en "
                            f"**{m.league.code} {m.home.name} vs {m.away.name}** "
                            f"({m.date_utc:%Y-%m-%d %H:%M} UTC) "
                            f"(cuota {p.odds:.2f}, {p.provider}) — p≈{p.prob:.2%}\n"
                        )
                    else:
                        f.write(
                            f"- **{MARKET_LABELS.get(p.market,p.market)}** en partido `{p.match_id}` "
                            f"(cuota {p.odds:.2f}, {p.provider}) — p≈{p.prob:.2%}\n"
                        )
                f.write(f"\n**Cuota total:** {_get_total_odds(slip):.3f}\n\n")
                f.write(f"**Prob conjunta (modelo):** {_get_joint_prob(slip):.2%}\n")
            typer.echo(f"Exportado a Markdown: {path}")
        else:
            raise typer.BadParameter("Formato no soportado: use json|md")


@app.command()
def singles(
    leagues: str = typer.Option(..., help="Códigos de ligas separados por coma."),
    days: int = typer.Option(7, help="Ventana de días hacia adelante."),
    provider_stats: str = typer.Option("mock"),
    provider_odds: str = typer.Option("mock"),
    markets: str = typer.Option("auto", help="both | fh-only | auto"),
    top_n: int = typer.Option(20, help="Cantidad de picks a listar."),
    min_odds: Optional[float] = typer.Option(None, help="Filtra cuota mínima."),
    max_odds: Optional[float] = typer.Option(None, help="Filtra cuota máxima."),
    explain: str = typer.Option("no", help="yes/no"),
    export_format: Optional[str] = typer.Option(None, "--export-format", case_sensitive=False),
    cache: str = typer.Option("readwrite", help="off|read|write|readwrite|refresh"),
    cache_ttl: int = typer.Option(6, help="TTL cache en horas."),
    fifa_regions: Optional[str] = typer.Option(None, "--fifa-regions", help="uefa,conmebol"),
):
    """
    Lista singles (picks sueltos) rankeados por edge (prob - prob implícita).
    Incluye selecciones (fecha FIFA) si se pasa --fifa-regions.
    """
    _ = setup_logging()
    leagues_list = [x.strip() for x in leagues.split(",") if x.strip()]
    stats = _stats_provider(provider_stats)
    odds = _odds_provider(provider_odds)

    # ---------------- Fixtures (clubes + selecciones) con cache ----------------
    fx_key = make_key("fixtures", provider_stats, leagues_list, days, (fifa_regions or "-"))
    matches = None
    read_ok = cache in ("read", "readwrite") and cache != "refresh"
    write_ok = cache in ("write", "readwrite", "refresh")

    if read_ok:
        matches = load_pickle(fx_key, max_age_s=cache_ttl * 3600)

    if matches is None:
        # Clubes
        matches = stats.fixtures(leagues_list, days=days) or []

        # Selecciones (UEFA/CONMEBOL) si se pidió y el provider lo soporta
        if fifa_regions:
            confeds = [c.strip().lower() for c in fifa_regions.split(",") if c.strip()]
            nat_fn = getattr(stats, "national_fixtures", None) or getattr(stats, "fixtures_internationals", None)
            if callable(nat_fn):
                try:
                    nat = nat_fn(confeds=confeds, days=days)
                except TypeError:
                    nat = nat_fn(confeds)
                nat = nat or []
                seen = {m.id for m in matches}
                matches.extend([m for m in nat if m.id not in seen])
            else:
                typer.echo("Aviso: el provider de stats no implementa internacionales (national_fixtures/fixtures_internationals).")

        if write_ok:
            save_pickle(fx_key, matches)

    if not matches:
        typer.echo("No hay fixtures en la ventana solicitada.")
        raise typer.Exit(code=2)

    by_id = {m.id: m for m in matches}

    # ---------------- Odds con cache ----------------
    match_ids = sorted(m.id for m in matches)  # ordenados para key estable
    od_key = make_key("odds", provider_odds, match_ids, days)
    quotes = None

    if read_ok:
        quotes = load_pickle(od_key, max_age_s=cache_ttl * 3600)

    if quotes is None:
        quotes = odds.odds_for_matches(match_ids)
        if write_ok:
            save_pickle(od_key, quotes)

    # ---------------- Ranking de singles ----------------
    mode = markets.lower()
    picks = list_singles(
        matches=matches,
        quotes=quotes,
        p_cards_min=SETTINGS.p_cards_min,
        p_fhgoal_min=SETTINGS.p_fhgoal_min,
        mode=mode,
        value_check=False,
        min_odds=min_odds,
        max_odds=max_odds,
        top_n=top_n,
    )

    if not picks:
        typer.echo("No hay singles que cumplan umbrales/mercados.")
        raise typer.Exit(code=2)

    typer.echo(f"=== TOP {len(picks)} SINGLES ===")
    for i, p in enumerate(picks, 1):
        label = MARKET_LABELS.get(p.market, p.market)
        m = by_id.get(p.match_id)
        mline = (f"{m.league.code} | {m.home.name} vs {m.away.name} | {m.date_utc:%Y-%m-%d %H:%M} UTC"
                 if m else f"Partido {p.match_id}")
        typer.echo(f"{i}. {label} | {mline} | cuota {p.odds:.2f} [{p.provider}]")
        if explain.lower() == "yes":
            typer.echo(f"   - Prob. estimada: {p.prob:.2%}")
            if getattr(p, "rationale", None):
                typer.echo(f"   - Razones: {p.rationale}")

    # ---------------- Export opcional ----------------
    if export_format:
        os.makedirs(SETTINGS.cache_dir, exist_ok=True)
        ext = export_format.lower()
        path = os.path.join(SETTINGS.cache_dir, f"last_singles.{ext}")

        if ext == "json":
            payload = [{
                "market": p.market,
                "match_id": p.match_id,
                "odds": float(p.odds),
                "prob": float(p.prob),
                "provider": p.provider,
                "match": {
                    "league": by_id[p.match_id].league.code if p.match_id in by_id else None,
                    "home": by_id[p.match_id].home.name if p.match_id in by_id else None,
                    "away": by_id[p.match_id].away.name if p.match_id in by_id else None,
                    "utc": by_id[p.match_id].date_utc.isoformat() if p.match_id in by_id else None,
                }
            } for p in picks]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        elif ext == "md":
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# Top {len(picks)} singles\n\n")

                # Agrupar: fecha (UTC) -> competición -> lista de (pick, match)
                groups: Dict[str, Dict[str, List[Tuple]]] = {}
                for p in picks:
                    m = by_id.get(p.match_id)
                    date_key = m.date_utc.strftime("%Y-%m-%d") if m else "SIN_FECHA"
                    comp_key = (m.league.code if (m and m.league) else "DESCONOCIDA")
                    groups.setdefault(date_key, {}).setdefault(comp_key, []).append((p, m))

                # Recorrer en orden por fecha, luego por competición, y por hora dentro
                for date_key in sorted(groups.keys()):
                    f.write(f"## {date_key}\n\n")
                    for comp_key in sorted(groups[date_key].keys()):
                        f.write(f"### {comp_key}\n\n")
                        items = sorted(
                            groups[date_key][comp_key],
                            key=lambda pm: (pm[1].date_utc if pm[1] else datetime.datetime.max.replace(tzinfo=datetime.timezone.utc))
                        )
                        for p, m in items:
                            label = MARKET_LABELS.get(p.market, p.market)
                            if m:
                                f.write(
                                    f"- **{label}** — **{m.home.name} vs {m.away.name}** "
                                    f"({m.date_utc:%H:%M} UTC) "
                                    f"(cuota {p.odds:.2f}, {p.provider}) — p≈{p.prob:.2%}\n"
                                )
                            else:
                                f.write(
                                    f"- **{label}** — partido `{p.match_id}` "
                                    f"(cuota {p.odds:.2f}, {p.provider}) — p≈{p.prob:.2%}\n"
                                )
                        f.write("\n")  # espacio entre competiciones
        else:
            raise typer.BadParameter("Formato no soportado: use json|md")

        typer.echo(f"Exportado: {path}")


@app.command()
def dryrun(file: List[str] = typer.Option(..., "--file", help="Paths JSON samples (fixtures/odds/leagues).")):
    """
    Ejecuta pipeline en modo offline leyendo data de ejemplo.
    """
    setup_logging()
    typer.echo("Modo offline: usando datasets de ejemplo (mock).")
    stats = MockStats()
    odds = MockOdds()
    leagues = stats.list_leagues()
    matches = stats.fixtures([l.code for l in leagues], days=7)
    quotes = odds.odds_for_matches([m.id for m in matches])
    slip = best_double(
        matches,
        quotes,
        SETTINGS.p_cards_min,
        SETTINGS.p_fhgoal_min,
        target_total_odds=2.0,
        mode="both",
    )
    if not slip:
        typer.echo("No se pudo crear combinada con samples.")
        raise typer.Exit(3)
    typer.echo("Dryrun OK — combinada encontrada.")
    if hasattr(slip, "to_dict"):
        typer.echo(json.dumps(slip.to_dict(), ensure_ascii=False, indent=2))
    else:
        payload = {
            "picks": [p.__dict__ for p in slip.picks],
            "total_odds": getattr(slip, "total_odds", None),
            "joint_prob": getattr(slip, "joint_prob", None),
        }
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command()
def export(format: str = typer.Option("json", "--format", help="json|md")):
    """
    Exporta la última sugerencia guardada por --export-format desde suggest.
    """
    path = os.path.join(SETTINGS.cache_dir, f"last_slip.{format}")
    if not os.path.exists(path):
        typer.echo("No hay export previo. Ejecuta `betcomb suggest ... --export-format json|md`.")
        raise typer.Exit(4)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    typer.echo(content)


def main():
    app()


if __name__ == "__main__":
    main()
