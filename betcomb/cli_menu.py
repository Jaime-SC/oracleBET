# -*- coding: utf-8 -*-
"""
CLI Menu for oracleBET (Jaime)
- Opciones: refrescar caché API, ver estado de caché, ver singles 1er tiempo (submenú), salir.

Hooks por defecto (reales):
    - betcomb.hooks_api:refresh_cache_api
    - betcomb.hooks_api:cache_status_api
    - betcomb.hooks_api:singles_fh_over05

También puedes sobreescribir por ENV:
    ORACLEBET_HOOK_CACHE_REFRESH="pkg.mod:func"
    ORACLEBET_HOOK_CACHE_STATUS="pkg.mod:func"
    ORACLEBET_HOOK_SINGLES_FH_OVER05="pkg.mod:func"

Formatos esperados:
    CACHE_STATUS -> {
        "ok": True,
        "last_refresh": "2025-10-12T15:23:11Z" o "2025-10-12 12:23:11-03:00",
        "entries": 1234,
        "storage_path": ".betcomb_cache",
        "age_seconds": 321
    }
    SINGLES_FH_OVER05 -> list[{
        "date":"YYYY-MM-DD",
        "league":"Premier League",
        "match":"Team A vs Team B",
        "market":"1erT +0.5",
        "odd":1.55,
        "p":0.70
    }, ...]
"""

from __future__ import annotations
import os
import sys
import time
import json
import importlib
from dataclasses import dataclass
from typing import Callable, Optional, Any, Dict, List
import typer
from datetime import datetime, timezone
from collections import defaultdict
# === Cargar .env si existe (para API_FOOTBALL_KEY, etc.) ===
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

app = typer.Typer(add_completion=False, help="oracleBET – Menú interactivo")

# ====== Utils (estilo) ======
RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
GREEN = "\033[32m"; YELLOW = "\033[33m"; CYAN = "\033[36m"; RED = "\033[31m"
MIN_PROB = float(os.getenv("ORACLEBET_MIN_PROB", "0.80"))  # 0..1


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")

def pause(msg: str = "Presiona Enter para continuar...") -> None:
    try:
        input(f"\n{DIM}{msg}{RESET}")
    except EOFError:
        pass

# ====== Hook loader ======
def load_callable_from_env(env_key: str) -> Optional[Callable[..., Any]]:
    """
    Env format: 'package.module:callable'
    """
    spec = os.getenv(env_key, "").strip()
    if not spec:
        return None
    if ":" not in spec:
        typer.echo(f"{YELLOW}Advertencia:{RESET} {env_key} debe tener formato 'pkg.mod:func'")
        return None
    module_path, func_name = spec.split(":", 1)
    try:
        mod = importlib.import_module(module_path)
        fn = getattr(mod, func_name)
        if not callable(fn):
            typer.echo(f"{RED}Error:{RESET} {env_key} → {spec} no es invocable.")
            return None
        return fn
    except Exception as e:
        typer.echo(f"{RED}Error cargando {env_key}:{RESET} {spec}\n{DIM}{e}{RESET}")
        return None

def resolve_hook(env_key: str, default_callable: Optional[Callable[..., Any]]) -> Optional[Callable[..., Any]]:
    """
    Si existe un callable en ENV, úsalo. Si no, usa el default importado.
    """
    fn_env = load_callable_from_env(env_key)
    return fn_env or default_callable

def safe_call(fn: Optional[Callable[..., Any]], *args, **kwargs) -> Any:
    if not fn:
        return {"ok": False, "msg": "Hook no configurado."}
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return {"ok": False, "msg": f"Excepción al ejecutar hook: {e}"}

@dataclass
class MenuItem:
    key: str
    label: str
    handler: Callable[[], None]

# ====== Importar hooks reales por defecto ======
try:
    from betcomb.hooks_api import (
        refresh_cache_api,
        cache_status_api,
        singles_fh_over05,
    )
except Exception:
    # Si por algún motivo no existe hooks_api, caeremos 100% en ENV/placeholder
    refresh_cache_api = None
    cache_status_api = None
    singles_fh_over05 = None

# ====== Hooks efectivos (ENV sobrescribe; si no, defaults reales) ======
HOOK_CACHE_REFRESH   = resolve_hook("ORACLEBET_HOOK_CACHE_REFRESH",   refresh_cache_api)
HOOK_CACHE_STATUS    = resolve_hook("ORACLEBET_HOOK_CACHE_STATUS",    cache_status_api)
HOOK_SINGLES_FH_O05  = resolve_hook("ORACLEBET_HOOK_SINGLES_FH_OVER05", singles_fh_over05)

# ====== Competitions (filtro fijo) ======
COMP_EU5_UCL_UEL_CONMEBOL: List[str] = [
    # EU5
    "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1",
    # UEFA
    "UEFA Champions League", "Champions League",
    "UEFA Europa League", "Europa League",
    # CONMEBOL
    "Copa Libertadores", "CONMEBOL Libertadores",
    "Copa Sudamericana", "CONMEBOL Sudamericana",
]

# ====== Render helpers ======
def _print_kv(d: Dict[str, Any]) -> None:
    if not isinstance(d, dict) or not d:
        return
    width = max((len(str(k)) for k in d.keys()), default=10)
    for k, v in d.items():
        print(f"{BOLD}{str(k).rjust(width)}{RESET} : {v}")

def _render_predictions(rows: Any) -> None:
    if not rows:
        print(f"{YELLOW}No hay predicciones para mostrar.{RESET}")
        return
    print(f"{DIM}Fecha{' ' * 6}Liga{' ' * 22} Partido{' ' * 24} Mercado    Cuota   p{RESET}")
    print("-" * 78)
    for row in rows:
        date = str(row.get("date", ""))
        league = str(row.get("league", ""))[:20].ljust(20)
        match = str(row.get("match", ""))[:22].ljust(22)
        market = str(row.get("market", "")).ljust(9)
        odd = f"{row.get('odd','')}".ljust(6)
        try:
            p = float(row.get("p", 0))
        except Exception:
            p = 0.0
        p_str = f"{round(p*100, 2):.2f}%".rjust(6)
        print(f"{date}  {league}  {match}  {market}  {odd}  {p_str}")

def _render_predictions_grouped(rows: List[Dict[str, Any]]) -> None:
    """
    Muestra las singles agrupadas por competición (league), con orden preferido EU5+UCL+UEL+CONMEBOL.
    """
    if not rows:
        print(f"{YELLOW}No hay predicciones para mostrar.{RESET}")
        return

    preferred_order = [
        "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1",
        "UEFA Champions League", "Champions League",
        "UEFA Europa League", "Europa League",
        "Copa Libertadores", "CONMEBOL Libertadores",
        "Copa Sudamericana", "CONMEBOL Sudamericana",
    ]
    order_idx = {name: i for i, name in enumerate(preferred_order)}

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        league = str(r.get("league", "") or "")
        groups[league].append(r)

    # Orden de grupos: primero los de la lista preferida (en ese orden), luego alfabético
    sorted_leagues = sorted(groups.keys(), key=lambda k: (order_idx.get(k, 999), k))

    total = 0
    for league in sorted_leagues:
        items = groups[league]
        # Orden interno: fecha asc, prob desc, partido
        def _pval(x):
            try:
                return float(x.get("p", 0.0))
            except Exception:
                return 0.0
        items = sorted(items, key=lambda x: (str(x.get("date","")), -_pval(x), str(x.get("match",""))))

        print(f"\n{BOLD}{CYAN}{league}{RESET}  {DIM}({len(items)} picks){RESET}")
        print(f"{DIM}Fecha{' ' * 6}Partido{' ' * 28} Mercado    Cuota   p{RESET}")
        print("-" * 78)
        for row in items:
            date = str(row.get("date", ""))
            match = str(row.get("match", ""))[:32].ljust(32)
            market = str(row.get("market", "")).ljust(9)
            odd = f"{row.get('odd','')}".ljust(6)
            try:
                p = float(row.get("p", 0))
            except Exception:
                p = 0.0
            p_str = f"{round(p*100, 2):.2f}%".rjust(6)
            print(f"{date}  {match}  {market}  {odd}  {p_str}")
            total += 1

    print("\n" + "-" * 78)
    print(f"{BOLD}Total picks:{RESET} {total}  {DIM}(agrupadas por competición){RESET}")

def _render_predictions_by_day(rows: List[Dict[str, Any]]) -> None:
    """
    Muestra las singles agrupadas por día (YYYY-MM-DD).
    Dentro de cada día: orden por prob desc, luego liga y partido.
    """
    if not rows:
        print(f"{YELLOW}No hay predicciones para mostrar.{RESET}")
        return

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        d = str(r.get("date", "") or "")
        groups[d].append(r)

    def _pval(x):
        try:
            return float(x.get("p", 0.0))
        except Exception:
            return 0.0

    total = 0
    for d in sorted(groups.keys()):
        items = sorted(groups[d], key=lambda x: (-_pval(x), str(x.get("league","")), str(x.get("match",""))))
        print(f"\n{BOLD}{CYAN}{d}{RESET}  {DIM}({len(items)} picks){RESET}")
        print(f"{DIM}Liga{' ' * 20}Partido{' ' * 28} Mercado    Cuota   p{RESET}")
        print("-" * 78)
        for row in items:
            league = str(row.get("league", ""))[:22].ljust(22)
            match = str(row.get("match", ""))[:28].ljust(28)
            market = str(row.get("market", "")).ljust(9)
            odd = f"{row.get('odd','')}".ljust(6)
            try:
                p = float(row.get("p", 0))
            except Exception:
                p = 0.0
            p_str = f"{round(p*100, 2):.2f}%".rjust(6)
            print(f"{league}  {match}  {market}  {odd}  {p_str}")
            total += 1

    print("\n" + "-" * 78)
    print(f"{BOLD}Total picks:{RESET} {total}  {DIM}(agrupadas por día){RESET}")


# ====== Cache status fallback (.betcomb_cache) ======
def _local_cache_status() -> Dict[str, Any]:
    """
    Fallback por si no hay HOOK_CACHE_STATUS:
    - Mira el directorio .betcomb_cache
    - Devuelve last_modified y conteo de archivos
    """
    path = ".betcomb_cache"
    data = {"ok": True, "storage_path": path}
    try:
        if not os.path.isdir(path):
            return {"ok": False, "msg": "No existe .betcomb_cache"}
        # último mtime del directorio
        mtime = os.path.getmtime(path)
        last_dt = datetime.fromtimestamp(mtime).astimezone()
        # contar archivos
        total = 0
        for _root, _dirs, files in os.walk(path):
            total += len(files)
        data.update({
            "last_refresh": last_dt.isoformat(timespec="seconds"),
            "entries": total,
        })
        return data
    except Exception as e:
        return {"ok": False, "msg": f"Error leyendo caché local: {e}"}

# ====== Handlers ======
def handle_cache_refresh() -> None:
    clear_screen()
    print(f"{BOLD}Cargar/Actualizar datos y caché{RESET}\n")
    # → ahora pedimos el rango también aquí
    def _ask_days(min_days: int = 7) -> int:
        print(f"{DIM}Selecciona rango de días (mínimo {min_days}).{RESET}")
        print("  1) 7 días\n  2) 10 días\n  3) 20 días\n  4) Otro (>= 7)")
        choice = (input("\nOpción: ").strip() or "1")
        mapping = {"1": 7, "2": 10, "3": 20}
        if choice in mapping:
            return mapping[choice]
        while True:
            raw = input("Ingresa número de días (>= 7): ").strip()
            if not raw.isdigit():
                print(f"{RED}Debe ser un entero.{RESET}")
                continue
            val = int(raw)
            if val < min_days:
                print(f"{YELLOW}Mínimo {min_days} días.{RESET}")
                continue
            return val

    days = _ask_days(min_days=7)
    print()

    if HOOK_CACHE_REFRESH:
        # Pasamos días + competiciones fijas
        res = safe_call(HOOK_CACHE_REFRESH, days=days, competitions=COMP_EU5_UCL_UEL_CONMEBOL)
        if isinstance(res, dict) and not res.get("ok", True):
            print(f"{RED}{res.get('msg')}{RESET}")
        else:
            print(f"{GREEN}Actualización completada correctamente.{RESET}")
            if isinstance(res, dict) and res:
                print()
                _print_kv(res)
    else:
        print(f"{YELLOW}Hook no configurado:{RESET} ORACLEBET_HOOK_CACHE_REFRESH")
        print("Simulación: actualización ejecutada (placeholder).")

    print("\n" + f"{BOLD}Estado de caché tras la actualización:{RESET}\n")
    handle_cache_status(show_header=False)
    pause()


def handle_cache_status(show_header: bool = True) -> None:
    if show_header:
        clear_screen()
        print(f"{BOLD}Estado de la caché{RESET}\n")
    if HOOK_CACHE_STATUS:
        res = safe_call(HOOK_CACHE_STATUS)
        if isinstance(res, dict) and not res.get("ok", True):
            print(f"{RED}{res.get('msg')}{RESET}")
            print(f"\n{DIM}Intentando fallback local...{RESET}\n")
            local = _local_cache_status()
            if local.get("ok"):
                _print_kv(local)
            else:
                print(f"{RED}{local.get('msg')}{RESET}")
        else:
            _print_kv(res if isinstance(res, dict) else {"resultado": res})
    else:
        print(f"{YELLOW}Hook no configurado:{RESET} ORACLEBET_HOOK_CACHE_STATUS")
        print(f"{DIM}Usando fallback local .betcomb_cache{RESET}\n")
        local = _local_cache_status()
        if local.get("ok"):
            _print_kv(local)
        else:
            print(f"{RED}{local.get('msg')}{RESET}")

def _ask_days(min_days: int = 7) -> int:
    print(f"{DIM}Selecciona rango de días (mínimo {min_days}).{RESET}")
    print("  1) 7 días\n  2) 10 días\n  3) 20 días\n  4) Otro (>= 7)")
    choice = (input("\nOpción: ").strip() or "1")
    mapping = {"1": 7, "2": 10, "3": 20}
    if choice in mapping:
        return mapping[choice]
    # Otro
    while True:
        raw = input("Ingresa número de días (>= 7): ").strip()
        if not raw.isdigit():
            print(f"{RED}Debe ser un entero.{RESET}")
            continue
        val = int(raw)
        if val < min_days:
            print(f"{YELLOW}Mínimo {min_days} días.{RESET}")
            continue
        return val

def handle_singles_first_half() -> None:
    clear_screen()
    print(f"{BOLD}Singles – Gol en 1er tiempo (+0.5){RESET}\n")
    days = _ask_days(min_days=7)
    print()
    print(f"{DIM}Competiciones fijas:{RESET}")
    print("- EU5 (Premier, La Liga, Serie A, Bundesliga, Ligue 1)")
    print("- UEFA (Champions, Europa League)")
    print("- CONMEBOL (Libertadores, Sudamericana)\n")

    if HOOK_SINGLES_FH_O05:
        res = safe_call(
            HOOK_SINGLES_FH_O05,
            days=days,
            competitions=COMP_EU5_UCL_UEL_CONMEBOL,
            threshold=MIN_PROB,
        )
        if isinstance(res, dict) and not res.get("ok", True):
            print(f"{RED}{res.get('msg')}{RESET}")
        else:
            _render_predictions_by_day(res)   # ← ahora agrupado por día

    else:
        print(f"{YELLOW}Hook no configurado:{RESET} ORACLEBET_HOOK_SINGLES_FH_OVER05")
        print("Ejemplo (mock):\n")
        demo = [
            {"date":"2025-10-11","league":"Premier League","match":"Fulham vs Spurs","market":"1erT +0.5","odd":1.58,"p":0.71},
            {"date":"2025-10-12","league":"UEFA Champions League","match":"PSG vs Milan","market":"1erT +0.5","odd":1.52,"p":0.69},
            {"date":"2025-10-12","league":"Copa Libertadores","match":"Boca vs Palmeiras","market":"1erT +0.5","odd":1.60,"p":0.72},
        ]
        _render_predictions(demo)
        print("\nConfigura tu función real con:\n  set ORACLEBET_HOOK_SINGLES_FH_OVER05=betcomb.predictions:first_half_over05_filtered")

    pause()

def handle_exit() -> None:
    clear_screen()
    print(f"{GREEN}Hasta luego 👋{RESET}")
    time.sleep(0.30)
    raise typer.Exit(code=0)

# ====== Menú principal ======
def _menu_loop() -> None:
    items = [
        MenuItem("1", "Cargar/Actualizar data y caché", handle_cache_refresh),
        MenuItem("2", "Ver estado de caché (última actualización)", handle_cache_status),
        MenuItem("3", "Singles 1er tiempo (+0.5) – submenú por rango", handle_singles_first_half),
        MenuItem("0", "Salir", handle_exit),
    ]
    while True:
        clear_screen()
        print(f"{BOLD}oracleBET – Menú principal{RESET}")
        print(f"{DIM}Selecciona una opción ingresando su número.{RESET}\n")
        for it in items:
            print(f"  {CYAN}{it.key}{RESET}  {it.label}")
        choice = input(f"\n{BOLD}Opción:{RESET} ").strip()
        found = next((x for x in items if x.key == choice), None)
        if not found:
            print(f"{RED}Opción inválida.{RESET}")
            time.sleep(0.8)
            continue
        try:
            found.handler()
        except typer.Exit:
            raise
        except Exception as e:
            print(f"{RED}Error:{RESET} {e}")
            pause()

# ====== Typer wiring (sin subcomando y con "menu") ======
@app.command("menu")
def menu_cmd() -> None:
    _menu_loop()

@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        _menu_loop()

# Subcomandos directos (útiles en scripts/CI)
@app.command("cache-refresh")
def cache_refresh_cmd():
    handle_cache_refresh()

@app.command("cache-status")
def cache_status_cmd():
    handle_cache_status()

@app.command("singles-half")
def singles_half_cmd():
    handle_singles_first_half()

if __name__ == "__main__":
    app()
