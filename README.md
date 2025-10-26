# oracle-bet / betcomb

CLI para sugerir **combinadas (doubles)** y **singles** con heurísticas sobre mercados como:
- **1er tiempo +0.5 goles**
- (opcional) **CARDS_BTTS** (si tu proveedor de cuotas/estadísticas lo soporta)

Incluye:
- **Cache local** de fixtures y cuotas para ahorrar llamadas a APIs y acelerar la respuesta.
- **Exportación** a Markdown/JSON.
- Soporte para **partidos de selecciones (fecha FIFA)** vía `--fifa-regions` (UEFA / CONMEBOL).
- Agrupación en el export **por fecha** y **por competición**.

---

## Requisitos

- Python 3.10+
- Clave de API para [API-FOOTBALL](https://www.api-football.com/) si vas a usar `--provider-stats apifootball`.

Instala dependencias:

```bash
pip install -r requirements.txt
```

## README (extracto) – Mercado "Ambos equipos reciben ≥1 tarjeta"

Flujo end-to-end sugerido usando los nuevos comandos Typer:

1. **Construir datasets auxiliares**
   ```bash
   oraclebet agg-cards build-all --source cache/cards_bt_source.parquet
   ```
   El comando escribe `historicos_cards_bt.parquet`, `rollings.parquet`,
   `leagues.parquet` y `referees.parquet` dentro de `cache/`. Si no hay
   archivo `--source`, se usa un dataset de demostración incluido en el
   código para validar el pipeline.

2. **Normalizar cuotas del mercado BT Card**
   ```bash
   oraclebet odds-cards parse --in odds_raw.csv --out cache/odds_cards_bt.csv
   ```
   Admite CSV/JSON/Parquet y filtra aliases como “Both Teams Booked”. Con
   `--mode mean` se puede promediar las cuotas en lugar de elegir la mejor.

3. **Entrenar el modelo logístico**
   ```bash
   oraclebet cards-bt train \
     --data-path cache/historicos_cards_bt.parquet \
     --model-out models/cards_bt.joblib
   ```
   El modelo usa `scikit-learn` (pipeline con `OneHotEncoder` +
   `LogisticRegression`) y guarda métricas básicas en el archivo `.joblib`.

4. **Predecir próximos fixtures con EV mínimo**
   ```bash
   oraclebet cards-bt predict \
     --fixtures-parquet cache/fixtures_next.parquet \
     --rollings-parquet cache/rollings.parquet \
     --refs-parquet cache/referees.parquet \
     --league-parquet cache/leagues.parquet \
     --odds-csv cache/odds_cards_bt.csv \
     --ev-min 0.05
   ```
   Si faltan archivos, la CLI genera datasets demo para ilustrar el flujo.

### Notas de integración con proveedores existentes

- `agg-cards` busca automáticamente `cards_bt_source.(csv|parquet|json)` en
  `cache/` y reutiliza los cachés de `.betcomb_cache` producidos por
  `hooks_api`. Así se puede enchufar fácilmente el proveedor real
  (`providers/apifootball.py`) exportando históricos al formato esperado.
- El pipeline reconoce la ausencia de árbitros confirmados y rellena con el
  baseline liga/temporada (`ref_missing=1`).
- Los comandos aceptan `--source` (aggregator) y `--mode`/`--decimal`
  (parser) para adaptar distintos dumps según las necesidades de producción.
