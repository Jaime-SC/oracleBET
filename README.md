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
