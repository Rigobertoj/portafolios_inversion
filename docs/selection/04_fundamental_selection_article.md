# Selection: Artículo De Selección Fundamental

## Índice Del Documento

1. Propósito
2. Mapa De La Serie
3. Problema Que Resuelve
4. Fundamento Conceptual
5. Fuentes Académicas O Profesionales
6. Traducción Al Código
7. Value Y Growth En La Implementación
8. Interpretación De `score_coverage`
9. Ejemplo Interpretado
10. Supuestos Y Límites
11. Errores Comunes
12. Referencias

## Mapa De La Serie

| Orden | Documento | Rol | Cuándo Leerlo |
|---:|---|---|---|
| 00 | `00_index.md` | Entrada secuencial | Siempre primero. |
| 01 | `01_conceptual_model.md` | Modelo conceptual | Antes de usar o modificar `selection`. |
| 02 | `02_architecture.md` | Arquitectura del módulo | Antes de tocar código. |
| 03 | `03_data_contracts.md` | Contratos de datos | Cuando se integran entradas o salidas. |
| 04 | `04_fundamental_selection_article.md` | Artículo fundamental | Para entender value, growth y scoring. |
| 05 | `05_correlation_selection_article.md` | Artículo correlación | Para entender diversificación estadística. |
| 06 | `06_scoring_model.md` | Modelo de score | Para ajustar métricas, señales o pesos. |
| 07 | `07_workflows.md` | Flujos prácticos | Para ejecutar casos de uso. |
| 08 | `08_api_reference.md` | API narrativa | Para consultar clases, métodos y atributos. |
| 09 | `09_validation_and_edge_cases.md` | Validación | Para pruebas, errores y límites. |
| 10 | `10_glossary.md` | Glosario | Para unificar vocabulario. |

## Propósito

Este artículo explica la ruta fundamental de `selection`. La meta no es listar
métodos, sino explicar cómo una tesis de inversión se convierte en un ranking
auditable de compañías.

## Problema Que Resuelve

La selección fundamental intenta responder:

```text
¿Qué compañías del universo parecen más atractivas bajo una tesis financiera?
```

La implementación actual no estima valor intrínseco completo. Construye un
ranking cuantitativo con métricas fundamentales, señales históricas, direcciones
de preferencia, pesos y normalización cross-sectional.

## Fundamento Conceptual

La selección fundamental parte de la idea de que el precio de mercado puede
compararse con fundamentos económicos de una compañía. CFA Institute describe
la valuación de equity como un proceso basado en fundamentos, modelos y juicio
del analista. También distingue enfoques activos fundamentales y cuantitativos:
el primero enfatiza juicio e investigación profunda; el segundo enfatiza reglas,
datos y amplitud.

`selection` se ubica más cerca de un enfoque cuantitativo basado en fundamentos:
usa métricas, reglas y rankings para hacer explícita una tesis. No reemplaza el
juicio del analista; lo convierte en una configuración revisable.

## Fuentes Académicas O Profesionales

La documentación conceptual de esta ruta se apoya en:

- CFA Institute, *Equity Valuation: Concepts and Basic Tools*. Útil para
  distinguir valor intrínseco, fundamentos, múltiplos y juicio de valuación.
- CFA Institute, *Active Equity Investing: Strategies*. Útil para distinguir
  estrategias fundamentales, cuantitativas, value y growth.
- CFA Institute, *Market-Based Valuation: Price and Enterprise Value Multiples*.
  Útil para interpretar múltiplos como P/E, P/B, EV/EBITDA o P/FCF.

## Traducción Al Código

La ruta fundamental se implementa así:

```text
FundamentalSelector.rank(tickers)
  -> collect_metrics(tickers)
  -> YahooFundamentalsProvider.fetch_many(...)
  -> build_metrics_frame(...)
  -> build_metric_history_frame(...) si hay señales históricas
  -> score_fundamentals(...)
  -> ranking_ con fundamental_score y score_coverage
```

Los conceptos se traducen a código de esta forma:

| Concepto | Código |
|---|---|
| Universo de compañías | `tickers` |
| Datos fundamentales | `YahooFundamentalsProvider`, `FundamentalData` |
| Métricas comparables | `build_fundamental_metrics`, `build_metrics_frame` |
| Tesis value/growth | `ValueScoreConfig`, `GrowthScoreConfig` |
| Señal individual | `MetricSignalSpec` |
| Score compuesto | `score_fundamentals` |
| Interpretabilidad | `selection_report` |

## Value Y Growth En La Implementación

### Value

`ValueScoreConfig` pondera valuación, calidad, cash flow, rentabilidad,
leverage, liquidez y cambios históricos. En términos prácticos:

- Múltiplos más bajos suelen puntuar mejor cuando `higher_is_better=False`.
- Rentabilidad y cash flow suelen puntuar mejor cuando son más altos.
- Leverage excesivo suele penalizarse.
- Cambios recientes e históricos agregan una lectura dinámica.

Value aquí no significa "comprar barato" de forma aislada. Significa combinar
valuación con calidad financiera y señales de estabilidad.

### Growth

`GrowthScoreConfig` enfatiza crecimiento de revenue, EPS, net income y free cash
flow. También incluye rentabilidad, márgenes, ROIC, PEG, P/E, deuda e interés.

Growth aquí no significa ignorar precio o riesgo financiero. El score favorece
expansión, pero conserva disciplina de valuación y balance.

## Interpretación De `score_coverage`

`score_coverage` mide qué proporción del peso total de la configuración tuvo
datos disponibles para una compañía. No es una métrica de calidad financiera.
Es una métrica de completitud del score.

Ejemplo:

```text
fundamental_score = 86
score_coverage = 0.45
```

Esto significa que la compañía rankeó bien, pero solo con 45% del peso
observable. La conclusión no debería ser "compañía excelente", sino "candidata
interesante con evidencia incompleta".

## Ejemplo Interpretado

```python
from src.selection import FundamentalSelector

selector = FundamentalSelector(strategy="value")
ranking = selector.rank(["AAPL", "MSFT", "NVDA", "KO"])
selected = selector.select_top(ranking, top_k=3)
report = selector.selection_report(ranking, top_k=3)
```

Lectura del resultado:

- `ranking` ordena compañías por `fundamental_score`.
- `selected` conserva las mejores filas.
- `report["score_weights"]` explica qué métricas participaron.
- `report["score_components"]` permite ver qué señales empujaron el score.
- `score_coverage` advierte cuánta información respaldó la calificación.

## Supuestos Y Límites

- Yahoo Finance puede devolver campos faltantes o inconsistentes.
- El score es relativo al universo analizado, no absoluto.
- Los percentiles dependen de la muestra.
- Una métrica puede tener dirección distinta según la tesis.
- La configuración default no es una recomendación profesional.
- Un proveedor institucional requiere credenciales, licencia y una capa de
  ingesta compatible si se quiere usar como fuente real.

## Errores Comunes

- Interpretar `fundamental_score` como valor intrínseco.
- Comparar scores calculados sobre universos distintos sin advertencia.
- Ignorar `score_coverage`.
- Usar señales históricas sin suficiente historial.
- Tratar `FUNDAMENTAL_METRIC_SIGNAL_SPECS` como downloader de datos.

## Referencias

- CFA Institute, *Equity Valuation: Concepts and Basic Tools*. Consultado:
  2026-05-03.
  https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/equity-valuation-concepts-basic-tools
- CFA Institute, *Active Equity Investing: Strategies*. Consultado:
  2026-05-03.
  https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/active-equity-investing-strategies
- CFA Institute, *Market-Based Valuation: Price and Enterprise Value Multiples*.
  Consultado: 2026-05-03.
  https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/market-based-valuation-price-enterprise-value-multiples
