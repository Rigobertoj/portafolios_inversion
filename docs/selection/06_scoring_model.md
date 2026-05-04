# Selection: Modelo De Scoring Fundamental

## Índice Del Documento

1. Propósito
2. Mapa De La Serie
3. Unidad Básica: `MetricSignalSpec`
4. Configuración: `FundamentalScoreConfig`
5. Tipos De Señal
6. Normalizadores
7. Métodos De Cambio
8. Fórmula Operativa Del Score
9. Catálogo Fundamental
10. Cómo Modificar Un Score

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

Este documento explica cómo el módulo convierte métricas fundamentales en un
score compuesto. Es el documento principal para cambiar pesos, agregar métricas
o adaptar el modelo a otra fuente de datos.

## Unidad Básica: `MetricSignalSpec`

`MetricSignalSpec` representa un componente del score.

| Atributo | Interpretación |
|---|---|
| `metric` | Columna de métrica financiera. |
| `signal` | Transformación: nivel, cambio reciente, cambio interanual o promedio histórico. |
| `weight` | Peso relativo en el score compuesto. |
| `higher_is_better` | Dirección de preferencia. |
| `normalizer` | Método para convertir valores a escala común. |
| `change_method` | Cómo calcular cambios históricos. |
| `target`, `tolerance` | Parámetros para scoring contra objetivo. |
| `category` | Categoría financiera para reporteo. |

Una señal no es solo una métrica. Es una métrica más una interpretación.

## Configuración: `FundamentalScoreConfig`

`FundamentalScoreConfig` agrupa componentes y reglas del score:

- `name`: nombre de la estrategia.
- `metric_weights`: pesos legacy cuando no hay señales granulares.
- `higher_is_better`: dirección legacy.
- `signal_specs`: componentes granulares.
- `score_column`: nombre de la columna final.
- `winsorize_quantiles`: límites para reducir outliers.
- `missing_component_score`: score asignado cuando un componente falta.

`ValueScoreConfig` y `GrowthScoreConfig` son configuraciones default.

## Tipos De Señal

| Señal | Significado |
|---|---|
| `level` | Valor más reciente de la métrica. |
| `period_change` | Cambio contra el periodo anterior. |
| `yoy_change` | Cambio contra el mismo periodo del año anterior. |
| `historical_avg_change` | Promedio de cambios históricos recientes. |

Ejemplo:

```text
revenue__period_change
```

No significa revenue total. Significa cambio reciente de revenue.

## Normalizadores

| Normalizador | Uso |
|---|---|
| `percentile_rank` | Convierte valores a percentiles cross-sectionales. |
| `robust_zscore` | Usa mediana e IQR para reducir sensibilidad a outliers. |
| `target` | Premia cercanía a un valor objetivo. |

El default favorece comparabilidad dentro del universo analizado. Por eso el
score es relativo: depende de qué compañías estén en la muestra.

## Métodos De Cambio

| Método | Uso |
|---|---|
| `relative` | `(actual - previo) / abs(previo)` |
| `difference` | `actual - previo` |
| `auto` | Usa diferencia para ratios y relativo para métricas de escala. |

La distinción importa. Un margen pasa de 20% a 22% con cambio de 2 puntos
porcentuales, no necesariamente con lectura de crecimiento relativo.

## Fórmula Operativa Del Score

El flujo de `score_fundamentals` es:

```text
para cada MetricSignalSpec:
  calcular señal
  normalizar señal
  aplicar dirección higher_is_better
  multiplicar por weight

fundamental_score = suma(component_score * weight) / suma(weight) * 100
score_coverage = peso_con_datos_disponibles / peso_total
```

El score final queda en escala 0-100.

## Catálogo Fundamental

`FUNDAMENTAL_METRIC_SIGNAL_SPECS` expone un catálogo amplio de métricas
agrupadas por categorías como:

- profitability
- valuation
- per_share
- asset_turnover_analysis
- dupont_analysis
- operating_efficiency
- operating_cycle_days
- liquidity
- coverage
- leverage

Este catálogo no descarga datos. Solo define cómo podrían puntuarse columnas
fundamentales que ya estén disponibles en un `DataFrame`.

## Cómo Modificar Un Score

Antes de modificar `fundamental_scorers.py`, decidir:

1. Qué tesis se quiere representar.
2. Qué métricas la expresan.
3. Qué dirección tiene cada métrica.
4. Si importa el nivel, el cambio o ambos.
5. Qué peso tiene cada componente.
6. Qué normalizador es apropiado.
7. Cómo se validará con tests.

Ejemplo conceptual:

```python
from src.selection import FundamentalScoreConfig, MetricSignalSpec

quality_config = FundamentalScoreConfig(
    name="quality",
    signal_specs=[
        MetricSignalSpec("roe", weight=0.30, higher_is_better=True, category="profitability"),
        MetricSignalSpec("debt_to_equity", weight=0.20, higher_is_better=False, category="leverage"),
        MetricSignalSpec("free_cash_flow_margin", weight=0.25, higher_is_better=True, category="cash_flow"),
        MetricSignalSpec("operating_margin", weight=0.25, higher_is_better=True, category="profitability"),
    ],
)
```

Después se puede pasar a `FundamentalSelector(score_config=quality_config)`.
