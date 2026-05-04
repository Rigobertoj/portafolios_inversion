# Flujos De Trabajo

## Índice Del Documento

1. Ranking Fundamental Tipo Value
2. Ranking Fundamental Tipo Growth
3. Flujo Completo En Una Línea Conceptual
4. Reporte De Selección
5. Ranking Temporal
6. Selección Por Correlación
7. Actualización De Portafolio
8. Conexión Con Optimization

## Mapa De La Serie Piloto

La ruta nueva empieza en [00_index.md](00_index.md). Esta guía conserva ejemplos
extendidos; la entrada secuencial nueva para workflows vive en
[07_workflows.md](07_workflows.md).

Este documento muestra los casos de uso más orgánicos del módulo `selection`.
El énfasis está en cómo usar la selección de activos dentro del flujo completo de
investigación y construcción de portafolios.

## 1. Ranking Fundamental Tipo Value

Use este flujo cuando quiera buscar compañías con múltiplos razonables, buena
rentabilidad y estructura financiera sólida.

```python
from src.selection import FundamentalSelector

tickers = ["AAPL", "MSFT", "NVDA", "KO", "PEP", "JNJ"]

selector = FundamentalSelector(strategy="value")
ranking = selector.rank(tickers)
selected = selector.select_top(ranking, top_k=3)
```

Salida esperada:

- `ranking`: universo completo ordenado por `fundamental_score`.
- `selected`: subconjunto superior.
- `selector.metrics_`: tabla de métricas antes del scoring.
- `selector.raw_data`: datos crudos de Yahoo Finance.

Uso posterior:

```python
selected_tickers = selected["ticker"].tolist()
```

Ese arreglo de tickers ya puede alimentar `optimization`. A partir de ahí el
flujo posterior corresponde a portfolio construction, execution/backtesting,
performance analysis y risk/reporting.

## 2. Ranking Fundamental Tipo Growth

Use este flujo cuando quiera buscar compañías con crecimiento alto, expansión de
utilidades y valuación razonable frente a su crecimiento.

```python
from src.selection import FundamentalSelector

tickers = ["NVDA", "MSFT", "AMZN", "GOOGL", "META", "ADBE"]

selector = FundamentalSelector(strategy="growth")
ranking = selector.rank(tickers)
selected = selector.select_top(top_k=4)
```

En este caso `select_top` puede omitir el argumento `ranking`, porque el selector
guarda el último resultado en `selector.ranking_`.

```python
selector.ranking_.head()
```

## 3. Flujo Completo En Una Línea Conceptual

Cuando el objetivo es ejecutar la selección sin inspeccionar cada paso:

```python
from src.selection import FundamentalSelector

selector = FundamentalSelector(strategy="value")
result = selector.run_pipeline(["AAPL", "MSFT", "NVDA", "KO"], top_k=2)

selected_tickers = result["selected_tickers"]
```

`result` contiene:

- `metrics`
- `ranking`
- `selected`
- `selected_tickers`
- `report`

Este flujo es útil para notebooks de investigación rápida o para preparar una
entrada directa a otro módulo.

## 4. Inspección De Métricas Antes Del Ranking

Antes de confiar en un score, conviene observar las métricas crudas.

```python
from src.selection import FundamentalSelector

selector = FundamentalSelector(strategy="value")
metrics = selector.collect_metrics(["AAPL", "MSFT", "NVDA", "KO"])

metrics[
    [
        "ticker",
        "trailing_pe",
        "price_to_book",
        "roe",
        "profit_margin",
        "debt_to_equity",
    ]
]
```

Este flujo es recomendable cuando:

- El universo contiene compañías de sectores muy distintos.
- Yahoo Finance puede tener campos faltantes.
- Se quiere validar si el score está comparando compañías razonablemente.

## 5. Análisis Temporal Trimestral

Use este flujo para revisar cómo evolucionan las métricas y el score durante los
últimos cuatro trimestres.

```python
from src.selection import FundamentalSelector

tickers = ["AAPL", "MSFT", "NVDA", "KO"]

selector = FundamentalSelector(strategy="growth")
score_history = selector.rank_over_time(
    tickers,
    frequency="quarterly",
    trailing_periods=4,
)

score_history[
    [
        "ticker",
        "period",
        "fundamental_score",
        "revenue_period_growth",
        "eps_period_growth",
        "profit_margin",
        "roe",
    ]
]
```

Para reporteo, puede transformarse una métrica específica a formato matriz:

```python
score_table = selector.metric_evolution(
    metric="fundamental_score",
    top_k=5,
)

eps_growth_table = selector.metric_evolution(
    metric="eps_period_growth",
    tickers=["AAPL", "MSFT", "NVDA"],
)
```

La salida tiene tickers en filas, periodos en columnas y una sola métrica como
valor. Este formato es más cómodo para comparar visualmente la evolución de los
activos seleccionados.

Este flujo responde preguntas como:

- ¿El score de una compañía viene mejorando o deteriorándose?
- ¿El crecimiento de ingresos se sostiene trimestre contra trimestre?
- ¿El crecimiento de EPS acompaña al crecimiento de ventas?
- ¿La mejora ocurre con márgenes sanos o a costa de rentabilidad?

## 6. Análisis Temporal Anual

Para una mirada menos ruidosa, puede usarse información anual.

```python
annual_history = selector.rank_over_time(
    tickers,
    frequency="annual",
    trailing_periods=4,
)
```

En frecuencia anual, los campos `*_yoy_growth` y `*_period_growth` son
equivalentes en la práctica, porque cada periodo ya representa un año.

## 7. Reporte Interpretativo De Selección

Después de rankear, conviene explicar por qué se seleccionaron los top activos.

```python
ranking = selector.rank(tickers)
report = selector.selection_report(ranking, top_k=3)
```

El reporte incluye:

```python
report["selected"]
report["metric_snapshot"]
report["score_weights"]
report["score_components"]
```

`score_weights` muestra qué métrica y qué señal participaron en el score,
cuánto pesó cada una, su normalizador y si valores altos o bajos son
preferibles. `score_components` muestra el componente normalizado de cada señal
para los activos seleccionados.

## 8. Personalización De Pesos

Si la estrategia estándar no encaja con el análisis, puede crearse una
configuración propia.

```python
from src.selection import FundamentalScoreConfig, FundamentalSelector, MetricSignalSpec

quality_value = FundamentalScoreConfig(
    name="quality_value",
    signal_specs=(
        MetricSignalSpec("trailing_pe", "level", 0.20, False, category="valuation"),
        MetricSignalSpec("price_to_book", "level", 0.15, False, category="valuation"),
        MetricSignalSpec("roe", "level", 0.20, True, change_method="difference", category="quality"),
        MetricSignalSpec("roe", "historical_avg_change", 0.15, True, change_method="difference", category="quality"),
        MetricSignalSpec("profit_margin", "period_change", 0.15, True, change_method="difference", category="quality"),
        MetricSignalSpec("debt_to_equity", "level", 0.15, False, change_method="difference", category="leverage"),
    ),
)

selector = FundamentalSelector(
    strategy="value",
    score_config=quality_value,
)

ranking = selector.rank(["AAPL", "MSFT", "NVDA", "KO"])
```

Este patrón permite experimentar sin modificar el código fuente del módulo.
Si se entrega una configuración antigua con `metric_weights` y
`higher_is_better`, el score sigue funcionando como percentiles por métrica de
nivel.

Para partir del catálogo completo de métricas fundamentales:

```python
from src.selection import FundamentalScoreConfig, fundamental_metric_signal_specs

fundamental_config = FundamentalScoreConfig(
    name="fundamental_all_signals",
    signal_specs=fundamental_metric_signal_specs(
        signals=("level", "period_change", "yoy_change", "historical_avg_change")
    ),
)
```

## 9. Selección Fundamental Más Diversificación Por Correlación

Un flujo más robusto puede combinar calidad fundamental y diversificación.

Primero se rankean compañías por fundamentos:

```python
from src.selection import FundamentalSelector

fundamental_selector = FundamentalSelector(strategy="value")
ranking = fundamental_selector.rank(["AAPL", "MSFT", "NVDA", "KO", "PEP", "JNJ"])
fundamental_candidates = fundamental_selector.select_top(ranking, top_k=5)
```

Después se puede usar selección por correlación para evitar candidatos demasiado
parecidos:

```python
from src.selection import CorrelationPortfolioSelector

correlation_selector = CorrelationPortfolioSelector(
    start_date="2020-01-01",
    use_absolute_corr=True,
)

result = correlation_selector.update_portfolio(
    current_tickers=[],
    candidate_tickers=fundamental_candidates["ticker"].tolist(),
    max_new_assets=3,
)

final_tickers = result["updated_tickers"]
```

Este flujo responde dos preguntas distintas:

1. ¿Qué compañías se ven atractivas por fundamentos?
2. ¿Cuáles agregan diversificación al portafolio?

## 10. Selección Por Grupos

Cuando el universo está organizado por sectores o industrias, el selector de
correlación puede trabajar por grupos.

```python
from src.selection import CorrelationPortfolioSelector

grouped_tickers = {
    "technology": ["AAPL", "MSFT", "NVDA", "ADBE"],
    "consumer": ["KO", "PEP", "PG", "COST"],
    "healthcare": ["JNJ", "PFE", "MRK", "ABBV"],
}

selector = CorrelationPortfolioSelector(start_date="2020-01-01")
result = selector.run_pipeline(
    grouped_tickers=grouped_tickers,
    top_k_in_group=2,
    final_size=5,
)

result["final_tickers"]
```

Este flujo es útil cuando se quiere evitar que un solo sector domine la selección
inicial.

## 11. Flujo Recomendado Para Investigación

Para análisis en notebooks, el flujo más saludable suele ser:

```text
1. Definir universo inicial de tickers.
2. Descargar y revisar métricas fundamentales.
3. Rankear con value o growth.
4. Seleccionar top_k.
5. Revisar evolución trimestral o anual si la estrategia depende de crecimiento.
6. Generar reporte interpretativo de selección.
7. Revisar correlación entre candidatos.
8. Pasar tickers finales a optimization.
9. Continuar hacia portfolio construction, execution/backtesting, performance
   analysis y risk/reporting.
```

Ejemplo:

```python
from src.selection import FundamentalSelector, CorrelationPortfolioSelector

tickers = ["AAPL", "MSFT", "NVDA", "KO", "PEP", "JNJ", "PG", "COST"]

fundamental = FundamentalSelector(strategy="value")
ranking = fundamental.rank(tickers)
top_candidates = fundamental.select_top(ranking, top_k=6)
report = fundamental.selection_report(ranking, top_k=6)

correlation = CorrelationPortfolioSelector(start_date="2020-01-01")
diversified = correlation.update_portfolio(
    current_tickers=[],
    candidate_tickers=top_candidates["ticker"].tolist(),
    max_new_assets=4,
)

selected_tickers = diversified["updated_tickers"]
```

## Buenas Prácticas

### Revisar Datos Faltantes

Yahoo Finance puede no devolver todos los campos para todas las compañías. Antes
de interpretar el ranking, revise columnas clave:

```python
ranking[["ticker", "trailing_pe", "price_to_book", "roe", "fundamental_score"]]
```

### Comparar Compañías Similares

Los scores son cross-sectionales. Son más expresivos cuando las compañías son
comparables por sector, industria, tamaño o modelo de negocio.

### No Confundir Selección Con Asignación

`selection` elige candidatos. La decisión de pesos corresponde a
`src.optimization`; la construcción, ejecución, evaluación y reporte pertenecen
a etapas posteriores del flujo.

### Usar Los Diagramas Como Mapa Del Flujo

Los diagramas de [diagrams](diagrams/README.md) son el mapa visual de esta
documentación. Si una decisión de workflow parece ambigua, revise primero:

1. `selection_module_architecture.drawio` para ubicar la ruta dentro del flujo
   completo.
2. `selection_class_architecture.drawio` para ubicar clases, funciones y datos.
3. `selection_workflow.drawio` para ubicar gates, handoff y feedback.

### Guardar El Universo Inicial

El ranking depende del universo de entrada. Para reproducibilidad, documente qué
tickers se usaron y en qué fecha se ejecutó el análisis.

### Usar El Score Como Punto De Partida

`fundamental_score` ayuda a ordenar candidatos, pero no sustituye interpretación.
Conviene revisar las métricas que explican el resultado antes de tomar una
decisión de inversión.
