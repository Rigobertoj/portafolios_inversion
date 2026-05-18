# Selection: Flujos De Trabajo

## Índice Del Documento

1. Propósito
2. Mapa De La Serie
3. Flujo Fundamental Value
4. Flujo Fundamental Growth
5. Flujo De Scoring Temporal
6. Flujo De Correlación Por Grupos
7. Flujo De Actualización De Portafolio
8. Flujo Fundamental Aprendido
9. Relación Con La Guía Heredada

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
| 11 | `11_learned_fundamental_scoring.md` | Scoring aprendido | Para research con XGBoost y pesos aprendidos. |

## Propósito

Este documento muestra cómo usar `selection` en flujos reales. La guía heredada
con ejemplos más extensos sigue disponible en `workflows.md`.

## Flujo Fundamental Value

```python
from src.selection import FundamentalSelector

tickers = ["AAPL", "MSFT", "NVDA", "KO"]

selector = FundamentalSelector(strategy="value")
ranking = selector.rank(tickers, frequency="quarterly", trailing_periods=8)
selected = selector.select_top(ranking, top_k=3)
report = selector.selection_report(ranking, top_k=3)

selected_tickers = selected["ticker"].tolist()
```

Lectura:

- `ranking` ordena el universo bajo tesis value.
- `selected_tickers` puede alimentar `optimization`.
- `report` explica por qué esos tickers quedaron arriba.

## Flujo Fundamental Growth

```python
from src.selection import FundamentalSelector

selector = FundamentalSelector(strategy="growth")
ranking = selector.rank(["AAPL", "MSFT", "NVDA", "AMZN"])
selected = selector.select_top(ranking, top_k=2)
```

Growth enfatiza expansión de métricas como revenue, EPS, net income y free cash
flow, sin ignorar calidad, leverage y disciplina de valuación.

## Flujo De Scoring Temporal

```python
history = selector.rank_over_time(
    ["AAPL", "MSFT", "NVDA", "KO"],
    frequency="quarterly",
    trailing_periods=6,
)

evolution = selector.metric_evolution(
    metric="fundamental_score",
    top_k=3,
    history=history,
)
```

Este flujo sirve para revisar si una selección fue estable o si el último score
depende de un cambio reciente.

## Flujo De Correlación Por Grupos

```python
from src.selection import CorrelationPortfolioSelector

selector = CorrelationPortfolioSelector(
    start_date="2020-01-01",
    end_date="2025-01-01",
    min_coverage=0.80,
)

result = selector.run_pipeline(
    grouped_tickers={
        "technology": ["AAPL", "MSFT", "NVDA"],
        "defensive": ["KO", "PG", "WMT"],
    },
    top_k_in_group=1,
    final_size=2,
)

result["final_tickers"]
```

Este flujo es útil cuando se quiere una lista candidata que reduzca redundancia
estadística entre grupos.

## Flujo De Actualización De Portafolio

```python
selector = CorrelationPortfolioSelector(start_date="2020-01-01")

update = selector.update_portfolio(
    current_tickers=["AAPL", "MSFT"],
    candidate_tickers=["NVDA", "KO", "PG"],
    max_new_assets=1,
)

update["updated_tickers"]
```

Este flujo sugiere nuevos tickers con menor correlación promedio frente al
portafolio actual.

## Flujo Fundamental Aprendido

```python
from src.selection import (
    FundamentalLearningPanelBuilder,
    LearnedFundamentalSelector,
    LearnedScoreConfigFactory,
    XGBoostFundamentalModel,
)

panel = FundamentalLearningPanelBuilder(
    frequency="quarterly",
    trailing_periods=20,
    horizon_months=12,
    reporting_lag_days=60,
).build(["AAPL", "MSFT", "NVDA", "KO"])

model = XGBoostFundamentalModel(group_by="sector")
model.fit(panel, target="forward_total_return_12m")

configs = LearnedScoreConfigFactory(max_features=15).from_impact_report(
    model.impact_report_
)

selector = LearnedFundamentalSelector(configs, group_by="sector")
ranking = selector.rank(["AAPL", "MSFT", "NVDA", "KO"])
selected = selector.select_top(ranking, top_k=3)
```

Este flujo aprende pesos con datos históricos, pero el ranking final sigue
pasando por `score_fundamentals`.

## Relación Con La Guía Heredada

La guía anterior `workflows.md` conserva ejemplos y notas del flujo original.
Este documento funciona como ruta secuencial nueva. Cuando haya duplicación, la
serie numerada debe considerarse la entrada recomendada.
