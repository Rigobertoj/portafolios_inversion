# Selection: Contratos De Datos

## Índice Del Documento

1. Propósito
2. Mapa De La Serie
3. Contratos De Entrada
4. Contratos Intermedios
5. Contratos De Salida
6. Estados Auditables
7. Integración Con El Catálogo Fundamental
8. Riesgos De Contrato

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

Este documento define las formas de datos que entran, se transforman y salen de
`selection`. Es el documento que se debe leer antes de conectar una nueva fuente
de datos o antes de consumir salidas en `optimization`.

## Contratos De Entrada

### Tickers

La entrada más común es un iterable de strings:

```python
["AAPL", "MSFT", "NVDA", "KO"]
```

`FundamentalSelector` normaliza tickers con mayúsculas, elimina espacios y
deduplica conservando orden.

### Grupos De Tickers

La ruta de correlación espera un diccionario:

```python
{
    "tech": ["AAPL", "MSFT", "NVDA"],
    "defensive": ["KO", "PG", "WMT"],
}
```

Cada grupo debe tener al menos dos tickers útiles después de descarga y filtro
de cobertura.

### Fechas Y Campo De Precio

`CorrelationPortfolioSelector` recibe:

- `start_date`
- `end_date`
- `price_field`
- `min_coverage`
- `use_absolute_corr`

La combinación define el universo estadístico sobre el que se calculan retornos.

## Contratos Intermedios

| Objeto | Tipo | Descripción |
|---|---|---|
| `FundamentalData` | dataclass | Registro fundamental por ticker. |
| `metrics_` | `pandas.DataFrame` | Métricas comparables por compañía. |
| `metric_history_` | `pandas.DataFrame` | Métricas por ticker y periodo. |
| `ranking_` | `pandas.DataFrame` | Ranking fundamental actual. |
| `ranking_history_` | `pandas.DataFrame` | Ranking fundamental temporal. |
| `prices_by_group` | `dict[str, DataFrame]` | Precios usados por grupo en correlación. |
| `returns_by_group` | `dict[str, DataFrame]` | Retornos usados por grupo. |
| `ranking_by_group` | `pandas.DataFrame` | Ranking de baja correlación por grupo. |
| `learning_panel` | `pandas.DataFrame` | Panel histórico con señales y targets forward. |
| `impact_report_` | `pandas.DataFrame` | Impactos aprendidos por modelo, grupo y señal. |

## Contratos De Salida

### Ranking Fundamental

`FundamentalSelector.rank(...)` devuelve un `DataFrame` ordenado por
`fundamental_score`.

Columnas esperadas:

- `ticker`
- métricas financieras disponibles
- señales calculadas, como `revenue__period_change`
- componentes, como `revenue__period_change_score`
- `fundamental_score`
- `score_coverage`
- `strategy`

### Panel Fundamental Aprendido

`FundamentalLearningPanelBuilder.build(...)` devuelve una tabla larga con:

- identificadores: `ticker`, `period`, `sector`, `industry`
- fechas de target: `available_at`, `target_end_at`
- señales: columnas como `roe__level` o `revenue__yoy_change`
- precios de target: `entry_price`, `exit_price`
- targets: `forward_price_return_12m`, `forward_dividend_return_12m`,
  `forward_total_return_12m`

El sufijo del target cambia según `horizon_months`.

### Selección Top-K

`select_top(...)` devuelve las primeras filas del ranking y reinicia el índice.
Este objeto es una lista corta auditable, no una cartera optimizada.

### Reporte De Selección

`selection_report(...)` devuelve un diccionario:

| Llave | Significado |
|---|---|
| `selected` | Resumen de compañías seleccionadas. |
| `metric_snapshot` | Métricas usadas por los seleccionados. |
| `score_weights` | Configuración de pesos y direcciones. |
| `score_components` | Componentes individuales de score. |

### Salida De Correlación

`CorrelationPortfolioSelector.run_pipeline(...)` devuelve:

| Llave | Significado |
|---|---|
| `group_ranking` | Ranking por grupo. |
| `per_group_selection` | Primeros candidatos por grupo. |
| `candidate_tickers` | Universo candidato consolidado. |
| `candidate_corr_matrix` | Matriz de correlación de candidatos. |
| `candidate_scores` | Scores de correlación promedio. |
| `final_tickers` | Lista final de tickers. |
| `final_corr_matrix` | Matriz de los tickers finales. |
| `final_mean_offdiag_corr` | Correlación promedio fuera de diagonal. |

## Estados Auditables

Los atributos terminados en `_` siguen una convención común en Python: son
resultados aprendidos o calculados después de ejecutar métodos.

En `FundamentalSelector`:

- `metrics_` existe después de `collect_metrics` o `rank`.
- `metric_history_` existe después de `collect_metric_history`,
  `rank_over_time` o `rank` cuando hay señales históricas.
- `ranking_` existe después de `rank`.
- `ranking_history_` existe después de `rank_over_time`.

En `XGBoostFundamentalModel`:

- `impact_report_` existe después de `fit`.
- `training_summary_` resume observaciones, features y rank IC por grupo.
- `models_` contiene modelos ajustados por grupo cuando la dependencia
  opcional `xgboost` o un estimador inyectado está disponible.

Estos estados no deben tratarse como parámetros de entrada. Son evidencia
auditable del último flujo ejecutado.

## Integración Con El Catálogo Fundamental

`FUNDAMENTAL_METRIC_SIGNAL_SPECS` y `fundamental_metric_signal_specs(...)` no
descargan datos. Funcionan como catálogo de componentes para métricas
fundamentales ya disponibles en un `DataFrame`.

La integración real con un proveedor institucional requiere credenciales,
licencia y una capa de ingesta que produzca columnas con los nombres esperados
por `MetricSignalSpec`.

## Riesgos De Contrato

- Un ticker puede no devolver información suficiente.
- Dos proveedores pueden nombrar métricas de forma distinta.
- Las señales históricas necesitan `metric_history_`.
- `score_coverage` bajo implica que el score se calculó con información
  parcial.
- En correlación, una cobertura de precios insuficiente elimina activos antes
  del ranking.
