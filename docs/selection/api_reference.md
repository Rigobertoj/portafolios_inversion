# Referencia API

## Índice Del Documento

1. `FundamentalSelector`
2. Datos Fundamentales
3. Métricas Fundamentales
4. Scoring Fundamental
5. Selección Por Correlación
6. Alias Públicos

## Mapa De La Serie Piloto

La ruta nueva empieza en [00_index.md](00_index.md). Esta referencia mantiene el
detalle de firmas, parámetros, atributos y retornos; la lectura narrativa de API
vive en [08_api_reference.md](08_api_reference.md).

Esta referencia describe las clases y funciones públicas de `src.selection`.
La estructura sigue una forma similar a NumPy y pandas: descripción, parámetros,
atributos, métodos, retornos, notas y ejemplos.

## `FundamentalSelector`

```python
class FundamentalSelector(strategy="value", provider=None, score_config=None)
```

Selector de alto nivel para rankear compañías con criterios fundamentales.

### Parámetros

`strategy` : `str`, default `"value"`
Estrategia fundamental a utilizar. Valores soportados:

- `"value"`
- `"growth"`

`provider` : `YahooFundamentalsProvider`, optional
Proveedor de datos. Si no se entrega, se crea un `YahooFundamentalsProvider` por
defecto.

`score_config` : `FundamentalScoreConfig`, optional
Configuración de scoring. Si se entrega, reemplaza la configuración asociada a
`strategy`.

### Atributos

`strategy` : `str`
Nombre normalizado de la estrategia.

`provider` : `YahooFundamentalsProvider`
Proveedor usado para descargar o recuperar datos fundamentales.

`score_config` : `FundamentalScoreConfig`
Configuración de pesos y dirección de cada métrica.

`raw_data` : `dict[str, FundamentalData]`
Datos fundamentales crudos descargados para cada ticker.

`metrics_` : `pandas.DataFrame`
Métricas calculadas después de llamar `collect_metrics` o `rank`.

`metric_history_` : `pandas.DataFrame`
Métricas por periodo calculadas después de llamar `collect_metric_history` o
`rank_over_time`.

`ranking_` : `pandas.DataFrame`
Ranking calculado después de llamar `rank`.

`ranking_history_` : `pandas.DataFrame`
Ranking por periodo calculado después de llamar `rank_over_time`.

### Métodos

#### `set_strategy`

```python
set_strategy(strategy, score_config=None, clear_rankings=True)
```

Actualiza la estrategia y la configuración de pesos usada para futuros rankings.

Parámetros:

`strategy` : `str`
Nombre de la estrategia. Si `score_config` no se entrega, debe ser una estrategia
soportada por defecto, como `"value"` o `"growth"`.

`score_config` : `FundamentalScoreConfig`, optional
Configuración custom de pesos y dirección de métricas.

`clear_rankings` : `bool`, default `True`
Si es `True`, elimina `ranking_` y `ranking_history_` porque fueron calculados
con la estrategia anterior. Conserva `raw_data`, `metrics_` y
`metric_history_`, ya que esos datos base pueden reutilizarse.

#### `collect_metrics`

```python
collect_metrics(tickers)
```

Descarga datos y calcula métricas fundamentales para el universo de tickers.

Parámetros:

`tickers` : iterable of `str`
Universo de compañías.

Retorna:

`pandas.DataFrame`
Tabla indexada por ticker con métricas fundamentales.

#### `collect_metric_history`

```python
collect_metric_history(tickers, frequency="quarterly", trailing_periods=4)
```

Descarga datos y calcula métricas fundamentales por periodo.

Parámetros:

`tickers` : iterable of `str`
Universo de compañías.

`frequency` : {"annual", "quarterly"}, default `"quarterly"`
Frecuencia de estados financieros.

`trailing_periods` : `int`, default `4`
Número de periodos recientes a conservar.

Retorna:

`pandas.DataFrame`
Tabla con una fila por ticker y periodo.

#### `rank`

```python
rank(tickers, frequency="quarterly", trailing_periods=8)
```

Calcula métricas y devuelve el ranking ordenado por `fundamental_score`.
Si la configuración usa señales históricas, también construye
`metric_history_` para calcular cambios recientes, interanuales e históricos.

Parámetros:

`tickers` : iterable of `str`
Universo de compañías.

`frequency` : {"annual", "quarterly"}, default `"quarterly"`
Frecuencia usada para señales históricas.

`trailing_periods` : `int`, default `8`
Número máximo de periodos recientes usados para señales históricas.

Retorna:

`pandas.DataFrame`
Ranking de compañías con métricas, componentes de score, `fundamental_score` y
`strategy`.

#### `rank_over_time`

```python
rank_over_time(tickers, frequency="quarterly", trailing_periods=4)
```

Calcula el score fundamental por periodo. Cada periodo se puntúa de forma
transversal, comparando las compañías disponibles dentro de ese mismo trimestre o
año.

Parámetros:

`tickers` : iterable of `str`
Universo de compañías.

`frequency` : {"annual", "quarterly"}, default `"quarterly"`
Frecuencia de estados financieros.

`trailing_periods` : `int`, default `4`
Número de periodos recientes a evaluar.

Retorna:

`pandas.DataFrame`
Historial de scores y métricas por ticker y periodo.

#### `select_top`

```python
select_top(ranking=None, top_k=5)
```

Selecciona las primeras `top_k` compañías de un ranking.

Parámetros:

`ranking` : `pandas.DataFrame`, optional
Ranking producido por `rank`. Si no se entrega, usa `self.ranking_`.

`top_k` : `int`, default `5`
Número de compañías a seleccionar.

Retorna:

`pandas.DataFrame`
Subconjunto superior del ranking.

#### `selection_report`

```python
selection_report(ranking=None, top_k=5)
```

Construye un reporte interpretativo para los top seleccionados.

Parámetros:

`ranking` : `pandas.DataFrame`, optional
Ranking producido por `rank`. Si no se entrega, usa `self.ranking_`.

`top_k` : `int`, default `5`
Número de compañías a incluir en el reporte.

Retorna:

`dict[str, pandas.DataFrame]`
Diccionario con:

- `selected`: score resumido de las compañías seleccionadas.
- `metric_snapshot`: métricas usadas por el score.
- `score_weights`: pesos y dirección de las métricas.
- `score_components`: percentiles por métrica.

#### `metric_evolution`

```python
metric_evolution(metric="fundamental_score", tickers=None, top_k=None, history=None)
```

Convierte un historial largo en una matriz de reporteo para una sola métrica.

Parámetros:

`metric` : `str`, default `"fundamental_score"`
Métrica a visualizar en el tiempo.

`tickers` : iterable of `str`, optional
Subconjunto explícito de tickers.

`top_k` : `int`, optional
Selecciona los mejores tickers del último periodo antes de construir la matriz.

`history` : `pandas.DataFrame`, optional
Historial custom en formato largo. Si no se entrega, usa `ranking_history_` y,
si no existe, `metric_history_`.

Retorna:

`pandas.DataFrame`
Matriz con tickers en filas, periodos en columnas y la métrica solicitada como
valor.

#### `run_pipeline`

```python
run_pipeline(tickers, top_k=5)
```

Ejecuta el flujo completo: métricas, ranking y selección final.

Parámetros:

`tickers` : iterable of `str`
Universo de compañías.

`top_k` : `int`, default `5`
Número de compañías a seleccionar.

Retorna:

`dict`
Diccionario con:

- `metrics`
- `ranking`
- `selected`
- `selected_tickers`
- `report`

### Ejemplo

```python
from src.selection import FundamentalSelector

selector = FundamentalSelector(strategy="value")
ranking = selector.rank(["AAPL", "MSFT", "NVDA", "KO"])
selected = selector.select_top(ranking, top_k=2)
```

## `YahooFundamentalsProvider`

```python
class YahooFundamentalsProvider(
    start="2020-01-01",
    end=None,
    price_field="Close",
    ticker_factory=None,
)
```

Proveedor de datos fundamentales desde Yahoo Finance.

### Parámetros

`start` : `str`, default `"2020-01-01"`
Fecha inicial para descargar precios históricos.

`end` : `str`, optional
Fecha final para descargar precios históricos.

`price_field` : `str`, default `"Close"`
Campo de precio usado desde el historial de Yahoo Finance.

`ticker_factory` : callable, optional
Fábrica para crear objetos tipo `yf.Ticker`. Se usa principalmente para pruebas o
para inyectar un proveedor compatible.

### Métodos

#### `fetch`

```python
fetch(ticker)
```

Descarga datos de una compañía y devuelve un `FundamentalData`.

Parámetros:

`ticker` : `str`
Símbolo bursátil.

Retorna:

`FundamentalData`

#### `fetch_many`

```python
fetch_many(tickers)
```

Descarga datos para un grupo de tickers.

Parámetros:

`tickers` : iterable of `str`

Retorna:

`dict[str, FundamentalData]`

#### `clear_cache`

```python
clear_cache()
```

Limpia la caché de objetos `yf.Ticker` y registros `FundamentalData`
descargados. Es útil cuando se quiere forzar una nueva consulta para el mismo
universo, rango de fechas y campo de precio.

Retorna:

`None`

## `FundamentalData`

```python
@dataclass(frozen=True)
class FundamentalData
```

Contenedor inmutable con los datos crudos de una compañía.

### Atributos

`ticker` : `str`
Ticker normalizado.

`info` : `dict`
Diccionario de metadatos y ratios provistos por Yahoo Finance.

`income_statement` : `pandas.DataFrame`
Estado de resultados.

`balance_sheet` : `pandas.DataFrame`
Balance general.

`cash_flow` : `pandas.DataFrame`
Estado de flujo de efectivo.

`prices` : `pandas.Series`
Serie histórica de precios.

`quarterly_income_statement` : `pandas.DataFrame`
Estado de resultados trimestral.

`quarterly_balance_sheet` : `pandas.DataFrame`
Balance general trimestral.

`quarterly_cash_flow` : `pandas.DataFrame`
Estado de flujo de efectivo trimestral.

## `FundamentalScoreConfig`

```python
@dataclass(frozen=True)
class FundamentalScoreConfig
```

Configuración base para puntuar compañías.

### Atributos

`name` : `str`
Nombre de la estrategia.

`metric_weights` : mapping
Pesos asignados a cada métrica.

`higher_is_better` : mapping
Indica si valores altos son preferibles para cada métrica.

`signal_specs` : iterable of `MetricSignalSpec`
Componentes granulares del score. Cuando se entrega, reemplaza el score plano
por métrica y permite usar `level`, `period_change`, `yoy_change` e
`historical_avg_change`.

`score_column` : `str`, default `"fundamental_score"`
Nombre de la columna final de score.

`winsorize_quantiles` : tuple, default `(0.05, 0.95)`
Límites de winsorización transversal antes de normalizar.

`missing_component_score` : `float`, default `0.0`
Score asignado a un componente disponible en el universo pero faltante para una
compañía específica.

## `MetricSignalSpec`

```python
@dataclass(frozen=True)
class MetricSignalSpec
```

Define un componente individual de scoring.

### Atributos

`metric` : `str`
Métrica fuente.

`signal` : {"level", "period_change", "yoy_change", "historical_avg_change"}
Transformación temporal usada antes de normalizar.

`weight` : `float`
Peso relativo del componente.

`higher_is_better` : `bool`
Dirección del ranking.

`normalizer` : {"percentile_rank", "robust_zscore", "target"}
Método para llevar la señal a escala comparable.

`change_method` : {"relative", "difference", "auto"}
Método para calcular cambios. En márgenes y ratios de rentabilidad suele usarse
`difference`; en ventas, EPS y cash flow suele usarse `relative`.

## `FUNDAMENTAL_METRIC_SIGNAL_SPECS`

Catálogo de 91 componentes `MetricSignalSpec` en señal `level`, alineado con
las métricas fundamentales disponibles como referencia de nomenclatura.

## `fundamental_metric_signal_specs`

```python
fundamental_metric_signal_specs(signals=("level",))
```

Devuelve el catálogo de métricas fundamentales como componentes de score. Puede
expandirse a las familias temporales:

```python
specs = fundamental_metric_signal_specs(
    signals=("level", "period_change", "yoy_change", "historical_avg_change")
)
```

## `ValueScoreConfig`

```python
class ValueScoreConfig(FundamentalScoreConfig)
```

Configuración por defecto para selección tipo `value`.

### Señales Incluidas

Incluye señales de valuación (`trailing_pe`, `price_to_book`,
`enterprise_value_to_ebitda`, `price_to_free_cash_flow`), calidad y cash flow
(`roe`, `return_on_invested_capital`, `profit_margin`,
`free_cash_flow_margin`), leverage/liquidez (`debt_to_equity`,
`total_debt_to_ebitda`, `current_ratio`) y cambios recientes/históricos de
ventas, EPS, flujo libre y márgenes.

### Interpretación

La configuración favorece compañías con múltiplos bajos, deuda razonable,
rentabilidad positiva y generación de flujo de efectivo.

## `GrowthScoreConfig`

```python
class GrowthScoreConfig(FundamentalScoreConfig)
```

Configuración por defecto para selección tipo `growth`.

### Señales Incluidas

Incluye crecimiento reciente, interanual e histórico de ingresos, EPS, utilidad
neta y flujo libre, además de calidad (`gross_margin`, `operating_margin`,
`roe`, `return_on_invested_capital`), disciplina de valuación (`peg_ratio`,
`trailing_pe`) y salud financiera (`debt_to_equity`, `interest_coverage`).

### Interpretación

La configuración favorece compañías con crecimiento fuerte, márgenes sanos,
rentabilidad y valuación razonable frente al crecimiento.

## `build_fundamental_metrics`

```python
build_fundamental_metrics(data)
```

Convierte un `FundamentalData` en una fila de métricas fundamentales.

Parámetros:

`data` : `FundamentalData`

Retorna:

`pandas.Series`

## `build_fundamental_metric_history`

```python
build_fundamental_metric_history(data, frequency="quarterly", trailing_periods=4)
```

Convierte un `FundamentalData` en una tabla temporal de métricas.

Parámetros:

`data` : `FundamentalData`

`frequency` : {"annual", "quarterly"}, default `"quarterly"`

`trailing_periods` : `int`, default `4`

Retorna:

`pandas.DataFrame`

## `build_metrics_frame`

```python
build_metrics_frame(records)
```

Convierte una colección de `FundamentalData` en una tabla de métricas.

Parámetros:

`records` : iterable of `FundamentalData`

Retorna:

`pandas.DataFrame`

## `build_metric_history_frame`

```python
build_metric_history_frame(records, frequency="quarterly", trailing_periods=4)
```

Convierte una colección de `FundamentalData` en una tabla temporal de métricas.

Parámetros:

`records` : iterable of `FundamentalData`

`frequency` : {"annual", "quarterly"}, default `"quarterly"`

`trailing_periods` : `int`, default `4`

Retorna:

`pandas.DataFrame`

## `score_fundamentals`

```python
score_fundamentals(metrics, config, metric_history=None, frequency=None)
```

Calcula componentes normalizados y un score final ponderado.

Parámetros:

`metrics` : `pandas.DataFrame`
Tabla de métricas por compañía.

`config` : `FundamentalScoreConfig`
Configuración de scoring.

`metric_history` : `pandas.DataFrame`, optional
Historial usado por señales de cambio.

`frequency` : `str`, optional
Frecuencia usada para inferir el rezago interanual.

Retorna:

`pandas.DataFrame`
Tabla ordenada por `fundamental_score` de mayor a menor.

## `score_fundamentals_over_time`

```python
score_fundamentals_over_time(metric_history, config, frequency=None)
```

Calcula scores por periodo. Dentro de cada periodo, las compañías se comparan
entre sí usando los mismos pesos definidos en `config`.
Si hay señales históricas, cada periodo solo usa información disponible hasta
ese punto.

Parámetros:

`metric_history` : `pandas.DataFrame`
Tabla con métricas por ticker y periodo.

`config` : `FundamentalScoreConfig`
Configuración de scoring.

Retorna:

`pandas.DataFrame`
Historial ordenado de scores por periodo.

## `CorrelationPortfolioSelector`

```python
class CorrelationPortfolioSelector(
    start_date,
    end_date=None,
    price_field="Close",
    use_absolute_corr=True,
    min_coverage=0.80,
)
```

Selector basado en correlación para construir candidatos diversificados.

### Atributos

`start_date` : `str`
Fecha inicial de descarga.

`end_date` : `str`, optional
Fecha final de descarga.

`price_field` : `str`
Campo de precio usado.

`use_absolute_corr` : `bool`
Si `True`, usa correlaciones absolutas.

`min_coverage` : `float`
Cobertura mínima requerida de datos.

`prices_by_group` : `dict[str, pandas.DataFrame]`
Precios descargados por grupo.

`returns_by_group` : `dict[str, pandas.DataFrame]`
Retornos calculados por grupo.

`ranking_by_group` : `pandas.DataFrame`
Ranking interno por grupo.

### Métodos

`rank_within_groups(grouped_tickers, intra_group_weights=None, top_k=1)`
Rankea activos dentro de cada grupo por menor correlación promedio.

`build_multigroup_portfolio(top_k_per_group=1, final_size=None)`
Construye una selección final entre grupos.

`update_portfolio(current_tickers, candidate_tickers, max_new_assets=1)`
Sugiere nuevos activos de baja correlación frente a un portafolio existente.

`run_pipeline(grouped_tickers, top_k_in_group=1, final_size=None, intra_group_weights=None)`
Ejecuta el flujo completo de selección por correlación.

## Alias Públicos

`CorrelationSelector` es un alias de `CorrelationPortfolioSelector`.

```python
from src.selection import CorrelationSelector
```

## Nota Sobre Funciones Internas De Módulo

`config_for_strategy` forma parte de `fundamental_scorers.py` y es utilizada por
`FundamentalSelector` para resolver las configuraciones por defecto de `value` y
`growth`. No se reexporta desde `src.selection.__init__`, por lo que la API
pública recomendada para usuarios es construir `FundamentalSelector` con
`strategy="value"` o `strategy="growth"`, o pasar un `FundamentalScoreConfig`
custom.
