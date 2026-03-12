# AssetsResearch API Reference

Documentación de referencia para la clase `AssetsResearch` con estructura de API estilo NumPy/SciPy.

## Import

```python
from portfolio_utils import AssetsResearch
```

## Class

```python
AssetsResearch(
    tickers,
    start,
    end=None,
    price_field="Close",
)
```

Clase base para investigación cuantitativa de activos: descarga precios, calcula retornos y genera métricas descriptivas.

## Parameters

- `tickers` : `str | Iterable[str]`
  - Ticker único o lista de tickers.
- `start` : `str`
  - Fecha inicial en formato `YYYY-MM-DD`.
- `end` : `str | None`, default `None`
  - Fecha final en formato `YYYY-MM-DD`.
- `price_field` : `str`, default `"Close"`
  - Campo de precio solicitado a Yahoo Finance (por ejemplo `"Close"` o `"Adj Close"`).

## Attributes (Properties)

- `tickers` : `list[str]`
  - Lista normalizada de símbolos.
- `start` : `str`
  - Fecha de inicio.
- `end` : `str | None`
  - Fecha de fin.
- `price_field` : `str`
  - Campo de precio activo.
- `prices` : `pd.DataFrame`
  - Caché de precios.
- `returns` : `pd.DataFrame`
  - Caché de retornos diarios.

Note:

- Al modificar `tickers`, `start`, `end` o `price_field`, se limpia automáticamente el caché (`prices` y `returns`).

## Public Methods

### `download_prices()`

```python
download_prices()
```

Descarga precios desde Yahoo Finance y los guarda en `self.prices`.

Returns:

- `pd.DataFrame`
  - Índice temporal y columnas por ticker.

Raises:

- `ValueError`
  - Si `price_field` no existe en la estructura descargada.

### `compute_returns()`

```python
compute_returns()
```

Calcula retornos diarios como:

`prices.pct_change().dropna()`

Si no hay precios en caché, ejecuta `download_prices()` primero.

Returns:

- `pd.DataFrame`

### `get_prices(tickers=None)`

```python
get_prices(tickers=None)
```

Devuelve precios para todos los tickers o para una selección.

Parameters:

- `tickers` : `None | str | Sequence[str]`, default `None`

Returns:

- `pd.DataFrame`

Raises:

- `ValueError`
  - Si algún ticker solicitado no existe en los datos.

### `get_returns(tickers=None)`

```python
get_returns(tickers=None)
```

Devuelve retornos diarios para todos los tickers o para una selección.

Si no existe caché de retornos, ejecuta `compute_returns()`.

Parameters:

- `tickers` : `None | str | Sequence[str]`, default `None`

Returns:

- `pd.DataFrame`

Raises:

- `ValueError`
  - Si algún ticker solicitado no existe en los datos.

### `annual_return(tickers=None)`

```python
annual_return(tickers=None)
```

Retorno anualizado por activo:

`mean_daily_return * 252`

Returns:

- `pd.Series`

### `annual_volatility(tickers=None)`

```python
annual_volatility(tickers=None)
```

Volatilidad anualizada por activo:

`std_daily_return * sqrt(252)`

Returns:

- `pd.Series`

### `skew(tickers=None)`

```python
skew(tickers=None)
```

Asimetría de retornos diarios.

Returns:

- `pd.Series`

### `vol_over_mean(tickers=None)`

```python
vol_over_mean(tickers=None)
```

Razón volatilidad/rendimiento anualizado:

`annual_volatility / annual_return`

`annual_return == 0` se reemplaza por `NaN` para evitar división por cero.

Returns:

- `pd.Series`

### `return_interval_pct(tickers=None, z=2.65)`

```python
return_interval_pct(tickers=None, z=2.65)
```

Intervalo porcentual estimado:

`annual_return_pct +/- z * annual_volatility_pct`

Parameters:

- `tickers` : `None | str | Sequence[str]`, default `None`
- `z` : `float`, default `2.65`
  - Multiplicador del intervalo.

Returns:

- `pd.DataFrame`
  - Columnas: `low`, `high`.

### `metrics(tickers=None)`

```python
metrics(tickers=None)
```

Tabla consolidada de métricas por activo.

Columns:

- `annual_return`
- `annual_volatility`
- `annual_return_pct`
- `annual_volatility_pct`
- `skew`
- `vol_over_mean`
- `return_interval_low_pct`
- `return_interval_high_pct`

Returns:

- `pd.DataFrame`

Note:

- El método usa `z = 2.65` fijo para las columnas de intervalo.

### `describe_returns(tickers=None)`

```python
describe_returns(tickers=None)
```

Estadística descriptiva de retornos (`DataFrame.describe()`).

Returns:

- `pd.DataFrame`

### `covariance(tickers=None)`

```python
covariance(tickers=None)
```

Matriz de covarianza de retornos diarios.

Returns:

- `pd.DataFrame`

### `correlation(tickers=None)`

```python
correlation(tickers=None)
```

Matriz de correlación de retornos diarios.

Returns:

- `pd.DataFrame`

## Validation and Error Behavior

- `tickers` vacío -> `ValueError`.
- `start` no válido (string vacío) -> `ValueError`.
- `end` no válido (no `None` y string vacío) -> `ValueError`.
- `price_field` inválido (string vacío) -> `ValueError`.
- Selección de tickers inexistentes en `get_prices/get_returns` -> `ValueError`.
- Campo de precios ausente en Yahoo data -> `ValueError`.

## Typical Workflow

```python
research = AssetsResearch(
    tickers=["JPM", "V", "GS", "PGR"],
    start="2020-01-01",
    end="2025-01-01",
    price_field="Close",
)

prices = research.download_prices()
returns = research.compute_returns()

annual_ret = research.annual_return()
annual_vol = research.annual_volatility()
summary = research.metrics()

cov_subset = research.covariance(["JPM", "V"])
corr_all = research.correlation()
```

## See Also

- [PortfolioElementaryMetrics API](./portfolio_elementary_metrics_api.md)
- [PortfolioElementaryAnalysis API](./portfolio_elementary_analysis_api.md)
- [PortfolioOptimization API](./portfolio_optimization_api.md)
