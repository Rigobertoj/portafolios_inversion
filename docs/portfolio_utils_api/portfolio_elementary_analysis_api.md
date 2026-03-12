# PortfolioElementaryAnalysis API Reference

Documentación de referencia para la clase `PortfolioElementaryAnalysis` con estructura de API estilo NumPy/SciPy.

## Import

```python
from portfolio_utils import PortfolioElementaryAnalysis
```

## Class

```python
PortfolioElementaryAnalysis(
    tickers,
    start,
    end=None,
    price_field="Close",
    *,
    weight,
    benchmark,
)
```

Clase para análisis comparativo portafolio vs benchmark y métricas base de CAPM.

Herencia:

`AssetsResearch -> PortfolioElementaryMetrics -> PortfolioElementaryAnalysis`

Esto implica que conserva:

- métodos de datos y métricas por activo (`get_prices`, `get_returns`, `annual_return`, `covariance`, etc.),
- métricas agregadas de portafolio (`portfolio_annual_return`, `portfolio_annual_volatility`, `portfolio_sharpe_ratio`, etc.),
- y agrega cálculo de beta y retornos esperados CAPM contra benchmark.

## Parameters

- `tickers` : `Iterable[str]`
  - Lista de activos del portafolio.
- `start` : `str`
  - Fecha inicial en formato `YYYY-MM-DD`.
- `end` : `str | None`, default `None`
  - Fecha final en formato `YYYY-MM-DD`.
- `price_field` : `str`, default `"Close"`
  - Campo de precios utilizado para descarga.
- `weight` : `Iterable[float]`
  - Vector de pesos del portafolio. Debe sumar `1` y coincidir en longitud con `tickers`.
- `benchmark` : `str`
  - Ticker del benchmark (por ejemplo `^GSPC`).

## Attributes

- `benchmark` : `str`
  - Ticker benchmark configurado.
- `weight` : `np.ndarray` (heredado)
- `tickers` : `list[str]` (heredado)
- `start` : `str` (heredado)
- `end` : `str | None` (heredado)
- `price_field` : `str` (heredado)

## Constructors

### `with_equal_weights(...)`

```python
PortfolioElementaryAnalysis.with_equal_weights(
    tickers,
    start,
    benchmark,
    end=None,
    price_field="Close",
)
```

Construye una instancia asignando pesos iguales a todos los activos.

Returns:

- `PortfolioElementaryAnalysis`

Raises:

- `ValueError`
  - Si `tickers` está vacío.

## Public Methods

### `benchmark_returns()`

```python
benchmark_returns()
```

Descarga/calcula retornos diarios del benchmark y usa caché interno para llamadas posteriores.

Returns:

- `pd.Series`

### `benchmark_annual_return()`

```python
benchmark_annual_return()
```

Retorno anualizado del benchmark (`mean * 252`).

Returns:

- `float`

Raises:

- `ValueError`
  - Si la serie de retornos del benchmark está vacía.

### `portfolio_beta()`

```python
portfolio_beta()
```

Calcula beta del portafolio contra benchmark:

`cov(portfolio, benchmark) / var(benchmark)`

Returns:

- `float`

Raises:

- `ValueError`
  - Si no hay traslape suficiente de fechas (menos de 2 observaciones).
  - Si la varianza del benchmark es cero.

### `assets_beta()`

```python
assets_beta()
```

Calcula beta individual por activo contra benchmark.

Returns:

- `pd.Series`
  - Índice: ticker.
  - Nombre de serie: `"beta"`.

Raises:

- `ValueError`
  - Si retornos de activos o benchmark están vacíos.
  - Si la varianza del benchmark es cero.

### `portfolio_capm_expected_return(risk_free_rate=0.0, market_expected_return=None)`

```python
portfolio_capm_expected_return(
    risk_free_rate=0.0,
    market_expected_return=None,
)
```

Calcula rendimiento esperado del portafolio con CAPM:

`Rf + beta_portfolio * (Rm - Rf)`

Si `market_expected_return=None`, usa `benchmark_annual_return()`.

Returns:

- `float`

### `assets_capm_expected_return(risk_free_rate=0.0, market_expected_return=None)`

```python
assets_capm_expected_return(
    risk_free_rate=0.0,
    market_expected_return=None,
)
```

Calcula rendimiento esperado CAPM por activo:

`Rf + beta_asset * (Rm - Rf)`

Si `market_expected_return=None`, usa `benchmark_annual_return()`.

Returns:

- `pd.Series`

Note:

- En la implementación actual el `type hint` del método indica `float`, pero el valor retornado real es una `pd.Series`.

## Validation and Error Behavior

- `benchmark` debe ser `str` no vacío; de lo contrario lanza `ValueError`.
- Al cambiar `benchmark`, se limpia caché interno de retornos del benchmark.
- Métodos CAPM validan que exista traslape suficiente entre series y que la varianza del benchmark no sea cero.

## Computational Notes

- Todas las anualizaciones usan `252` días hábiles.
- `portfolio_beta()` alinea retornos del portafolio y benchmark por intersección de fechas y elimina `NaN`.
- El portafolio diario se calcula como combinación lineal de retornos de activos:
  `assets_returns @ weight`.

## Examples

### 1) Construcción directa

```python
import numpy as np
from portfolio_utils import PortfolioElementaryAnalysis

analysis = PortfolioElementaryAnalysis(
    tickers=["AAPL", "MSFT", "JPM"],
    start="2020-01-01",
    end="2025-12-31",
    weight=np.array([0.4, 0.4, 0.2]),
    benchmark="^GSPC",
)

portfolio_beta = analysis.portfolio_beta()
assets_beta = analysis.assets_beta()
```

### 2) Construcción con pesos iguales

```python
analysis = PortfolioElementaryAnalysis.with_equal_weights(
    tickers=["AAPL", "MSFT", "JPM"],
    start="2020-01-01",
    end="2025-12-31",
    benchmark="^GSPC",
)
```

### 3) CAPM de portafolio y activos

```python
rf = 0.04

capm_portfolio = analysis.portfolio_capm_expected_return(risk_free_rate=rf)
capm_assets = analysis.assets_capm_expected_return(risk_free_rate=rf)
```

## See Also

- [AssetsResearch API](./assets_research_api.md)
- [PortfolioElementaryMetrics API](./portfolio_elementary_metrics_api.md)
- [PortfolioOptimization API](./portfolio_optimization_api.md)
