# PortfolioElementaryMetrics API Reference

Documentación de referencia para la clase `PortfolioElementaryMetrics` con estructura de API estilo NumPy/SciPy.

## Import

```python
from portfolio_utils import PortfolioElementaryMetrics
```

## Class

```python
PortfolioElementaryMetrics(
    tickers,
    start,
    end=None,
    price_field="Close",
    *,
    weight,
)
```

Clase para calcular métricas básicas de un portafolio a partir de una combinación de activos.

Herencia:

`AssetsResearch -> PortfolioElementaryMetrics`

Esto implica que además de los métodos del portafolio, también hereda utilidades de investigación de activos (`get_prices`, `get_returns`, `annual_return`, `covariance`, etc.).

## Parameters

- `tickers` : `Iterable[str]`
  - Lista de símbolos de los activos.
- `start` : `str`
  - Fecha inicial en formato `YYYY-MM-DD`.
- `end` : `str | None`, default `None`
  - Fecha final en formato `YYYY-MM-DD`. Si es `None`, Yahoo Finance usa su comportamiento por defecto hasta fecha actual.
- `price_field` : `str`, default `"Close"`
  - Campo de precios usado en la descarga (por ejemplo `"Close"` o `"Adj Close"`).
- `weight` : `Iterable[float]`
  - Vector de pesos del portafolio.
  - Debe tener la misma longitud que `tickers`.
  - Debe sumar `1`.

## Attributes

- `weight` : `np.ndarray`
  - Pesos actuales del portafolio (copia defensiva al leer).
- `tickers` : `list[str]` (heredado)
- `start` : `str` (heredado)
- `end` : `str | None` (heredado)
- `price_field` : `str` (heredado)
- `prices` : `pd.DataFrame` (cache heredado)
- `returns` : `pd.DataFrame` (cache heredado)

## Methods

### `portfolio_path()`

```python
portfolio_path()
```

Construye la trayectoria de riqueza/precio del portafolio como combinación lineal de precios:

`assets_prices @ weight`

Returns:

- `pd.Series | pd.DataFrame` según el backend de precios devuelto por Yahoo.

### `portfolio_annual_return()`

```python
portfolio_annual_return()
```

Calcula rendimiento anualizado del portafolio:

`weight @ annual_return_assets`

Returns:

- `float`

### `portfolio_annual_volatility()`

```python
portfolio_annual_volatility()
```

Calcula volatilidad anualizada:

1. `portfolio_variance_daily = weight.T @ cov_daily @ weight`
2. `portfolio_variance_annual = portfolio_variance_daily * 252`
3. `vol = sqrt(portfolio_variance_annual)`

Returns:

- `float`

### `portfolio_variance_coeficience()`

```python
portfolio_variance_coeficience()
```

Calcula la razón volatilidad/rendimiento anual del portafolio.

Returns:

- `float`

Raises:

- `ValueError`
  - Si el rendimiento anual del portafolio es cero.

### `portfolio_sharpe_ratio(free_rate)`

```python
portfolio_sharpe_ratio(free_rate)
```

Calcula ratio de Sharpe del portafolio:

`(portfolio_return - free_rate) / portfolio_volatility`

Parameters:

- `free_rate` : `float`
  - Tasa libre de riesgo en base anual para que sea consistente con las métricas anualizadas.

Returns:

- `float`

Raises:

- `ValueError`
  - Si la volatilidad anual del portafolio es cero.

## Validation Rules

Durante la construcción y asignación de `weight`:

- El vector debe ser 1D.
- La longitud debe coincidir con número de `tickers`.
- La suma debe ser `1` (`np.isclose`).

Si no se cumple, se lanza `ValueError`.

## Notes

- Las métricas anualizadas usan `252` días hábiles.
- La clase depende de `AssetsResearch` para descarga y cálculo de retornos diarios.
- `portfolio_variance_coeficience` mantiene el nombre actual del código (incluye la grafía `coeficience`).

## Examples

```python
import numpy as np
from portfolio_utils import PortfolioElementaryMetrics

portfolio = PortfolioElementaryMetrics(
    tickers=["AAPL", "MSFT", "JPM"],
    start="2020-01-01",
    end="2025-12-31",
    weight=np.array([0.4, 0.4, 0.2]),
)

annual_ret = portfolio.portfolio_annual_return()
annual_vol = portfolio.portfolio_annual_volatility()
sharpe = portfolio.portfolio_sharpe_ratio(free_rate=0.04)
path = portfolio.portfolio_path()
```

## See Also

- [AssetsResearch API](./assets_research_api.md)
- [PortfolioElementaryAnalysis API](./portfolio_elementary_analysis_api.md)
- [PortfolioOptimization API](./portfolio_optimization_api.md)
