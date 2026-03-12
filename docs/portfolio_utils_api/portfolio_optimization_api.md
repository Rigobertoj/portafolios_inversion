# PortfolioOptimization API Reference

Documentación de referencia para optimización de portafolios con estructura de API estilo NumPy/SciPy.

## Import

```python
from portfolio_utils import (
    PortfolioOptimization,
    OptimizationConfig,
    OptimizationResult,
)
```

## Overview

`PortfolioOptimization` extiende `PortfolioElementaryAnalysis` y agrega métodos para resolver problemas de asignación de pesos usando `scipy.optimize.minimize`.

Herencia:

`AssetsResearch -> PortfolioElementaryMetrics -> PortfolioElementaryAnalysis -> PortfolioOptimization`

## `OptimizationConfig`

```python
OptimizationConfig(
    objective="max_sharpe",
    expected_return_model="historical",
    risk_free_rate=0.0,
    market_expected_return=None,
    target_return=None,
    annualize_covariance=True,
    allow_short=False,
    bounds=None,
    initial_weights=None,
    solver_method="SLSQP",
    solver_options={"maxiter": 500, "ftol": 1e-9, "disp": False},
)
```

Configura el problema de optimización.

### Parameters

- `objective` : `{"min_variance", "max_sharpe", "target_return"}`, default `"max_sharpe"`
  - Objetivo de optimización.
- `expected_return_model` : `{"historical", "capm_assets"}`, default `"historical"`
  - Modelo para estimar retornos esperados por activo.
- `risk_free_rate` : `float`, default `0.0`
  - Tasa libre de riesgo anual (utilizada en Sharpe y CAPM).
- `market_expected_return` : `float | None`, default `None`
  - Retorno esperado del mercado para CAPM. Si es `None`, se estima con benchmark anual.
- `target_return` : `float | None`, default `None`
  - Retorno objetivo anual. Requerido cuando `objective="target_return"`.
- `annualize_covariance` : `bool`, default `True`
  - Si `True`, multiplica covarianza diaria por `252`.
- `allow_short` : `bool`, default `False`
  - Si `True` y no se definen `bounds`, usa `(-1, 1)` por activo.
- `bounds` : `Sequence[Tuple[float, float]] | None`, default `None`
  - Límites por activo. Si se provee, debe coincidir en longitud con el número de activos.
- `initial_weights` : `Iterable[float] | None`, default `None`
  - Pesos iniciales. Deben sumar `1`.
- `solver_method` : `str`, default `"SLSQP"`
  - Método de optimización para `scipy.optimize.minimize`.
- `solver_options` : `dict[str, object]`
  - Opciones del solver de SciPy.

## `OptimizationResult`

Resultado estructurado de la optimización.

### Attributes

- `objective` : `str`
- `success` : `bool`
- `status` : `int`
- `message` : `str`
- `weights` : `np.ndarray`
- `weights_by_ticker` : `pd.Series`
- `expected_return` : `float`
- `volatility` : `float`
- `sharpe` : `float`
- `objective_value` : `float`
- `iterations` : `int`

## Class `PortfolioOptimization`

```python
PortfolioOptimization(
    tickers,
    start,
    end=None,
    price_field="Close",
    *,
    weight,
    benchmark,
)
```

### Parameters

Los parámetros de construcción siguen la jerarquía heredada:

- `tickers`, `start`, `end`, `price_field` desde `AssetsResearch`.
- `weight` desde `PortfolioElementaryMetrics`.
- `benchmark` desde `PortfolioElementaryAnalysis`.

## Public Methods

### `optimize(config=None)`

```python
optimize(config=None)
```

Resuelve el problema de optimización y devuelve un `OptimizationResult`.

Si `config` es `None`, usa `OptimizationConfig()` por defecto.

### `optimize_and_set_weights(config=None)`

```python
optimize_and_set_weights(config=None)
```

Ejecuta `optimize` y, si `success=True`, actualiza `self.weight` con los pesos óptimos.

Returns:

- `OptimizationResult`

Raises:

- `ValueError`
  - Si la optimización falla (`success=False`).

## Optimization Logic

La función `optimize` aplica esta lógica:

1. Construye vector de retornos esperados según `expected_return_model`.
2. Construye matriz de covarianza (anualizada o no).
3. Define restricciones:
   - Siempre: `sum(weights) == 1`.
   - Si `objective="target_return"`: además `weights @ mu == target_return`.
4. Define objetivo:
   - `"min_variance"`: minimiza varianza.
   - `"max_sharpe"`: minimiza Sharpe negativo.
   - `"target_return"`: minimiza varianza con retorno objetivo.
5. Ejecuta `scipy.optimize.minimize`.
6. Empaqueta salida en `OptimizationResult`.

## Validation and Error Behavior

- `expected_return_model` inválido -> `ValueError`.
- `objective` inválido -> `ValueError`.
- `objective="target_return"` sin `target_return` -> `ValueError`.
- `bounds` con longitud distinta a número de activos -> `ValueError`.
- `initial_weights` con longitud distinta o suma distinta de `1` -> `ValueError`.

## Examples

### 1) Máximo Sharpe (default)

```python
import numpy as np
from portfolio_utils import PortfolioOptimization, OptimizationConfig

opt = PortfolioOptimization(
    tickers=["AAPL", "MSFT", "JPM", "V"],
    start="2020-01-01",
    end="2025-12-31",
    weight=np.array([0.25, 0.25, 0.25, 0.25]),
    benchmark="^GSPC",
)

result = opt.optimize(
    OptimizationConfig(
        objective="max_sharpe",
        risk_free_rate=0.04,
        expected_return_model="historical",
    )
)

result.weights_by_ticker
```

### 2) Mínima varianza sin ventas en corto

```python
result = opt.optimize(
    OptimizationConfig(
        objective="min_variance",
        allow_short=False,
    )
)
```

### 3) Mínima varianza con retorno objetivo

```python
cfg = OptimizationConfig(
    objective="target_return",
    target_return=0.12,
    expected_return_model="historical",
)

result = opt.optimize(cfg)
```

### 4) Optimizar y actualizar pesos del objeto

```python
result = opt.optimize_and_set_weights(
    OptimizationConfig(objective="max_sharpe", risk_free_rate=0.04)
)

opt.weight  # ahora contiene pesos óptimos
```

## Notes

- Por defecto se usa `SLSQP`, adecuado para restricciones lineales y límites.
- El Sharpe reportado en el resultado usa la volatilidad resultante y `risk_free_rate` del config.
- Si la volatilidad óptima es ~0, el Sharpe devuelto es `NaN`.

## See Also

- [AssetsResearch API](./assets_research_api.md)
- [PortfolioElementaryMetrics API](./portfolio_elementary_metrics_api.md)
- [PortfolioElementaryAnalysis API](./portfolio_elementary_analysis_api.md)
