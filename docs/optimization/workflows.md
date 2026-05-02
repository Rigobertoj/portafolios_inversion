# Flujos De Trabajo

Esta guía muestra cómo usar `src.optimization` en situaciones reales del
proyecto: investigación en notebooks, construcción de portafolio, optimización
post-moderna y backtesting.

## 1. Optimizar Un `Portfolio` Existente

Este es el flujo recomendado cuando ya tienes precios descargados o preparados.

```python
import pandas as pd

from src.portfolio import Portfolio
from src.optimization import MeanVarianceOptimizer, OptimizationConfig

prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
initial_weights = [1 / len(prices.columns)] * len(prices.columns)

portfolio = Portfolio(
    prices=prices,
    weights=initial_weights,
    name="Tech Portfolio",
)

optimizer = MeanVarianceOptimizer(portfolio=portfolio)
result = optimizer.optimize_maximum_sharpe(
    OptimizationConfig(risk_free_rate=0.045),
)

print(result.success)
print(result.weights_by_ticker)
print(result.sharpe)
```

Qué sucede internamente:

1. `Portfolio` valida precios, retornos y pesos.
2. `MeanVarianceOptimizer` construye un optimizador compatible.
3. Se calcula `mu` y `Sigma` anualizados.
4. Se resuelve el máximo Sharpe con `SLSQP`.
5. Si `result.success=True`, `portfolio.weights` queda actualizado.

## 2. Mínima Varianza Con Retorno Mínimo

Cuando no basta con minimizar riesgo, puede pedirse un retorno esperado mínimo.

```python
from src.optimization import MeanVarianceOptimizer, MinimumVarianceConfig

config = MinimumVarianceConfig(
    minimum_return=0.10,
    allow_short=False,
)

optimizer = MeanVarianceOptimizer(portfolio=portfolio)
result = optimizer.optimize_minimum_variance(config)

if result.success:
    allocation = result.weights_by_ticker.sort_values(ascending=False)
```

Este problema se interpreta así:

```text
minimizar    w.T @ Sigma @ w
sujeto a     sum(w) = 1
             w @ mu >= 0.10
             0 <= w_i <= 1
```

Es útil cuando el inversionista tiene un retorno requerido y quiere encontrar la
forma menos volátil de alcanzarlo.

## 3. Definir Límites Por Activo

Los `bounds` permiten expresar reglas de concentración. Por ejemplo, ningún
activo puede pesar más de 40% y ninguno menos de 5%.

```python
from src.optimization import OptimizationConfig

n_assets = len(portfolio.tickers)

config = OptimizationConfig(
    risk_free_rate=0.04,
    bounds=[(0.05, 0.40)] * n_assets,
)

result = MeanVarianceOptimizer(portfolio).optimize_maximum_sharpe(config)
```

Si los pesos iniciales no respetan esos límites, el solver no se ejecuta y se
lanza `ValueError`. En ese caso conviene pasar `initial_weights` coherentes:

```python
config = OptimizationConfig(
    bounds=[(0.05, 0.40)] * n_assets,
    initial_weights=[1 / n_assets] * n_assets,
)
```

## 4. Permitir Ventas En Corto

Si `allow_short=True` y no se entregan `bounds`, el módulo usa `[-1, 1]` por
activo.

```python
config = OptimizationConfig(
    risk_free_rate=0.04,
    allow_short=True,
)

result = MeanVarianceOptimizer(portfolio).optimize_maximum_sharpe(config)
```

Este modo debe interpretarse con cuidado: permite pesos negativos y posiciones
apalancadas relativas, aunque la restricción `sum(w)=1` se mantiene.

## 5. Mínima Semivarianza

La semivarianza mide solo desviaciones desfavorables. Es apropiada cuando el
usuario no quiere castigar la volatilidad positiva.

```python
from src.optimization import PostModernOptimizer, MinimumSemivarianceConfig

config = MinimumSemivarianceConfig(
    threshold=0.0,
    minimum_return=0.08,
)

optimizer = PostModernOptimizer(portfolio)
result = optimizer.optimize_minimum_semivariance(config)

print(result.weights_by_ticker)
print(result.downside_risk)
```

Con `threshold=0.0`, el módulo evalúa retornos debajo de cero. Con un umbral
positivo, mide retornos debajo de una meta diaria o referencia mínima.

## 6. Mínima Semivarianza Contra Benchmark

También se puede medir downside contra un benchmark. En ese caso la referencia
diaria es:

```text
benchmark_return + threshold
```

Ejemplo:

```python
benchmark_returns = benchmark_prices.pct_change().dropna()

config = MinimumSemivarianceConfig(threshold=0.0)

result = PostModernOptimizer(portfolio).optimize_minimum_semivariance(
    config=config,
    benchmark_returns=benchmark_returns,
)
```

Esto sirve cuando el riesgo no significa perder dinero absoluto, sino quedar por
debajo de un índice o una estrategia pasiva.

## 7. Máximo Omega

Omega compara potencial al alza contra riesgo downside. En este módulo el Omega
de portafolio se calcula como una combinación ponderada del Omega por activo.

```python
from src.optimization import PostModernOptimizer, MaximumOmegaConfig

config = MaximumOmegaConfig(threshold=0.0)

result = PostModernOptimizer(portfolio).optimize_maximum_omega(config)

print(result.omega)
print(result.weights_by_ticker)
```

Si algún activo tiene downside risk igual a cero, Omega puede quedar indefinido.
En ese caso el módulo lanza `ValueError` para evitar una optimización basada en
valores no finitos.

## 8. Usar La API Histórica En Notebooks

Los notebooks antiguos pueden seguir trabajando directamente con tickers y
fechas.

```python
from src.optimization import PortfolioOptimization, MinimumVarianceConfig

optimizer = PortfolioOptimization(
    tickers=["AAPL", "MSFT", "NVDA", "KO"],
    start="2020-01-01",
    end="2025-01-01",
    weight=[0.25, 0.25, 0.25, 0.25],
)

result = optimizer.optimize_minimum_variance(
    MinimumVarianceConfig(minimum_return=0.08),
)

optimizer.weight
```

Esta forma es cómoda para exploración, pero para código reusable conviene
preferir `Portfolio` + `MeanVarianceOptimizer`.

## 9. Conectar Con Backtesting

El módulo de backtesting envuelve optimización mediante estrategias.

```python
from src.backtesting import BacktestConfig, Backtester, MeanVarianceStrategy
from src.optimization import OptimizationConfig

config = BacktestConfig(
    tickers=["AAPL", "MSFT", "NVDA"],
    initial_capital=1_000_000,
    optimization_start="2018-01-01",
    backtest_start="2021-01-01",
    end="2025-01-01",
)

strategy = MeanVarianceStrategy(
    objective="maximum_sharpe",
    config=OptimizationConfig(risk_free_rate=0.04),
)

result = Backtester(config).run(strategy, prices=prices)
```

En este flujo, el usuario no llama directamente al optimizador. La estrategia lo
hace por dentro y devuelve pesos al motor de backtesting.

Para semivarianza:

```python
from src.backtesting import PostModernStrategy
from src.optimization import MinimumSemivarianceConfig

strategy = PostModernStrategy(
    objective="minimum_semivariance",
    config=MinimumSemivarianceConfig(threshold=0.0),
)

result = Backtester(config).run(strategy, prices=prices)
```

## 10. Interpretar Resultados

Después de optimizar, conviene revisar tres niveles:

```python
result.success
result.message
result.weights_by_ticker
```

Luego se revisan métricas financieras:

```python
result.expected_return
result.volatility      # media-varianza
result.sharpe          # media-varianza
result.downside_risk   # post-moderno
result.omega           # post-moderno
```

Una buena práctica es no usar pesos si `success` es falso:

```python
if not result.success:
    raise RuntimeError(result.message)
```

## 11. Errores Frecuentes

### Pesos Que No Suman 1

```python
Portfolio(prices=prices, weights=[0.2, 0.2, 0.2])
```

Si hay tres activos, esos pesos suman 0.6 y se lanza `ValueError`. Deben sumar
1.

### Bounds Con Longitud Incorrecta

```python
OptimizationConfig(bounds=[(0, 1), (0, 1)])
```

Si el portafolio tiene cuatro activos, los bounds deben tener cuatro tuplas.

### Punto Inicial Fuera De Bounds

```python
OptimizationConfig(
    bounds=[(0.10, 0.40)] * 4,
    initial_weights=[0.70, 0.10, 0.10, 0.10],
)
```

El primer peso viola el máximo de 0.40. El módulo lo detecta antes de llamar al
solver.

### Benchmark Sin Suficiente Intersección De Fechas

En semivarianza contra benchmark, los retornos de activos y benchmark se alinean
por índice. Si la intersección tiene menos de dos observaciones, se lanza
`ValueError`.

## 12. Patrón Recomendado Para Investigación

Un patrón robusto para notebooks es:

```python
from src.portfolio import Portfolio
from src.optimization import (
    MeanVarianceOptimizer,
    PostModernOptimizer,
    OptimizationConfig,
    MinimumSemivarianceConfig,
)

portfolio = Portfolio(prices=prices, weights=initial_weights)

mv_result = MeanVarianceOptimizer(portfolio.with_weights(initial_weights)).optimize_maximum_sharpe(
    OptimizationConfig(risk_free_rate=0.04),
)

pm_result = PostModernOptimizer(portfolio.with_weights(initial_weights)).optimize_minimum_semivariance(
    MinimumSemivarianceConfig(threshold=0.0),
)

comparison = {
    "max_sharpe": mv_result.weights_by_ticker,
    "min_semivariance": pm_result.weights_by_ticker,
}
```

Usar `with_weights(...)` evita que una optimización sobrescriba los pesos que
otra necesita como punto de partida. Esto es especialmente útil al comparar
estrategias en una misma celda.
