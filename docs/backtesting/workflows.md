# Flujos De Trabajo

Este documento muestra cómo usar `src.backtesting` dentro del proyecto.

## 1. Backtesting Estático

Use este flujo cuando quiera optimizar pesos una sola vez y probarlos en una
ventana posterior.

```python
from src.backtesting import BacktestConfig, Backtester, MeanVarianceStrategy

config = BacktestConfig(
    tickers=["AAPL", "MSFT", "NVDA"],
    initial_capital=1_000_000,
    optimization_start="2018-01-01",
    backtest_start="2020-01-01",
    end="2025-01-01",
    benchmark_ticker="^GSPC",
    benchmark_label="S&P 500",
    risk_free_rate=0.035,
)

strategies = [
    MeanVarianceStrategy(objective="minimum_variance"),
    MeanVarianceStrategy(objective="maximum_sharpe"),
]

result = Backtester(config).run(
    strategies,
    prices=prices,
    benchmark_prices=benchmark,
)
```

Objetos principales:

```python
result.evolution
result.returns
result.metrics
result.strategy_results
```

Este flujo es el equivalente modular del notebook `16.Backtesting.ipynb`.

## 2. Backtesting Dinámico Mensual

Use este flujo cuando quiera reoptimizar pesos periódicamente.

```python
from src.backtesting import (
    BacktestConfig,
    DynamicBacktester,
    MeanVarianceStrategy,
    RebalanceConfig,
)

config = BacktestConfig(
    tickers=["AAPL", "MSFT", "GOOG", "JPM"],
    initial_capital=1_000_000,
    optimization_start="2018-01-01",
    backtest_start="2020-01-01",
    end="2025-01-01",
    benchmark_ticker="^GSPC",
    benchmark_label="S&P 500",
    risk_free_rate=0.035,
)

rebalance = RebalanceConfig(
    lookback_months=12,
    rebalance_months=1,
    expanding_window=False,
    transaction_cost=0.001,
)

result = DynamicBacktester(config, rebalance).run(
    MeanVarianceStrategy(objective="maximum_sharpe"),
    prices=prices,
    benchmark_prices=benchmark,
)
```

Objetos principales:

```python
result.evolution
result.metrics
result.weights_history
result.turnover
result.transaction_costs
```

Este flujo representa la idea central de `21.- Backtesting Dinámico.ipynb`, pero
con una arquitectura más extensible.

## 3. Comparar Varias Estrategias Dinámicas

El motor dinámico acepta una lista de estrategias.

```python
from src.backtesting import MeanVarianceStrategy, PostModernStrategy

strategies = [
    MeanVarianceStrategy(objective="minimum_variance"),
    MeanVarianceStrategy(objective="maximum_sharpe"),
    PostModernStrategy(objective="minimum_semivariance"),
    PostModernStrategy(objective="maximum_omega"),
]

result = DynamicBacktester(config, rebalance).run(
    strategies,
    prices=prices,
    benchmark_prices=benchmark,
    optimization_benchmark_prices=benchmark,
)
```

`optimization_benchmark_prices` se usa para estrategias que requieren una
referencia durante la optimización, por ejemplo mínima semivarianza relativa.

## 4. Interpretar `weights_history`

`weights_history` muestra los pesos objetivo calculados en cada rebalanceo.

```python
result.weights_history.head()
```

Ejemplo conceptual:

```text
date        strategy      AAPL   MSFT   GOOG   JPM
2020-01-01  Max Sharpe    0.30   0.20   0.40   0.10
2020-02-03  Max Sharpe    0.25   0.25   0.35   0.15
```

Esto permite auditar cómo fue cambiando la asignación en el tiempo.

## 5. Interpretar Turnover Y Costos

`turnover` mide cuánto cambian los pesos entre rebalanceos.

```python
result.turnover
```

`transaction_costs` muestra el costo descontado del capital.

```python
result.transaction_costs
```

Con `transaction_cost=0.001`, un turnover de `0.40` sobre un portafolio de
1,000,000 implica:

```text
costo = 1,000,000 * 0.40 * 0.001 = 400
```

## 6. Rolling Window Vs Expanding Window

Rolling window:

```python
RebalanceConfig(
    lookback_months=12,
    rebalance_months=1,
    expanding_window=False,
)
```

Usa siempre los últimos 12 meses. Es más sensible a cambios recientes.

Expanding window:

```python
RebalanceConfig(
    lookback_months=12,
    rebalance_months=1,
    expanding_window=True,
)
```

Usa desde `optimization_start` hasta la fecha de rebalanceo. Es más estable, pero
puede reaccionar más lento a cambios de régimen.

## 7. Flujo Completo Del Proyecto

Un flujo orgánico podría ser:

```python
from src.selection import FundamentalSelector
from src.backtesting import BacktestConfig, DynamicBacktester, RebalanceConfig
from src.backtesting import MeanVarianceStrategy

selector = FundamentalSelector(strategy="value")
ranking = selector.rank(universe)
selected = selector.select_top(ranking, top_k=8)

tickers = selected["ticker"].tolist()

config = BacktestConfig(
    tickers=tickers,
    initial_capital=1_000_000,
    optimization_start="2018-01-01",
    backtest_start="2020-01-01",
    end="2025-01-01",
)

rebalance = RebalanceConfig(
    lookback_months=12,
    rebalance_months=3,
)

result = DynamicBacktester(config, rebalance).run(
    MeanVarianceStrategy(objective="minimum_variance"),
    prices=prices[tickers],
)
```

Así el proyecto conserva su secuencia natural:

```text
seleccionar activos -> optimizar pesos -> simular -> medir riesgo
```

## Buenas Prácticas

### Evitar Look-Ahead Bias

La ventana de entrenamiento debe terminar antes de aplicar retornos futuros. El
motor dinámico usa precios anteriores a la fecha de rebalanceo para estimar
pesos.

### Revisar Fechas De Inicio

`backtest_start` debe dejar suficiente historia previa para `lookback_months`.
Si no hay al menos dos precios en la ventana de entrenamiento, el motor levanta
un error.

### Separar Benchmark De Optimización Y Benchmark Pasivo

`benchmark_prices` se usa para comparar evolución.  
`optimization_benchmark_prices` se usa dentro de estrategias que necesitan una
referencia para optimizar.

### Empezar Simple

Antes de usar rebalanceo mensual con costos, conviene validar un backtest
estático y luego pasar al dinámico.
