# Backtesting Module

El módulo `src.backtesting` contiene la capa que evalúa estrategias de
asignación de activos sobre datos históricos. Su papel dentro del proyecto es
tomar un universo ya seleccionado, estimar pesos con los módulos de
optimización y simular cómo habría evolucionado el capital.

El módulo cubre dos formas de evaluación:

- Backtesting estático: optimiza una vez y evalúa una ventana fuera de muestra.
- Backtesting dinámico: reoptimiza pesos periódicamente durante la simulación.

## Estructura

```text
src/backtesting/
├── __init__.py
├── _helpers.py
├── engine_static.py
├── engine_dynamic.py
├── results.py
└── strategies.py
```

## Documentación

- [architecture.md](architecture.md): relación entre motores, estrategias,
  resultados y optimización.
- [api_reference.md](api_reference.md): clases, atributos, parámetros y métodos.
- [workflows.md](workflows.md): casos de uso estáticos y dinámicos.

## Flujo General

El flujo orgánico del proyecto queda así:

```text
selection -> optimization -> backtesting -> risk
```

En backtesting, `optimization` no se reimplementa. Se reutiliza mediante
estrategias:

```python
from src.backtesting import BacktestConfig, Backtester, MeanVarianceStrategy

config = BacktestConfig(
    tickers=["AAPL", "MSFT", "NVDA"],
    initial_capital=1_000_000,
    optimization_start="2018-01-01",
    backtest_start="2020-01-01",
    end="2025-01-01",
)

result = Backtester(config).run(
    MeanVarianceStrategy(objective="minimum_variance"),
    prices=prices,
)
```

Para rebalanceo dinámico:

```python
from src.backtesting import DynamicBacktester, RebalanceConfig

rebalance = RebalanceConfig(
    lookback_months=12,
    rebalance_months=1,
    expanding_window=False,
    transaction_cost=0.001,
)

result = DynamicBacktester(config, rebalance).run(
    MeanVarianceStrategy(objective="maximum_sharpe"),
    prices=prices,
)
```

## Principio De Diseño

El motor de backtesting decide **cuándo** simular y rebalancear. Las estrategias
deciden **cómo** calcular pesos.

Esto permite que el motor dinámico use cualquier estrategia que implemente
`AllocationStrategy.optimize(...)`, incluyendo las actuales:

- `MeanVarianceStrategy`
- `PostModernStrategy`

y futuras estrategias sin cambiar el motor.
