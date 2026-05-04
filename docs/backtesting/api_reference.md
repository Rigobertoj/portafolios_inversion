# Referencia API

Esta referencia describe la API pública de `src.backtesting`.

## `BacktestConfig`

```python
class BacktestConfig(
    tickers,
    initial_capital,
    optimization_start,
    backtest_start,
    end=None,
    price_field="Close",
    benchmark_ticker=None,
    benchmark_label=None,
    reuse_optimization_window=False,
    risk_free_rate=0.0,
    trading_days=252,
)
```

Configuración general compartida por los motores estático y dinámico.

### Parámetros

`tickers` : sequence of `str`  
Activos del universo a simular.

`initial_capital` : `float`  
Capital inicial de la simulación.

`optimization_start` : `str`  
Fecha desde la cual hay datos disponibles para optimizar.

`backtest_start` : `str`  
Fecha desde la cual inicia la evaluación.

`end` : `str`, optional  
Fecha final exclusiva.

`price_field` : `str`, default `"Close"`  
Campo de precio usado si el motor descarga datos.

`benchmark_ticker` : `str`, optional  
Ticker del benchmark pasivo.

`benchmark_label` : `str`, optional  
Nombre usado para mostrar el benchmark.

`reuse_optimization_window` : `bool`, default `False`  
Si es `True`, la ventana de optimización y la de evaluación coinciden.

`risk_free_rate` : `float`, default `0.0`  
Tasa libre de riesgo anual usada en métricas como Sharpe.

`trading_days` : `int`, default `252`  
Días de mercado usados para anualizar.

## `RebalanceConfig`

```python
class RebalanceConfig(
    lookback_months=12,
    rebalance_months=1,
    expanding_window=False,
    transaction_cost=0.0,
)
```

Configuración específica del backtesting dinámico.

### Parámetros

`lookback_months` : `int`, default `12`  
Meses de historia usados para optimizar en cada rebalanceo.

`rebalance_months` : `int`, default `1`  
Frecuencia de rebalanceo en meses.

`expanding_window` : `bool`, default `False`  
Si `False`, usa una ventana móvil de tamaño `lookback_months`. Si `True`, la
ventana inicia en `optimization_start` y crece con cada rebalanceo.

`transaction_cost` : `float`, default `0.0`  
Costo proporcional aplicado al turnover. Por ejemplo, `0.001` representa 0.10%.

## `Backtester`

```python
class Backtester(config)
```

Motor de backtesting estático.

### Parámetros

`config` : `BacktestConfig`

### Métodos

#### `run`

```python
run(strategies, prices=None, benchmark_prices=None, optimization_benchmark_prices=None)
```

Ejecuta el backtest para una o varias estrategias.

Parámetros:

`strategies` : `AllocationStrategy` or sequence of `AllocationStrategy`  
Estrategias a evaluar.

`prices` : `pandas.DataFrame`, optional  
Precios de activos. Si no se entrega, el motor intenta descargarlos.

`benchmark_prices` : `pandas.Series` or `pandas.DataFrame`, optional  
Precios del benchmark pasivo.

`optimization_benchmark_prices` : `pandas.Series` or `pandas.DataFrame`, optional  
Benchmark usado durante optimizaciones que requieren referencia.

Retorna:

`BacktestResult`

## `DynamicBacktester`

```python
class DynamicBacktester(config, rebalance_config)
```

Motor de backtesting dinámico con rebalanceo periódico.

### Parámetros

`config` : `BacktestConfig`  
Configuración general.

`rebalance_config` : `RebalanceConfig`  
Configuración de rebalanceo.

### Métodos

#### `run`

```python
run(strategies, prices=None, benchmark_prices=None, optimization_benchmark_prices=None)
```

Ejecuta una simulación dinámica. En cada fecha de rebalanceo, el motor construye
una ventana de entrenamiento y llama `strategy.optimize(training_prices)`.

Retorna:

`DynamicBacktestResult`

## `AllocationStrategy`

```python
class AllocationStrategy
```

Interfaz común para estrategias que calculan pesos.

### Atributos

`name` : `str`  
Nombre usado en resultados y columnas.

### Métodos

`optimize(prices, optimization_benchmark_prices=None)`  
Calcula pesos con la ventana de precios recibida.

## `MeanVarianceStrategy`

```python
class MeanVarianceStrategy(objective, config=None, name=None)
```

Adaptador para objetivos de media-varianza.

### Parámetros

`objective` : {"minimum_variance", "maximum_sharpe"}  
Objetivo de optimización.

`config` : `OptimizationConfig`, optional  
Configuración del optimizador.

`name` : `str`, optional  
Nombre mostrado en resultados.

## `PostModernStrategy`

```python
class PostModernStrategy(objective, config=None, name=None)
```

Adaptador para objetivos downside/post-modernos.

### Parámetros

`objective` : {"minimum_semivariance", "maximum_omega"}  
Objetivo de optimización.

`config` : `PostModernOptimizationConfig`, optional

`name` : `str`, optional

## Resultados

### `BacktestResult`

Atributos:

- `config`
- `prices_optimization`
- `prices_backtest`
- `strategy_results`
- `returns`
- `evolution`
- `metrics`

### `DynamicBacktestResult`

Atributos:

- `config`
- `rebalance_config`
- `prices`
- `prices_backtest`
- `strategy_results`
- `returns`
- `evolution`
- `metrics`
- `weights_history`
- `turnover`
- `transaction_costs`

### `StrategyAllocation`

Resultado de una estrategia en una fecha de optimización.

Atributos:

- `name`
- `weights`
- `weights_by_ticker`
- `optimization_result`

### `DynamicBacktestStrategyResult`

Resultado de una estrategia dinámica.

Atributos:

- `name`
- `allocations`
- `portfolio_returns`
- `evolution`
- `weights_history`
- `turnover`
- `transaction_costs`

## Alias Públicos

`StaticBacktestEngine` es alias de `Backtester`.

`DynamicBacktestEngine` es alias de `DynamicBacktester`.
