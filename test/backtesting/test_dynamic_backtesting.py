import numpy as np
import pandas as pd

from src.backtesting import (
    AllocationStrategy,
    BacktestConfig,
    DynamicBacktester,
    MeanVarianceStrategy,
    RebalanceConfig,
    StrategyAllocation,
)
from src.optimization import MinimumVarianceConfig


class CyclingStrategy(AllocationStrategy):
    """Deterministic strategy used to test dynamic orchestration."""

    def __init__(self):
        super().__init__(name="Cycling")
        self.calls = 0

    @property
    def default_name(self):
        return "Cycling"

    def optimize(self, prices, optimization_benchmark_prices=None):
        del optimization_benchmark_prices
        weights_by_call = [
            np.array([0.70, 0.30]),
            np.array([0.40, 0.60]),
            np.array([0.55, 0.45]),
        ]
        weights = weights_by_call[min(self.calls, len(weights_by_call) - 1)]
        self.calls += 1
        return StrategyAllocation(
            name=self.name,
            weights=weights,
            weights_by_ticker=pd.Series(weights, index=prices.columns, name=self.name),
            optimization_result={"call": self.calls},
        )


def _sample_prices():
    dates = pd.bdate_range("2024-01-01", "2024-04-15")
    step = np.arange(len(dates), dtype=float)
    return pd.DataFrame(
        {
            "AAA": 100.0 + step * 0.20,
            "BBB": 80.0 + np.sin(step / 4.0) + step * 0.05,
        },
        index=dates,
    )


def test_dynamic_backtester_rebalances_and_records_turnover_and_costs():
    config = BacktestConfig(
        tickers=["AAA", "BBB"],
        initial_capital=100_000.0,
        optimization_start="2024-01-01",
        backtest_start="2024-02-01",
        end="2024-04-16",
    )
    rebalance_config = RebalanceConfig(
        lookback_months=1,
        rebalance_months=1,
        transaction_cost=0.001,
    )
    strategy = CyclingStrategy()
    result = DynamicBacktester(config, rebalance_config).run(
        strategy,
        prices=_sample_prices(),
    )

    assert strategy.calls >= 2
    assert list(result.strategy_results) == ["Cycling"]
    assert "Cycling" in result.returns.columns
    assert "Cycling" in result.evolution.columns
    assert not result.evolution.empty

    weights = result.weights_history
    assert "strategy" in weights.columns
    assert {"AAA", "BBB"}.issubset(weights.columns)
    assert len(weights) >= 2

    turnover = result.turnover["Cycling"]
    costs = result.transaction_costs["Cycling"]
    assert turnover.iloc[0] == 0.0
    assert costs.iloc[0] == 0.0
    assert turnover.iloc[1] > 0.0
    assert costs.iloc[1] > 0.0
    assert not result.pre_back_metrics.empty
    assert "Rendimiento esperado" in result.pre_back_metrics.index
    assert "Rendimiento esperado" in result.gross_metrics.index
    assert "Rendimiento esperado" in result.net_metrics.index
    assert result.metrics.loc["Turnover promedio", "Cycling"] == turnover.mean()
    assert result.metrics.loc["Turnover acumulado", "Cycling"] == turnover.sum()
    assert result.metrics.loc["Costos de transacción", "Cycling"] == costs.sum()
    assert result.execution_metrics.loc["Turnover promedio", "Cycling"] == turnover.mean()
    assert result.execution_metrics.loc["Turnover acumulado", "Cycling"] == turnover.sum()
    assert result.execution_metrics.loc["Costos de transacción", "Cycling"] == costs.sum()
    assert (
        result.metrics.loc["Impacto de costos", "Cycling"]
        == costs.sum() / config.initial_capital
    )
    assert (
        result.gross_metrics.loc["Rendimiento realizado", "Cycling"]
        > result.net_metrics.loc["Rendimiento realizado", "Cycling"]
    )


def test_dynamic_backtester_caps_minimum_return_per_rebalance_window():
    config = BacktestConfig(
        tickers=["AAA", "BBB"],
        initial_capital=100_000.0,
        optimization_start="2024-01-01",
        backtest_start="2024-02-01",
        end="2024-04-16",
    )
    rebalance_config = RebalanceConfig(
        lookback_months=1,
        rebalance_months=1,
        transaction_cost=0.001,
    )
    requested_return = 10.0

    result = DynamicBacktester(config, rebalance_config).run(
        MeanVarianceStrategy(
            objective="minimum_variance",
            config=MinimumVarianceConfig(minimum_return=requested_return),
        ),
        prices=_sample_prices(),
    )

    allocations = result.strategy_results["Min Var"].allocations
    assert len(allocations) >= 2
    for allocation in allocations.values():
        optimization_result = allocation.optimization_result

        assert optimization_result.success
        assert optimization_result.minimum_return_was_capped
        assert optimization_result.requested_minimum_return == requested_return
        assert optimization_result.effective_minimum_return < requested_return
        assert optimization_result.maximum_feasible_return < requested_return
        assert (
            optimization_result.expected_return
            >= optimization_result.effective_minimum_return - 1e-8
        )


def test_rebalance_config_validates_positive_windows():
    try:
        RebalanceConfig(lookback_months=0)
    except ValueError as exc:
        assert "lookback_months" in str(exc)
    else:
        raise AssertionError("RebalanceConfig should reject lookback_months=0")
