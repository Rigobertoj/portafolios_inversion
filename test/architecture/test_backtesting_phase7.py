import numpy as np
import pandas as pd

from src.backtesting import (
    BacktestConfig,
    Backtester,
    MeanVarianceStrategy,
    PostModernStrategy,
)
from src.optimization import OptimizationConfig


def _sample_prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 102.5, 101.5, 103.0, 104.0, 103.5, 105.0, 106.5, 107.0],
            "BBB": [80.0, 79.5, 80.5, 81.0, 80.0, 79.0, 79.5, 80.0, 79.0, 78.5],
            "CCC": [50.0, 49.5, 50.5, 51.5, 52.0, 51.0, 52.5, 53.5, 53.0, 54.0],
        },
        index=dates,
    )


def _sample_benchmark() -> pd.Series:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.Series(
        [4000.0, 4010.0, 4025.0, 4015.0, 4030.0, 4040.0, 4035.0, 4050.0, 4060.0, 4075.0],
        index=dates,
        name="^GSPC",
    )


def test_new_backtesting_package_runs_mean_variance_strategies_with_legacy_config():
    config = BacktestConfig(
        tickers=["AAA", "BBB", "CCC"],
        initial_capital=1_000_000.0,
        optimization_start="2024-01-01",
        backtest_start="2024-01-07",
        end="2024-01-11",
        benchmark_ticker="^GSPC",
        benchmark_label="Pasiva",
        risk_free_rate=0.02,
    )
    backtester = Backtester(config)

    result = backtester.run(
        [
            MeanVarianceStrategy(objective="minimum_variance"),
            MeanVarianceStrategy(
                objective="maximum_sharpe",
                config=OptimizationConfig(risk_free_rate=0.02),
            ),
        ],
        prices=_sample_prices(),
        benchmark_prices=_sample_benchmark(),
    )

    assert set(result.strategy_results) == {"Min Var", "Max Sharpe"}
    assert "Pasiva" in result.returns.columns
    assert "Pasiva" in result.evolution.columns
    for strategy_result in result.strategy_results.values():
        assert strategy_result.optimization_result.success
        np.testing.assert_allclose(strategy_result.weights.sum(), 1.0)


def test_new_backtesting_package_runs_postmodern_strategy_with_optimization_benchmark():
    config = BacktestConfig(
        tickers=["AAA", "BBB", "CCC"],
        initial_capital=500_000.0,
        optimization_start="2024-01-01",
        backtest_start="2024-01-07",
        end="2024-01-11",
    )
    backtester = Backtester(config)

    result = backtester.run(
        PostModernStrategy(objective="minimum_semivariance"),
        prices=_sample_prices(),
        optimization_benchmark_prices=_sample_benchmark(),
    )

    assert list(result.strategy_results) == ["Min Semivar"]
    strategy_result = result.strategy_results["Min Semivar"]
    assert strategy_result.optimization_result.success
    assert not strategy_result.portfolio_returns.empty
    assert not strategy_result.evolution.empty
