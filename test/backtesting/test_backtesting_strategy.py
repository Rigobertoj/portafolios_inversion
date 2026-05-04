import numpy as np
import pandas as pd
import pytest

from src.backtesting import (
    BacktestConfig,
    Backtester,
    MeanVarianceStrategy,
    PostModernStrategy,
)
from src.optimization import (
    MinimumSemivarianceConfig,
    MinimumVarianceConfig,
    OptimizationConfig,
)
from src.portfolio import Portfolio, PortfolioPerformanceAnalysis


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


def test_backtester_runs_mean_variance_strategies_with_benchmark_from_provided_prices():
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

    strategies = [
        MeanVarianceStrategy(objective="minimum_variance"),
        MeanVarianceStrategy(
            objective="maximum_sharpe",
            config=OptimizationConfig(risk_free_rate=0.02),
        ),
    ]

    result = backtester.run(
        strategies,
        prices=_sample_prices(),
        benchmark_prices=_sample_benchmark(),
    )

    assert list(result.prices_optimization.columns) == config.tickers
    assert result.prices_optimization.index.max() < pd.Timestamp(config.backtest_start)
    assert result.prices_backtest.index.min() >= pd.Timestamp(config.backtest_start)

    assert set(result.strategy_results) == {"Min Var", "Max Sharpe"}
    for strategy_result in result.strategy_results.values():
        assert strategy_result.evolution.name == strategy_result.name
        assert strategy_result.portfolio_returns.name == strategy_result.name
        assert strategy_result.optimization_result.success
        np.testing.assert_allclose(strategy_result.weights.sum(), 1.0)

    assert "Pasiva" in result.evolution.columns
    assert "Pasiva" in result.returns.columns
    assert list(result.metrics.index) == [
        "Rendimiento esperado",
        "Rendimiento realizado",
        "Volatilidad",
        "Ratio de sharpe",
        "Downside risk",
        "Upside risk",
        "Omega",
        "Beta",
        "Alpha de Jensen",
        "Ratio de Treynor",
        "Ratio de Sortino",
    ]
    assert set(result.metrics.columns) == {"Min Var", "Max Sharpe", "Pasiva"}
    assert set(result.pre_back_metrics.columns) == {"Min Var", "Max Sharpe"}
    assert set(result.gross_metrics.columns) == {"Min Var", "Max Sharpe", "Pasiva"}
    assert set(result.net_metrics.columns) == {"Min Var", "Max Sharpe", "Pasiva"}
    assert result.execution_metrics.empty
    np.testing.assert_allclose(
        result.metrics,
        result.net_metrics,
        equal_nan=True,
    )

    min_var = result.strategy_results["Min Var"]
    portfolio = Portfolio(
        prices=result.prices_backtest,
        weights=min_var.weights,
        name=min_var.name,
    )
    portfolio_metrics = PortfolioPerformanceAnalysis(
        portfolio=portfolio,
        benchmark_returns=result.returns["Pasiva"],
    ).metrics_table(risk_free_rate=config.risk_free_rate)
    np.testing.assert_allclose(
        result.metrics["Min Var"],
        portfolio_metrics["value"],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result.pre_back_metrics.loc["Rendimiento esperado", "Min Var"],
        min_var.optimization_result.expected_return,
    )


def test_backtester_runs_post_modern_strategy_from_provided_prices():
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
    )

    assert list(result.strategy_results) == ["Min Semivar"]
    strategy_result = result.strategy_results["Min Semivar"]

    assert strategy_result.optimization_result.success
    assert not strategy_result.portfolio_returns.empty
    assert not strategy_result.evolution.empty
    np.testing.assert_allclose(strategy_result.weights.sum(), 1.0)
    np.testing.assert_allclose(
        result.metrics.loc["Rendimiento realizado", "Min Semivar"],
        result.evolution.iloc[-1, 0] / config.initial_capital - 1.0,
    )
    assert "Rendimiento esperado" in result.pre_back_metrics.index
    assert "Rendimiento esperado" in result.gross_metrics.index
    assert "Rendimiento esperado" in result.net_metrics.index


def test_backtester_runs_post_modern_strategy_with_optimization_benchmark():
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


def test_backtester_caps_infeasible_minimum_return_constraints():
    config = BacktestConfig(
        tickers=["AAA", "BBB", "CCC"],
        initial_capital=500_000.0,
        optimization_start="2024-01-01",
        backtest_start="2024-01-07",
        end="2024-01-11",
    )
    requested_return = 10.0
    backtester = Backtester(config)

    result = backtester.run(
        [
            MeanVarianceStrategy(
                objective="minimum_variance",
                config=MinimumVarianceConfig(minimum_return=requested_return),
            ),
            PostModernStrategy(
                objective="minimum_semivariance",
                config=MinimumSemivarianceConfig(minimum_return=requested_return),
            ),
        ],
        prices=_sample_prices(),
    )

    assert set(result.strategy_results) == {"Min Var", "Min Semivar"}
    for strategy_result in result.strategy_results.values():
        optimization_result = strategy_result.optimization_result

        assert optimization_result.success
        assert optimization_result.minimum_return_was_capped
        assert optimization_result.requested_minimum_return == requested_return
        assert optimization_result.effective_minimum_return < requested_return
        assert optimization_result.maximum_feasible_return < requested_return
        assert (
            optimization_result.effective_minimum_return
            < optimization_result.maximum_feasible_return
        )
        assert (
            optimization_result.expected_return
            >= optimization_result.effective_minimum_return - 1e-8
        )
        np.testing.assert_allclose(strategy_result.weights.sum(), 1.0)


def test_backtester_can_reuse_optimization_window_for_in_sample_evaluation():
    config = BacktestConfig(
        tickers=["AAA", "BBB", "CCC"],
        initial_capital=250_000.0,
        optimization_start="2024-01-01",
        backtest_start="2024-01-01",
        end="2024-01-11",
        benchmark_ticker="^GSPC",
        benchmark_label="Pasiva",
        reuse_optimization_window=True,
    )
    backtester = Backtester(config)

    result = backtester.run(
        MeanVarianceStrategy(objective="minimum_variance"),
        prices=_sample_prices(),
        benchmark_prices=_sample_benchmark(),
    )

    pd.testing.assert_frame_equal(result.prices_optimization, result.prices_backtest)
    assert result.prices_optimization.index.min() == pd.Timestamp(config.optimization_start)
    assert list(result.strategy_results) == ["Min Var"]
    assert "Pasiva" in result.returns.columns


def test_backtest_config_rejects_equal_dates_without_reuse_optimization_window():
    with pytest.raises(
        ValueError,
        match="backtest_start must be later than optimization_start.",
    ):
        BacktestConfig(
            tickers=["AAA", "BBB", "CCC"],
            initial_capital=100_000.0,
            optimization_start="2024-01-01",
            backtest_start="2024-01-01",
            end="2024-01-11",
        )


def test_backtest_config_requires_matching_dates_when_reusing_optimization_window():
    with pytest.raises(
        ValueError,
        match="backtest_start must match optimization_start when reuse_optimization_window is enabled.",
    ):
        BacktestConfig(
            tickers=["AAA", "BBB", "CCC"],
            initial_capital=100_000.0,
            optimization_start="2024-01-01",
            backtest_start="2024-01-07",
            end="2024-01-11",
            reuse_optimization_window=True,
        )
