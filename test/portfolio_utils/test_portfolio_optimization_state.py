import numpy as np
import pandas as pd

from src.asset_allocation import PortfolioOptimization, PortfolioOptimizationPostModern


def _sample_prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    return pd.DataFrame(
        {
            "AAA": [100.0, 98.0, 99.5, 97.0, 101.0, 102.0],
            "BBB": [50.0, 49.0, 48.5, 49.5, 47.0, 46.5],
            "CCC": [30.0, 31.0, 29.0, 30.5, 28.0, 29.0],
        },
        index=dates,
    )


def _sample_benchmark_returns() -> pd.Series:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    benchmark = pd.Series(
        [200.0, 201.5, 202.0, 201.0, 203.0, 204.0],
        index=dates,
        name="BM",
    )
    return benchmark.pct_change().dropna()


def test_minimum_variance_updates_instance_weight_after_successful_optimization():
    portfolio = PortfolioOptimization(
        ["AAA", "BBB", "CCC"],
        start="2020-01-01",
        weight=np.array([1 / 3, 1 / 3, 1 / 3]),
    )
    portfolio._set_prices_cache(_sample_prices())
    portfolio.compute_returns()

    result = portfolio.optimize_minimum_variance()

    assert result.success
    np.testing.assert_allclose(portfolio.weight, result.weights)
    assert np.isclose(portfolio.portfolio_variance(), result.variance)
    assert np.isclose(portfolio.portfolio_annual_volatility(), result.volatility)


def test_minimum_semivariance_updates_instance_weight_after_successful_optimization():
    portfolio = PortfolioOptimizationPostModern(
        ["AAA", "BBB", "CCC"],
        start="2020-01-01",
        weight=np.array([1 / 3, 1 / 3, 1 / 3]),
    )
    portfolio._set_prices_cache(_sample_prices())
    portfolio.compute_returns()

    result = portfolio.optimize_minimum_semivariance()

    assert result.success
    np.testing.assert_allclose(portfolio.weight, result.weights)
    assert np.isclose(portfolio.portfolio_semivariance(), result.semivariance)
    assert np.isclose(portfolio.portfolio_downside_risk(), result.downside_risk)


def test_minimum_semivariance_supports_benchmark_return_reference():
    portfolio = PortfolioOptimizationPostModern(
        ["AAA", "BBB", "CCC"],
        start="2020-01-01",
        weight=np.array([1 / 3, 1 / 3, 1 / 3]),
    )
    portfolio._set_prices_cache(_sample_prices())
    portfolio.compute_returns()
    benchmark_returns = _sample_benchmark_returns()

    result = portfolio.optimize_minimum_semivariance(
        benchmark_returns=benchmark_returns,
    )

    assert result.success
    np.testing.assert_allclose(portfolio.weight, result.weights)
    assert np.isclose(
        portfolio.portfolio_semivariance(
            weight=result.weights,
            benchmark_returns=benchmark_returns,
        ),
        result.semivariance,
    )
