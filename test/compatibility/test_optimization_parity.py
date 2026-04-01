import numpy as np
import pandas as pd

from src.asset_allocation import PortfolioOptimization, PortfolioOptimizationPostModern
from src.optimization import (
    MaximumOmegaConfig,
    MeanVarianceOptimizer,
    MinimumSemivarianceConfig,
    MinimumVarianceConfig,
    OptimizationResult,
    PortfolioOptimization as NewPortfolioOptimization,
    PortfolioOptimizationPostModern as NewPortfolioOptimizationPostModern,
    PostModernOptimizationResult,
    PostModernOptimizer,
)
from src.portfolio import Portfolio


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


def _legacy_mean_variance_optimizer() -> PortfolioOptimization:
    portfolio = PortfolioOptimization(
        ["AAA", "BBB", "CCC"],
        start="2024-01-01",
        end="2024-01-07",
        weight=np.array([1 / 3, 1 / 3, 1 / 3]),
    )
    portfolio._set_prices_cache(_sample_prices())
    portfolio._set_returns_cache(_sample_prices().pct_change().dropna())
    return portfolio


def _legacy_postmodern_optimizer() -> PortfolioOptimizationPostModern:
    portfolio = PortfolioOptimizationPostModern(
        ["AAA", "BBB", "CCC"],
        start="2024-01-01",
        end="2024-01-07",
        weight=np.array([1 / 3, 1 / 3, 1 / 3]),
    )
    portfolio._set_prices_cache(_sample_prices())
    portfolio._set_returns_cache(_sample_prices().pct_change().dropna())
    return portfolio


def test_mean_variance_optimizer_updates_shared_portfolio_weights():
    portfolio = Portfolio(
        prices=_sample_prices(),
        weights=np.array([1 / 3, 1 / 3, 1 / 3]),
        name="Demo",
    )
    optimizer = MeanVarianceOptimizer(portfolio=portfolio)

    result = optimizer.optimize_minimum_variance(
        config=MinimumVarianceConfig(),
    )
    legacy_result = _legacy_mean_variance_optimizer().optimize_minimum_variance()

    assert isinstance(result, OptimizationResult)
    assert result.success
    np.testing.assert_allclose(portfolio.weight, result.weights)
    np.testing.assert_allclose(result.weights, legacy_result.weights)
    assert np.isclose(result.variance, legacy_result.variance)
    assert np.isclose(result.volatility, legacy_result.volatility)


def test_direct_mean_variance_compatibility_optimizer_matches_legacy_solver():
    optimizer = NewPortfolioOptimization(
        ["AAA", "BBB", "CCC"],
        start="2024-01-01",
        end="2024-01-07",
        weight=np.array([1 / 3, 1 / 3, 1 / 3]),
    )
    optimizer._set_prices_cache(_sample_prices())
    optimizer._set_returns_cache(_sample_prices().pct_change().dropna())

    result = optimizer.optimize_minimum_variance()
    legacy_result = _legacy_mean_variance_optimizer().optimize_minimum_variance()

    assert isinstance(result, OptimizationResult)
    np.testing.assert_allclose(result.weights, legacy_result.weights)
    assert np.isclose(result.variance, legacy_result.variance)
    assert np.isclose(result.volatility, legacy_result.volatility)


def test_mean_variance_optimizer_supports_maximum_sharpe_configs():
    portfolio = Portfolio(
        prices=_sample_prices(),
        weights=np.array([1 / 3, 1 / 3, 1 / 3]),
        name="Demo",
    )
    optimizer = MeanVarianceOptimizer(portfolio=portfolio)

    result = optimizer.optimize_maximum_sharpe()
    legacy_result = _legacy_mean_variance_optimizer().optimize_maximum_sharpe()

    assert isinstance(result, OptimizationResult)
    assert result.success
    np.testing.assert_allclose(portfolio.weight, result.weights)
    np.testing.assert_allclose(result.weights, legacy_result.weights)
    assert np.isclose(result.sharpe, legacy_result.sharpe, equal_nan=True)


def test_postmodern_optimizer_updates_shared_portfolio_weights():
    portfolio = Portfolio(
        prices=_sample_prices(),
        weights=np.array([1 / 3, 1 / 3, 1 / 3]),
        name="Demo",
    )
    optimizer = PostModernOptimizer(portfolio=portfolio)

    result = optimizer.optimize_minimum_semivariance(
        config=MinimumSemivarianceConfig(),
    )
    legacy_result = _legacy_postmodern_optimizer().optimize_minimum_semivariance()

    assert isinstance(result, PostModernOptimizationResult)
    assert result.success
    np.testing.assert_allclose(portfolio.weight, result.weights)
    np.testing.assert_allclose(result.weights, legacy_result.weights)
    assert np.isclose(result.semivariance, legacy_result.semivariance)
    assert np.isclose(result.downside_risk, legacy_result.downside_risk)


def test_direct_postmodern_compatibility_optimizer_matches_legacy_solver():
    optimizer = NewPortfolioOptimizationPostModern(
        ["AAA", "BBB", "CCC"],
        start="2024-01-01",
        end="2024-01-07",
        weight=np.array([1 / 3, 1 / 3, 1 / 3]),
    )
    optimizer._set_prices_cache(_sample_prices())
    optimizer._set_returns_cache(_sample_prices().pct_change().dropna())

    result = optimizer.optimize_minimum_semivariance()
    legacy_result = _legacy_postmodern_optimizer().optimize_minimum_semivariance()

    assert isinstance(result, PostModernOptimizationResult)
    np.testing.assert_allclose(result.weights, legacy_result.weights)
    assert np.isclose(result.semivariance, legacy_result.semivariance)
    assert np.isclose(result.downside_risk, legacy_result.downside_risk)


def test_postmodern_optimizer_supports_benchmark_relative_semivariance():
    portfolio = Portfolio(
        prices=_sample_prices(),
        weights=np.array([1 / 3, 1 / 3, 1 / 3]),
        name="Demo",
    )
    optimizer = PostModernOptimizer(portfolio=portfolio)
    benchmark_returns = _sample_benchmark_returns()

    result = optimizer.optimize_minimum_semivariance(
        config=MinimumSemivarianceConfig(),
        benchmark_returns=benchmark_returns,
    )
    legacy_result = _legacy_postmodern_optimizer().optimize_minimum_semivariance(
        benchmark_returns=benchmark_returns,
    )

    assert isinstance(result, PostModernOptimizationResult)
    assert result.success
    np.testing.assert_allclose(portfolio.weight, result.weights)
    np.testing.assert_allclose(result.weights, legacy_result.weights)
    assert np.isclose(result.semivariance, legacy_result.semivariance)


def test_postmodern_optimizer_supports_maximum_omega():
    portfolio = Portfolio(
        prices=_sample_prices(),
        weights=np.array([1 / 3, 1 / 3, 1 / 3]),
        name="Demo",
    )
    optimizer = PostModernOptimizer(portfolio=portfolio)

    result = optimizer.optimize_maximum_omega(config=MaximumOmegaConfig())
    legacy_result = _legacy_postmodern_optimizer().optimize_maximum_omega()

    assert isinstance(result, PostModernOptimizationResult)
    assert result.success
    np.testing.assert_allclose(portfolio.weight, result.weights)
    np.testing.assert_allclose(result.weights, legacy_result.weights)
    assert np.isclose(result.omega, legacy_result.omega, equal_nan=True)
