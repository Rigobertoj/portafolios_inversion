import numpy as np
import pandas as pd

from src.asset_allocation import PortfolioPostModernMetrics


def _sample_prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "AAA": [100.0, 98.0, 99.0, 97.0, 101.0],
            "BBB": [50.0, 49.0, 48.0, 49.5, 47.0],
            "CCC": [30.0, 31.0, 29.0, 30.0, 28.0],
        },
        index=dates,
    )


def _sample_benchmark_returns() -> pd.Series:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    benchmark = pd.Series(
        [200.0, 201.0, 199.5, 202.0, 201.5],
        index=dates,
        name="BM",
    )
    return benchmark.pct_change().dropna()


def test_returns_below_replaces_non_downside_values_with_zero():
    metrics = PortfolioPostModernMetrics(
        ["AAA", "BBB", "CCC"],
        start="2020-01-01",
        weight=np.array([1 / 3, 1 / 3, 1 / 3]),
    )
    metrics._set_prices_cache(_sample_prices())

    returns = metrics.compute_returns()
    expected = returns.where(returns < 0.0, 0.0)

    filtered = metrics.returns_below()

    pd.testing.assert_frame_equal(filtered, expected)
    pd.testing.assert_frame_equal(metrics.returns_down, expected)


def test_returns_below_supports_benchmark_reference_with_threshold_spread():
    metrics = PortfolioPostModernMetrics(
        ["AAA", "BBB", "CCC"],
        start="2020-01-01",
        weight=np.array([1 / 3, 1 / 3, 1 / 3]),
    )
    metrics._set_prices_cache(_sample_prices())

    returns = metrics.compute_returns()
    benchmark_returns = _sample_benchmark_returns().loc[returns.index]
    threshold = 0.002

    expected = returns.sub(benchmark_returns + threshold, axis=0)
    expected = expected.where(expected < 0.0, 0.0)

    filtered = metrics.returns_below(
        threshold=threshold,
        benchmark_returns=benchmark_returns,
    )

    pd.testing.assert_frame_equal(filtered, expected)


def test_downside_risk_reuses_the_last_selected_threshold():
    metrics = PortfolioPostModernMetrics(
        ["AAA", "BBB", "CCC"],
        start="2020-01-01",
        weight=np.array([1 / 3, 1 / 3, 1 / 3]),
    )
    metrics._set_prices_cache(_sample_prices())

    returns = metrics.compute_returns()
    threshold = -0.015
    threshold_adjusted = returns - threshold
    expected = (
        threshold_adjusted.where(threshold_adjusted < 0.0, 0.0).std()
        * np.sqrt(252.0)
    ).to_numpy(dtype=float)

    metrics.returns_below(threshold)
    risk = metrics.downside_risk()

    np.testing.assert_allclose(risk, expected)
    np.testing.assert_allclose(metrics.shortfall_risk, expected)


def test_semivariance_matrix_matches_manual_formula():
    metrics = PortfolioPostModernMetrics(
        ["AAA", "BBB", "CCC"],
        start="2020-01-01",
        weight=np.array([1 / 3, 1 / 3, 1 / 3]),
    )
    metrics._set_prices_cache(_sample_prices())

    returns = metrics.compute_returns()
    downside_risk = returns.where(returns < 0.0, 0.0).std() * np.sqrt(252.0)
    downside_risk_matrix = pd.DataFrame(
        np.outer(downside_risk.to_numpy(dtype=float), downside_risk.to_numpy(dtype=float)),
        index=downside_risk.index,
        columns=downside_risk.index,
    )
    expected = downside_risk_matrix * returns.corr()

    semivariance = metrics.semivarianza_down()

    pd.testing.assert_frame_equal(semivariance, expected)
