import numpy as np
import pandas as pd

from src.portfolio import Portfolio
from src.risk import (
    PortfolioDrawdownAnalysis,
    PortfolioRelativeRisk,
    PortfolioTailRisk,
    PortfolioVolatilityAnalysis,
    RiskAnalyzer,
)


def _sample_prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    return pd.DataFrame(
        {
            "AAA": [100.0, 102.0, 99.0, 101.0, 98.0, 100.0],
            "BBB": [80.0, 79.0, 81.0, 82.0, 81.0, 83.0],
            "CCC": [50.0, 51.0, 49.5, 50.5, 52.0, 51.0],
        },
        index=dates,
    )


def _sample_benchmark_prices() -> pd.Series:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    return pd.Series(
        [4000.0, 4020.0, 3995.0, 4010.0, 3980.0, 3990.0],
        index=dates,
        name="^GSPC",
    )


def test_risk_exports_are_available_from_public_api():
    assert PortfolioDrawdownAnalysis.__module__ == "src.risk.drawdown"
    assert PortfolioTailRisk.__module__ == "src.risk.var_cvar"
    assert PortfolioRelativeRisk.__module__ == "src.risk.tracking"
    assert PortfolioVolatilityAnalysis.__module__ == "src.risk.volatility"
    assert RiskAnalyzer.__module__ == "src.risk.report"


def test_drawdown_analysis_matches_manual_formulas():
    portfolio = Portfolio(
        prices=_sample_prices(),
        weights=np.array([0.5, 0.3, 0.2]),
        name="Demo",
    )
    analysis = PortfolioDrawdownAnalysis(portfolio=portfolio)

    portfolio_returns = portfolio.portfolio_returns()
    expected_wealth = (1.0 + portfolio_returns).cumprod()
    expected_wealth.name = "Demo"
    expected_peak = expected_wealth.cummax()
    expected_peak.name = "Demo"
    expected_drawdown = expected_wealth / expected_peak - 1.0
    expected_drawdown.name = "Demo"

    pd.testing.assert_series_equal(analysis.wealth_index(), expected_wealth)
    pd.testing.assert_series_equal(analysis.running_peak(), expected_peak)
    pd.testing.assert_series_equal(analysis.drawdown_series(), expected_drawdown)
    assert np.isclose(analysis.max_drawdown(), expected_drawdown.min())


def test_tail_risk_matches_manual_formulas():
    portfolio = Portfolio(
        prices=_sample_prices(),
        weights=np.array([0.5, 0.3, 0.2]),
        name="Demo",
    )
    tail_risk = PortfolioTailRisk(portfolio=portfolio)

    portfolio_returns = portfolio.portfolio_returns()
    cutoff = float(portfolio_returns.quantile(0.05))
    expected_var = max(-cutoff, 0.0)
    expected_cvar = max(-portfolio_returns[portfolio_returns <= cutoff].mean(), 0.0)

    assert np.isclose(tail_risk.historical_var(confidence_level=0.95), expected_var)
    assert np.isclose(
        tail_risk.historical_cvar(confidence_level=0.95),
        expected_cvar,
    )


def test_volatility_analysis_matches_manual_ewma_formulas():
    portfolio = Portfolio(
        prices=_sample_prices(),
        weights=np.array([0.5, 0.3, 0.2]),
        name="Demo",
    )
    volatility = PortfolioVolatilityAnalysis(
        portfolio=portfolio,
        trading_days=252,
    )
    analyzer = RiskAnalyzer(portfolio=portfolio)

    portfolio_returns = portfolio.portfolio_returns()
    squared_returns = portfolio_returns.pow(2).to_numpy(dtype=float)

    expected_variance = np.empty(len(squared_returns), dtype=float)
    expected_variance[0] = squared_returns[0]
    for idx in range(1, len(squared_returns)):
        expected_variance[idx] = (
            0.94 * expected_variance[idx - 1] + 0.06 * squared_returns[idx]
        )

    expected_variance_series = pd.Series(
        expected_variance,
        index=portfolio_returns.index,
        name="Demo",
    )
    expected_volatility_series = (expected_variance_series * 252.0).pow(0.5)
    expected_volatility_series.name = "Demo"
    expected_historical_variance = portfolio_returns.var() * 252.0
    expected_historical_volatility = portfolio_returns.std() * np.sqrt(252.0)

    pd.testing.assert_series_equal(
        volatility.ewma_variance_series(decay=0.94),
        expected_variance_series,
    )
    pd.testing.assert_series_equal(
        volatility.ewma_volatility_series(decay=0.94),
        expected_volatility_series,
    )
    assert np.isclose(volatility.historical_variance(), expected_historical_variance)
    assert np.isclose(
        volatility.historical_volatility(),
        expected_historical_volatility,
    )
    assert np.isclose(
        analyzer.latest_ewma_variance(decay=0.94),
        expected_variance_series.iloc[-1],
    )
    assert np.isclose(
        analyzer.latest_ewma_volatility(decay=0.94),
        expected_volatility_series.iloc[-1],
    )
    assert np.isclose(
        analyzer.historical_volatility(),
        expected_historical_volatility,
    )


def test_relative_risk_and_summary_match_manual_formulas():
    portfolio = Portfolio(
        prices=_sample_prices(),
        weights=np.array([0.5, 0.3, 0.2]),
        name="Demo",
    )
    benchmark_prices = _sample_benchmark_prices()
    relative_risk = PortfolioRelativeRisk(
        portfolio=portfolio,
        benchmark_prices=benchmark_prices,
    )
    analyzer = RiskAnalyzer(
        portfolio=portfolio,
        benchmark_prices=benchmark_prices,
    )

    portfolio_returns = portfolio.portfolio_returns()
    benchmark_returns = benchmark_prices.pct_change().dropna()
    aligned = pd.concat(
        [
            portfolio_returns.rename("portfolio"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    active_returns = aligned["portfolio"] - aligned["benchmark"]
    expected_tracking_error = active_returns.std() * np.sqrt(252.0)
    expected_information_ratio = (active_returns.mean() * 252.0) / expected_tracking_error

    pd.testing.assert_series_equal(relative_risk.active_returns(), active_returns.rename("Demo"))
    assert np.isclose(relative_risk.tracking_error(), expected_tracking_error)
    assert np.isclose(relative_risk.information_ratio(), expected_information_ratio)

    summary = analyzer.summary(confidence_level=0.95)
    assert list(summary.index) == [
        "Max Drawdown",
        "VaR 95%",
        "CVaR 95%",
        "Tracking Error",
        "Information Ratio",
    ]
    assert np.isclose(summary.loc["Tracking Error", "value"], expected_tracking_error)
    assert np.isclose(
        summary.loc["Information Ratio", "value"],
        expected_information_ratio,
    )


def test_risk_analyzer_summary_gracefully_handles_missing_benchmark():
    portfolio = Portfolio(
        prices=_sample_prices(),
        weights=np.array([0.5, 0.3, 0.2]),
        name="Demo",
    )
    analyzer = RiskAnalyzer(portfolio=portfolio)

    summary = analyzer.summary(confidence_level=0.95)

    assert np.isnan(summary.loc["Tracking Error", "value"])
    assert np.isnan(summary.loc["Information Ratio", "value"])
