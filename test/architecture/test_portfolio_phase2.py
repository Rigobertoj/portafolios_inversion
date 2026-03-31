import numpy as np
import pandas as pd

from src.portfolio import Portfolio
from src.research import AssetsResearch


def _sample_prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    return pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 98.0, 99.5, 103.0, 104.0],
            "BBB": [80.0, 79.5, 79.0, 81.0, 80.5, 80.0],
            "CCC": [50.0, 50.5, 49.0, 51.0, 52.0, 51.5],
        },
        index=dates,
    )

def test_portfolio_from_research_reuses_cached_data():
    prices = _sample_prices()
    research = AssetsResearch(
        ["AAA", "BBB", "CCC"],
        start="2024-01-01",
        end="2024-01-07",
    )
    research._set_prices_cache(prices)
    research._set_returns_cache(prices.pct_change().dropna())

    portfolio = Portfolio.from_research(
        research,
        weights=np.array([0.5, 0.3, 0.2]),
        name="Demo",
    )

    assert portfolio.tickers == ["AAA", "BBB", "CCC"]
    assert portfolio.start == "2024-01-01"
    assert portfolio.end == "2024-01-07"
    assert portfolio.price_field == "Close"
    assert portfolio.name == "Demo"
    pd.testing.assert_frame_equal(portfolio.asset_prices(), prices)
    pd.testing.assert_frame_equal(
        portfolio.asset_returns(),
        prices.pct_change().dropna(),
    )


def test_portfolio_computes_weighted_returns_and_wealth_index():
    prices = _sample_prices()
    weights = np.array([0.5, 0.3, 0.2])

    portfolio = Portfolio(
        prices=prices,
        weights=weights,
        name="Demo",
    )

    manual_returns = prices.pct_change().dropna() @ weights
    manual_returns.name = "Demo"
    pd.testing.assert_series_equal(portfolio.portfolio_returns(), manual_returns)

    wealth = portfolio.wealth_index(initial_value=100.0)
    expected_wealth = 100.0 * (1.0 + manual_returns).cumprod()
    expected_wealth.name = "Demo"
    pd.testing.assert_series_equal(wealth, expected_wealth)
    assert np.isclose(portfolio.realized_return(), (1.0 + manual_returns).prod() - 1.0)
