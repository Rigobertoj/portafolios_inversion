import importlib

import numpy as np
import pandas as pd
import pytest

from src.research import AssetsResearch

assets_research_module = importlib.import_module(AssetsResearch.__module__)


def _sample_prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "AAA": [100.0, 102.0, 101.0, 103.0],
            "BBB": [50.0, 49.0, 51.0, 52.0],
        },
        index=dates,
    )


def test_constructor_validation_for_tickers_and_start():
    with pytest.raises(ValueError):
        AssetsResearch([], start="2020-01-01")

    with pytest.raises(ValueError):
        AssetsResearch(["AAA"], start="")


def test_compute_returns_uses_cached_prices_without_download(monkeypatch):
    research = AssetsResearch(["AAA", "BBB"], start="2020-01-01")
    research._set_prices_cache(_sample_prices())

    def _fail_download(*args, **kwargs):
        raise AssertionError("download_prices no debería ejecutarse")

    monkeypatch.setattr(assets_research_module.yf, "download", _fail_download)

    returns = research.compute_returns()
    expected = _sample_prices().pct_change().dropna()

    pd.testing.assert_frame_equal(returns, expected)


def test_download_prices_with_multiindex(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    columns = pd.MultiIndex.from_product([["Close", "Open"], ["AAA", "BBB"]])
    raw = pd.DataFrame(
        [
            [100.0, 50.0, 99.0, 49.5],
            [101.0, 51.0, 100.0, 50.0],
            [102.0, 52.0, 101.0, 51.0],
        ],
        index=dates,
        columns=columns,
    )

    monkeypatch.setattr(assets_research_module.yf, "download", lambda *a, **k: raw)

    research = AssetsResearch(["AAA", "BBB"], start="2020-01-01")
    prices = research.download_prices()

    expected = raw["Close"].dropna()
    pd.testing.assert_frame_equal(prices, expected)


def test_get_returns_with_unknown_ticker_raises_value_error():
    research = AssetsResearch(["AAA", "BBB"], start="2020-01-01")
    research._set_prices_cache(_sample_prices())
    research.compute_returns()

    with pytest.raises(ValueError, match="Tickers not found"):
        research.get_returns(["ZZZ"])


def test_metrics_output_matches_manual_calculation():
    research = AssetsResearch(["AAA", "BBB"], start="2020-01-01")
    research._set_prices_cache(_sample_prices())
    returns = research.compute_returns()

    metrics = research.metrics()

    expected_annual_return = returns.mean() * 252
    expected_annual_vol = returns.std() * np.sqrt(252)

    np.testing.assert_allclose(metrics["annual_return"].values, expected_annual_return.values)
    np.testing.assert_allclose(metrics["annual_volatility"].values, expected_annual_vol.values)

    assert set(metrics.columns) == {
        "annual_return",
        "annual_volatility",
        "annual_return_pct",
        "annual_volatility_pct",
        "skew",
        "vol_over_mean",
        "return_interval_low_pct",
        "return_interval_high_pct",
    }
