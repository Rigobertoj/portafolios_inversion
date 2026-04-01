"""Shared helpers used by the composition-based backtesting layer."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..portfolio.portfolio import Portfolio


def default_weights(n_assets: int) -> np.ndarray:
    """Return an equally weighted feasible portfolio."""
    return np.ones(n_assets, dtype=float) / float(n_assets)


def normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a price table used by the backtester."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")

    cleaned = prices.sort_index().dropna().copy()
    if cleaned.empty:
        raise ValueError("prices must contain at least one complete observation.")
    if len(cleaned.columns) == 0:
        raise ValueError("prices must contain at least one asset column.")
    return cleaned


def normalize_benchmark_prices(
    prices: pd.Series | pd.DataFrame,
    label: str,
) -> pd.Series:
    """Normalize benchmark prices into a clean named series."""
    if isinstance(prices, pd.Series):
        benchmark = prices.sort_index().dropna().copy()
    elif isinstance(prices, pd.DataFrame):
        cleaned = prices.sort_index().dropna().copy()
        if cleaned.shape[1] != 1:
            raise ValueError("benchmark_prices must contain exactly one column.")
        benchmark = cleaned.iloc[:, 0]
    else:
        raise TypeError("benchmark_prices must be a pandas Series or DataFrame.")

    if benchmark.empty:
        raise ValueError("benchmark_prices must contain at least one observation.")

    benchmark.name = label
    return benchmark


def slice_time_window(
    prices: pd.DataFrame | pd.Series,
    *,
    start: str | pd.Timestamp,
    end: Optional[str] = None,
) -> pd.DataFrame | pd.Series:
    """Return a copy restricted to the configured [start, end) window."""
    start_timestamp = pd.Timestamp(start)
    mask = prices.index >= start_timestamp

    if end is not None:
        end_timestamp = pd.Timestamp(end)
        mask &= prices.index < end_timestamp

    return prices.loc[mask].copy()


def window_bounds(prices: pd.DataFrame) -> tuple[str, str]:
    """Infer constructor dates for in-memory portfolio instances."""
    start = pd.Timestamp(prices.index.min()).date().isoformat()
    end = (pd.Timestamp(prices.index.max()) + pd.Timedelta(days=1)).date().isoformat()
    return start, end


def build_portfolio(
    prices: pd.DataFrame,
    *,
    initial_weights: Optional[Iterable[float]] = None,
    name: Optional[str] = None,
) -> Portfolio:
    """Build a `Portfolio` instance from an in-memory price table."""
    normalized_prices = normalize_prices(prices)
    tickers = list(normalized_prices.columns)
    weights = default_weights(len(tickers))
    if initial_weights is not None:
        weights = np.asarray(initial_weights, dtype=float)

    start, end = window_bounds(normalized_prices)
    return Portfolio(
        prices=normalized_prices,
        returns=normalized_prices.pct_change().dropna(),
        tickers=tickers,
        weights=weights,
        start=start,
        end=end,
        price_field="Close",
        name=name,
    )
