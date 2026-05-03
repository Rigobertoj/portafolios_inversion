"""Shared helpers used by the backtesting package.

The functions in this module normalize user-supplied price data, align time
windows, infer portfolio constructor dates, and build in-memory `Portfolio`
instances for strategy optimization.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..portfolio.portfolio import Portfolio


def default_weights(n_assets: int) -> np.ndarray:
    """
    Return an equally weighted portfolio vector.

    Parameters
    ----------
    n_assets : int
        Number of assets in the portfolio.

    Returns
    -------
    numpy.ndarray
        Vector of length `n_assets` where every asset receives weight
        `1 / n_assets`.
    """
    return np.ones(n_assets, dtype=float) / float(n_assets)


def normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean a price table used by the backtester.

    Parameters
    ----------
    prices : pandas.DataFrame
        Raw asset prices indexed by date, with one column per asset.

    Returns
    -------
    pandas.DataFrame
        Sorted copy with rows containing missing values removed.

    Raises
    ------
    TypeError
        If `prices` is not a pandas DataFrame.
    ValueError
        If no complete observations or no asset columns remain.
    """
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
    """
    Normalize benchmark prices into a clean named series.

    Parameters
    ----------
    prices : pandas.Series or pandas.DataFrame
        Benchmark price history. DataFrames must contain exactly one column.
    label : str
        Name assigned to the returned benchmark series.

    Returns
    -------
    pandas.Series
        Sorted benchmark prices with missing values removed and `name` set to
        `label`.

    Raises
    ------
    TypeError
        If `prices` is neither a pandas Series nor a pandas DataFrame.
    ValueError
        If a DataFrame contains more than one column, or if no observations
        remain after cleaning.
    """
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
    """
    Return a copy restricted to a half-open time window.

    Parameters
    ----------
    prices : pandas.DataFrame or pandas.Series
        Time-indexed price data to slice.
    start : str or pandas.Timestamp
        Inclusive window start.
    end : str, optional
        Exclusive window end. If omitted, the slice includes all observations
        from `start` onward.

    Returns
    -------
    pandas.DataFrame or pandas.Series
        Copy of `prices` restricted to `[start, end)`.
    """
    start_timestamp = pd.Timestamp(start)
    mask = prices.index >= start_timestamp

    if end is not None:
        end_timestamp = pd.Timestamp(end)
        mask &= prices.index < end_timestamp

    return prices.loc[mask].copy()


def window_bounds(prices: pd.DataFrame) -> tuple[str, str]:
    """
    Infer constructor dates for in-memory portfolio instances.

    Parameters
    ----------
    prices : pandas.DataFrame
        Clean price table indexed by date.

    Returns
    -------
    tuple of str
        ISO date strings `(start, end)`, where `end` is one day after the last
        available price date so it can be used as an exclusive boundary.
    """
    start = pd.Timestamp(prices.index.min()).date().isoformat()
    end = (pd.Timestamp(prices.index.max()) + pd.Timedelta(days=1)).date().isoformat()
    return start, end


def build_portfolio(
    prices: pd.DataFrame,
    *,
    initial_weights: Optional[Iterable[float]] = None,
    name: Optional[str] = None,
) -> Portfolio:
    """
    Build a `Portfolio` instance from an in-memory price table.

    Parameters
    ----------
    prices : pandas.DataFrame
        Asset prices indexed by date, with one column per ticker.
    initial_weights : iterable of float, optional
        Initial weights passed to the portfolio. If omitted, equal weights are
        used.
    name : str, optional
        Portfolio name forwarded to the `Portfolio` constructor.

    Returns
    -------
    Portfolio
        Portfolio object with prices, returns, tickers, weights, inferred date
        bounds, and `price_field="Close"`.

    Raises
    ------
    TypeError
        If `prices` is not a pandas DataFrame.
    ValueError
        If the normalized price table is empty or has no asset columns.
    """
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
