"""Forward-return target builders for learned fundamental selection.

The functions in this module align point-in-time fundamental observations with
future price and dividend outcomes. Targets are deliberately built after a
reporting lag so a model cannot learn from financial statements before they
would plausibly have been available to an investor.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from .fundamentals import FundamentalData


def _records_by_ticker(
    records: Mapping[str, FundamentalData] | Iterable[FundamentalData],
) -> dict[str, FundamentalData]:
    if isinstance(records, Mapping):
        values = records.values()
    else:
        values = records
    return {record.ticker.upper(): record for record in values}


def _market_series(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series, pd.Series) or series.empty:
        return pd.Series(dtype=float, name=name)

    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return pd.Series(dtype=float, name=name)

    index = pd.to_datetime(clean.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert(None)
    clean = pd.Series(clean.to_numpy(dtype=float), index=index, name=name)
    return clean.sort_index()


def price_at_or_after(prices: pd.Series, date: object) -> float:
    """Return the first observed price at or after `date`."""
    prices_clean = _market_series(prices, getattr(prices, "name", "price") or "price")
    if prices_clean.empty:
        return np.nan

    date_ts = pd.Timestamp(date)
    eligible = prices_clean[prices_clean.index >= date_ts]
    if eligible.empty:
        return np.nan
    return float(eligible.iloc[0])


def dividends_between(dividends: pd.Series, start: object, end: object) -> float:
    """Return cash dividends paid after `start` and on or before `end`."""
    dividends_clean = _market_series(dividends, "Dividends")
    if dividends_clean.empty:
        return 0.0

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    window = dividends_clean[(dividends_clean.index > start_ts) & (dividends_clean.index <= end_ts)]
    if window.empty:
        return 0.0
    return float(window.sum())


def target_column_names(horizon_months: int) -> dict[str, str]:
    """Return canonical target column names for a forward horizon."""
    horizon = int(horizon_months)
    return {
        "price": f"forward_price_return_{horizon}m",
        "dividend": f"forward_dividend_return_{horizon}m",
        "total": f"forward_total_return_{horizon}m",
        "dividends": f"forward_dividends_{horizon}m",
    }


def build_forward_return_targets(
    metric_history: pd.DataFrame,
    records: Mapping[str, FundamentalData] | Iterable[FundamentalData],
    horizon_months: int = 12,
    reporting_lag_days: int = 60,
) -> pd.DataFrame:
    """Append forward price, dividend, and total-return targets.

    Parameters
    ----------
    metric_history : pandas.DataFrame
        Long-format fundamental metrics with `ticker` and `period` columns.
    records : mapping or iterable of FundamentalData
        Raw records containing prices and optional dividends by ticker.
    horizon_months : int, default 12
        Forward horizon used for the target.
    reporting_lag_days : int, default 60
        Delay between the statement period date and the assumed information
        availability date.

    Returns
    -------
    pandas.DataFrame
        Copy of `metric_history` with entry/exit dates, prices, dividends, and
        canonical forward-return targets.
    """
    if horizon_months < 1:
        raise ValueError("horizon_months must be >= 1.")
    if reporting_lag_days < 0:
        raise ValueError("reporting_lag_days must be >= 0.")
    if metric_history.empty:
        return metric_history.copy()
    required = {"ticker", "period"}
    missing = required.difference(metric_history.columns)
    if missing:
        raise ValueError(f"metric_history must contain columns: {sorted(missing)}")

    names = target_column_names(horizon_months)
    by_ticker = _records_by_ticker(records)
    panel = metric_history.copy()
    panel["ticker"] = panel["ticker"].astype(str).str.upper()
    panel["period"] = pd.to_datetime(panel["period"])
    panel["available_at"] = panel["period"] + pd.to_timedelta(reporting_lag_days, unit="D")
    panel["target_end_at"] = panel["available_at"] + pd.DateOffset(months=int(horizon_months))

    entry_prices: list[float] = []
    exit_prices: list[float] = []
    dividend_amounts: list[float] = []
    price_returns: list[float] = []
    dividend_returns: list[float] = []
    total_returns: list[float] = []

    for row in panel.itertuples(index=False):
        record = by_ticker.get(str(row.ticker).upper())
        if record is None:
            entry = exit_ = dividends_paid = np.nan
        else:
            entry = price_at_or_after(record.prices, row.available_at)
            exit_ = price_at_or_after(record.prices, row.target_end_at)
            dividends_paid = dividends_between(record.dividends, row.available_at, row.target_end_at)

        if pd.notna(entry) and not np.isclose(float(entry), 0.0):
            dividend_return = float(dividends_paid) / float(entry) if pd.notna(dividends_paid) else np.nan
        else:
            dividend_return = np.nan

        if pd.notna(entry) and pd.notna(exit_) and not np.isclose(float(entry), 0.0):
            price_return = float(exit_) / float(entry) - 1.0
            total_return = (float(exit_) - float(entry) + float(dividends_paid)) / float(entry)
        else:
            price_return = np.nan
            total_return = np.nan

        entry_prices.append(entry)
        exit_prices.append(exit_)
        dividend_amounts.append(dividends_paid)
        price_returns.append(price_return)
        dividend_returns.append(dividend_return)
        total_returns.append(total_return)

    panel["entry_price"] = entry_prices
    panel["exit_price"] = exit_prices
    panel[names["dividends"]] = dividend_amounts
    panel[names["price"]] = price_returns
    panel[names["dividend"]] = dividend_returns
    panel[names["total"]] = total_returns
    return panel


__all__ = [
    "build_forward_return_targets",
    "dividends_between",
    "price_at_or_after",
    "target_column_names",
]
