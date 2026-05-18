"""Yahoo Finance fundamental-data access for security selection.

This module fetches profile data, financial statements, cash-flow statements,
balance sheets, quarterly statements, and price history used by fundamental
selectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional, Tuple

import pandas as pd
import numpy as np

from .assets_research import yf


@dataclass(frozen=True)
class FundamentalData:
    """
    Container with the raw Yahoo Finance inputs for one company.

    Parameters
    ----------
    ticker : str
        Normalized ticker symbol.
    info : dict
        Yahoo Finance profile and quote metadata.
    income_statement, balance_sheet, cash_flow : pandas.DataFrame
        Annual financial statements.
    quarterly_income_statement, quarterly_balance_sheet, quarterly_cash_flow : pandas.DataFrame
        Quarterly financial statements.
    prices : pandas.Series
        Price series used for point-in-time ratio calculations.
    dividends : pandas.Series, default empty
        Dividend cash-flow series used to build forward shareholder-return
        targets. Empty when the provider does not expose dividend history.
    """

    ticker: str
    info: Dict[str, object]
    income_statement: pd.DataFrame
    balance_sheet: pd.DataFrame
    cash_flow: pd.DataFrame
    quarterly_income_statement: pd.DataFrame
    quarterly_balance_sheet: pd.DataFrame
    quarterly_cash_flow: pd.DataFrame
    prices: pd.Series
    dividends: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float, name="Dividends")
    )


class YahooFundamentalsProvider:
    """
    Fetch and normalize Yahoo Finance data used by fundamental selectors.

    Parameters
    ----------
    start : str, default "2020-01-01"
        First date requested for price history.
    end : str, optional
        Optional exclusive end date requested for price history.
    price_field : str, default "Close"
        Yahoo Finance history column used as the price series.
    ticker_factory : callable, optional
        Factory used to create ticker objects. Defaults to `yfinance.Ticker`.
    """

    def __init__(
        self,
        start: str = "2020-01-01",
        end: Optional[str] = None,
        price_field: str = "Close",
        ticker_factory: Optional[Callable[[str], object]] = None,
    ) -> None:
        self.start = start
        self.end = end
        self.price_field = price_field
        self._ticker_factory = ticker_factory or yf.Ticker
        self._ticker_cache: Dict[str, object] = {}
        self._data_cache: Dict[Tuple[str, str, Optional[str], str], FundamentalData] = {}

    @staticmethod
    def _normalize_ticker(ticker: str) -> str:
        ticker_clean = str(ticker).strip().upper()
        if not ticker_clean:
            raise ValueError("ticker must be a non-empty string.")
        return ticker_clean

    @staticmethod
    def _copy_statement(ticker_obj: object, *names: str) -> pd.DataFrame:
        for name in names:
            value = getattr(ticker_obj, name, None)
            if isinstance(value, pd.DataFrame):
                return value.copy()
        return pd.DataFrame()

    def _fetch_market_history(self, ticker_obj: object) -> tuple[pd.Series, pd.Series]:
        history = ticker_obj.history(start=self.start, end=self.end)
        if not isinstance(history, pd.DataFrame) or history.empty:
            return (
                pd.Series(dtype=float, name=self.price_field),
                pd.Series(dtype=float, name="Dividends"),
            )
        if self.price_field not in history.columns:
            raise ValueError(
                f"price_field '{self.price_field}' not found in Yahoo history data."
            )
        prices = pd.to_numeric(history[self.price_field], errors="coerce").dropna()
        prices.name = self.price_field
        if "Dividends" in history.columns:
            dividends = pd.to_numeric(history["Dividends"], errors="coerce").dropna()
            dividends = dividends[~np.isclose(dividends, 0.0)]
        else:
            dividends = pd.Series(dtype=float, name="Dividends")
        dividends.name = "Dividends"
        return prices, dividends

    def _fetch_prices(self, ticker_obj: object) -> pd.Series:
        prices, _ = self._fetch_market_history(ticker_obj)
        return prices

    def clear_cache(self) -> None:
        """
        Clear cached ticker objects and downloaded fundamental records.

        Returns
        -------
        None
        """
        self._ticker_cache.clear()
        self._data_cache.clear()

    def _ticker_object(self, ticker: str) -> object:
        if ticker not in self._ticker_cache:
            self._ticker_cache[ticker] = self._ticker_factory(ticker)
        return self._ticker_cache[ticker]

    def _cache_key(self, ticker: str) -> Tuple[str, str, Optional[str], str]:
        return (ticker, self.start, self.end, self.price_field)

    def fetch(self, ticker: str) -> FundamentalData:
        """
        Fetch raw financial statements, profile data, and prices for a ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol to fetch.

        Returns
        -------
        FundamentalData
            Cached or newly fetched fundamental record.
        """
        ticker_clean = self._normalize_ticker(ticker)
        cache_key = self._cache_key(ticker_clean)
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]

        ticker_obj = self._ticker_object(ticker_clean)
        info = getattr(ticker_obj, "info", {}) or {}
        if not isinstance(info, dict):
            info = dict(info)

        prices, dividends = self._fetch_market_history(ticker_obj)

        record = FundamentalData(
            ticker=ticker_clean,
            info=info,
            income_statement=self._copy_statement(ticker_obj, "income_stmt", "financials"),
            balance_sheet=self._copy_statement(ticker_obj, "balance_sheet", "balancesheet"),
            cash_flow=self._copy_statement(ticker_obj, "cash_flow", "cashflow"),
            quarterly_income_statement=self._copy_statement(
                ticker_obj,
                "quarterly_income_stmt",
                "quarterly_financials",
            ),
            quarterly_balance_sheet=self._copy_statement(
                ticker_obj,
                "quarterly_balance_sheet",
                "quarterly_balancesheet",
            ),
            quarterly_cash_flow=self._copy_statement(
                ticker_obj,
                "quarterly_cash_flow",
                "quarterly_cashflow",
            ),
            prices=prices,
            dividends=dividends,
        )
        self._data_cache[cache_key] = record
        return record

    def fetch_many(self, tickers: Iterable[str]) -> Dict[str, FundamentalData]:
        """
        Fetch a group of tickers, preserving normalized ticker keys.

        Parameters
        ----------
        tickers : iterable of str
            Ticker symbols to fetch.

        Returns
        -------
        dict of str to FundamentalData
            Fundamental records keyed by normalized ticker.
        """
        data: Dict[str, FundamentalData] = {}
        for ticker in tickers:
            record = self.fetch(ticker)
            data[record.ticker] = record
        return data


__all__ = [
    "FundamentalData",
    "YahooFundamentalsProvider",
]
