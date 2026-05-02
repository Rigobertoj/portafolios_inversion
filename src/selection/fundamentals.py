"""Yahoo Finance fundamental-data access for security selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import pandas as pd

from ..research.assets_research import yf


@dataclass(frozen=True)
class FundamentalData:
    """Container with the raw Yahoo Finance inputs for one company."""

    ticker: str
    info: Dict[str, object]
    income_statement: pd.DataFrame
    balance_sheet: pd.DataFrame
    cash_flow: pd.DataFrame
    quarterly_income_statement: pd.DataFrame
    quarterly_balance_sheet: pd.DataFrame
    quarterly_cash_flow: pd.DataFrame
    prices: pd.Series


class YahooFundamentalsProvider:
    """Fetch and normalize Yahoo Finance data used by fundamental selectors."""

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

    def _fetch_prices(self, ticker_obj: object) -> pd.Series:
        history = ticker_obj.history(start=self.start, end=self.end)
        if not isinstance(history, pd.DataFrame) or history.empty:
            return pd.Series(dtype=float, name=self.price_field)
        if self.price_field not in history.columns:
            raise ValueError(
                f"price_field '{self.price_field}' not found in Yahoo history data."
            )
        prices = pd.to_numeric(history[self.price_field], errors="coerce").dropna()
        prices.name = self.price_field
        return prices

    def clear_cache(self) -> None:
        """Clear cached ticker objects and downloaded fundamental records."""
        self._ticker_cache.clear()
        self._data_cache.clear()

    def _ticker_object(self, ticker: str) -> object:
        if ticker not in self._ticker_cache:
            self._ticker_cache[ticker] = self._ticker_factory(ticker)
        return self._ticker_cache[ticker]

    def _cache_key(self, ticker: str) -> Tuple[str, str, Optional[str], str]:
        return (ticker, self.start, self.end, self.price_field)

    def fetch(self, ticker: str) -> FundamentalData:
        """Fetch raw financial statements, profile data, and prices for a ticker."""
        ticker_clean = self._normalize_ticker(ticker)
        cache_key = self._cache_key(ticker_clean)
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]

        ticker_obj = self._ticker_object(ticker_clean)
        info = getattr(ticker_obj, "info", {}) or {}
        if not isinstance(info, dict):
            info = dict(info)

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
            prices=self._fetch_prices(ticker_obj),
        )
        self._data_cache[cache_key] = record
        return record

    def fetch_many(self, tickers: Iterable[str]) -> Dict[str, FundamentalData]:
        """Fetch a group of tickers, preserving the normalized ticker keys."""
        data: Dict[str, FundamentalData] = {}
        for ticker in tickers:
            record = self.fetch(ticker)
            data[record.ticker] = record
        return data


__all__ = [
    "FundamentalData",
    "YahooFundamentalsProvider",
]
