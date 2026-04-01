"""Asset-level market research utilities for the new package layout."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ModuleNotFoundError:
    class _MissingYFinance:
        """Fallback used when yfinance is unavailable in the environment."""

        @staticmethod
        def download(*args, **kwargs):
            raise ModuleNotFoundError(
                "yfinance is required to download market data. "
                "Install it to use AssetsResearch.download_prices()."
            )

    yf = _MissingYFinance()


class AssetsResearch:
    """
    Research helper centered on market prices and descriptive return metrics.

    The class keeps the legacy public surface used across the project while
    relocating the actual implementation to `src.research`.
    """

    def __init__(
        self,
        tickers: Iterable[str],
        start: str,
        end: Optional[str] = None,
        price_field: str = "Close",
    ) -> None:
        self.__tickers: list[str] = []
        self.__start = ""
        self.__end: Optional[str] = None
        self.__price_field = "Close"
        self.__prices = pd.DataFrame()
        self.__returns = pd.DataFrame()

        self.tickers = tickers
        self.start = start
        self.end = end
        self.price_field = price_field

    def _reset_cache(self) -> None:
        """Reset cached prices and returns after configuration changes."""
        self.__prices = pd.DataFrame()
        self.__returns = pd.DataFrame()

    @staticmethod
    def _normalize_tickers(tickers: Iterable[str]) -> list[str]:
        """Normalize a ticker input into a non-empty string list."""
        if isinstance(tickers, str):
            return [tickers]
        return [str(ticker) for ticker in tickers]

    @staticmethod
    def _normalize_select(
        tickers: Optional[Union[str, Sequence[str]]],
    ) -> Optional[list[str]]:
        """Normalize an optional ticker subset selection."""
        if tickers is None:
            return None
        if isinstance(tickers, str):
            return [tickers]
        return list(tickers)

    @staticmethod
    def _validate_date_range(start: str, end: Optional[str]) -> None:
        """Validate parseable dates and an ordered download window."""
        if not start:
            return

        try:
            start_ts = pd.Timestamp(start)
        except Exception as exc:
            raise ValueError(
                f"start must be a valid date string, got {start!r}."
            ) from exc

        if end is None:
            return

        try:
            end_ts = pd.Timestamp(end)
        except Exception as exc:
            raise ValueError(
                f"end must be a valid date string, got {end!r}."
            ) from exc

        if end_ts <= start_ts:
            raise ValueError(
                "end must be later than start. "
                "Yahoo Finance treats end as exclusive, so end=start returns no rows."
            )

    @staticmethod
    def _select_columns(
        df: pd.DataFrame,
        tickers: Optional[list[str]],
    ) -> pd.DataFrame:
        """Return the requested ticker columns, validating all names."""
        if tickers is None:
            return df
        missing = [ticker for ticker in tickers if ticker not in df.columns]
        if missing:
            raise ValueError(f"Tickers not found in data: {missing}")
        return df[tickers]

    @property
    def tickers(self) -> list[str]:
        """Normalized list of tickers used by the research instance."""
        return list(self.__tickers)

    @tickers.setter
    def tickers(self, value: Iterable[str]) -> None:
        normalized = self._normalize_tickers(value)
        if not normalized:
            raise ValueError("tickers must contain at least one symbol.")
        self.__tickers = normalized
        self._reset_cache()

    @property
    def start(self) -> str:
        """Start date used to download market data."""
        return self.__start

    @start.setter
    def start(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("start must be a non-empty date string.")
        start = value.strip()
        self._validate_date_range(start=start, end=self.__end)
        self.__start = start
        self._reset_cache()

    @property
    def end(self) -> Optional[str]:
        """Optional exclusive end date used to download market data."""
        return self.__end

    @end.setter
    def end(self, value: Optional[str]) -> None:
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("end must be None or a non-empty date string.")
        end = value.strip() if value is not None else None
        self._validate_date_range(start=self.__start, end=end)
        self.__end = end
        self._reset_cache()

    @property
    def price_field(self) -> str:
        """Yahoo Finance price column used to build the research dataset."""
        return self.__price_field

    @price_field.setter
    def price_field(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("price_field must be a non-empty string.")
        self.__price_field = value.strip()
        self._reset_cache()

    @property
    def prices(self) -> pd.DataFrame:
        """Cached prices indexed by date."""
        return self.__prices.copy()

    @property
    def returns(self) -> pd.DataFrame:
        """Cached daily returns indexed by date."""
        return self.__returns.copy()

    def _set_prices_cache(self, value: pd.DataFrame) -> None:
        """Internal cache update for prices."""
        if not isinstance(value, pd.DataFrame):
            raise TypeError("prices cache must be a pandas DataFrame.")
        self.__prices = value.copy()

    def _set_returns_cache(self, value: pd.DataFrame) -> None:
        """Internal cache update for returns."""
        if not isinstance(value, pd.DataFrame):
            raise TypeError("returns cache must be a pandas DataFrame.")
        self.__returns = value.copy()

    def download_prices(self) -> pd.DataFrame:
        """Download market prices and cache them inside the instance."""
        data = yf.download(
            self.tickers,
            start=self.start,
            end=self.end,
            progress=False,
        )

        if isinstance(data.columns, pd.MultiIndex):
            if self.price_field not in data.columns.levels[0]:
                raise ValueError(
                    f"price_field '{self.price_field}' not found in downloaded data."
                )
            prices = data[self.price_field]
        else:
            if self.price_field not in data.columns:
                raise ValueError(
                    f"price_field '{self.price_field}' not found in downloaded data."
                )
            prices = data[self.price_field]

        cleaned_prices = prices.dropna()
        if cleaned_prices.empty:
            tickers = ", ".join(self.tickers)
            raise ValueError(
                "No price data was downloaded for "
                f"{tickers} between start={self.start!r} and end={self.end!r}. "
                "Check the ticker symbols and date range."
            )

        if isinstance(cleaned_prices, pd.Series):
            cleaned_prices = cleaned_prices.to_frame(name=self.tickers[0])

        self._set_prices_cache(cleaned_prices)
        return self.prices

    def compute_returns(self) -> pd.DataFrame:
        """Compute daily percentage returns from cached prices."""
        if self.prices.empty:
            self.download_prices()

        self._set_returns_cache(self.prices.pct_change().dropna())
        return self.returns

    def get_prices(
        self,
        tickers: Optional[Union[str, Sequence[str]]] = None,
    ) -> pd.DataFrame:
        """Return cached prices for all tickers or a validated subset."""
        if self.prices.empty:
            self.download_prices()
        return self._select_columns(self.prices, self._normalize_select(tickers))

    def get_returns(
        self,
        tickers: Optional[Union[str, Sequence[str]]] = None,
    ) -> pd.DataFrame:
        """Return cached returns for all tickers or a validated subset."""
        if self.returns.empty:
            self.compute_returns()
        return self._select_columns(self.returns, self._normalize_select(tickers))

    def annual_return(
        self,
        tickers: Optional[Union[str, Sequence[str]]] = None,
    ) -> pd.Series:
        """Return annualized expected returns using 252 trading days."""
        return self.get_returns(tickers).mean() * 252

    def annual_volatility(
        self,
        tickers: Optional[Union[str, Sequence[str]]] = None,
    ) -> pd.Series:
        """Return annualized volatility using 252 trading days."""
        return self.get_returns(tickers).std() * np.sqrt(252)

    def skew(
        self,
        tickers: Optional[Union[str, Sequence[str]]] = None,
    ) -> pd.Series:
        """Return skewness of daily returns."""
        return self.get_returns(tickers).skew()

    def vol_over_mean(
        self,
        tickers: Optional[Union[str, Sequence[str]]] = None,
    ) -> pd.Series:
        """Return annualized volatility divided by annualized return."""
        annual_ret = self.annual_return(tickers)
        annual_vol = self.annual_volatility(tickers)
        return annual_vol / annual_ret.replace(0, np.nan)

    def return_interval_pct(
        self,
        tickers: Optional[Union[str, Sequence[str]]] = None,
        z: float = 2.65,
    ) -> pd.DataFrame:
        """Estimate a simple annual return interval in percentage terms."""
        annual_ret_pct = self.annual_return(tickers) * 100
        annual_vol_pct = self.annual_volatility(tickers) * 100
        return pd.DataFrame(
            {
                "low": annual_ret_pct - z * annual_vol_pct,
                "high": annual_ret_pct + z * annual_vol_pct,
            }
        )

    def metrics(
        self,
        tickers: Optional[Union[str, Sequence[str]]] = None,
    ) -> pd.DataFrame:
        """Build the consolidated per-ticker metrics table used in the project."""
        data = self.get_returns(tickers)
        annual_return = data.mean() * 252
        annual_vol = data.std() * np.sqrt(252)
        annual_return_pct = annual_return * 100
        annual_vol_pct = annual_vol * 100

        return pd.DataFrame(
            {
                "annual_return": annual_return,
                "annual_volatility": annual_vol,
                "annual_return_pct": annual_return_pct,
                "annual_volatility_pct": annual_vol_pct,
                "skew": data.skew(),
                "vol_over_mean": annual_vol / annual_return.replace(0, np.nan),
                "return_interval_low_pct": annual_return_pct - 2.65 * annual_vol_pct,
                "return_interval_high_pct": annual_return_pct + 2.65 * annual_vol_pct,
            }
        )

    def describe_returns(
        self,
        tickers: Optional[Union[str, Sequence[str]]] = None,
    ) -> pd.DataFrame:
        """Return descriptive statistics for daily returns."""
        return self.get_returns(tickers).describe()

    def covariance(
        self,
        tickers: Optional[Union[str, Sequence[str]]] = None,
    ) -> pd.DataFrame:
        """Return the covariance matrix of daily returns."""
        return self.get_returns(tickers).cov()

    def correlation(
        self,
        tickers: Optional[Union[str, Sequence[str]]] = None,
    ) -> pd.DataFrame:
        """Return the correlation matrix of daily returns."""
        return self.get_returns(tickers).corr()


__all__ = [
    "AssetsResearch",
]
