from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class AssetsResearch:
    """
    Research helper for asset prices and return metrics.

    This class centralizes a simple workflow:
    1) Download close prices from Yahoo Finance.
    2) Compute daily returns.
    3) Compute summary metrics (return, volatility, skew, intervals, etc.).

    Quick start:
        research = AssetsResearch(["JPM", "V"], start="2020-01-01")
        research.download_prices()
        metrics_df = research.metrics()

    Public API map:
    - Data loading: download_prices, compute_returns, get_prices, get_returns
    - Metrics: annual_return, annual_volatility, skew, vol_over_mean
    - Intervals/reporting: return_interval_pct, metrics
    - Statistics: describe_returns, covariance
    """

    tickers: Iterable[str]
    start: str
    end: Optional[str] = None
    price_field: str = "Close"

    prices: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    returns: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        """Normalize ticker input right after object creation."""
        self.tickers = self._normalize_tickers(self.tickers)

    @staticmethod
    def _normalize_tickers(tickers: Iterable[str]) -> List[str]:
        """Normalize constructor tickers into a list."""
        if isinstance(tickers, str):
            return [tickers]
        return [t for t in tickers]

    @staticmethod
    def _normalize_select(
        tickers: Optional[Union[str, Sequence[str]]],
    ) -> Optional[List[str]]:
        """Normalize optional ticker selection to list or None."""
        if tickers is None:
            return None
        if isinstance(tickers, str):
            return [tickers]
        return list(tickers)

    @staticmethod
    def _select_columns(df: pd.DataFrame, tickers: Optional[List[str]]) -> pd.DataFrame:
        """Return selected ticker columns, validating all requested names."""
        if tickers is None:
            return df
        missing = [t for t in tickers if t not in df.columns]
        if missing:
            raise ValueError(f"Tickers not found in data: {missing}")
        return df[tickers]

    def download_prices(self) -> pd.DataFrame:
        """
        Download price data from Yahoo Finance and cache it in ``self.prices``.

        Returns:
            pd.DataFrame: Price table indexed by date, columns are tickers.
        """
        data = yf.download(
            self.tickers,
            start=self.start,
            end=self.end,
            progress=False,
        )

        if isinstance(data.columns, pd.MultiIndex):
            if self.price_field in data.columns.levels[0]:
                prices = data[self.price_field]
            else:
                raise ValueError(
                    f"price_field '{self.price_field}' not found in downloaded data."
                )
        else:
            if self.price_field in data.columns:
                prices = data[self.price_field]
            else:
                raise ValueError(
                    f"price_field '{self.price_field}' not found in downloaded data."
                )

        self.prices = prices.dropna(how="all")
        return self.prices

    def compute_returns(self) -> pd.DataFrame:
        """
        Compute daily percentage returns from cached prices.

        If prices are not available, it triggers ``download_prices()`` first.

        Returns:
            pd.DataFrame: Daily returns by ticker.
        """
        if self.prices.empty:
            self.download_prices()

        self.returns = self.prices.pct_change().dropna()
        return self.returns

    def get_prices(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.DataFrame:
        """
        Return cached prices for all tickers or a subset.

        Args:
            tickers: Optional ticker or list/tuple of tickers to filter.

        Returns:
            pd.DataFrame: Filtered price table.
        """
        if self.prices.empty:
            self.download_prices()

        select = self._normalize_select(tickers)
        return self._select_columns(self.prices, select)

    def get_returns(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.DataFrame:
        """
        Return cached returns for all tickers or a subset.

        Args:
            tickers: Optional ticker or list/tuple of tickers to filter.

        Returns:
            pd.DataFrame: Filtered return table.
        """
        if self.returns.empty:
            self.compute_returns()
        select = self._normalize_select(tickers)
        return self._select_columns(self.returns, select)

    def annual_return(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.Series:
        """
        Compute annualized expected return using 252 trading days.

        Returns:
            pd.Series: Annualized mean return by ticker.
        """
        data = self.get_returns(tickers)
        return data.mean() * 252

    def annual_volatility(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.Series:
        """
        Compute annualized volatility using 252 trading days.

        Returns:
            pd.Series: Annualized standard deviation by ticker.
        """
        data = self.get_returns(tickers)
        return data.std() * np.sqrt(252)

    def skew(self, tickers: Optional[Union[str, Sequence[str]]] = None) -> pd.Series:
        """
        Compute skewness of daily returns.

        Returns:
            pd.Series: Return skewness by ticker.
        """
        data = self.get_returns(tickers)
        return data.skew()

    def vol_over_mean(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.Series:
        """
        Compute volatility divided by annualized return.

        A lower value can indicate a better risk/return profile
        for this simple ratio.

        Returns:
            pd.Series: Volatility over annual return by ticker.
        """
        annual_ret = self.annual_return(tickers)
        annual_vol = self.annual_volatility(tickers)
        return annual_vol / annual_ret.replace(0, np.nan)

    def return_interval_pct(
        self, tickers: Optional[Union[str, Sequence[str]]] = None, z: float = 2.65
    ) -> pd.DataFrame:
        """
        Estimate a return interval in percentage terms.

        Interval is computed as:
            annual_return_pct +/- z * annual_volatility_pct

        Args:
            tickers: Optional ticker selection.
            z: Z-score multiplier (default 2.65).

        Returns:
            pd.DataFrame: DataFrame with ``low`` and ``high`` columns.
        """
        annual_ret_pct = self.annual_return(tickers) * 100
        annual_vol_pct = self.annual_volatility(tickers) * 100
        low = annual_ret_pct - z * annual_vol_pct
        high = annual_ret_pct + z * annual_vol_pct
        return pd.DataFrame({"low": low, "high": high})

    def metrics(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.DataFrame:
        """
        Build a consolidated metrics table per ticker.

        Columns:
        - annual_return
        - annual_volatility
        - annual_return_pct
        - annual_volatility_pct
        - skew
        - vol_over_mean
        - return_interval_low_pct
        - return_interval_high_pct

        Returns:
            pd.DataFrame: Multi-metric summary by ticker.
        """
        if self.returns.empty:
            self.compute_returns()

        data = self.get_returns(tickers)

        annual_return = data.mean() * 252
        annual_vol = data.std() * np.sqrt(252)

        annual_return_pct = annual_return * 100
        annual_vol_pct = annual_vol * 100

        vol_over_mean = annual_vol / annual_return.replace(0, np.nan)

        interval_low = annual_return_pct - 2.65 * annual_vol_pct
        interval_high = annual_return_pct + 2.65 * annual_vol_pct

        metrics_df = pd.DataFrame(
            {
                "annual_return": annual_return,
                "annual_volatility": annual_vol,
                "annual_return_pct": annual_return_pct,
                "annual_volatility_pct": annual_vol_pct,
                "skew": data.skew(),
                "vol_over_mean": vol_over_mean,
                "return_interval_low_pct": interval_low,
                "return_interval_high_pct": interval_high,
            }
        )

        return metrics_df

    def describe_returns(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.DataFrame:
        """
        Return descriptive statistics for daily returns.

        Returns:
            pd.DataFrame: ``pandas.DataFrame.describe()`` over returns.
        """
        data = self.get_returns(tickers)
        return data.describe()

    def covariance(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.DataFrame:
        """
        Compute covariance matrix of daily returns.

        Returns:
            pd.DataFrame: Covariance matrix.
        """
        data = self.get_returns(tickers)
        return data.cov()


# Optional alias with standard class naming
if __name__ == "__main__":
    gspc = AssetsResearch(["^GSPC"], start="2020-09-30")

    print(gspc.annual_return())
