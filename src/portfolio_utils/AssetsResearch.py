from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class asset_research:
    """
    Utility class to research multiple assets using the same metrics
    covered in module 01 notebooks.
    """

    tickers: Iterable[str]
    start: str
    end: Optional[str] = None
    price_field: str = "Close"

    prices: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    returns: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        self.tickers = self._normalize_tickers(self.tickers)

    @staticmethod
    def _normalize_tickers(tickers: Iterable[str]) -> List[str]:
        if isinstance(tickers, str):
            return [tickers]
        return [t for t in tickers]

    @staticmethod
    def _normalize_select(
        tickers: Optional[Union[str, Sequence[str]]],
    ) -> Optional[List[str]]:
        if tickers is None:
            return None
        if isinstance(tickers, str):
            return [tickers]
        return list(tickers)

    @staticmethod
    def _select_columns(df: pd.DataFrame, tickers: Optional[List[str]]) -> pd.DataFrame:
        if tickers is None:
            return df
        missing = [t for t in tickers if t not in df.columns]
        if missing:
            raise ValueError(f"Tickers not found in data: {missing}")
        return df[tickers]

    def download_prices(self) -> pd.DataFrame:
        data = yf.download(self.tickers, start=self.start, end=self.end)

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
        if self.prices.empty:
            self.download_prices()

        self.returns = self.prices.pct_change().dropna()
        return self.returns

    def get_prices(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.DataFrame:
        if self.prices.empty:
            self.download_prices()
            
        select = self._normalize_select(tickers)
        return self._select_columns(self.prices, select)

    def get_returns(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.DataFrame:
        if self.returns.empty:
            self.compute_returns()
        select = self._normalize_select(tickers)
        return self._select_columns(self.returns, select)

    def annual_return(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.Series:
        data = self.get_returns(tickers)
        return data.mean() * 252

    def annual_volatility(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.Series:
        data = self.get_returns(tickers)
        return data.std() * np.sqrt(252)

    def skew(self, tickers: Optional[Union[str, Sequence[str]]] = None) -> pd.Series:
        data = self.get_returns(tickers)
        return data.skew()

    def vol_over_mean(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.Series:
        annual_ret = self.annual_return(tickers)
        annual_vol = self.annual_volatility(tickers)
        return annual_vol / annual_ret.replace(0, np.nan)

    def return_interval_pct(
        self, tickers: Optional[Union[str, Sequence[str]]] = None, z: float = 2.65
    ) -> pd.DataFrame:
        annual_ret_pct = self.annual_return(tickers) * 100
        annual_vol_pct = self.annual_volatility(tickers) * 100
        low = annual_ret_pct - z * annual_vol_pct
        high = annual_ret_pct + z * annual_vol_pct
        return pd.DataFrame({"low": low, "high": high})

    def metrics(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.DataFrame:
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
        data = self.get_returns(tickers)
        return data.describe()

    def covariance(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.DataFrame:
        data = self.get_returns(tickers)
        return data.cov()


# Optional alias with standard class naming
if __name__ == "__main__":
    AssetResearch = asset_research(["^GSPC"], start="2020-09-30")
    
    print(AssetResearch.annual_return())
