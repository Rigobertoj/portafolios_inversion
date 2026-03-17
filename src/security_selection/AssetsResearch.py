from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Iterable, List, Optional, Sequence, Union

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


@dataclass
class AssetsResearch:
    """
    Research and analytics helper for a group of financial assets.

    `AssetsResearch` centralizes the basic workflow used across the project:
    downloading market prices, computing daily returns, and exposing common
    descriptive statistics for one or many tickers. The class keeps a lazy
    in-memory cache for downloaded prices and computed returns, so most public
    methods can be called directly without manually preparing intermediate data.

    Parameters
    ----------
    tickers : Iterable[str]
        Asset symbols to analyze. A single ticker string is also accepted and
        normalized internally to a list.
    start : str
        Start date passed to `yfinance.download`.
    end : str | None, default None
        Optional end date passed to `yfinance.download`.
    price_field : str, default "Close"
        Price column extracted from Yahoo Finance data. Typical values are
        `"Close"` and `"Adj Close"`.

    Attributes
    ----------
    tickers : list[str]
        Normalized list of ticker symbols used by the instance.
    start : str
        Start date used for downloads.
    end : str | None
        Optional end date used for downloads.
    price_field : str
        Selected price column from the downloaded market data.
    prices : pandas.DataFrame
        Cached prices indexed by date with one column per ticker.
    returns : pandas.DataFrame
        Cached daily percentage returns indexed by date.

    Methods
    -------
    download_prices()
        Download and cache prices from Yahoo Finance.
    compute_returns()
        Compute and cache daily percentage returns from prices.
    get_prices(tickers=None)
        Return all cached prices or a ticker subset.
    get_returns(tickers=None)
        Return all cached returns or a ticker subset.
    annual_return(tickers=None)
        Compute annualized mean return using 252 trading days.
    annual_volatility(tickers=None)
        Compute annualized volatility using 252 trading days.
    skew(tickers=None)
        Compute skewness of daily returns.
    vol_over_mean(tickers=None)
        Compute the ratio between annualized volatility and annualized return.
    return_interval_pct(tickers=None, z=2.65)
        Estimate a simple return interval in percentage terms.
    metrics(tickers=None)
        Build a consolidated per-ticker metrics table.
    describe_returns(tickers=None)
        Return `DataFrame.describe()` over daily returns.
    covariance(tickers=None)
        Compute the covariance matrix of daily returns.
    correlation(tickers=None)
        Compute the correlation matrix of daily returns.

    Notes
    -----
    - The caches stored in `prices` and `returns` are reset automatically when
      `tickers`, `start`, `end`, or `price_field` change.
    - Public methods that depend on prices or returns load the required data
      lazily when it is not already available.

    Examples
    --------
    >>> research = AssetsResearch(["JPM", "V"], start="2020-01-01")
    >>> research.get_prices().head()
    >>> research.annual_return()
    >>> research.correlation()
    """

    tickers: InitVar[Iterable[str]]
    start: InitVar[str]
    end: InitVar[Optional[str]] = None
    price_field: InitVar[str] = "Close"

    __tickers: List[str] = field(init=False, repr=False, default_factory=list)
    __start: str = field(init=False, repr=False, default="")
    __end: Optional[str] = field(init=False, repr=False, default=None)
    __price_field: str = field(init=False, repr=False, default="Close")
    __prices: pd.DataFrame = field(init=False, repr=False, default_factory=pd.DataFrame)
    __returns: pd.DataFrame = field(
        init=False, repr=False, default_factory=pd.DataFrame
    )

    def __post_init__(
        self,
        tickers: Iterable[str],
        start: str,
        end: Optional[str],
        price_field: str,
    ) -> None:
        """
        Initialize validated public configuration and reset caches.

        Parameters
        ----------
        tickers : Iterable[str]
            Symbols to analyze.
        start : str
            Start date for the historical data window.
        end : str | None
            Optional end date for the historical data window.
        price_field : str
            Market data column used to build the research dataset.
        """
        self.tickers = tickers
        self.start = start
        self.end = end
        self.price_field = price_field

    def _reset_cache(self) -> None:
        """Reset derived cached data when source configuration changes."""
        self.__prices = pd.DataFrame()
        self.__returns = pd.DataFrame()

    def _get_tickers(self) -> List[str]:
        """Public read access to ticker symbols."""
        return list(self.__tickers)

    def _set_tickers(self, value: Iterable[str]) -> None:
        normalized = self._normalize_tickers(value)
        if not normalized:
            raise ValueError("tickers must contain at least one symbol.")
        self.__tickers = normalized
        self._reset_cache()

    def _get_start(self) -> str:
        """Public read access to start date."""
        return self.__start

    def _set_start(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("start must be a non-empty date string.")
        start = value.strip()
        self._validate_date_range(start=start, end=self.__end)
        self.__start = start
        self._reset_cache()

    def _get_end(self) -> Optional[str]:
        """Public read access to end date."""
        return self.__end

    def _set_end(self, value: Optional[str]) -> None:
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("end must be None or a non-empty date string.")
        end = value.strip() if value is not None else None
        self._validate_date_range(start=self.__start, end=end)
        self.__end = end
        self._reset_cache()

    def _get_price_field(self) -> str:
        """Public read access to selected price field."""
        return self.__price_field

    def _set_price_field(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("price_field must be a non-empty string.")
        self.__price_field = value
        self._reset_cache()

    def _get_prices(self) -> pd.DataFrame:
        """Read-only style accessor for cached prices."""
        return self.__prices.copy()

    def _set_prices_cache(self, value: pd.DataFrame) -> None:
        """Internal cache update for prices."""
        if not isinstance(value, pd.DataFrame):
            raise TypeError("prices cache must be a pandas DataFrame.")
        self.__prices = value.copy()

    def _get_returns(self) -> pd.DataFrame:
        """Read-only style accessor for cached returns."""
        return self.__returns.copy()

    def _set_returns_cache(self, value: pd.DataFrame) -> None:
        """Internal cache update for returns."""
        if not isinstance(value, pd.DataFrame):
            raise TypeError("returns cache must be a pandas DataFrame.")
        self.__returns = value.copy()

    @staticmethod
    def _normalize_tickers(tickers: Iterable[str]) -> List[str]:
        """Normalize constructor tickers into a list."""
        if isinstance(tickers, str):
            return [tickers]
        return [str(t) for t in tickers]

    @staticmethod
    def _validate_date_range(start: str, end: Optional[str]) -> None:
        """Validate parseable dates and ensure the download window is ordered."""
        if not start:
            return

        try:
            start_ts = pd.Timestamp(start)
        except Exception as exc:
            raise ValueError(f"start must be a valid date string, got {start!r}.") from exc

        if end is None:
            return

        try:
            end_ts = pd.Timestamp(end)
        except Exception as exc:
            raise ValueError(f"end must be a valid date string, got {end!r}.") from exc

        if end_ts <= start_ts:
            raise ValueError(
                "end must be later than start. "
                "Yahoo Finance treats end as exclusive, so end=start returns no rows."
            )

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
        Download price data from Yahoo Finance and cache it in `prices`.

        The returned table is indexed by date and contains one column per
        ticker. If Yahoo Finance returns a multi-indexed table, the selected
        `price_field` is extracted first.

        Returns
        -------
        pandas.DataFrame
            Price table indexed by date, with one column per ticker.

        Raises
        ------
        ValueError
            If `price_field` is not present in the downloaded dataset.
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

        cleaned_prices = prices.dropna()
        if cleaned_prices.empty:
            tickers = ", ".join(self.tickers)
            raise ValueError(
                "No price data was downloaded for "
                f"{tickers} between start={self.start!r} and end={self.end!r}. "
                "Check the ticker symbols and date range."
            )

        self._set_prices_cache(cleaned_prices)
        return self.prices

    def compute_returns(self) -> pd.DataFrame:
        """
        Compute daily percentage returns from cached prices.

        If prices are not available, this method triggers `download_prices()`
        first.

        Returns
        -------
        pandas.DataFrame
            Daily percentage returns by ticker.
        """
        if self.prices.empty:
            self.download_prices()

        self._set_returns_cache(self.prices.pct_change().dropna())
        return self.returns

    def get_prices(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.DataFrame:
        """
        Return cached prices for all tickers or a subset.

        Parameters
        ----------
        tickers : str | Sequence[str] | None, default None
            Optional ticker or collection of tickers to filter.

        Returns
        -------
        pandas.DataFrame
            Cached price table for the requested tickers.

        Raises
        ------
        ValueError
            If any requested ticker is not present in the cached data.
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

        Parameters
        ----------
        tickers : str | Sequence[str] | None, default None
            Optional ticker or collection of tickers to filter.

        Returns
        -------
        pandas.DataFrame
            Cached return table for the requested tickers.

        Raises
        ------
        ValueError
            If any requested ticker is not present in the cached data.
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

        Parameters
        ----------
        tickers : str | Sequence[str] | None, default None
            Optional ticker or collection of tickers to include.

        Returns
        -------
        pandas.Series
            Annualized mean return by ticker.
        """
        data = self.get_returns(tickers)
        return data.mean() * 252

    def annual_volatility(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.Series:
        """
        Compute annualized volatility using 252 trading days.

        Parameters
        ----------
        tickers : str | Sequence[str] | None, default None
            Optional ticker or collection of tickers to include.

        Returns
        -------
        pandas.Series
            Annualized standard deviation by ticker.
        """
        data = self.get_returns(tickers)
        return data.std() * np.sqrt(252)

    def skew(self, tickers: Optional[Union[str, Sequence[str]]] = None) -> pd.Series:
        """
        Compute skewness of daily returns.

        Parameters
        ----------
        tickers : str | Sequence[str] | None, default None
            Optional ticker or collection of tickers to include.

        Returns
        -------
        pandas.Series
            Return skewness by ticker.
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

        Parameters
        ----------
        tickers : str | Sequence[str] | None, default None
            Optional ticker or collection of tickers to include.

        Returns
        -------
        pandas.Series
            Volatility over annual return by ticker.
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

        Parameters
        ----------
        tickers : str | Sequence[str] | None, default None
            Optional ticker or collection of tickers to include.
        z : float, default 2.65
            Multiplier used to build the interval around the annualized return.

        Returns
        -------
        pandas.DataFrame
            DataFrame with `low` and `high` columns in percentage terms.
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

        Parameters
        ----------
        tickers : str | Sequence[str] | None, default None
            Optional ticker or collection of tickers to include.

        Returns
        -------
        pandas.DataFrame
            Multi-metric summary indexed by ticker.
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

        Parameters
        ----------
        tickers : str | Sequence[str] | None, default None
            Optional ticker or collection of tickers to include.

        Returns
        -------
        pandas.DataFrame
            Result of calling `DataFrame.describe()` on daily returns.
        """
        data = self.get_returns(tickers)
        return data.describe()

    def covariance(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.DataFrame:
        """
        Compute covariance matrix of daily returns.

        Parameters
        ----------
        tickers : str | Sequence[str] | None, default None
            Optional ticker or collection of tickers to include.

        Returns
        -------
        pandas.DataFrame
            Covariance matrix of daily returns.
        """
        data = self.get_returns(tickers)
        return data.cov()

    def correlation(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.DataFrame:
        """
        Compute correlation matrix of daily returns.

        Parameters
        ----------
        tickers : str | Sequence[str] | None, default None
            Optional ticker or collection of tickers to include.

        Returns
        -------
        pandas.DataFrame
            Correlation matrix of daily returns.
        """
        data = self.get_returns(tickers)
        return data.corr()


# Bind properties after dataclass processing to keep constructor parameters
# (`tickers`, `start`, `end`, `price_field`) and still encapsulate state.
AssetsResearch.tickers = property(
    AssetsResearch._get_tickers,
    AssetsResearch._set_tickers,
    doc="Normalized list of ticker symbols used by the research instance.",
)
AssetsResearch.start = property(
    AssetsResearch._get_start,
    AssetsResearch._set_start,
    doc="Start date used to download historical market data.",
)
AssetsResearch.end = property(
    AssetsResearch._get_end,
    AssetsResearch._set_end,
    doc="Optional end date used to download historical market data.",
)
AssetsResearch.price_field = property(
    AssetsResearch._get_price_field,
    AssetsResearch._set_price_field,
    doc="Selected Yahoo Finance price column, for example 'Close'.",
)
AssetsResearch.prices = property(
    AssetsResearch._get_prices,
    doc="Cached price table indexed by date with one column per ticker.",
)
AssetsResearch.returns = property(
    AssetsResearch._get_returns,
    doc="Cached daily percentage returns indexed by date.",
)

# Optional alias with standard class naming
if __name__ == "__main__":
    gspc = AssetsResearch(["^GSPC"], start="2020-09-30")

    print(gspc.annual_return())
