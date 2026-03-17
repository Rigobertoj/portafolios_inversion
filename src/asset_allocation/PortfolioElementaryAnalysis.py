from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from security_selection.AssetsResearch import AssetsResearch
    from PortfolioElementaryMetrics import PortfolioElementaryMetrics
else:
    try:
        from ..security_selection.AssetsResearch import AssetsResearch
    except ImportError:
        from security_selection.AssetsResearch import AssetsResearch
    from .PortfolioElementaryMetrics import PortfolioElementaryMetrics


@dataclass
class PortfolioElementaryAnalysis(PortfolioElementaryMetrics):
    """
    Compare a weighted portfolio against a benchmark and compute CAPM metrics.

    `PortfolioElementaryAnalysis` extends `PortfolioElementaryMetrics` by adding
    a benchmark asset and the analytics required to compare the portfolio
    against that benchmark. The class provides benchmark returns, beta
    estimation, and CAPM-based expected return calculations for both the whole
    portfolio and each individual asset.

    Parameters
    ----------
    tickers : Iterable[str]
        Asset symbols included in the portfolio.
    start : str
        Start date used to download historical market data.
    end : str | None, default None
        Optional end date used to download historical market data.
    price_field : str, default "Close"
        Price column extracted from Yahoo Finance data.
    weight : Iterable[float]
        Portfolio weights aligned with `tickers`. The vector must sum to 1.
    benchmark : str
        Ticker used as the market benchmark, for example `^GSPC`.

    Attributes
    ----------
    benchmark : str
        Benchmark ticker used in portfolio-vs-market analysis.

    Methods
    -------
    with_equal_weights(tickers, start, benchmark, end=None, price_field="Close")
        Build an instance using equal weights across all assets.
    benchmark_returns()
        Download or return cached daily benchmark returns.
    benchmark_annual_return()
        Compute the benchmark annualized return.
    portfolio_beta()
        Compute the beta of the weighted portfolio versus the benchmark.
    assets_beta()
        Compute the beta of each asset versus the benchmark.
    portfolio_capm_expected_return(risk_free_rate=0.0, market_expected_return=None)
        Compute the expected portfolio return under CAPM.
    assets_capm_expected_return(risk_free_rate=0.0, market_expected_return=None)
        Compute the expected return of each asset under CAPM.

    Inherited API
    -------------
    Since this class inherits from `PortfolioElementaryMetrics`, the IDE will
    also expose portfolio-level metrics such as `portfolio_annual_return`,
    `portfolio_annual_volatility`, and `portfolio_sharpe_ratio`, plus the
    research API inherited from `AssetsResearch`.

    Notes
    -----
    - Benchmark returns are cached internally and refreshed automatically when
      `benchmark` changes.
    - CAPM calculations use the benchmark annualized return when
      `market_expected_return` is not provided explicitly.

    Examples
    --------
    >>> weights = np.array([0.4, 0.35, 0.25])
    >>> analysis = PortfolioElementaryAnalysis(
    ...     tickers=["AAPL", "JPM", "MSFT"],
    ...     start="2024-01-01",
    ...     weight=weights,
    ...     benchmark="^GSPC",
    ... )
    >>> analysis.portfolio_beta()
    >>> analysis.assets_capm_expected_return(risk_free_rate=0.05)
    """
    benchmark: InitVar[str] = field(kw_only=True)

    __benchmark: str = field(init=False, repr=False)
    __benchmark_returns_cache: pd.DataFrame = field(
        init=False, repr=False, default_factory=pd.DataFrame
    )

    def __post_init__(
        self,
        tickers: Iterable[str],
        start: str,
        end: Optional[str],
        price_field: str,
        weight: Iterable[float],
        benchmark: str,
    ) -> None:
        """
        Initialize portfolio metrics inputs and validate the benchmark ticker.

        Parameters
        ----------
        tickers : Iterable[str]
            Symbols included in the portfolio.
        start : str
            Start date for the historical window.
        end : str | None
            Optional end date for the historical window.
        price_field : str
            Price column used to build the dataset.
        weight : Iterable[float]
            Portfolio weights associated with `tickers`.
        benchmark : str
            Ticker used as the market benchmark.
        """
        super().__post_init__(
            tickers=tickers,
            start=start,
            end=end,
            price_field=price_field,
            weight=weight,
        )
        self._set_benchmark(benchmark)

    @classmethod
    def with_equal_weights(
        cls,
        tickers: Iterable[str],
        start: str,
        benchmark: str,
        end: Optional[str] = None,
        price_field: str = "Close",
    ) -> "PortfolioElementaryAnalysis":
        """
        Create an analysis instance with equal weights across all assets.

        Parameters
        ----------
        tickers : Iterable[str]
            Symbols included in the portfolio.
        start : str
            Start date for the historical window.
        benchmark : str
            Ticker used as the market benchmark.
        end : str | None, default None
            Optional end date for the historical window.
        price_field : str, default "Close"
            Price column used to build the dataset.

        Returns
        -------
        PortfolioElementaryAnalysis
            Instance configured with uniform weights.

        Raises
        ------
        ValueError
            If no valid tickers are provided.
        """
        normalized_tickers = AssetsResearch._normalize_tickers(tickers)
        if not normalized_tickers:
            raise ValueError("tickers must contain at least one symbol.")
        weights = np.ones(len(normalized_tickers), dtype=float) / len(normalized_tickers)
        return cls(
            tickers=normalized_tickers,
            start=start,
            end=end,
            price_field=price_field,
            weight=weights,
            benchmark=benchmark,
        )

    def _get_benchmark(self) -> str:
        return self.__benchmark

    def _set_benchmark(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("benchmark must be a non-empty ticker string.")
        self.__benchmark = value.strip()
        self.__benchmark_returns_cache = pd.DataFrame()

    def benchmark_returns(self) -> pd.Series:
        """
        Return daily benchmark returns, downloading them if needed.

        Returns
        -------
        pandas.Series
            Daily return series for the configured benchmark.
        """
        if self.__benchmark_returns_cache.empty:
            benchmark_research = AssetsResearch(
                tickers=[self.benchmark],
                start=self.start,
                end=self.end,
                price_field=self.price_field,
            )
            self.__benchmark_returns_cache = benchmark_research.get_returns()

        if self.benchmark in self.__benchmark_returns_cache.columns:
            return self.__benchmark_returns_cache[self.benchmark].copy()

        return self.__benchmark_returns_cache.iloc[:, 0].copy()

    def benchmark_annual_return(self) -> float:
        """
        Compute the benchmark annualized expected return.

        Returns
        -------
        float
            Mean benchmark daily return multiplied by 252 trading days.

        Raises
        ------
        ValueError
            If benchmark returns are empty.
        """
        returns = self.benchmark_returns()
        if returns.empty:
            raise ValueError("benchmark returns are empty.")
        return float(returns.mean() * 252)

    def _portfolio_daily_returns(self) -> pd.Series:
        """Compute daily portfolio returns from asset returns and weights."""
        assets_returns = self.get_returns()
        if assets_returns.empty:
            raise ValueError("portfolio returns are empty.")
        values = assets_returns.to_numpy(dtype=float) @ self.weight
        return pd.Series(values, index=assets_returns.index, name="portfolio")

    def _aligned_portfolio_benchmark_returns(self) -> pd.DataFrame:
        """Align portfolio and benchmark returns on their shared dates."""
        aligned = pd.concat(
            [
                self._portfolio_daily_returns(),
                self.benchmark_returns().rename("benchmark"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        if len(aligned) < 2:
            raise ValueError("not enough overlapping returns for CAPM metrics.")
        return aligned

    def portfolio_beta(self) -> float:
        """
        Compute the beta of the weighted portfolio against the benchmark.

        Returns
        -------
        float
            Portfolio beta defined as covariance with the benchmark divided by
            benchmark variance.

        Raises
        ------
        ValueError
            If the benchmark variance is zero or there is not enough overlap in
            the return series.
        """
        aligned = self._aligned_portfolio_benchmark_returns()
        benchmark_variance = float(aligned["benchmark"].var())
        if np.isclose(benchmark_variance, 0.0):
            raise ValueError("benchmark variance is zero, beta is undefined.")
        
        covariance = float(aligned["portfolio"].cov(aligned["benchmark"]))
        
        return float(covariance / benchmark_variance)

    def assets_beta(self) -> pd.Series:
        """
        Compute the beta of each asset against the benchmark.

        Returns
        -------
        pandas.Series
            Beta per asset, indexed by ticker.

        Raises
        ------
        ValueError
            If asset returns are empty, benchmark returns are empty, or the
            benchmark variance is zero.
        """
        assets_returns = self.get_returns()
        if assets_returns.empty:
            raise ValueError("assets returns are empty.")

        benchmark_returns = self.benchmark_returns()
        if benchmark_returns.empty:
            raise ValueError("benchmark returns are empty.")

        benchmark_variance = float(benchmark_returns.var())
        if np.isclose(benchmark_variance, 0.0):
            raise ValueError("benchmark variance is zero, assets beta is undefined.")

        covariances = assets_returns.apply(
            lambda asset_returns: asset_returns.cov(benchmark_returns)
        )
        betas = covariances / benchmark_variance
        betas.name = "beta"
        return betas

    def portfolio_capm_expected_return(
        self,
        risk_free_rate: float = 0.0,
        market_expected_return: Optional[float] = None,
    ) -> float:
        """
        Compute the portfolio expected return under CAPM.

        Parameters
        ----------
        risk_free_rate : float, default 0.0
            Risk-free rate used by the CAPM formula.
        market_expected_return : float | None, default None
            Expected market return. If omitted, the annualized benchmark return
            is used.

        Returns
        -------
        float
            Expected portfolio return according to CAPM.
        """
        
        if market_expected_return is None:
            market_expected_return = self.benchmark_annual_return()
        
        beta = self.portfolio_beta()
        
        expected_return = risk_free_rate + beta * (market_expected_return - risk_free_rate)
        
        return float(expected_return)
    
    def assets_capm_expected_return(
        self,
        risk_free_rate: float = 0.0,
        market_expected_return: Optional[float] = None,
    ) -> pd.Series:
        """
        Compute the expected return of each asset under CAPM.

        Parameters
        ----------
        risk_free_rate : float, default 0.0
            Risk-free rate used by the CAPM formula.
        market_expected_return : float | None, default None
            Expected market return. If omitted, the annualized benchmark return
            is used.

        Returns
        -------
        pandas.Series
            CAPM expected return per asset, indexed by ticker.
        """
        
        if market_expected_return is None:
                market_expected_return = self.benchmark_annual_return()
        
        betas = self.assets_beta()
        
        expected_return = risk_free_rate + betas * (market_expected_return - risk_free_rate)

        return expected_return


PortfolioElementaryAnalysis.benchmark = property(
    PortfolioElementaryAnalysis._get_benchmark,
    PortfolioElementaryAnalysis._set_benchmark,
    doc="Ticker used as the benchmark in beta and CAPM calculations.",
)


def _main_() -> None:
    tickets = ["BOH", "AAPL", "JPM"]
    start_date = "2025-01-01"
    end_date = "2026-02-20"

    weight = np.ones(len(tickets)) / len(tickets)

    benchmark = "^GSPC"

    risk_rate_free = 0.055

    Portfolio1 = PortfolioElementaryAnalysis(
        tickers=tickets,
        start=start_date,
        end=end_date,
        weight=weight,
        benchmark=benchmark
    )
    
    #portfolio_beta = Portfolio1.portfolio_beta()
    #assets_beta = Portfolio1.assets_beta()
    
    #print("portfolio beta: ",portfolio_beta)
    #print("assets beta: ", assets_beta)
    
    portfolio_capm = Portfolio1.portfolio_capm_expected_return(risk_free_rate = risk_rate_free)
    assets_capm = Portfolio1.assets_capm_expected_return(risk_free_rate = risk_rate_free)
    
    print("CAPM Portfolio: ",portfolio_capm )
    print("CAPM Assets: ", assets_capm)
    
    print(Portfolio1.correlation())
    return

if __name__ == "__main__":
    _main_()
