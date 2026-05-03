"""Compatibility adapters that preserve legacy portfolio interfaces.

The classes in this module keep historical names available while delegating the
actual implementation to the newer research, optimization, and portfolio
modules.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..optimization.mean_variance import PortfolioOptimization
from ..optimization.postmodern import PortfolioOptimizationPostModern
from ..research.assets_research import AssetsResearch


class PortfolioElementaryMetrics(PortfolioOptimization):
    """
    Backward-compatible portfolio metrics adapter.

    The historical `PortfolioElementaryMetrics` API expects the constructor and
    data workflow from the old `asset_allocation` package. This adapter keeps
    that public contract while delegating the actual implementation to the new
    optimization and portfolio layers.

    Notes
    -----
    This class intentionally inherits all behavior from `PortfolioOptimization`.
    """


class PortfolioPostModernMetrics(PortfolioOptimizationPostModern):
    """
    Backward-compatible downside metrics adapter.

    The adapter preserves the legacy asset-level post-modern methods while the
    real implementation now lives in `src.optimization.postmodern`.

    Notes
    -----
    This class intentionally inherits all behavior from
    `PortfolioOptimizationPostModern`.
    """


class PortfolioElementaryAnalysis(PortfolioElementaryMetrics):
    """
    Backward-compatible benchmark analysis adapter.

    The class retains the legacy constructor and CAPM-oriented methods while
    reusing the new research and portfolio foundations underneath.

    Parameters
    ----------
    tickers : iterable of str
        Asset symbols included in the portfolio.
    start : str
        First date requested for market data.
    end : str, optional
        Optional exclusive end date requested for market data.
    price_field : str, default "Close"
        Yahoo Finance price field used by the research layer.
    weight : iterable of float, optional
        Portfolio weights. The parent optimizer requires a valid weight vector.
    benchmark : str
        Benchmark ticker used for CAPM-oriented metrics.

    Raises
    ------
    ValueError
        If `benchmark` is omitted or blank.
    """

    def __init__(
        self,
        tickers: Iterable[str],
        start: str,
        end: Optional[str] = None,
        price_field: str = "Close",
        weight: Optional[Iterable[float]] = None,
        benchmark: Optional[str] = None,
    ) -> None:
        if benchmark is None:
            raise ValueError("benchmark must be provided.")

        super().__init__(
            tickers=tickers,
            start=start,
            end=end,
            price_field=price_field,
            weight=weight,
        )
        self.__benchmark = ""
        self.__benchmark_returns_cache = pd.DataFrame()
        self.benchmark = benchmark

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
        Build a benchmark analysis instance with equal asset weights.

        Parameters
        ----------
        tickers : iterable of str
            Asset symbols included in the portfolio.
        start : str
            First date requested for market data.
        benchmark : str
            Benchmark ticker used for CAPM-oriented metrics.
        end : str, optional
            Optional exclusive end date requested for market data.
        price_field : str, default "Close"
            Yahoo Finance price field used by the research layer.

        Returns
        -------
        PortfolioElementaryAnalysis
            Analysis instance initialized with equal weights.
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

    @property
    def benchmark(self) -> str:
        return self.__benchmark

    @benchmark.setter
    def benchmark(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("benchmark must be a non-empty ticker string.")
        self.__benchmark = value.strip()
        self.__benchmark_returns_cache = pd.DataFrame()

    def benchmark_returns(self) -> pd.Series:
        """
        Return benchmark daily returns.

        Returns
        -------
        pandas.Series
            Benchmark return series downloaded through `AssetsResearch`.
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
        Return annualized benchmark expected return.

        Returns
        -------
        float
            Mean daily benchmark return multiplied by 252.
        """
        returns = self.benchmark_returns()
        if returns.empty:
            raise ValueError("benchmark returns are empty.")
        return float(returns.mean() * 252.0)

    def _portfolio_daily_returns(self) -> pd.Series:
        assets_returns = self.get_returns()
        if assets_returns.empty:
            raise ValueError("portfolio returns are empty.")
        values = assets_returns.to_numpy(dtype=float) @ self.weight
        return pd.Series(values, index=assets_returns.index, name="portfolio")

    def _aligned_portfolio_benchmark_returns(self) -> pd.DataFrame:
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
        Return portfolio beta relative to the configured benchmark.

        Returns
        -------
        float
            Covariance between portfolio and benchmark returns divided by
            benchmark variance.
        """
        aligned = self._aligned_portfolio_benchmark_returns()
        benchmark_variance = float(aligned["benchmark"].var())
        if np.isclose(benchmark_variance, 0.0):
            raise ValueError("benchmark variance is zero, beta is undefined.")

        covariance = float(aligned["portfolio"].cov(aligned["benchmark"]))
        return float(covariance / benchmark_variance)

    def assets_beta(self) -> pd.Series:
        """
        Return per-asset beta relative to the configured benchmark.

        Returns
        -------
        pandas.Series
            Beta value for each asset ticker.
        """
        assets_returns = self.get_returns()
        if assets_returns.empty:
            raise ValueError("assets returns are empty.")

        benchmark_returns = self.benchmark_returns()
        if benchmark_returns.empty:
            raise ValueError("benchmark returns are empty.")

        aligned_assets, aligned_benchmark = assets_returns.align(
            benchmark_returns,
            join="inner",
            axis=0,
        )
        if len(aligned_assets) < 2:
            raise ValueError("not enough overlapping returns for CAPM metrics.")

        benchmark_variance = float(aligned_benchmark.var())
        if np.isclose(benchmark_variance, 0.0):
            raise ValueError("benchmark variance is zero, assets beta is undefined.")

        covariances = aligned_assets.apply(
            lambda asset_returns: asset_returns.cov(aligned_benchmark)
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
        Return portfolio expected return under CAPM.

        Parameters
        ----------
        risk_free_rate : float, default 0.0
            Annual risk-free rate.
        market_expected_return : float, optional
            Annual market return. If omitted, benchmark annual return is used.

        Returns
        -------
        float
            CAPM expected portfolio return.
        """
        if market_expected_return is None:
            market_expected_return = self.benchmark_annual_return()

        beta = self.portfolio_beta()
        expected_return = risk_free_rate + beta * (
            market_expected_return - risk_free_rate
        )
        return float(expected_return)

    def assets_capm_expected_return(
        self,
        risk_free_rate: float = 0.0,
        market_expected_return: Optional[float] = None,
    ) -> pd.Series:
        """
        Return per-asset expected returns under CAPM.

        Parameters
        ----------
        risk_free_rate : float, default 0.0
            Annual risk-free rate.
        market_expected_return : float, optional
            Annual market return. If omitted, benchmark annual return is used.

        Returns
        -------
        pandas.Series
            CAPM expected return for each asset ticker.
        """
        if market_expected_return is None:
            market_expected_return = self.benchmark_annual_return()

        betas = self.assets_beta()
        expected_return = risk_free_rate + betas * (
            market_expected_return - risk_free_rate
        )
        return expected_return


__all__ = [
    "PortfolioElementaryAnalysis",
    "PortfolioElementaryMetrics",
    "PortfolioPostModernMetrics",
]
