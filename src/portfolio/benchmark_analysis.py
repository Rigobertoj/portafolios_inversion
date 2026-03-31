"""Benchmark-aware portfolio analysis built on top of the composition layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..asset_allocation.PortfolioElementaryAnalysis import PortfolioElementaryAnalysis
from .metrics_basic import PortfolioBasicMetrics
from .portfolio import Portfolio


@dataclass
class PortfolioBenchmarkAnalysis:
    """
    Compute benchmark-relative portfolio analysis metrics by composition.

    The class focuses on portfolio-level benchmark analysis only. Legacy
    asset-level functionality remains available through
    `PortfolioElementaryAnalysis` while the new public API is migrated.
    """

    portfolio: Portfolio
    benchmark_returns: Optional[pd.Series | pd.DataFrame] = None
    benchmark_prices: Optional[pd.Series | pd.DataFrame] = None
    benchmark_name: Optional[str] = None
    trading_days: int = 252

    def __post_init__(self) -> None:
        if self.trading_days <= 0:
            raise ValueError("trading_days must be greater than zero.")
        if self.benchmark_returns is not None and self.benchmark_prices is not None:
            raise ValueError("provide benchmark_returns or benchmark_prices, not both.")
        self._basic_metrics = PortfolioBasicMetrics(
            portfolio=self.portfolio,
            trading_days=self.trading_days,
        )

    @staticmethod
    def _normalize_benchmark_series(
        data: pd.Series | pd.DataFrame,
        *,
        label: Optional[str] = None,
    ) -> pd.Series:
        if isinstance(data, pd.Series):
            series = data.sort_index().dropna().copy()
        elif isinstance(data, pd.DataFrame):
            cleaned = data.sort_index().dropna().copy()
            if cleaned.shape[1] != 1:
                raise ValueError("benchmark data must contain exactly one column.")
            series = cleaned.iloc[:, 0]
        else:
            raise TypeError("benchmark data must be a pandas Series or DataFrame.")

        series = pd.to_numeric(series, errors="coerce").dropna()
        if series.empty:
            raise ValueError("benchmark data must contain at least one observation.")
        if label is not None:
            series.name = label
        elif series.name is None:
            series.name = "Benchmark"
        return series

    def resolved_benchmark_returns(self) -> Optional[pd.Series]:
        """Return the normalized benchmark return series."""
        if self.benchmark_returns is not None:
            return self._normalize_benchmark_series(
                self.benchmark_returns,
                label=self.benchmark_name,
            )

        if self.benchmark_prices is not None:
            prices = self._normalize_benchmark_series(
                self.benchmark_prices,
                label=self.benchmark_name,
            )
            returns = prices.pct_change().dropna()
            returns.name = prices.name
            return returns

        return None

    def _aligned_portfolio_benchmark_returns(self) -> pd.DataFrame:
        benchmark = self.resolved_benchmark_returns()
        if benchmark is None:
            raise ValueError("benchmark returns are required for benchmark metrics.")

        aligned = pd.concat(
            [
                self.portfolio.portfolio_returns().rename("portfolio"),
                benchmark.rename("benchmark"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        if len(aligned) < 2:
            raise ValueError("not enough overlapping returns for benchmark metrics.")
        return aligned

    def benchmark_annual_return(self) -> float:
        """Return the annualized benchmark expected return."""
        benchmark = self.resolved_benchmark_returns()
        if benchmark is None:
            return float("nan")
        return float(benchmark.mean() * self.trading_days)

    def portfolio_beta(self) -> float:
        """Return the portfolio beta relative to the configured benchmark."""
        aligned = self._aligned_portfolio_benchmark_returns()
        benchmark_variance = float(aligned["benchmark"].var())
        if np.isclose(benchmark_variance, 0.0):
            return float("nan")

        covariance = float(aligned["portfolio"].cov(aligned["benchmark"]))
        return float(covariance / benchmark_variance)

    def portfolio_capm_expected_return(
        self,
        risk_free_rate: float = 0.0,
        market_expected_return: Optional[float] = None,
    ) -> float:
        """Return the portfolio expected return under CAPM."""
        if market_expected_return is None:
            market_expected_return = self.benchmark_annual_return()

        beta = self.portfolio_beta()
        if np.isnan(beta):
            return float("nan")
        return float(
            risk_free_rate + beta * (market_expected_return - risk_free_rate)
        )

    def portfolio_jensen_alpha(self, risk_free_rate: float = 0.0) -> float:
        """Return the Jensen alpha of the portfolio."""
        capm_expected = self.portfolio_capm_expected_return(
            risk_free_rate=risk_free_rate,
        )
        if np.isnan(capm_expected):
            return float("nan")
        return float(self._basic_metrics.portfolio_annual_return() - capm_expected)

    def portfolio_treynor_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Return the Treynor ratio of the portfolio."""
        beta = self.portfolio_beta()
        if np.isnan(beta) or np.isclose(beta, 0.0):
            return float("nan")
        return float(
            (self._basic_metrics.portfolio_annual_return() - risk_free_rate) / beta
        )

    def beta(self) -> float:
        """Backward-friendly alias around `portfolio_beta()`."""
        return self.portfolio_beta()

    def capm_expected_return(
        self,
        risk_free_rate: float = 0.0,
        market_expected_return: Optional[float] = None,
    ) -> float:
        """Backward-friendly alias around `portfolio_capm_expected_return()`."""
        return self.portfolio_capm_expected_return(
            risk_free_rate=risk_free_rate,
            market_expected_return=market_expected_return,
        )

    def jensen_alpha(self, risk_free_rate: float = 0.0) -> float:
        """Backward-friendly alias around `portfolio_jensen_alpha()`."""
        return self.portfolio_jensen_alpha(risk_free_rate=risk_free_rate)

    def treynor_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Backward-friendly alias around `portfolio_treynor_ratio()`."""
        return self.portfolio_treynor_ratio(risk_free_rate=risk_free_rate)


__all__ = [
    "PortfolioBenchmarkAnalysis",
    "PortfolioElementaryAnalysis",
]
