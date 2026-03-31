"""Benchmark-relative risk analysis built on the portfolio composition layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..portfolio import Portfolio, PortfolioBenchmarkAnalysis


@dataclass
class PortfolioRelativeRisk:
    """Compute benchmark-relative risk metrics such as tracking error."""

    portfolio: Portfolio
    benchmark_returns: Optional[pd.Series | pd.DataFrame] = None
    benchmark_prices: Optional[pd.Series | pd.DataFrame] = None
    benchmark_name: Optional[str] = None
    trading_days: int = 252

    def __post_init__(self) -> None:
        self._benchmark_analysis = PortfolioBenchmarkAnalysis(
            portfolio=self.portfolio,
            benchmark_returns=self.benchmark_returns,
            benchmark_prices=self.benchmark_prices,
            benchmark_name=self.benchmark_name,
            trading_days=self.trading_days,
        )

    def active_returns(self) -> pd.Series:
        """Return the aligned daily active-return series of the portfolio."""
        benchmark = self._benchmark_analysis.resolved_benchmark_returns()
        if benchmark is None:
            raise ValueError("benchmark returns are required for relative-risk metrics.")

        aligned = pd.concat(
            [
                self.portfolio.portfolio_returns().rename("portfolio"),
                benchmark.rename("benchmark"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        if len(aligned) < 2:
            raise ValueError("not enough overlapping returns for relative-risk metrics.")

        active = aligned["portfolio"] - aligned["benchmark"]
        active.name = self.portfolio.name
        return active

    def tracking_error(self) -> float:
        """Return the annualized tracking error of the portfolio."""
        active = self.active_returns()
        return float(active.std() * np.sqrt(self.trading_days))

    def information_ratio(self) -> float:
        """Return the annualized information ratio of the portfolio."""
        tracking_error = self.tracking_error()
        if np.isclose(tracking_error, 0.0):
            return float("nan")

        active = self.active_returns()
        annualized_active_return = float(active.mean() * self.trading_days)
        return float(annualized_active_return / tracking_error)


__all__ = [
    "PortfolioRelativeRisk",
]
