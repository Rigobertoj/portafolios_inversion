"""Tail-risk analysis built on the portfolio composition layer."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..portfolio.portfolio import Portfolio


@dataclass
class PortfolioTailRisk:
    """Compute historical tail-risk metrics from portfolio returns."""

    portfolio: Portfolio

    @staticmethod
    def _validate_confidence_level(confidence_level: float) -> float:
        level = float(confidence_level)
        if not 0.0 < level < 1.0:
            raise ValueError("confidence_level must be strictly between 0 and 1.")
        return level

    def portfolio_returns(self) -> pd.Series:
        """Return the daily realized return series used by tail-risk metrics."""
        return self.portfolio.portfolio_returns()

    def historical_var(self, confidence_level: float = 0.95) -> float:
        """
        Return the historical Value at Risk as a positive loss threshold.

        The metric is computed from the lower tail of historical daily returns.
        For example, a 95% VaR of `0.02` means a 2% one-period loss threshold.
        """
        level = self._validate_confidence_level(confidence_level)
        returns = self.portfolio_returns()
        quantile = float(returns.quantile(1.0 - level))
        return float(max(-quantile, 0.0))

    def historical_cvar(self, confidence_level: float = 0.95) -> float:
        """
        Return the historical Conditional VaR as a positive expected tail loss.

        CVaR is the average loss of observations at or beyond the VaR cutoff.
        """
        level = self._validate_confidence_level(confidence_level)
        returns = self.portfolio_returns()
        cutoff = float(returns.quantile(1.0 - level))
        tail = returns[returns <= cutoff]
        if tail.empty:
            return float("nan")
        return float(max(-tail.mean(), 0.0))


__all__ = [
    "PortfolioTailRisk",
]
