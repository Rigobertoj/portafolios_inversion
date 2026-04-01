"""Aggregate portfolio risk analysis built on the composition layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..portfolio.portfolio import Portfolio
from .drawdown import PortfolioDrawdownAnalysis
from .tracking import PortfolioRelativeRisk
from .var_cvar import PortfolioTailRisk


@dataclass
class RiskAnalyzer:
    """
    Aggregate drawdown, tail-risk, and benchmark-relative risk metrics.

    The analyzer keeps a single portfolio-centered API for the most common
    risk measures while delegating the actual formulas to narrower classes.
    """

    portfolio: Portfolio
    benchmark_returns: Optional[pd.Series | pd.DataFrame] = None
    benchmark_prices: Optional[pd.Series | pd.DataFrame] = None
    benchmark_name: Optional[str] = None
    trading_days: int = 252

    def __post_init__(self) -> None:
        if self.trading_days <= 0:
            raise ValueError("trading_days must be greater than zero.")
        self._drawdown = PortfolioDrawdownAnalysis(portfolio=self.portfolio)
        self._tail_risk = PortfolioTailRisk(portfolio=self.portfolio)
        self._relative_risk = PortfolioRelativeRisk(
            portfolio=self.portfolio,
            benchmark_returns=self.benchmark_returns,
            benchmark_prices=self.benchmark_prices,
            benchmark_name=self.benchmark_name,
            trading_days=self.trading_days,
        )

    def wealth_index(self, initial_value: float = 1.0) -> pd.Series:
        """Return the compounded wealth path used by the risk report."""
        return self._drawdown.wealth_index(initial_value=initial_value)

    def drawdown_series(self, initial_value: float = 1.0) -> pd.Series:
        """Return the drawdown series of the portfolio."""
        return self._drawdown.drawdown_series(initial_value=initial_value)

    def max_drawdown(self, initial_value: float = 1.0) -> float:
        """Return the worst realized drawdown of the portfolio."""
        return self._drawdown.max_drawdown(initial_value=initial_value)

    def historical_var(self, confidence_level: float = 0.95) -> float:
        """Return the historical daily Value at Risk."""
        return self._tail_risk.historical_var(confidence_level=confidence_level)

    def historical_cvar(self, confidence_level: float = 0.95) -> float:
        """Return the historical daily Conditional Value at Risk."""
        return self._tail_risk.historical_cvar(confidence_level=confidence_level)

    def active_returns(self) -> pd.Series:
        """Return the aligned active-return series against the configured benchmark."""
        return self._relative_risk.active_returns()

    def tracking_error(self) -> float:
        """Return the annualized tracking error against the configured benchmark."""
        return self._relative_risk.tracking_error()

    def information_ratio(self) -> float:
        """Return the annualized information ratio against the configured benchmark."""
        return self._relative_risk.information_ratio()

    def summary(
        self,
        *,
        initial_value: float = 1.0,
        confidence_level: float = 0.95,
    ) -> pd.DataFrame:
        """Return a compact risk summary table for the portfolio."""
        confidence_pct = int(round(float(confidence_level) * 100.0))
        metrics = {
            "Max Drawdown": self.max_drawdown(initial_value=initial_value),
            f"VaR {confidence_pct}%": self.historical_var(
                confidence_level=confidence_level,
            ),
            f"CVaR {confidence_pct}%": self.historical_cvar(
                confidence_level=confidence_level,
            ),
            "Tracking Error": float("nan"),
            "Information Ratio": float("nan"),
        }

        try:
            metrics["Tracking Error"] = self.tracking_error()
            metrics["Information Ratio"] = self.information_ratio()
        except ValueError:
            pass

        return pd.DataFrame({"value": pd.Series(metrics, dtype=float)})


__all__ = [
    "RiskAnalyzer",
]
