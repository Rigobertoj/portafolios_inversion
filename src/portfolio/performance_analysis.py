"""Portfolio-level performance analysis built on the new composition layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .benchmark_analysis import PortfolioBenchmarkAnalysis
from .metrics_basic import PortfolioBasicMetrics
from .metrics_downside import PortfolioDownsideMetrics
from .portfolio import Portfolio


@dataclass
class PortfolioPerformanceAnalysis:
    """
    Aggregate performance metrics for a weighted portfolio.

    The analysis works directly from the realized daily portfolio return series,
    which keeps the API independent from the legacy inheritance hierarchy while
    still supporting the same style of metrics used in the course notebooks.
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
        self._downside_metrics = PortfolioDownsideMetrics(
            portfolio=self.portfolio,
            trading_days=self.trading_days,
        )
        self._benchmark_analysis = PortfolioBenchmarkAnalysis(
            portfolio=self.portfolio,
            benchmark_returns=self.benchmark_returns,
            benchmark_prices=self.benchmark_prices,
            benchmark_name=self.benchmark_name,
            trading_days=self.trading_days,
        )

    def expected_return(self) -> float:
        """Return the annualized mean return of the portfolio."""
        return self._basic_metrics.portfolio_annual_return()

    def realized_return(self) -> float:
        """Return the effective realized return of the portfolio."""
        return self._basic_metrics.portfolio_realized_return()

    def volatility(self) -> float:
        """Return the annualized volatility of the portfolio."""
        return self._basic_metrics.portfolio_annual_volatility()

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Return the annualized Sharpe ratio of the portfolio."""
        try:
            return self._basic_metrics.portfolio_sharpe_ratio(
                free_rate=risk_free_rate,
            )
        except ValueError:
            return float("nan")

    def downside_risk(
        self,
        threshold: float = 0.0,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> float:
        """Return annualized downside risk for the portfolio."""
        return self._downside_metrics.portfolio_downside_risk(
            threshold=threshold,
            benchmark_returns=benchmark_returns,
        )

    def upside_risk(
        self,
        threshold: float = 0.0,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> float:
        """Return annualized upside risk for the portfolio."""
        return self._downside_metrics.portfolio_upside_risk(
            threshold=threshold,
            benchmark_returns=benchmark_returns,
        )

    def omega_ratio(
        self,
        threshold: float = 0.0,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> float:
        """Return the Omega ratio of the portfolio."""
        return self._downside_metrics.portfolio_omega_ratio(
            threshold=threshold,
            benchmark_returns=benchmark_returns,
        )

    def sortino_ratio(
        self,
        risk_free_rate: float = 0.0,
        threshold: float = 0.0,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> float:
        """Return the Sortino ratio of the portfolio."""
        return self._downside_metrics.portfolio_sortino_ratio(
            risk_free_rate=risk_free_rate,
            threshold=threshold,
            benchmark_returns=benchmark_returns,
        )

    def beta(self) -> float:
        """Return the portfolio beta relative to the configured benchmark."""
        return self._benchmark_analysis.portfolio_beta()

    def benchmark_annual_return(self) -> float:
        """Return the annualized mean return of the configured benchmark."""
        return self._benchmark_analysis.benchmark_annual_return()

    def jensen_alpha(self, risk_free_rate: float = 0.0) -> float:
        """Return the Jensen alpha of the portfolio."""
        return self._benchmark_analysis.portfolio_jensen_alpha(
            risk_free_rate=risk_free_rate,
        )

    def treynor_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Return the Treynor ratio of the portfolio."""
        return self._benchmark_analysis.portfolio_treynor_ratio(
            risk_free_rate=risk_free_rate,
        )

    def metrics_table(
        self,
        *,
        risk_free_rate: float = 0.0,
        threshold: float = 0.0,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build a metrics table aligned with the naming used in the course work.
        """
        metrics = {
            "Rendimiento esperado": self.expected_return(),
            "Rendimiento realizado": self.realized_return(),
            "Volatilidad": self.volatility(),
            "Ratio de sharpe": self.sharpe_ratio(risk_free_rate=risk_free_rate),
            "Downside risk": self.downside_risk(
                threshold=threshold,
                benchmark_returns=benchmark_returns,
            ),
            "Upside risk": self.upside_risk(
                threshold=threshold,
                benchmark_returns=benchmark_returns,
            ),
            "Omega": self.omega_ratio(
                threshold=threshold,
                benchmark_returns=benchmark_returns,
            ),
            "Beta": float("nan"),
            "Alpha de Jensen": float("nan"),
            "Ratio de Treynor": float("nan"),
            "Ratio de Sortino": self.sortino_ratio(
                risk_free_rate=risk_free_rate,
                threshold=threshold,
                benchmark_returns=benchmark_returns,
            ),
        }

        if self._benchmark_analysis.resolved_benchmark_returns() is not None:
            metrics["Beta"] = self.beta()
            metrics["Alpha de Jensen"] = self.jensen_alpha(
                risk_free_rate=risk_free_rate,
            )
            metrics["Ratio de Treynor"] = self.treynor_ratio(
                risk_free_rate=risk_free_rate,
            )

        return pd.DataFrame(
            {"value": pd.Series(metrics, dtype=float)}
        )

    def summary(
        self,
        *,
        risk_free_rate: float = 0.0,
        threshold: float = 0.0,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Alias around `metrics_table()` for a more report-oriented API."""
        return self.metrics_table(
            risk_free_rate=risk_free_rate,
            threshold=threshold,
            benchmark_returns=benchmark_returns,
        )


__all__ = [
    "PortfolioPerformanceAnalysis",
]
