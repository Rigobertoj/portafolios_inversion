"""Aggregate portfolio performance analysis.

The analysis object combines basic return metrics, downside metrics, and
benchmark-aware ratios into a single report-oriented interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .benchmark_analysis import PortfolioBenchmarkAnalysis
from .metrics_basic import PortfolioBasicMetrics
from .metrics_downside import PortfolioDownsideMetrics
from .performance_metrics import PerformanceMetricsCalculator
from .portfolio import Portfolio


@dataclass
class PortfolioPerformanceAnalysis:
    """
    Aggregate performance metrics for a weighted portfolio.

    The analysis works directly from the realized daily portfolio return series,
    which keeps the API independent from the legacy inheritance hierarchy while
    still supporting the same style of metrics used in the course notebooks.

    Parameters
    ----------
    portfolio : Portfolio
        Portfolio object containing aligned returns and weights.
    benchmark_returns : pandas.Series or pandas.DataFrame, optional
        Benchmark returns used for beta, Jensen alpha, and Treynor ratio.
    benchmark_prices : pandas.Series or pandas.DataFrame, optional
        Benchmark prices converted to returns when `benchmark_returns` is not
        supplied.
    benchmark_name : str, optional
        Display name assigned to normalized benchmark series.
    trading_days : int, default 252
        Number of trading days used to annualize metrics.

    Raises
    ------
    ValueError
        If both benchmark returns and prices are supplied, or if `trading_days`
        is non-positive.
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

        Parameters
        ----------
        risk_free_rate : float, default 0.0
            Annual risk-free rate used by Sharpe, Jensen alpha, Treynor, and
            Sortino calculations.
        threshold : float, default 0.0
            Minimum acceptable return used by downside metrics.
        benchmark_returns : pandas.Series or pandas.DataFrame, optional
            Optional benchmark returns used specifically for downside and Omega
            calculations in this table.

        Returns
        -------
        pandas.DataFrame
            One-column metrics table with portfolio performance and optional
            benchmark-relative measures.
        """
        calculator = PerformanceMetricsCalculator(
            returns=self.portfolio.portfolio_returns(),
            evolution=self.portfolio.wealth_index(initial_value=1.0),
            initial_value=1.0,
            benchmark_returns=self._benchmark_analysis.resolved_benchmark_returns(),
            downside_benchmark_returns=benchmark_returns,
            risk_free_rate=risk_free_rate,
            threshold=threshold,
            trading_days=self.trading_days,
            use_evolution_returns=True,
        )
        table = calculator.metrics_table()
        return pd.DataFrame({"value": table.iloc[:, 0].astype(float)})

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
