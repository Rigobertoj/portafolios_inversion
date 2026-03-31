"""Portfolio-level performance analysis built on the new composition layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

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

    def _resolved_benchmark_returns(self) -> Optional[pd.Series]:
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
        benchmark = self._resolved_benchmark_returns()
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
        aligned = self._aligned_portfolio_benchmark_returns()
        benchmark_variance = float(aligned["benchmark"].var())
        if np.isclose(benchmark_variance, 0.0):
            return float("nan")
        covariance = float(aligned["portfolio"].cov(aligned["benchmark"]))
        return float(covariance / benchmark_variance)

    def benchmark_annual_return(self) -> float:
        """Return the annualized mean return of the configured benchmark."""
        benchmark = self._resolved_benchmark_returns()
        if benchmark is None:
            return float("nan")
        return float(benchmark.mean() * self.trading_days)

    def jensen_alpha(self, risk_free_rate: float = 0.0) -> float:
        """Return the Jensen alpha of the portfolio."""
        beta = self.beta()
        if np.isnan(beta):
            return float("nan")
        market_expected_return = self.benchmark_annual_return()
        capm_expected = risk_free_rate + beta * (market_expected_return - risk_free_rate)
        return float(self.expected_return() - capm_expected)

    def treynor_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Return the Treynor ratio of the portfolio."""
        beta = self.beta()
        if np.isnan(beta) or np.isclose(beta, 0.0):
            return float("nan")
        return float((self.expected_return() - risk_free_rate) / beta)

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

        if self._resolved_benchmark_returns() is not None:
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
