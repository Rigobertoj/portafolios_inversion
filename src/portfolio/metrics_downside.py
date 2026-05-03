"""Downside-oriented portfolio metrics.

This module computes portfolio-level semivariance, downside risk, upside risk,
Omega, and Sortino ratios from realized portfolio returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .metrics_basic import PortfolioBasicMetrics
from .portfolio import Portfolio


@dataclass
class PortfolioDownsideMetrics:
    """
    Compute downside-oriented metrics from the realized portfolio return series.

    Unlike the legacy post-modern layer, this class focuses on the portfolio as
    the unit of analysis. Asset-level post-modern functionality remains
    available through `PortfolioPostModernMetrics` during the migration.

    Parameters
    ----------
    portfolio : Portfolio
        Portfolio object containing aligned returns and weights.
    trading_days : int, default 252
        Number of trading days used to annualize downside and upside risk.

    Raises
    ------
    ValueError
        If `trading_days` is non-positive.
    """

    portfolio: Portfolio
    trading_days: int = 252

    def __post_init__(self) -> None:
        if self.trading_days <= 0:
            raise ValueError("trading_days must be greater than zero.")
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

    @staticmethod
    def _resolve_threshold(threshold: Optional[float]) -> float:
        return 0.0 if threshold is None else float(threshold)

    def _reference_adjusted_portfolio_returns(
        self,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> pd.Series:
        adjusted_threshold = self._resolve_threshold(threshold)
        portfolio_returns = self.portfolio.portfolio_returns()

        if benchmark_returns is None:
            return portfolio_returns - adjusted_threshold

        benchmark = self._normalize_benchmark_series(benchmark_returns)
        aligned = pd.concat(
            [
                portfolio_returns.rename("portfolio"),
                benchmark.rename("benchmark"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        if len(aligned) < 2:
            raise ValueError(
                "benchmark_returns must overlap portfolio returns with at least two observations."
            )
        return aligned["portfolio"] - (aligned["benchmark"] + adjusted_threshold)

    def portfolio_semivariance(
        self,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> float:
        """
        Return annualized portfolio semivariance under the selected reference.

        The reference is either a scalar minimum acceptable return or a
        benchmark-relative hurdle when `benchmark_returns` is provided.

        Parameters
        ----------
        threshold : float, optional
            Minimum acceptable return. Defaults to zero when omitted.
        benchmark_returns : pandas.Series or pandas.DataFrame, optional
            Benchmark return series used for benchmark-relative downside.

        Returns
        -------
        float
            Annualized portfolio semivariance.
        """
        adjusted = self._reference_adjusted_portfolio_returns(
            threshold=threshold,
            benchmark_returns=benchmark_returns,
        )
        annualized = adjusted.where(adjusted < 0.0, 0.0).var() * self.trading_days
        return float(max(annualized, 0.0))

    def portfolio_downside_risk(
        self,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> float:
        """
        Return annualized portfolio downside risk.

        Parameters
        ----------
        threshold : float, optional
            Minimum acceptable return. Defaults to zero when omitted.
        benchmark_returns : pandas.Series or pandas.DataFrame, optional
            Benchmark return series used for benchmark-relative downside.

        Returns
        -------
        float
            Square root of annualized semivariance.
        """
        semivariance = self.portfolio_semivariance(
            threshold=threshold,
            benchmark_returns=benchmark_returns,
        )
        return float(np.sqrt(semivariance))

    def portfolio_upside_risk(
        self,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> float:
        """
        Return annualized portfolio upside risk.

        Parameters
        ----------
        threshold : float, optional
            Minimum acceptable return. Defaults to zero when omitted.
        benchmark_returns : pandas.Series or pandas.DataFrame, optional
            Benchmark return series used for benchmark-relative upside.

        Returns
        -------
        float
            Annualized standard deviation of positive deviations.
        """
        adjusted = self._reference_adjusted_portfolio_returns(
            threshold=threshold,
            benchmark_returns=benchmark_returns,
        )
        upside = adjusted.where(adjusted > 0.0, 0.0).std() * np.sqrt(self.trading_days)
        return float(upside)

    def portfolio_omega_ratio(
        self,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> float:
        """
        Return the portfolio Omega ratio as upside risk over downside risk.

        Parameters
        ----------
        threshold : float, optional
            Minimum acceptable return. Defaults to zero when omitted.
        benchmark_returns : pandas.Series or pandas.DataFrame, optional
            Benchmark return series used for benchmark-relative risk.

        Returns
        -------
        float
            Upside risk divided by downside risk, or NaN when downside risk is
            zero.
        """
        downside = self.portfolio_downside_risk(
            threshold=threshold,
            benchmark_returns=benchmark_returns,
        )
        if np.isclose(downside, 0.0):
            return float("nan")

        upside = self.portfolio_upside_risk(
            threshold=threshold,
            benchmark_returns=benchmark_returns,
        )
        return float(upside / downside)

    def portfolio_sortino_ratio(
        self,
        risk_free_rate: float = 0.0,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> float:
        """
        Return the portfolio Sortino ratio.

        Parameters
        ----------
        risk_free_rate : float, default 0.0
            Annual risk-free rate subtracted from expected return.
        threshold : float, optional
            Minimum acceptable return used for downside risk.
        benchmark_returns : pandas.Series or pandas.DataFrame, optional
            Benchmark return series used for benchmark-relative downside.

        Returns
        -------
        float
            Excess annualized return divided by downside risk, or NaN when
            downside risk is zero.
        """
        downside = self.portfolio_downside_risk(
            threshold=threshold,
            benchmark_returns=benchmark_returns,
        )
        if np.isclose(downside, 0.0):
            return float("nan")
        expected_return = self._basic_metrics.portfolio_annual_return()
        return float((expected_return - risk_free_rate) / downside)

    def downside_risk(
        self,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> float:
        """Backward-friendly alias around `portfolio_downside_risk()`."""
        return self.portfolio_downside_risk(
            threshold=threshold,
            benchmark_returns=benchmark_returns,
        )

    def upside_risk(
        self,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> float:
        """Backward-friendly alias around `portfolio_upside_risk()`."""
        return self.portfolio_upside_risk(
            threshold=threshold,
            benchmark_returns=benchmark_returns,
        )

    def omega_ratio(
        self,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> float:
        """Backward-friendly alias around `portfolio_omega_ratio()`."""
        return self.portfolio_omega_ratio(
            threshold=threshold,
            benchmark_returns=benchmark_returns,
        )

    def sortino_ratio(
        self,
        risk_free_rate: float = 0.0,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> float:
        """Backward-friendly alias around `portfolio_sortino_ratio()`."""
        return self.portfolio_sortino_ratio(
            risk_free_rate=risk_free_rate,
            threshold=threshold,
            benchmark_returns=benchmark_returns,
        )


__all__ = [
    "PortfolioDownsideMetrics",
]
