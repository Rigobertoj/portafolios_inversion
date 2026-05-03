"""Aggregate portfolio risk analysis.

`RiskAnalyzer` combines drawdown, tail-risk, volatility, and benchmark-relative
risk helpers behind a single portfolio-centered reporting API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..portfolio.portfolio import Portfolio
from .drawdown import PortfolioDrawdownAnalysis
from .tracking import PortfolioRelativeRisk
from .var_cvar import PortfolioTailRisk
from .volatility import PortfolioVolatilityAnalysis


@dataclass
class RiskAnalyzer:
    """
    Aggregate drawdown, tail-risk, and benchmark-relative risk metrics.

    The analyzer keeps a single portfolio-centered API for the most common
    risk measures while delegating the actual formulas to narrower classes.

    Parameters
    ----------
    portfolio : Portfolio
        Portfolio object used to generate realized returns and wealth paths.
    benchmark_returns : pandas.Series or pandas.DataFrame, optional
        Benchmark returns used for tracking error and information ratio.
    benchmark_prices : pandas.Series or pandas.DataFrame, optional
        Benchmark prices converted to returns when `benchmark_returns` is not
        supplied.
    benchmark_name : str, optional
        Display name assigned to normalized benchmark series.
    trading_days : int, default 252
        Number of trading days used to annualize volatility and relative-risk
        metrics.

    Raises
    ------
    ValueError
        If `trading_days` is non-positive.
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
        self._volatility = PortfolioVolatilityAnalysis(
            portfolio=self.portfolio,
            trading_days=self.trading_days,
        )
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

    def historical_variance(self) -> float:
        """Return the annualized historical variance of the portfolio."""
        return self._volatility.historical_variance()

    def historical_volatility(self) -> float:
        """Return the annualized historical volatility of the portfolio."""
        return self._volatility.historical_volatility()

    def ewma_variance_series(
        self,
        decay: float = 0.94,
        *,
        mean_adjust: bool = False,
        annualize: bool = False,
        initial_variance: Optional[float] = None,
    ) -> pd.Series:
        """Return the EWMA variance series of the portfolio."""
        return self._volatility.ewma_variance_series(
            decay=decay,
            mean_adjust=mean_adjust,
            annualize=annualize,
            initial_variance=initial_variance,
        )

    def ewma_volatility_series(
        self,
        decay: float = 0.94,
        *,
        mean_adjust: bool = False,
        annualize: bool = True,
        initial_variance: Optional[float] = None,
    ) -> pd.Series:
        """Return the EWMA volatility series of the portfolio."""
        return self._volatility.ewma_volatility_series(
            decay=decay,
            mean_adjust=mean_adjust,
            annualize=annualize,
            initial_variance=initial_variance,
        )

    def latest_ewma_variance(
        self,
        decay: float = 0.94,
        *,
        mean_adjust: bool = False,
        annualize: bool = False,
        initial_variance: Optional[float] = None,
    ) -> float:
        """Return the latest EWMA variance estimate of the portfolio."""
        return self._volatility.latest_ewma_variance(
            decay=decay,
            mean_adjust=mean_adjust,
            annualize=annualize,
            initial_variance=initial_variance,
        )

    def latest_ewma_volatility(
        self,
        decay: float = 0.94,
        *,
        mean_adjust: bool = False,
        annualize: bool = True,
        initial_variance: Optional[float] = None,
    ) -> float:
        """Return the latest EWMA volatility estimate of the portfolio."""
        return self._volatility.latest_ewma_volatility(
            decay=decay,
            mean_adjust=mean_adjust,
            annualize=annualize,
            initial_variance=initial_variance,
        )

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
        """
        Return a compact risk summary table for the portfolio.

        Parameters
        ----------
        initial_value : float, default 1.0
            Starting value used for drawdown calculations.
        confidence_level : float, default 0.95
            Confidence level used for VaR and CVaR.

        Returns
        -------
        pandas.DataFrame
            One-column table with max drawdown, VaR, CVaR, tracking error, and
            information ratio.
        """
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
