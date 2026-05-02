"""Volatility-oriented risk analysis built on the portfolio composition layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..portfolio.portfolio import Portfolio


@dataclass
class PortfolioVolatilityAnalysis:
    """Compute historical and EWMA volatility measures from portfolio returns."""

    portfolio: Portfolio
    trading_days: int = 252

    def __post_init__(self) -> None:
        if self.trading_days <= 0:
            raise ValueError("trading_days must be greater than zero.")

    @staticmethod
    def _validate_decay(decay: float) -> float:
        value = float(decay)
        if not 0.0 < value < 1.0:
            raise ValueError("decay must be strictly between 0 and 1.")
        return value

    @staticmethod
    def _validate_initial_variance(initial_variance: Optional[float]) -> Optional[float]:
        if initial_variance is None:
            return None

        value = float(initial_variance)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("initial_variance must be a finite non-negative value.")
        return value

    def portfolio_returns(self) -> pd.Series:
        """Return the daily realized return series used by volatility metrics."""
        return self.portfolio.portfolio_returns()

    def historical_variance(self) -> float:
        """Return the annualized variance of the portfolio."""
        returns = self.portfolio_returns()
        variance = float(returns.var() * self.trading_days)
        return max(variance, 0.0)

    def historical_volatility(self) -> float:
        """Return the annualized historical volatility of the portfolio."""
        returns = self.portfolio_returns()
        return float(returns.std() * np.sqrt(self.trading_days))

    def _adjusted_returns(self, mean_adjust: bool = False) -> pd.Series:
        returns = self.portfolio_returns()
        if mean_adjust:
            returns = returns - returns.mean()
        return returns

    def ewma_variance_series(
        self,
        decay: float = 0.94,
        *,
        mean_adjust: bool = False,
        annualize: bool = False,
        initial_variance: Optional[float] = None,
    ) -> pd.Series:
        """
        Return the EWMA variance series aligned with the portfolio returns.

        Each point updates the previous conditional variance estimate using the
        current realized return and the selected decay parameter.
        """
        lambda_ = self._validate_decay(decay)
        seed_variance = self._validate_initial_variance(initial_variance)
        adjusted_returns = self._adjusted_returns(mean_adjust=mean_adjust)
        squared_returns = adjusted_returns.pow(2)

        ewma_values = np.empty(len(squared_returns), dtype=float)
        previous_variance = (
            float(squared_returns.iloc[0])
            if seed_variance is None
            else float(seed_variance)
        )

        for idx, squared_return in enumerate(squared_returns.to_numpy(dtype=float)):
            previous_variance = (
                lambda_ * previous_variance + (1.0 - lambda_) * squared_return
            )
            ewma_values[idx] = previous_variance

        variance_series = pd.Series(
            ewma_values,
            index=adjusted_returns.index,
            name=self.portfolio.name,
        )
        if annualize:
            variance_series = variance_series * self.trading_days
            variance_series.name = self.portfolio.name
        return variance_series

    def ewma_volatility_series(
        self,
        decay: float = 0.94,
        *,
        mean_adjust: bool = False,
        annualize: bool = True,
        initial_variance: Optional[float] = None,
    ) -> pd.Series:
        """Return the EWMA volatility series of the portfolio."""
        variance_series = self.ewma_variance_series(
            decay=decay,
            mean_adjust=mean_adjust,
            annualize=annualize,
            initial_variance=initial_variance,
        )
        volatility_series = variance_series.clip(lower=0.0).pow(0.5)
        volatility_series.name = self.portfolio.name
        return volatility_series

    def latest_ewma_variance(
        self,
        decay: float = 0.94,
        *,
        mean_adjust: bool = False,
        annualize: bool = False,
        initial_variance: Optional[float] = None,
    ) -> float:
        """Return the latest EWMA variance estimate of the portfolio."""
        return float(
            self.ewma_variance_series(
                decay=decay,
                mean_adjust=mean_adjust,
                annualize=annualize,
                initial_variance=initial_variance,
            ).iloc[-1]
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
        return float(
            self.ewma_volatility_series(
                decay=decay,
                mean_adjust=mean_adjust,
                annualize=annualize,
                initial_variance=initial_variance,
            ).iloc[-1]
        )


__all__ = [
    "PortfolioVolatilityAnalysis",
]
