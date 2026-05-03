"""Basic portfolio return and volatility metrics.

This module computes portfolio-level paths, returns, variance, volatility, and
Sharpe-style ratios from a composed `Portfolio`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .portfolio import Portfolio


@dataclass
class PortfolioBasicMetrics:
    """
    Compute basic return and volatility metrics from a composed `Portfolio`.

    This class is the first non-alias metrics implementation in the new
    `portfolio` package. It exposes a naming style compatible with the existing
    project while removing the need to inherit directly from `AssetsResearch`.

    Parameters
    ----------
    portfolio : Portfolio
        Portfolio object containing aligned prices, returns, tickers, and
        weights.
    trading_days : int, default 252
        Number of trading days used to annualize returns, variance, and
        volatility.

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

    def _resolve_portfolio(
        self,
        weight: Optional[Iterable[float]] = None,
    ) -> Portfolio:
        if weight is None:
            return self.portfolio
        return self.portfolio.with_weights(weight)

    def portfolio_path(
        self,
        weight: Optional[Iterable[float]] = None,
    ) -> pd.Series:
        """
        Return the weighted price path of the portfolio assets.

        Parameters
        ----------
        weight : iterable of float, optional
            Candidate weights. If omitted, the portfolio's current weights are
            used.

        Returns
        -------
        pandas.Series
            Weighted asset price path indexed by date.
        """
        resolved = self._resolve_portfolio(weight)
        values = resolved.asset_prices().to_numpy(dtype=float) @ resolved.weights
        return pd.Series(values, index=resolved.prices.index, name=resolved.name)

    def portfolio_returns(
        self,
        weight: Optional[Iterable[float]] = None,
    ) -> pd.Series:
        """
        Return the weighted daily return series of the portfolio.

        Parameters
        ----------
        weight : iterable of float, optional
            Candidate weights. If omitted, the portfolio's current weights are
            used.

        Returns
        -------
        pandas.Series
            Daily portfolio returns indexed by date.
        """
        resolved = self._resolve_portfolio(weight)
        return resolved.portfolio_returns()

    def portfolio_annual_return(
        self,
        weight: Optional[Iterable[float]] = None,
    ) -> float:
        """
        Return the annualized mean return of the portfolio.

        Parameters
        ----------
        weight : iterable of float, optional
            Candidate weights. If omitted, current weights are used.

        Returns
        -------
        float
            Mean daily portfolio return multiplied by `trading_days`.
        """
        returns = self.portfolio_returns(weight=weight)
        return float(returns.mean() * self.trading_days)

    def portfolio_realized_return(
        self,
        weight: Optional[Iterable[float]] = None,
    ) -> float:
        """
        Return the effective realized return of the portfolio.

        Parameters
        ----------
        weight : iterable of float, optional
            Candidate weights. If omitted, current weights are used.

        Returns
        -------
        float
            Compounded return over the available return window.
        """
        returns = self.portfolio_returns(weight=weight)
        return float((1.0 + returns).prod() - 1.0)

    def portfolio_variance(
        self,
        weight: Optional[Iterable[float]] = None,
    ) -> float:
        """
        Return the annualized variance of the portfolio.

        Parameters
        ----------
        weight : iterable of float, optional
            Candidate weights. If omitted, current weights are used.

        Returns
        -------
        float
            Daily return variance multiplied by `trading_days`.
        """
        returns = self.portfolio_returns(weight=weight)
        variance = float(returns.var() * self.trading_days)
        return max(variance, 0.0)

    def portfolio_annual_volatility(
        self,
        weight: Optional[Iterable[float]] = None,
    ) -> float:
        """
        Return the annualized volatility of the portfolio.

        Parameters
        ----------
        weight : iterable of float, optional
            Candidate weights. If omitted, current weights are used.

        Returns
        -------
        float
            Daily return standard deviation annualized by `sqrt(trading_days)`.
        """
        returns = self.portfolio_returns(weight=weight)
        return float(returns.std() * np.sqrt(self.trading_days))

    def portfolio_variance_coeficience(
        self,
        weight: Optional[Iterable[float]] = None,
    ) -> float:
        """
        Return the volatility-to-return coefficient of the portfolio.

        Parameters
        ----------
        weight : iterable of float, optional
            Candidate weights. If omitted, current weights are used.

        Returns
        -------
        float
            Annualized volatility divided by annualized expected return.

        Raises
        ------
        ValueError
            If the annualized expected return is zero.
        """
        expected_return = self.portfolio_annual_return(weight=weight)
        volatility = self.portfolio_annual_volatility(weight=weight)
        if np.isclose(expected_return, 0.0):
            raise ValueError("portfolio annual return is zero, coefficient is undefined.")
        return float(volatility / expected_return)

    def portfolio_variance_coefficient(
        self,
        weight: Optional[Iterable[float]] = None,
    ) -> float:
        """Return a correctly spelled alias around `portfolio_variance_coeficience()`."""
        return self.portfolio_variance_coeficience(weight=weight)

    def portfolio_sharpe_ratio(
        self,
        free_rate: float,
        weight: Optional[Iterable[float]] = None,
    ) -> float:
        """
        Return the Sharpe ratio of the portfolio.

        Parameters
        ----------
        free_rate : float
            Annual risk-free rate.
        weight : iterable of float, optional
            Candidate weights. If omitted, current weights are used.

        Returns
        -------
        float
            Annualized Sharpe ratio.

        Raises
        ------
        ValueError
            If annualized portfolio volatility is zero.
        """
        volatility = self.portfolio_annual_volatility(weight=weight)
        if np.isclose(volatility, 0.0):
            raise ValueError("portfolio annual volatility is zero, coefficient is undefined")
        expected_return = self.portfolio_annual_return(weight=weight)
        return float((expected_return - free_rate) / volatility)


__all__ = [
    "PortfolioBasicMetrics",
]
