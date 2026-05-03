"""Drawdown-oriented risk analysis.

This module computes wealth paths, running peaks, drawdown series, and maximum
drawdown directly from a composed `Portfolio`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..portfolio.portfolio import Portfolio


@dataclass
class PortfolioDrawdownAnalysis:
    """
    Compute drawdown metrics directly from the realized portfolio path.

    Parameters
    ----------
    portfolio : Portfolio
        Portfolio object used to generate returns and wealth paths.
    """

    portfolio: Portfolio

    def wealth_index(self, initial_value: float = 1.0) -> pd.Series:
        """
        Return the compounded wealth path used by drawdown calculations.

        Parameters
        ----------
        initial_value : float, default 1.0
            Starting value of the wealth index.

        Returns
        -------
        pandas.Series
            Compounded portfolio wealth path.
        """
        return self.portfolio.wealth_index(initial_value=initial_value)

    def running_peak(self, initial_value: float = 1.0) -> pd.Series:
        """
        Return the running maximum of the wealth index.

        Parameters
        ----------
        initial_value : float, default 1.0
            Starting value of the wealth index.

        Returns
        -------
        pandas.Series
            Cumulative maximum of the wealth path.
        """
        wealth = self.wealth_index(initial_value=initial_value)
        peak = wealth.cummax()
        peak.name = wealth.name
        return peak

    def drawdown_series(self, initial_value: float = 1.0) -> pd.Series:
        """
        Return the percentage drawdown series of the portfolio wealth path.

        Parameters
        ----------
        initial_value : float, default 1.0
            Starting value of the wealth index.

        Returns
        -------
        pandas.Series
            Drawdown series expressed as wealth divided by running peak minus
            one.
        """
        wealth = self.wealth_index(initial_value=initial_value)
        peak = self.running_peak(initial_value=initial_value)
        drawdown = wealth / peak - 1.0
        drawdown.name = wealth.name
        return drawdown

    def max_drawdown(self, initial_value: float = 1.0) -> float:
        """
        Return the worst realized drawdown of the portfolio.

        Parameters
        ----------
        initial_value : float, default 1.0
            Starting value of the wealth index.

        Returns
        -------
        float
            Minimum value of the drawdown series.
        """
        return float(self.drawdown_series(initial_value=initial_value).min())


__all__ = [
    "PortfolioDrawdownAnalysis",
]
