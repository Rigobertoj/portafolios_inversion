from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    from .AssetsResearch import AssetsResearch
except ImportError:
    from AssetsResearch import AssetsResearch


@dataclass
class PortfolioElementaryMetrics(AssetsResearch):
    """
    Compute portfolio-level metrics from asset returns and portfolio weights.

    `PortfolioElementaryMetrics` extends `AssetsResearch` by adding a weight
    vector and portfolio aggregation formulas. Instead of analyzing each asset
    independently, this class combines the underlying assets into a single
    weighted portfolio and exposes return, volatility, Sharpe ratio, and wealth
    path calculations.

    Parameters
    ----------
    tickers : Iterable[str]
        Asset symbols included in the portfolio.
    start : str
        Start date used to download historical market data.
    end : str | None, default None
        Optional end date used to download historical market data.
    price_field : str, default "Close"
        Price column extracted from Yahoo Finance data.
    weight : Iterable[float]
        Portfolio weights aligned with `tickers`. The vector must have the same
        length as the number of assets and must sum to 1.

    Attributes
    ----------
    weight : numpy.ndarray
        Normalized portfolio weight vector.

    Methods
    -------
    portfolio_path()
        Compute the historical wealth path of the weighted portfolio.
    portfolio_annual_return()
        Compute the annualized portfolio return.
    portfolio_annual_volatility()
        Compute the annualized portfolio volatility.
    portfolio_variance_coeficience()
        Compute volatility divided by annualized return.
    portfolio_sharpe_ratio(free_rate)
        Compute the Sharpe ratio of the portfolio.

    Inherited API
    -------------
    Since this class inherits from `AssetsResearch`, the IDE will also expose
    data access and per-asset analytics such as `get_prices`, `get_returns`,
    `annual_return`, `annual_volatility`, `covariance`, and `correlation`.

    Notes
    -----
    - The order of `weight` must match the order of `tickers`.
    - The `weight` setter validates dimensionality, length, and that the
      weights add up to 1.

    Examples
    --------
    >>> weights = np.array([0.4, 0.35, 0.25])
    >>> portfolio = PortfolioElementaryMetrics(
    ...     tickers=["AAPL", "JPM", "MSFT"],
    ...     start="2024-01-01",
    ...     weight=weights,
    ... )
    >>> portfolio.portfolio_annual_return()
    >>> portfolio.portfolio_annual_volatility()
    """
    weight: InitVar[Iterable[float]] = field(kw_only=True)
    __weight: np.ndarray = field(init=False, repr=False)

    def __post_init__(
        self,
        tickers: Iterable[str],
        start: str,
        end: Optional[str],
        price_field: str,
        weight: Iterable[float],
    ) -> None:
        """
        Initialize research inputs and validate the portfolio weight vector.

        Parameters
        ----------
        tickers : Iterable[str]
            Symbols included in the portfolio.
        start : str
            Start date for the historical window.
        end : str | None
            Optional end date for the historical window.
        price_field : str
            Price column used to build the research dataset.
        weight : Iterable[float]
            Portfolio weights associated with `tickers`.
        """
        super().__post_init__(tickers=tickers, start=start, end=end, price_field=price_field)
        self._set_weight(weight)

    def _get_weight(self) -> np.ndarray:
        return self.__weight.copy()

    def _set_weight(self, value: Iterable[float]) -> None:
        vector = self._normalize_weight(value)
        if vector.ndim != 1:
            raise ValueError("weight must be a 1D iterable.")
        if len(vector) != len(self.tickers):
            raise ValueError("weight length must match tickers length.")
        if not np.isclose(vector.sum(), 1.0):
            raise ValueError("the sum of weight must be equal to 1.")
        self.__weight = vector

    @staticmethod
    def _normalize_weight(weight: Iterable[float]) -> np.ndarray:
        """Convert the provided weights into a one-dimensional NumPy array."""
        vector = np.asarray(weight, dtype=float)
        if vector.ndim == 0:
            return vector.reshape(1)
        return vector
    
    def portfolio_path(self) -> pd.Series:
        """
        Compute the historical wealth path of the weighted portfolio.

        Returns
        -------
        pandas.Series
            Time series obtained by multiplying asset prices by the portfolio
            weight vector at each date.
        """
        assets_prices = self.get_prices()
        portfolio_wealth_path = assets_prices @ self.__weight 
        return portfolio_wealth_path

    def portfolio_annual_return(self) -> float:
        """
        Compute the annualized expected return of the portfolio.

        Returns
        -------
        float
            Weighted average of the assets' annualized returns.
        """
        assets_returns = self.annual_return().to_numpy(dtype=float)
        return float(self.weight @ assets_returns)

    def portfolio_annual_volatility(self) -> float:
        """
        Compute the annualized volatility of the portfolio.

        Returns
        -------
        float
            Portfolio standard deviation annualized using 252 trading days.
        """
        assets_cov = self.covariance().to_numpy(dtype=float)
        portfolio_variance = float(self.weight.T @ assets_cov @ self.weight)
        portfolio_annual_variance = portfolio_variance * 252
        return float(np.sqrt(portfolio_annual_variance))

    def portfolio_variance_coeficience(self) -> float:
        """
        Compute the volatility-to-return coefficient of the portfolio.

        Returns
        -------
        float
            Ratio between annualized portfolio volatility and annualized
            portfolio return.

        Raises
        ------
        ValueError
            If the annualized portfolio return is zero.
        """
        portfolio_return = self.portfolio_annual_return()
        portfolio_volatility = self.portfolio_annual_volatility()
        if np.isclose(portfolio_return, 0.0):
            raise ValueError("portfolio annual return is zero, coefficient is undefined.")
        return float(portfolio_volatility / portfolio_return)
    
    def portfolio_sharpe_ratio(self, free_rate : float) -> float:
        """
        Compute the Sharpe ratio of the portfolio.

        Parameters
        ----------
        free_rate : float
            Risk-free rate used as the benchmark return.

        Returns
        -------
        float
            Sharpe ratio computed from annualized portfolio return and
            annualized portfolio volatility.

        Raises
        ------
        ValueError
            If the annualized portfolio volatility is zero.
        """
        portfolio_return = self.portfolio_annual_return()
        portfolio_volatility = self.portfolio_annual_volatility()
        if np.isclose(portfolio_volatility, 0.0):
            raise ValueError("portfolio annual volatility is zero, coefficient is undefined")
        sharpe_ratio = (portfolio_return - free_rate)/portfolio_volatility        
        return float(sharpe_ratio)


PortfolioElementaryMetrics.weight = property(
    PortfolioElementaryMetrics._get_weight,
    PortfolioElementaryMetrics._set_weight,
    doc="Portfolio weight vector aligned with the order of `tickers`.",
)


def _main_():
    tickets = ["BOH", "AAPL", "JPM"]
    start_date = "2018-01-01"
    end_date = "2026-02-20"

    weight = np.ones(len(tickets)) / len(tickets)

    Portfolio1 = PortfolioElementaryMetrics(
        tickers=tickets,
        start=start_date,
        end=end_date,
        weight=weight
    )
    
    #print(Portfolio1.portfolio_annual_return())
    #print(Portfolio1.annual_return())
    #print(Portfolio1.portfolio_annual_volatility())
    print(Portfolio1.portfolio_path())
    print(Portfolio1.get_prices())

    print(Portfolio1.correlation())
    
    return


if __name__ == "__main__":
    _main_()
