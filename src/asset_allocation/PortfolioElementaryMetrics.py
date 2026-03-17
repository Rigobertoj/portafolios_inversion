from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    SRC_ROOT = Path(__file__).resolve().parents[1]
    
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
        
    from security_selection.AssetsResearch import AssetsResearch
else:
    try:
        from ..security_selection.AssetsResearch import AssetsResearch
    except ImportError:
        from security_selection.AssetsResearch import AssetsResearch

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
    portfolio_variance()
        Compute the portfolio variance from the covariance matrix.
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
        self.__weight = self._validate_weight(value)

    @staticmethod
    def _normalize_weight(weight: Iterable[float]) -> np.ndarray:
        """Convert the provided weights into a one-dimensional NumPy array."""
        vector = np.asarray(weight, dtype=float)
        if vector.ndim == 0:
            return vector.reshape(1)
        return vector

    def _validate_weight(self, weight: Iterable[float]) -> np.ndarray:
        """Validate that a weight vector is one-dimensional and sums to one."""
        vector = self._normalize_weight(weight)
        if vector.ndim != 1:
            raise ValueError("weight must be a 1D iterable.")
        if len(vector) != len(self.tickers):
            raise ValueError("weight length must match tickers length.")
        if not np.isclose(vector.sum(), 1.0):
            raise ValueError("the sum of weight must be equal to 1.")
        return vector

    def _resolve_weight(self, weight: Optional[Iterable[float]] = None) -> np.ndarray:
        """Return a validated weight vector, defaulting to the instance weights."""
        if weight is None:
            return self.weight
        return self._validate_weight(weight)

    def _assets_order(self) -> list[str]:
        """Return the canonical asset order used by portfolio calculations."""
        return list(self.tickers)

    def _annual_returns_vector(self) -> np.ndarray:
        """Return annualized asset returns aligned with the portfolio weights."""
        asset_order = self._assets_order()
        annual_returns = self.annual_return(asset_order).loc[asset_order]
        return annual_returns.to_numpy(dtype=float)

    def _annual_covariance_matrix(self) -> np.ndarray:
        """Return the annualized covariance matrix aligned with the weights."""
        asset_order = self._assets_order()
        covariance_matrix = self.covariance(asset_order).loc[asset_order, asset_order]
        return covariance_matrix.to_numpy(dtype=float) * 252.0

    @staticmethod
    def _portfolio_return_from_inputs(
        weight: np.ndarray,
        expected_returns: np.ndarray,
    ) -> float:
        """Compute portfolio expected return from weights and asset returns."""
        return float(weight @ expected_returns)

    @staticmethod
    def _portfolio_variance_from_inputs(
        weight: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> float:
        """Compute portfolio variance from weights and covariance matrix."""
        return float(weight.T @ covariance_matrix @ weight)

    @classmethod
    def _portfolio_volatility_from_inputs(
        cls,
        weight: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> float:
        """Compute portfolio volatility from weights and covariance matrix."""
        variance = max(cls._portfolio_variance_from_inputs(weight, covariance_matrix), 0.0)
        return float(np.sqrt(variance))

    def portfolio_path(self, weight: Optional[Iterable[float]] = None) -> pd.Series:
        """
        Compute the historical wealth path of the weighted portfolio.

        Parameters
        ----------
        weight : Iterable[float] | None, default None
            Optional weight vector used instead of the instance weights.

        Returns
        -------
        pandas.Series
            Time series obtained by multiplying asset prices by the portfolio
            weight vector at each date.
        """
        vector = self._resolve_weight(weight)
        asset_order = self._assets_order()
        assets_prices = self.get_prices(asset_order)
        portfolio_wealth_path = assets_prices @ vector
        return portfolio_wealth_path

    def portfolio_annual_return(self, weight: Optional[Iterable[float]] = None) -> float:
        """
        Compute the annualized expected return of the portfolio.

        Parameters
        ----------
        weight : Iterable[float] | None, default None
            Optional weight vector used instead of the instance weights.

        Returns
        -------
        float
            Weighted average of the assets' annualized returns.
        """
        vector = self._resolve_weight(weight)
        assets_returns = self._annual_returns_vector()
        return self._portfolio_return_from_inputs(vector, assets_returns)

    def portfolio_variance(self, weight: Optional[Iterable[float]] = None) -> float:
        """
        Compute the annualized variance of the portfolio.

        Parameters
        ----------
        weight : Iterable[float] | None, default None
            Optional weight vector used instead of the instance weights.

        Returns
        -------
        float
            Annualized portfolio variance.
        """
        vector = self._resolve_weight(weight)
        assets_cov = self._annual_covariance_matrix()
        variance = self._portfolio_variance_from_inputs(vector, assets_cov)
        return max(variance, 0.0)

    def portfolio_annual_volatility(
        self,
        weight: Optional[Iterable[float]] = None,
    ) -> float:
        """
        Compute the annualized volatility of the portfolio.

        Parameters
        ----------
        weight : Iterable[float] | None, default None
            Optional weight vector used instead of the instance weights.

        Returns
        -------
        float
            Portfolio standard deviation annualized using 252 trading days.
        """
        vector = self._resolve_weight(weight)
        assets_cov = self._annual_covariance_matrix()
        return self._portfolio_volatility_from_inputs(vector, assets_cov)

    def portfolio_variance_coeficience(
        self,
        weight: Optional[Iterable[float]] = None,
    ) -> float:
        """
        Compute the volatility-to-return coefficient of the portfolio.

        Parameters
        ----------
        weight : Iterable[float] | None, default None
            Optional weight vector used instead of the instance weights.

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
        portfolio_return = self.portfolio_annual_return(weight=weight)
        portfolio_volatility = self.portfolio_annual_volatility(weight=weight)
        if np.isclose(portfolio_return, 0.0):
            raise ValueError("portfolio annual return is zero, coefficient is undefined.")
        return float(portfolio_volatility / portfolio_return)

    def portfolio_sharpe_ratio(
        self,
        free_rate: float,
        weight: Optional[Iterable[float]] = None,
    ) -> float:
        """
        Compute the Sharpe ratio of the portfolio.

        Parameters
        ----------
        free_rate : float
            Risk-free rate used as the benchmark return.
        weight : Iterable[float] | None, default None
            Optional weight vector used instead of the instance weights.

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
        portfolio_return = self.portfolio_annual_return(weight=weight)
        portfolio_volatility = self.portfolio_annual_volatility(weight=weight)
        if np.isclose(portfolio_volatility, 0.0):
            raise ValueError("portfolio annual volatility is zero, coefficient is undefined")
        sharpe_ratio = (portfolio_return - free_rate) / portfolio_volatility
        return float(sharpe_ratio)


PortfolioElementaryMetrics.weight = property(
    PortfolioElementaryMetrics._get_weight,
    PortfolioElementaryMetrics._set_weight,
    doc="Portfolio weight vector aligned with the order of `tickers`.",
)
if __name__ == "__main__":
    pass
