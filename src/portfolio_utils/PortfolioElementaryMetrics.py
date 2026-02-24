from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Iterable, Optional

import numpy as np

try:
    from .AssetsResearch import AssetsResearch
except ImportError:
    from AssetsResearch import AssetsResearch


@dataclass
class PortfolioElementaryMetrics(AssetsResearch):
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
        vector = np.asarray(weight, dtype=float)
        if vector.ndim == 0:
            return vector.reshape(1)
        return vector
    
    def portfolio_path(self):
        assets_prices = self.get_prices()
        portfolio_wealth_path = assets_prices @ self.__weight 
        return portfolio_wealth_path

    def portfolio_annual_return(self) -> float:
        assets_returns = self.annual_return().to_numpy(dtype=float)
        return float(self.weight @ assets_returns)

    def portfolio_annual_volatility(self) -> float:
        assets_cov = self.covariance().to_numpy(dtype=float)
        portfolio_variance = float(self.weight.T @ assets_cov @ self.weight)
        portfolio_annual_variance = portfolio_variance * 252
        return float(np.sqrt(portfolio_annual_variance))

    def portfolio_variance_coeficience(self) -> float:
        portfolio_return = self.portfolio_annual_return()
        portfolio_volatility = self.portfolio_annual_volatility()
        if np.isclose(portfolio_return, 0.0):
            raise ValueError("portfolio annual return is zero, coefficient is undefined.")
        return float(portfolio_volatility / portfolio_return)
    
    def portfolio_sharpe_ratio(self, free_rate : float) -> float:
        portfolio_return = self.portfolio_annual_return()
        portfolio_volatility = self.portfolio_annual_volatility()
        if np.isclose(portfolio_volatility, 0.0):
            raise ValueError("portfolio annual volatility is zero, coefficient is undefined")
        sharpe_ratio = (portfolio_return - free_rate)/portfolio_volatility        
        return float(sharpe_ratio)


PortfolioElementaryMetrics.weight = property(
    PortfolioElementaryMetrics._get_weight,
    PortfolioElementaryMetrics._set_weight,
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
