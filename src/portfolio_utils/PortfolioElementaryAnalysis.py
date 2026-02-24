from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    from .AssetsResearch import AssetsResearch
    from .PortfolioElementaryMetrics import PortfolioElementaryMetrics
except ImportError:
    from AssetsResearch import AssetsResearch
    from PortfolioElementaryMetrics import PortfolioElementaryMetrics


# Clase  para realizar un análisis comparativo
# portafolio vs benchmark
@dataclass
class PortfolioElementaryAnalysis(PortfolioElementaryMetrics):
    benchmark: InitVar[str] = field(kw_only=True)

    __benchmark: str = field(init=False, repr=False)
    __benchmark_returns_cache: pd.DataFrame = field(
        init=False, repr=False, default_factory=pd.DataFrame
    )

    def __post_init__(
        self,
        tickers: Iterable[str],
        start: str,
        end: Optional[str],
        price_field: str,
        weight: Iterable[float],
        benchmark: str,
    ) -> None:
        super().__post_init__(
            tickers=tickers,
            start=start,
            end=end,
            price_field=price_field,
            weight=weight,
        )
        self._set_benchmark(benchmark)

    @classmethod
    def with_equal_weights(
        cls,
        tickers: Iterable[str],
        start: str,
        benchmark: str,
        end: Optional[str] = None,
        price_field: str = "Close",
    ) -> "PortfolioElementaryAnalysis":
        normalized_tickers = AssetsResearch._normalize_tickers(tickers)
        if not normalized_tickers:
            raise ValueError("tickers must contain at least one symbol.")
        weights = np.ones(len(normalized_tickers), dtype=float) / len(normalized_tickers)
        return cls(
            tickers=normalized_tickers,
            start=start,
            end=end,
            price_field=price_field,
            weight=weights,
            benchmark=benchmark,
        )

    def _get_benchmark(self) -> str:
        return self.__benchmark

    def _set_benchmark(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("benchmark must be a non-empty ticker string.")
        self.__benchmark = value.strip()
        self.__benchmark_returns_cache = pd.DataFrame()

    def benchmark_returns(self) -> pd.Series:
        if self.__benchmark_returns_cache.empty:
            benchmark_research = AssetsResearch(
                tickers=[self.benchmark],
                start=self.start,
                end=self.end,
                price_field=self.price_field,
            )
            self.__benchmark_returns_cache = benchmark_research.get_returns()

        if self.benchmark in self.__benchmark_returns_cache.columns:
            return self.__benchmark_returns_cache[self.benchmark].copy()

        return self.__benchmark_returns_cache.iloc[:, 0].copy()

    def benchmark_annual_return(self) -> float:
        returns = self.benchmark_returns()
        if returns.empty:
            raise ValueError("benchmark returns are empty.")
        return float(returns.mean() * 252)

    def _portfolio_daily_returns(self) -> pd.Series:
        assets_returns = self.get_returns()
        if assets_returns.empty:
            raise ValueError("portfolio returns are empty.")
        values = assets_returns.to_numpy(dtype=float) @ self.weight
        return pd.Series(values, index=assets_returns.index, name="portfolio")

    def _aligned_portfolio_benchmark_returns(self) -> pd.DataFrame:
        aligned = pd.concat(
            [
                self._portfolio_daily_returns(),
                self.benchmark_returns().rename("benchmark"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        if len(aligned) < 2:
            raise ValueError("not enough overlapping returns for CAPM metrics.")
        return aligned

    def portfolio_beta(self) -> float:
        aligned = self._aligned_portfolio_benchmark_returns()
        benchmark_variance = float(aligned["benchmark"].var())
        if np.isclose(benchmark_variance, 0.0):
            raise ValueError("benchmark variance is zero, beta is undefined.")
        
        covariance = float(aligned["portfolio"].cov(aligned["benchmark"]))
        
        return float(covariance / benchmark_variance)

    def assets_beta(self) -> pd.Series:
        assets_returns = self.get_returns()
        if assets_returns.empty:
            raise ValueError("assets returns are empty.")

        benchmark_returns = self.benchmark_returns()
        if benchmark_returns.empty:
            raise ValueError("benchmark returns are empty.")

        benchmark_variance = float(benchmark_returns.var())
        if np.isclose(benchmark_variance, 0.0):
            raise ValueError("benchmark variance is zero, assets beta is undefined.")

        covariances = assets_returns.apply(
            lambda asset_returns: asset_returns.cov(benchmark_returns)
        )
        betas = covariances / benchmark_variance
        betas.name = "beta"
        return betas

    def portfolio_capm_expected_return(
        self,
        risk_free_rate: float = 0.0,
        market_expected_return: Optional[float] = None,
    ) -> float:
        
        if market_expected_return is None:
            market_expected_return = self.benchmark_annual_return()
        
        beta = self.portfolio_beta()
        
        expected_return = risk_free_rate + beta * (market_expected_return - risk_free_rate)
        
        return float(expected_return)
    
    def assets_capm_expected_return(
        self,
        risk_free_rate: float = 0.0,
        market_expected_return: Optional[float] = None,
    ) -> float:
        
        if market_expected_return is None:
                market_expected_return = self.benchmark_annual_return()
        
        betas = self.assets_beta()
        
        expected_return = risk_free_rate + betas * (market_expected_return - risk_free_rate)

        return expected_return


PortfolioElementaryAnalysis.benchmark = property(
    PortfolioElementaryAnalysis._get_benchmark,
    PortfolioElementaryAnalysis._set_benchmark,
)


def _main_() -> None:
    tickets = ["BOH", "AAPL", "JPM"]
    start_date = "2025-01-01"
    end_date = "2026-02-20"

    weight = np.ones(len(tickets)) / len(tickets)

    benchmark = "^GSPC"

    risk_rate_free = 0.055

    Portfolio1 = PortfolioElementaryAnalysis(
        tickers=tickets,
        start=start_date,
        end=end_date,
        weight=weight,
        benchmark=benchmark
    )
    
    portfolio_beta = Portfolio1.portfolio_beta()
    assets_beta = Portfolio1.assets_beta()
    
    #print("portfolio beta: ",portfolio_beta)
    #print("assets beta: ", assets_beta)
    
    portfolio_capm = Portfolio1.portfolio_capm_expected_return(risk_free_rate = risk_rate_free)
    assets_capm = Portfolio1.assets_capm_expected_return(risk_free_rate = risk_rate_free)
    
    print("CAPM Portfolio: ",portfolio_capm )
    print("CAPM Assets: ", assets_capm)
    
    print(Portfolio1.correlation())
    return

if __name__ == "__main__":
    _main_()
