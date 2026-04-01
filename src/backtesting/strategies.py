"""Strategy adapters used by the composition-based backtesting engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from ..optimization.configs import (
    MaximumOmegaConfig,
    MinimumSemivarianceConfig,
    MinimumVarianceConfig,
    OptimizationConfig,
    PostModernOptimizationConfig,
)
from ..optimization.mean_variance import MeanVarianceOptimizer
from ..optimization.postmodern import PostModernOptimizer
from ._helpers import build_portfolio, normalize_benchmark_prices
from .results import StrategyAllocation


class AllocationStrategy(ABC):
    """
    Common interface used by the backtesting engine.

    Concrete subclasses are responsible only for weight estimation. They do
    not perform the actual capital simulation; that work is delegated to
    `Backtester`.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self._name = name

    @property
    @abstractmethod
    def default_name(self) -> str:
        """Default human-readable label for the strategy."""

    @property
    def name(self) -> str:
        """Return the user-defined name or the strategy default label."""
        return self._name or self.default_name

    @abstractmethod
    def optimize(
        self,
        prices: pd.DataFrame,
        optimization_benchmark_prices: Optional[pd.Series | pd.DataFrame] = None,
    ) -> StrategyAllocation:
        """Estimate the portfolio allocation on the provided price window."""


class MeanVarianceStrategy(AllocationStrategy):
    """Adapter around the mean-variance optimization objectives."""

    _DEFAULT_NAMES = {
        "minimum_variance": "Min Var",
        "maximum_sharpe": "Max Sharpe",
    }

    def __init__(
        self,
        objective: str,
        config: Optional[OptimizationConfig] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        if objective not in self._DEFAULT_NAMES:
            raise ValueError(
                "objective must be 'minimum_variance' or 'maximum_sharpe'."
            )

        self.objective = objective
        self.config = config

    @property
    def default_name(self) -> str:
        """Default display name associated with the selected objective."""
        return self._DEFAULT_NAMES[self.objective]

    def optimize(
        self,
        prices: pd.DataFrame,
        optimization_benchmark_prices: Optional[pd.Series | pd.DataFrame] = None,
    ) -> StrategyAllocation:
        """Optimize portfolio weights on the provided in-sample window."""
        del optimization_benchmark_prices

        config = self.config
        if config is None:
            if self.objective == "minimum_variance":
                config = MinimumVarianceConfig()
            else:
                config = OptimizationConfig()

        portfolio = build_portfolio(
            prices,
            initial_weights=getattr(config, "initial_weights", None),
            name=self.name,
        )
        optimizer = MeanVarianceOptimizer(portfolio=portfolio)

        if self.objective == "minimum_variance":
            result = optimizer.optimize_minimum_variance(config=config)
        else:
            result = optimizer.optimize_maximum_sharpe(config=config)

        if not result.success:
            raise RuntimeError(f"{self.name} optimization failed: {result.message}")

        return StrategyAllocation(
            name=self.name,
            weights=result.weights,
            weights_by_ticker=result.weights_by_ticker,
            optimization_result=result,
        )


class PostModernStrategy(AllocationStrategy):
    """Adapter around the post-modern optimization objectives."""

    _DEFAULT_NAMES = {
        "minimum_semivariance": "Min Semivar",
        "maximum_omega": "Max Omega",
    }

    def __init__(
        self,
        objective: str,
        config: Optional[PostModernOptimizationConfig] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        if objective not in self._DEFAULT_NAMES:
            raise ValueError(
                "objective must be 'minimum_semivariance' or 'maximum_omega'."
            )

        self.objective = objective
        self.config = config

    @property
    def default_name(self) -> str:
        """Default display name associated with the selected objective."""
        return self._DEFAULT_NAMES[self.objective]

    def optimize(
        self,
        prices: pd.DataFrame,
        optimization_benchmark_prices: Optional[pd.Series | pd.DataFrame] = None,
    ) -> StrategyAllocation:
        """Optimize post-modern portfolio weights on the provided price window."""
        config = self.config
        if config is None:
            if self.objective == "minimum_semivariance":
                config = MinimumSemivarianceConfig()
            else:
                config = MaximumOmegaConfig()

        portfolio = build_portfolio(
            prices,
            initial_weights=getattr(config, "initial_weights", None),
            name=self.name,
        )
        optimizer = PostModernOptimizer(portfolio=portfolio)

        benchmark_returns = None
        if (
            self.objective == "minimum_semivariance"
            and optimization_benchmark_prices is not None
        ):
            benchmark_label = "Optimization Benchmark"
            if isinstance(optimization_benchmark_prices, pd.Series):
                benchmark_label = str(
                    optimization_benchmark_prices.name or benchmark_label
                )
            elif (
                isinstance(optimization_benchmark_prices, pd.DataFrame)
                and optimization_benchmark_prices.shape[1] == 1
            ):
                benchmark_label = str(
                    optimization_benchmark_prices.columns[0] or benchmark_label
                )

            benchmark_prices = normalize_benchmark_prices(
                optimization_benchmark_prices,
                label=benchmark_label,
            )
            benchmark_returns = benchmark_prices.pct_change().dropna()

        if self.objective == "minimum_semivariance":
            result = optimizer.optimize_minimum_semivariance(
                config=config,
                benchmark_returns=benchmark_returns,
            )
        else:
            result = optimizer.optimize_maximum_omega(config=config)

        if not result.success:
            raise RuntimeError(f"{self.name} optimization failed: {result.message}")

        return StrategyAllocation(
            name=self.name,
            weights=result.weights,
            weights_by_ticker=result.weights_by_ticker,
            optimization_result=result,
        )


__all__ = [
    "AllocationStrategy",
    "MeanVarianceStrategy",
    "PostModernStrategy",
]
