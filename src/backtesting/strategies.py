"""Strategy adapters used by the backtesting engines.

This module provides a common `AllocationStrategy` interface plus adapters for
the mean-variance and post-modern optimizers. The engines depend only on this
interface, so each strategy is responsible for estimating weights while the
engines handle simulation, benchmark comparison, and metrics.
"""

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

    Parameters
    ----------
    name : str, optional
        Custom display name used in result dictionaries and output columns.
        When omitted, `default_name` is used.

    Attributes
    ----------
    name : str
        Effective strategy name, either user supplied or derived from the
        concrete strategy implementation.

    Notes
    -----
    Subclasses must implement `default_name` and `optimize`. The `optimize`
    method must return a `StrategyAllocation` whose weight order matches the
    columns of the price window supplied by the engine.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self._name = name

    @property
    @abstractmethod
    def default_name(self) -> str:
        """
        Default human-readable label for the strategy.

        Returns
        -------
        str
            Fallback strategy label used when `name` was not provided.
        """

    @property
    def name(self) -> str:
        """
        Return the effective strategy label.

        Returns
        -------
        str
            User-defined label when available; otherwise `default_name`.
        """
        return self._name or self.default_name

    @abstractmethod
    def optimize(
        self,
        prices: pd.DataFrame,
        optimization_benchmark_prices: Optional[pd.Series | pd.DataFrame] = None,
    ) -> StrategyAllocation:
        """
        Estimate portfolio weights on the provided price window.

        Parameters
        ----------
        prices : pandas.DataFrame
            In-sample price window used to estimate the allocation.
        optimization_benchmark_prices : pandas.Series or pandas.DataFrame, optional
            Benchmark price window for objectives that require a reference
            benchmark.

        Returns
        -------
        StrategyAllocation
            Optimized allocation ready for simulation by the backtesting engine.
        """


class MeanVarianceStrategy(AllocationStrategy):
    """
    Adapter around mean-variance optimization objectives.

    Parameters
    ----------
    objective : {"minimum_variance", "maximum_sharpe"}
        Mean-variance objective to optimize.
    config : OptimizationConfig, optional
        Optimizer configuration. If omitted, a default
        `MinimumVarianceConfig` is used for `"minimum_variance"` and a default
        `OptimizationConfig` is used for `"maximum_sharpe"`.
    name : str, optional
        Custom display name for result tables.

    Raises
    ------
    ValueError
        If `objective` is not supported.
    RuntimeError
        If the underlying optimizer reports an unsuccessful result.
    """

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
        """
        Default display name associated with the selected objective.

        Returns
        -------
        str
            `"Min Var"` for minimum variance or `"Max Sharpe"` for maximum
            Sharpe.
        """
        return self._DEFAULT_NAMES[self.objective]

    def optimize(
        self,
        prices: pd.DataFrame,
        optimization_benchmark_prices: Optional[pd.Series | pd.DataFrame] = None,
    ) -> StrategyAllocation:
        """
        Optimize mean-variance portfolio weights.

        Parameters
        ----------
        prices : pandas.DataFrame
            In-sample asset prices. Columns define the ticker order used by the
            returned weights.
        optimization_benchmark_prices : pandas.Series or pandas.DataFrame, optional
            Accepted for interface compatibility and ignored by this strategy.

        Returns
        -------
        StrategyAllocation
            Mean-variance allocation with weights, readable ticker weights, and
            the raw optimizer result.

        Raises
        ------
        RuntimeError
            If the optimizer does not converge successfully.
        """
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
    """
    Adapter around post-modern optimization objectives.

    Parameters
    ----------
    objective : {"minimum_semivariance", "maximum_omega"}
        Post-modern objective to optimize.
    config : PostModernOptimizationConfig, optional
        Optimizer configuration. If omitted, the strategy creates a default
        `MinimumSemivarianceConfig` or `MaximumOmegaConfig` according to
        `objective`.
    name : str, optional
        Custom display name for result tables.

    Notes
    -----
    `minimum_semivariance` can consume `optimization_benchmark_prices`; the
    benchmark is converted to returns and passed to the optimizer. `maximum_omega`
    does not use a benchmark.

    Raises
    ------
    ValueError
        If `objective` is not supported.
    RuntimeError
        If the underlying optimizer reports an unsuccessful result.
    """

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
        """
        Default display name associated with the selected objective.

        Returns
        -------
        str
            `"Min Semivar"` for minimum semivariance or `"Max Omega"` for
            maximum Omega.
        """
        return self._DEFAULT_NAMES[self.objective]

    def optimize(
        self,
        prices: pd.DataFrame,
        optimization_benchmark_prices: Optional[pd.Series | pd.DataFrame] = None,
    ) -> StrategyAllocation:
        """
        Optimize post-modern portfolio weights.

        Parameters
        ----------
        prices : pandas.DataFrame
            In-sample asset prices. Columns define the ticker order used by the
            returned weights.
        optimization_benchmark_prices : pandas.Series or pandas.DataFrame, optional
            Benchmark prices used by the minimum-semivariance objective when a
            benchmark-relative downside measure is desired.

        Returns
        -------
        StrategyAllocation
            Post-modern allocation with weights, readable ticker weights, and
            the raw optimizer result.

        Raises
        ------
        ValueError
            If benchmark prices are supplied as a DataFrame with more than one
            column.
        RuntimeError
            If the optimizer does not converge successfully.
        """
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
