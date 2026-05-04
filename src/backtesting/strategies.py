"""Strategy adapters used by the backtesting engines.

This module provides a common `AllocationStrategy` interface plus adapters for
the mean-variance and post-modern optimizers. The engines depend only on this
interface, so each strategy is responsible for estimating weights while the
engines handle simulation, benchmark comparison, and metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Optional, Sequence, Tuple

import numpy as np
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


_TRADING_DAYS = 252.0
_RETURN_CAP_TOLERANCE = 1e-10


def _resolve_config_bounds(
    config: OptimizationConfig | PostModernOptimizationConfig,
    n_assets: int,
) -> Sequence[Tuple[float, float]]:
    if config.bounds is not None:
        if len(config.bounds) != n_assets:
            raise ValueError("bounds length must match number of assets.")
        return [tuple(map(float, bound)) for bound in config.bounds]

    if config.allow_short:
        return [(-1.0, 1.0)] * n_assets

    return [(0.0, 1.0)] * n_assets


def _maximum_feasible_return(
    expected_returns: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
) -> float:
    lower = np.asarray([bound[0] for bound in bounds], dtype=float)
    upper = np.asarray([bound[1] for bound in bounds], dtype=float)

    if np.any(upper < lower):
        raise ValueError(
            "bounds upper values must be greater than or equal to lower values."
        )

    residual = 1.0 - float(lower.sum())
    capacity = upper - lower
    if (
        residual < -_RETURN_CAP_TOLERANCE
        or residual > float(capacity.sum()) + _RETURN_CAP_TOLERANCE
    ):
        raise ValueError("bounds must allow portfolio weights to sum to 1.")

    weights = lower.copy()
    remaining = max(residual, 0.0)
    for index in np.argsort(expected_returns)[::-1]:
        allocation = min(float(capacity[index]), remaining)
        if allocation > 0.0:
            weights[index] += allocation
            remaining -= allocation
        if remaining <= _RETURN_CAP_TOLERANCE:
            break

    return float(weights @ expected_returns)


def _annual_returns_vector(prices: pd.DataFrame) -> np.ndarray:
    normalized_prices = prices.sort_index().dropna()
    returns = normalized_prices.pct_change().dropna()
    if returns.empty:
        raise ValueError("optimization window must contain at least one return row.")

    expected_returns = (
        returns.mean().loc[normalized_prices.columns].to_numpy(dtype=float)
        * _TRADING_DAYS
    )
    if not np.isfinite(expected_returns).all():
        raise ValueError("expected returns must be finite to cap minimum_return.")
    return expected_returns


def _minimum_return_tolerance(maximum_return: float) -> float:
    return max(_RETURN_CAP_TOLERANCE, abs(maximum_return) * _RETURN_CAP_TOLERANCE)


def _cap_minimum_return_for_window(
    config: OptimizationConfig | PostModernOptimizationConfig,
    prices: pd.DataFrame,
) -> tuple[
    OptimizationConfig | PostModernOptimizationConfig,
    Optional[float],
    Optional[float],
    Optional[float],
    bool,
]:
    requested = getattr(config, "minimum_return", None)
    if requested is None:
        return config, None, None, None, False

    requested_return = float(requested)
    if not np.isfinite(requested_return):
        raise ValueError("minimum_return must be finite.")

    expected_returns = _annual_returns_vector(prices)
    bounds = _resolve_config_bounds(config, len(expected_returns))
    maximum_return = _maximum_feasible_return(expected_returns, bounds)
    effective_return = requested_return
    was_capped = False

    if requested_return > maximum_return:
        effective_return = maximum_return - _minimum_return_tolerance(maximum_return)
        config = replace(config, minimum_return=effective_return)
        was_capped = True

    return config, requested_return, effective_return, maximum_return, was_capped


def _annotate_minimum_return(
    result,
    *,
    requested: Optional[float],
    effective: Optional[float],
    maximum: Optional[float],
    was_capped: bool,
) -> None:
    result.requested_minimum_return = requested
    result.effective_minimum_return = effective
    result.maximum_feasible_return = maximum
    result.minimum_return_was_capped = was_capped


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

        (
            config,
            requested_minimum_return,
            effective_minimum_return,
            maximum_feasible_return,
            minimum_return_was_capped,
        ) = _cap_minimum_return_for_window(config, prices)

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

        _annotate_minimum_return(
            result,
            requested=requested_minimum_return,
            effective=effective_minimum_return,
            maximum=maximum_feasible_return,
            was_capped=minimum_return_was_capped,
        )

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

        (
            config,
            requested_minimum_return,
            effective_minimum_return,
            maximum_feasible_return,
            minimum_return_was_capped,
        ) = _cap_minimum_return_for_window(config, prices)

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

        _annotate_minimum_return(
            result,
            requested=requested_minimum_return,
            effective=effective_minimum_return,
            maximum=maximum_feasible_return,
            was_capped=minimum_return_was_capped,
        )

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
