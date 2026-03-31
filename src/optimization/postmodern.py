"""Post-modern optimization built on top of the portfolio composition layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..asset_allocation.PortfolioOptimizationPostModern import (
    PortfolioOptimizationPostModern,
)
from ..portfolio import Portfolio
from .configs import (
    MaximumOmegaConfig,
    MinimumSemivarianceConfig,
    PostModernOptimizationConfig,
)
from .results import PostModernOptimizationResult


@dataclass
class PostModernOptimizer:
    """
    Optimize a `Portfolio` using the existing post-modern solver backend.

    The wrapper keeps the new optimization API consistent with the composition
    design while preserving the current numerical behavior of the legacy
    downside-based solver.
    """

    portfolio: Portfolio

    def _build_legacy_optimizer(self) -> PortfolioOptimizationPostModern:
        """Create a legacy post-modern optimizer preloaded with portfolio data."""
        optimizer = PortfolioOptimizationPostModern(
            tickers=self.portfolio.tickers,
            start=self.portfolio.start,
            end=self.portfolio.end,
            price_field=self.portfolio.price_field,
            weight=self.portfolio.weight,
        )
        optimizer._set_prices_cache(self.portfolio.asset_prices())
        optimizer._set_returns_cache(self.portfolio.asset_returns())
        return optimizer

    def _apply_optimized_weights(self, result: PostModernOptimizationResult) -> None:
        """Persist optimized weights back into the shared portfolio instance."""
        if result.success:
            self.portfolio.update_weights(result.weights)

    def optimize_minimum_semivariance(
        self,
        config: Optional[MinimumSemivarianceConfig] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> PostModernOptimizationResult:
        """Minimize annualized semivariance under the selected downside reference."""
        active_config = MinimumSemivarianceConfig() if config is None else config
        legacy_optimizer = self._build_legacy_optimizer()
        legacy_result = legacy_optimizer.optimize_minimum_semivariance(
            config=active_config.to_legacy(),
            benchmark_returns=benchmark_returns,
        )
        result = PostModernOptimizationResult.from_legacy(legacy_result)
        self._apply_optimized_weights(result)
        return result

    def optimize_maximum_omega(
        self,
        config: Optional[MaximumOmegaConfig] = None,
    ) -> PostModernOptimizationResult:
        """Maximize the portfolio Omega ratio."""
        active_config = MaximumOmegaConfig() if config is None else config
        legacy_optimizer = self._build_legacy_optimizer()
        legacy_result = legacy_optimizer.optimize_maximum_omega(
            config=active_config.to_legacy(),
        )
        result = PostModernOptimizationResult.from_legacy(legacy_result)
        self._apply_optimized_weights(result)
        return result


__all__ = [
    "MaximumOmegaConfig",
    "MinimumSemivarianceConfig",
    "PortfolioOptimizationPostModern",
    "PostModernOptimizationConfig",
    "PostModernOptimizationResult",
    "PostModernOptimizer",
]
