"""Mean-variance optimization built on top of the portfolio composition layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..asset_allocation.PortfolioOptimization import PortfolioOptimization
from ..portfolio import Portfolio
from .configs import MinimumVarianceConfig, OptimizationConfig
from .results import OptimizationResult


@dataclass
class MeanVarianceOptimizer:
    """
    Optimize a `Portfolio` using the existing mean-variance solver backend.

    The class keeps the new public API centered on `Portfolio` while reusing
    the mature solver already implemented in the legacy asset-allocation
    module. This lets the migration preserve numerical behavior and
    compatibility while progressively improving the structure around it.
    """

    portfolio: Portfolio

    def _build_legacy_optimizer(self) -> PortfolioOptimization:
        """Create a legacy optimizer preloaded with the current portfolio data."""
        optimizer = PortfolioOptimization(
            tickers=self.portfolio.tickers,
            start=self.portfolio.start,
            end=self.portfolio.end,
            price_field=self.portfolio.price_field,
            weight=self.portfolio.weight,
        )
        optimizer._set_prices_cache(self.portfolio.asset_prices())
        optimizer._set_returns_cache(self.portfolio.asset_returns())
        return optimizer

    def _apply_optimized_weights(self, result: OptimizationResult) -> None:
        """Persist optimized weights back into the shared portfolio instance."""
        if result.success:
            self.portfolio.update_weights(result.weights)

    def optimize_minimum_variance(
        self,
        config: Optional[MinimumVarianceConfig] = None,
    ) -> OptimizationResult:
        """Minimize the annualized portfolio variance."""
        active_config = MinimumVarianceConfig() if config is None else config
        legacy_optimizer = self._build_legacy_optimizer()
        legacy_result = legacy_optimizer.optimize_minimum_variance(
            config=active_config.to_legacy(),
        )
        result = OptimizationResult.from_legacy(legacy_result)
        self._apply_optimized_weights(result)
        return result

    def optimize_maximum_sharpe(
        self,
        config: Optional[OptimizationConfig] = None,
    ) -> OptimizationResult:
        """Maximize the annualized Sharpe ratio of the portfolio."""
        active_config = OptimizationConfig() if config is None else config
        legacy_optimizer = self._build_legacy_optimizer()
        legacy_result = legacy_optimizer.optimize_maximum_sharpe(
            config=active_config.to_legacy(),
        )
        result = OptimizationResult.from_legacy(legacy_result)
        self._apply_optimized_weights(result)
        return result


__all__ = [
    "MeanVarianceOptimizer",
    "MinimumVarianceConfig",
    "OptimizationConfig",
    "OptimizationResult",
    "PortfolioOptimization",
]
