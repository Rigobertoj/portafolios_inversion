"""Mean-variance optimization aliases over the current optimizer layer."""

from ..asset_allocation.PortfolioOptimization import (
    MinimumVarianceConfig,
    OptimizationConfig,
    OptimizationResult,
    PortfolioOptimization,
)

MeanVarianceOptimizer = PortfolioOptimization

__all__ = [
    "MeanVarianceOptimizer",
    "MinimumVarianceConfig",
    "OptimizationConfig",
    "OptimizationResult",
    "PortfolioOptimization",
]
