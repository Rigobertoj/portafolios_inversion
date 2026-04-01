"""Legacy compatibility wrapper for mean-variance optimization."""

from ..optimization.mean_variance import (
    MinimumVarianceConfig,
    OptimizationConfig,
    OptimizationResult,
    PortfolioOptimization,
)

__all__ = [
    "MinimumVarianceConfig",
    "OptimizationConfig",
    "OptimizationResult",
    "PortfolioOptimization",
]
