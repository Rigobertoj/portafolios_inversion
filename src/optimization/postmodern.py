"""Post-modern optimization aliases over the current optimizer layer."""

from ..asset_allocation.PortfolioOptimizationPostModern import (
    MaximumOmegaConfig,
    MinimumSemivarianceConfig,
    PortfolioOptimizationPostModern,
    PostModernOptimizationConfig,
    PostModernOptimizationResult,
)

PostModernOptimizer = PortfolioOptimizationPostModern

__all__ = [
    "MaximumOmegaConfig",
    "MinimumSemivarianceConfig",
    "PortfolioOptimizationPostModern",
    "PostModernOptimizationConfig",
    "PostModernOptimizationResult",
    "PostModernOptimizer",
]
