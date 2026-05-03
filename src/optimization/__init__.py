"""Public API for portfolio optimization.

The package exports mean-variance and post-modern optimizers, their
configuration dataclasses, and serializable optimization results. Legacy class
names remain available so older notebooks can migrate without import changes.
"""

from .configs import (
    MaximumOmegaConfig,
    MinimumSemivarianceConfig,
    MinimumVarianceConfig,
    OptimizationConfig,
    PostModernOptimizationConfig,
)
from .mean_variance import MeanVarianceOptimizer, PortfolioOptimization
from .postmodern import (
    PortfolioOptimizationPostMordern,
    PortfolioOptimizationPostModern,
    PostModernOptimizer,
)
from .results import OptimizationResult, PostModernOptimizationResult

__all__ = [
    "MaximumOmegaConfig",
    "MeanVarianceOptimizer",
    "MinimumSemivarianceConfig",
    "MinimumVarianceConfig",
    "OptimizationConfig",
    "OptimizationResult",
    "PortfolioOptimization",
    "PortfolioOptimizationPostMordern",
    "PortfolioOptimizationPostModern",
    "PostModernOptimizationConfig",
    "PostModernOptimizationResult",
    "PostModernOptimizer",
]
