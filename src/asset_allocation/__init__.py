"""Legacy compatibility exports for the historical asset-allocation package."""

from .PortfolioElementaryMetrics import PortfolioElementaryMetrics

from .PortfolioOptimization import (
    MinimumVarianceConfig,
    OptimizationConfig,
    OptimizationResult,
    PortfolioOptimization,
)

from .PortfolioPostModernMetrics import PortfolioPostModernMetrics

from .PortfolioOptimizationPostModern import (
    MaximumOmegaConfig,
    MinimumSemivarianceConfig,
    PostModernOptimizationConfig,
    PostModernOptimizationResult,
    PortfolioOptimizationPostMordern,
    PortfolioOptimizationPostModern,
)


from .PortfolioElementaryAnalysis import PortfolioElementaryAnalysis


__all__ = [
    "PortfolioElementaryAnalysis",
    "PortfolioElementaryMetrics",
    "MinimumVarianceConfig",
    "OptimizationConfig",
    "OptimizationResult",
    "PortfolioOptimization",
    "PostModernOptimizationConfig",
    "MinimumSemivarianceConfig",
    "MaximumOmegaConfig",
    "PostModernOptimizationResult",
    "PortfolioOptimizationPostMordern",
    "PortfolioOptimizationPostModern",
    "PortfolioPostModernMetrics",
]
