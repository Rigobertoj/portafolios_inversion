"""Optimization result reexports for the new package layout."""

from ..asset_allocation.PortfolioOptimization import OptimizationResult
from ..asset_allocation.PortfolioOptimizationPostModern import (
    PostModernOptimizationResult,
)

__all__ = [
    "OptimizationResult",
    "PostModernOptimizationResult",
]
