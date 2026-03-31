"""Result models for the composition-based optimization layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..asset_allocation.PortfolioOptimization import (
    OptimizationResult as LegacyOptimizationResult,
)
from ..asset_allocation.PortfolioOptimizationPostModern import (
    PostModernOptimizationResult as LegacyPostModernOptimizationResult,
)


@dataclass
class OptimizationResult:
    """Serializable result of a mean-variance optimization run."""

    objective: str
    success: bool
    status: int
    message: str
    weights: np.ndarray
    weights_by_ticker: pd.Series
    expected_return: float
    variance: float
    volatility: float
    sharpe: float
    objective_value: float
    iterations: int

    @classmethod
    def from_legacy(cls, result: LegacyOptimizationResult) -> "OptimizationResult":
        """Build a new-layer result from the current legacy solver output."""
        return cls(
            objective=str(result.objective),
            success=bool(result.success),
            status=int(result.status),
            message=str(result.message),
            weights=np.asarray(result.weights, dtype=float).copy(),
            weights_by_ticker=result.weights_by_ticker.copy(),
            expected_return=float(result.expected_return),
            variance=float(result.variance),
            volatility=float(result.volatility),
            sharpe=float(result.sharpe),
            objective_value=float(result.objective_value),
            iterations=int(result.iterations),
        )


@dataclass
class PostModernOptimizationResult:
    """Serializable result of a post-modern optimization run."""

    objective: str
    success: bool
    status: int
    message: str
    weights: np.ndarray
    weights_by_ticker: pd.Series
    expected_return: float
    semivariance: float
    downside_risk: float
    omega: float
    objective_value: float
    iterations: int

    @classmethod
    def from_legacy(
        cls,
        result: LegacyPostModernOptimizationResult,
    ) -> "PostModernOptimizationResult":
        """Build a new-layer result from the current legacy solver output."""
        return cls(
            objective=str(result.objective),
            success=bool(result.success),
            status=int(result.status),
            message=str(result.message),
            weights=np.asarray(result.weights, dtype=float).copy(),
            weights_by_ticker=result.weights_by_ticker.copy(),
            expected_return=float(result.expected_return),
            semivariance=float(result.semivariance),
            downside_risk=float(result.downside_risk),
            omega=float(result.omega),
            objective_value=float(result.objective_value),
            iterations=int(result.iterations),
        )


__all__ = [
    "OptimizationResult",
    "PostModernOptimizationResult",
]
