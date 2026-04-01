"""Result models for the composition-based optimization layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


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
    def from_legacy(cls, result: Any) -> "OptimizationResult":
        """Build a new-layer result from any solver output with the same fields."""
        return cls(
            objective=str(getattr(result, "objective")),
            success=bool(getattr(result, "success")),
            status=int(getattr(result, "status")),
            message=str(getattr(result, "message")),
            weights=np.asarray(getattr(result, "weights"), dtype=float).copy(),
            weights_by_ticker=getattr(result, "weights_by_ticker").copy(),
            expected_return=float(getattr(result, "expected_return")),
            variance=float(getattr(result, "variance")),
            volatility=float(getattr(result, "volatility")),
            sharpe=float(getattr(result, "sharpe")),
            objective_value=float(getattr(result, "objective_value")),
            iterations=int(getattr(result, "iterations")),
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
        result: Any,
    ) -> "PostModernOptimizationResult":
        """Build a new-layer result from any solver output with the same fields."""
        return cls(
            objective=str(getattr(result, "objective")),
            success=bool(getattr(result, "success")),
            status=int(getattr(result, "status")),
            message=str(getattr(result, "message")),
            weights=np.asarray(getattr(result, "weights"), dtype=float).copy(),
            weights_by_ticker=getattr(result, "weights_by_ticker").copy(),
            expected_return=float(getattr(result, "expected_return")),
            semivariance=float(getattr(result, "semivariance")),
            downside_risk=float(getattr(result, "downside_risk")),
            omega=float(getattr(result, "omega")),
            objective_value=float(getattr(result, "objective_value")),
            iterations=int(getattr(result, "iterations")),
        )


__all__ = [
    "OptimizationResult",
    "PostModernOptimizationResult",
]
