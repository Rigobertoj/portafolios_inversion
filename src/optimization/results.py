"""Result models returned by portfolio optimization routines.

The dataclasses in this module provide serializable summaries of optimizer
outputs, including solver status, optimized weights, objective values, and the
portfolio statistics computed from the final allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class OptimizationResult:
    """
    Serializable result of a mean-variance optimization run.

    Parameters
    ----------
    objective : str
        Name of the optimized objective.
    success : bool
        Whether the numerical solver reported convergence.
    status : int
        Solver-specific numeric status code.
    message : str
        Solver message describing the termination condition.
    weights : numpy.ndarray
        Optimized portfolio weights aligned with the optimizer's asset order.
    weights_by_ticker : pandas.Series
        Optimized weights indexed by ticker.
    expected_return : float
        Annualized expected portfolio return at the optimized weights.
    variance : float
        Annualized portfolio variance at the optimized weights.
    volatility : float
        Annualized portfolio volatility at the optimized weights.
    sharpe : float
        Annualized Sharpe ratio at the optimized weights.
    objective_value : float
        Value used to summarize the optimized objective.
    iterations : int
        Number of solver iterations reported by the backend.
    """

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
        """
        Build a new-layer result from any solver output with matching fields.

        Parameters
        ----------
        result : Any
            Legacy or third-party solver result exposing the attributes required
            by `OptimizationResult`.

        Returns
        -------
        OptimizationResult
            Normalized dataclass copy of the supplied result.
        """
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
    """
    Serializable result of a post-modern optimization run.

    Parameters
    ----------
    objective : str
        Name of the optimized objective.
    success : bool
        Whether the numerical solver reported convergence.
    status : int
        Solver-specific numeric status code.
    message : str
        Solver message describing the termination condition.
    weights : numpy.ndarray
        Optimized portfolio weights aligned with the optimizer's asset order.
    weights_by_ticker : pandas.Series
        Optimized weights indexed by ticker.
    expected_return : float
        Annualized expected portfolio return at the optimized weights.
    semivariance : float
        Portfolio semivariance at the optimized weights.
    downside_risk : float
        Portfolio downside risk at the optimized weights.
    omega : float
        Portfolio Omega ratio at the optimized weights.
    objective_value : float
        Value used to summarize the optimized objective.
    iterations : int
        Number of solver iterations reported by the backend.
    """

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
        """
        Build a new-layer result from any solver output with matching fields.

        Parameters
        ----------
        result : Any
            Legacy or third-party solver result exposing the attributes required
            by `PostModernOptimizationResult`.

        Returns
        -------
        PostModernOptimizationResult
            Normalized dataclass copy of the supplied result.
        """
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
