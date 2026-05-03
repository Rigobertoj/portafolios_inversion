"""Configuration models for portfolio optimization routines.

The dataclasses in this module collect solver settings, constraints, initial
weights, bounds, and objective-specific parameters used by the mean-variance and
post-modern optimization engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple


@dataclass
class OptimizationConfig:
    """
    Shared configuration for mean-variance optimization routines.

    Parameters
    ----------
    risk_free_rate : float, default 0.0
        Annual risk-free rate used by maximum-Sharpe optimization.
    allow_short : bool, default False
        Whether portfolio weights may be negative when explicit `bounds` are not
        provided.
    bounds : sequence of tuple of float, optional
        Per-asset lower and upper weight bounds passed to SciPy's optimizer.
        Length must match the number of optimized assets.
    initial_weights : iterable of float, optional
        Initial portfolio weights used as the solver starting point. If omitted,
        the optimizer uses the current portfolio weights.
    solver_method : str, default "SLSQP"
        Optimization method passed to `scipy.optimize.minimize`.
    solver_options : dict, optional
        Additional solver options passed to `scipy.optimize.minimize`.
    """

    risk_free_rate: float = 0.0
    allow_short: bool = False
    bounds: Optional[Sequence[Tuple[float, float]]] = None
    initial_weights: Optional[Iterable[float]] = None
    solver_method: str = "SLSQP"
    solver_options: Dict[str, object] = field(
        default_factory=lambda: {
            "maxiter": 500,
            "ftol": 1e-9,
            "disp": False,
        }
    )

    def to_legacy(self) -> "OptimizationConfig":
        """
        Return a solver-compatible config view.

        The current backend only needs the config attributes, so the new
        dataclass can be passed directly while the migration remains active.
        """
        return self


@dataclass
class MinimumVarianceConfig(OptimizationConfig):
    """
    Configuration for the minimum-variance optimization routine.

    Parameters
    ----------
    minimum_return : float, optional
        Minimum annualized expected return enforced as an inequality constraint.
    """

    minimum_return: Optional[float] = None

    def to_legacy(self) -> "MinimumVarianceConfig":
        """Return a solver-compatible minimum-variance config view."""
        return self


@dataclass
class PostModernOptimizationConfig:
    """
    Shared configuration for post-modern optimization routines.

    Parameters
    ----------
    threshold : float, default 0.0
        Minimum acceptable return used to separate downside and upside returns.
    allow_short : bool, default False
        Whether portfolio weights may be negative when explicit `bounds` are not
        provided.
    bounds : sequence of tuple of float, optional
        Per-asset lower and upper weight bounds passed to SciPy's optimizer.
    initial_weights : iterable of float, optional
        Initial portfolio weights used as the solver starting point. If omitted,
        the optimizer uses the current portfolio weights.
    solver_method : str, default "SLSQP"
        Optimization method passed to `scipy.optimize.minimize`.
    solver_options : dict, optional
        Additional solver options passed to `scipy.optimize.minimize`.
    """

    threshold: float = 0.0
    allow_short: bool = False
    bounds: Optional[Sequence[Tuple[float, float]]] = None
    initial_weights: Optional[Iterable[float]] = None
    solver_method: str = "SLSQP"
    solver_options: Dict[str, object] = field(
        default_factory=lambda: {
            "maxiter": 500,
            "ftol": 1e-9,
            "disp": False,
        }
    )

    def to_legacy(self) -> "PostModernOptimizationConfig":
        """Return a solver-compatible post-modern config view."""
        return self


@dataclass
class MinimumSemivarianceConfig(PostModernOptimizationConfig):
    """
    Configuration for the minimum-semivariance optimization routine.

    Parameters
    ----------
    minimum_return : float, optional
        Minimum annualized expected return enforced as an inequality constraint.
    """

    minimum_return: Optional[float] = None

    def to_legacy(self) -> "MinimumSemivarianceConfig":
        """Return a solver-compatible minimum-semivariance config view."""
        return self


@dataclass
class MaximumOmegaConfig(PostModernOptimizationConfig):
    """
    Configuration for the maximum-Omega optimization routine.

    Notes
    -----
    This objective maximizes the weighted asset Omega ratio computed from
    upside and downside risk around `threshold`.
    """

    def to_legacy(self) -> "MaximumOmegaConfig":
        """Return a solver-compatible maximum-Omega config view."""
        return self


__all__ = [
    "MaximumOmegaConfig",
    "MinimumSemivarianceConfig",
    "MinimumVarianceConfig",
    "OptimizationConfig",
    "PostModernOptimizationConfig",
]
