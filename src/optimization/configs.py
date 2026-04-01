"""Configuration models for the composition-based optimization layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple


@dataclass
class OptimizationConfig:
    """Shared configuration for mean-variance optimization routines."""

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
    """Configuration for the minimum-variance optimization routine."""

    minimum_return: Optional[float] = None

    def to_legacy(self) -> "MinimumVarianceConfig":
        """Return a solver-compatible minimum-variance config view."""
        return self


@dataclass
class PostModernOptimizationConfig:
    """Shared configuration for post-modern optimization routines."""

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
    """Configuration for the minimum-semivariance optimization routine."""

    minimum_return: Optional[float] = None

    def to_legacy(self) -> "MinimumSemivarianceConfig":
        """Return a solver-compatible minimum-semivariance config view."""
        return self


@dataclass
class MaximumOmegaConfig(PostModernOptimizationConfig):
    """Configuration for the maximum-Omega optimization routine."""

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
