"""Configuration models for the composition-based optimization layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

from ..asset_allocation.PortfolioOptimization import (
    MinimumVarianceConfig as LegacyMinimumVarianceConfig,
)
from ..asset_allocation.PortfolioOptimization import (
    OptimizationConfig as LegacyOptimizationConfig,
)
from ..asset_allocation.PortfolioOptimizationPostModern import (
    MaximumOmegaConfig as LegacyMaximumOmegaConfig,
)
from ..asset_allocation.PortfolioOptimizationPostModern import (
    MinimumSemivarianceConfig as LegacyMinimumSemivarianceConfig,
)
from ..asset_allocation.PortfolioOptimizationPostModern import (
    PostModernOptimizationConfig as LegacyPostModernOptimizationConfig,
)


def _copy_bounds(
    bounds: Optional[Sequence[Tuple[float, float]]],
) -> Optional[list[tuple[float, float]]]:
    if bounds is None:
        return None
    return [tuple(bound) for bound in bounds]


def _copy_initial_weights(
    initial_weights: Optional[Iterable[float]],
) -> Optional[list[float]]:
    if initial_weights is None:
        return None
    return [float(weight) for weight in initial_weights]


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

    def to_legacy(self) -> LegacyOptimizationConfig:
        """Build the legacy config consumed by the current solver backend."""
        return LegacyOptimizationConfig(
            risk_free_rate=float(self.risk_free_rate),
            allow_short=bool(self.allow_short),
            bounds=_copy_bounds(self.bounds),
            initial_weights=_copy_initial_weights(self.initial_weights),
            solver_method=self.solver_method,
            solver_options=dict(self.solver_options),
        )


@dataclass
class MinimumVarianceConfig(OptimizationConfig):
    """Configuration for the minimum-variance optimization routine."""

    minimum_return: Optional[float] = None

    def to_legacy(self) -> LegacyMinimumVarianceConfig:
        """Build the legacy minimum-variance config."""
        return LegacyMinimumVarianceConfig(
            risk_free_rate=float(self.risk_free_rate),
            allow_short=bool(self.allow_short),
            bounds=_copy_bounds(self.bounds),
            initial_weights=_copy_initial_weights(self.initial_weights),
            solver_method=self.solver_method,
            solver_options=dict(self.solver_options),
            minimum_return=(
                None if self.minimum_return is None else float(self.minimum_return)
            ),
        )


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

    def to_legacy(self) -> LegacyPostModernOptimizationConfig:
        """Build the legacy config consumed by the current solver backend."""
        return LegacyPostModernOptimizationConfig(
            threshold=float(self.threshold),
            allow_short=bool(self.allow_short),
            bounds=_copy_bounds(self.bounds),
            initial_weights=_copy_initial_weights(self.initial_weights),
            solver_method=self.solver_method,
            solver_options=dict(self.solver_options),
        )


@dataclass
class MinimumSemivarianceConfig(PostModernOptimizationConfig):
    """Configuration for the minimum-semivariance optimization routine."""

    minimum_return: Optional[float] = None

    def to_legacy(self) -> LegacyMinimumSemivarianceConfig:
        """Build the legacy minimum-semivariance config."""
        return LegacyMinimumSemivarianceConfig(
            threshold=float(self.threshold),
            allow_short=bool(self.allow_short),
            bounds=_copy_bounds(self.bounds),
            initial_weights=_copy_initial_weights(self.initial_weights),
            solver_method=self.solver_method,
            solver_options=dict(self.solver_options),
            minimum_return=(
                None if self.minimum_return is None else float(self.minimum_return)
            ),
        )


@dataclass
class MaximumOmegaConfig(PostModernOptimizationConfig):
    """Configuration for the maximum-Omega optimization routine."""

    def to_legacy(self) -> LegacyMaximumOmegaConfig:
        """Build the legacy maximum-Omega config."""
        return LegacyMaximumOmegaConfig(
            threshold=float(self.threshold),
            allow_short=bool(self.allow_short),
            bounds=_copy_bounds(self.bounds),
            initial_weights=_copy_initial_weights(self.initial_weights),
            solver_method=self.solver_method,
            solver_options=dict(self.solver_options),
        )


__all__ = [
    "MaximumOmegaConfig",
    "MinimumSemivarianceConfig",
    "MinimumVarianceConfig",
    "OptimizationConfig",
    "PostModernOptimizationConfig",
]
