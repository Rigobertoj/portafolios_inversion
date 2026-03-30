from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy.optimize import minimize

if __package__ in {None, ""}:
    from PortfolioPostModernMetrics import PortfolioPostModernMetrics
else:
    from .PortfolioPostModernMetrics import PortfolioPostModernMetrics


@dataclass
class PostModernOptimizationConfig:
    """
    Common configuration shared by the post-modern optimization routines.

    Parameters
    ----------
    threshold : float, default 0.0
        Scalar downside hurdle used to separate downside and upside returns.
        When a benchmark return series is supplied to the minimum-semivariance
        optimization, the effective downside reference becomes
        `benchmark_returns + threshold`.
        In other words, this parameter can represent:

        - a standalone MAR when no benchmark is provided, or
        - a spread over the benchmark when `benchmark_returns` is supplied.
    allow_short : bool, default False
        Allow negative weights when custom bounds are not provided.
    bounds : Sequence[tuple[float, float]] | None, default None
        Optional bounds per asset. When omitted, defaults to `[0, 1]` or
        `[-1, 1]` depending on `allow_short`.
    initial_weights : Iterable[float] | None, default None
        Feasible initial point for the solver. When omitted, the instance
        portfolio weights are used.
    solver_method : str, default "SLSQP"
        SciPy solver method used by `scipy.optimize.minimize`.
    solver_options : dict[str, object]
        Additional keyword options passed to the solver.
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


@dataclass
class MinimumSemivarianceConfig(PostModernOptimizationConfig):
    """
    Configuration used by the minimum-semivariance optimization routine.

    Parameters
    ----------
    minimum_return : float | None, default None
        Optional lower bound for the annualized portfolio return. When omitted,
        the routine only minimizes portfolio semivariance.

    Notes
    -----
    `minimum_return` is intentionally different from `threshold`:

    - `minimum_return` is an annualized portfolio-level constraint,
    - `threshold` is the periodic downside hurdle used to classify downside
      and upside observations.
    """

    minimum_return: Optional[float] = None


@dataclass
class MaximumOmegaConfig(PostModernOptimizationConfig):
    """Configuration used by the maximum-Omega optimization routine."""


@dataclass
class PostModernOptimizationResult:
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


class PortfolioOptimizationPostModern(PortfolioPostModernMetrics):
    """Optimize portfolio weights using post-modern downside metrics."""

    @staticmethod
    def _minimum_semivariance_objective(
        weights: np.ndarray,
        semivariance_matrix: np.ndarray,
    ) -> float:
        return PortfolioPostModernMetrics._portfolio_variance_from_inputs(
            weights,
            semivariance_matrix,
        )

    @staticmethod
    def _negative_omega_objective(
        weights: np.ndarray,
        omega_vector: np.ndarray,
    ) -> float:
        return float(-(weights @ omega_vector))

    def _resolve_bounds(
        self,
        config: PostModernOptimizationConfig,
        n_assets: int,
    ) -> Sequence[Tuple[float, float]]:
        if config.bounds is not None:
            if len(config.bounds) != n_assets:
                raise ValueError("bounds length must match number of assets.")
            return [tuple(bound) for bound in config.bounds]

        if config.allow_short:
            return [(-1.0, 1.0)] * n_assets

        return [(0.0, 1.0)] * n_assets

    def _resolve_initial_weights(
        self,
        config: PostModernOptimizationConfig,
        bounds: Sequence[Tuple[float, float]],
    ) -> np.ndarray:
        if config.initial_weights is None:
            initial_weights = self.weight
        else:
            initial_weights = self._validate_weight(config.initial_weights)

        for weight, (lower, upper) in zip(initial_weights, bounds):
            if weight < lower or weight > upper:
                raise ValueError("initial_weights must satisfy the configured bounds.")

        return initial_weights

    @staticmethod
    def _sum_weights_constraint() -> dict[str, object]:
        return {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}

    @staticmethod
    def _minimum_return_constraint(
        expected_returns: np.ndarray,
        minimum_return: float,
    ) -> dict[str, object]:
        return {
            "type": "ineq",
            "fun": lambda weights, mu=expected_returns, target=minimum_return: (
                weights @ mu
            )
            - target,
        }

    def _solve(
        self,
        *,
        objective_function,
        config: PostModernOptimizationConfig,
        constraints: Sequence[dict[str, object]],
        bounds: Sequence[Tuple[float, float]],
    ):
        initial_weights = self._resolve_initial_weights(config, bounds)
        return minimize(
            fun=objective_function,
            x0=initial_weights,
            method=config.solver_method,
            bounds=bounds,
            constraints=list(constraints),
            options=config.solver_options,
        )

    def _build_result(
        self,
        *,
        objective: str,
        solution,
        threshold: float,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> PostModernOptimizationResult:
        optimal_weights = np.asarray(solution.x, dtype=float)
        expected_return = self.portfolio_annual_return(weight=optimal_weights)
        semivariance = self.portfolio_semivariance(
            weight=optimal_weights,
            threshold=threshold,
            benchmark_returns=benchmark_returns,
        )
        downside_risk = self.portfolio_downside_risk(
            weight=optimal_weights,
            threshold=threshold,
            benchmark_returns=benchmark_returns,
        )
        omega = self.portfolio_omega_ratio(
            weight=optimal_weights,
            threshold=threshold,
            benchmark_returns=benchmark_returns,
        )
        objective_value = semivariance if objective == "minimum_semivariance" else omega

        weights_by_ticker = pd.Series(
            optimal_weights,
            index=self._assets_order(),
            name="weight",
        )

        if solution.success:
            # Persist the successful optimum so subsequent portfolio methods
            # use the optimized allocation by default.
            self.weight = optimal_weights

        return PostModernOptimizationResult(
            objective=objective,
            success=bool(solution.success),
            status=int(solution.status),
            message=str(solution.message),
            weights=optimal_weights,
            weights_by_ticker=weights_by_ticker,
            expected_return=float(expected_return),
            semivariance=float(semivariance),
            downside_risk=float(downside_risk),
            omega=float(omega),
            objective_value=float(objective_value),
            iterations=int(getattr(solution, "nit", 0)),
        )

    def optimize_minimum_semivariance(
        self,
        config: Optional[MinimumSemivarianceConfig] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> PostModernOptimizationResult:
        """
        Minimize the annualized portfolio semivariance.

        When `minimum_return` is provided, the routine also enforces
        `portfolio_return >= minimum_return`.
        When `benchmark_returns` is provided, the downside reference becomes
        `benchmark_returns + threshold`.
        On successful optimization, `self.weight` is updated to the
        optimized allocation.

        Parameters
        ----------
        config : MinimumSemivarianceConfig | None, default None
            Optimization settings for the routine. If omitted, default
            long-only settings are used.
        benchmark_returns : pandas.Series | pandas.DataFrame | None
            Optional benchmark return series used to build a target
            semivariance problem. When omitted, `config.threshold` behaves as
            a scalar MAR. When provided, `config.threshold` behaves as a spread
            over the benchmark.

        Notes
        -----
        This method does not reinterpret `minimum_return` as a downside
        threshold. The downside reference is always controlled by
        `threshold` and, optionally, `benchmark_returns`.
        """
        if config is None:
            config = MinimumSemivarianceConfig()

        threshold = self._normalize_threshold(config.threshold)
        semivariance_matrix = self.semivariance_matrix(
            threshold,
            benchmark_returns=benchmark_returns,
        ).to_numpy(dtype=float)
        expected_returns = self._annual_returns_vector()
        bounds = self._resolve_bounds(config, len(self._assets_order()))
        constraints = [self._sum_weights_constraint()]

        if config.minimum_return is not None:
            constraints.append(
                self._minimum_return_constraint(
                    expected_returns,
                    float(config.minimum_return),
                )
            )

        def objective_function(
            weights: np.ndarray,
            semivariance: np.ndarray = semivariance_matrix,
        ) -> float:
            return self._minimum_semivariance_objective(weights, semivariance)

        solution = self._solve(
            objective_function=objective_function,
            config=config,
            constraints=constraints,
            bounds=bounds,
        )

        return self._build_result(
            objective="minimum_semivariance",
            solution=solution,
            threshold=threshold,
            benchmark_returns=benchmark_returns,
        )

    def optimize_maximum_omega(
        self,
        config: Optional[MaximumOmegaConfig] = None,
    ) -> PostModernOptimizationResult:
        """
        Maximize the portfolio Omega ratio.

        On successful optimization, `self.weight` is updated to the
        optimized allocation.
        """
        if config is None:
            config = MaximumOmegaConfig()

        threshold = self._normalize_threshold(config.threshold)
        omega_vector = self.asset_omega_ratio(threshold).to_numpy(dtype=float)
        if not np.isfinite(omega_vector).all():
            raise ValueError("omega ratio is undefined for assets with zero downside risk.")

        bounds = self._resolve_bounds(config, len(self._assets_order()))

        def objective_function(
            weights: np.ndarray,
            omega_values: np.ndarray = omega_vector,
        ) -> float:
            return self._negative_omega_objective(weights, omega_values)

        solution = self._solve(
            objective_function=objective_function,
            config=config,
            constraints=[self._sum_weights_constraint()],
            bounds=bounds,
        )

        return self._build_result(
            objective="maximum_omega",
            solution=solution,
            threshold=threshold,
        )


# Backward-compatible alias for the previous misspelled class export.
PortfolioOptimizationPostMordern = PortfolioOptimizationPostModern


if __name__ == "__main__":
    pass
