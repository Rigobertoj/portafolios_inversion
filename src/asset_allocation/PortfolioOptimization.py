from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy.optimize import minimize

if __package__ in {None, ""}:
    from PortfolioElementaryMetrics import PortfolioElementaryMetrics
else:
    from .PortfolioElementaryMetrics import PortfolioElementaryMetrics


@dataclass
class OptimizationConfig:
    """
    Common configuration shared by the portfolio optimization routines.

    Parameters
    ----------
    risk_free_rate : float, default 0.0
        Risk-free rate used to report the Sharpe ratio and optimize the
        maximum-Sharpe portfolio.
    allow_short : bool, default False
        Allow negative weights when custom bounds are not provided.
    bounds : Sequence[tuple[float, float]] | None, default None
        Optional bounds per asset. When omitted, defaults to `[0, 1]` or
        `[-1, 1]` depending on `allow_short`.
    initial_weights : Iterable[float] | None, default None
        Feasible initial point for the numerical solver. When omitted, the
        instance portfolio weights are used.
    solver_method : str, default "SLSQP"
        SciPy solver method used by `scipy.optimize.minimize`.
    solver_options : dict[str, object]
        Additional keyword options passed to the solver.
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


@dataclass
class MinimumVarianceConfig(OptimizationConfig):
    """
    Configuration used by the minimum-variance optimization routine.

    Parameters
    ----------
    minimum_return : float | None, default None
        Optional lower bound for the annualized portfolio return. When omitted,
        the optimization only minimizes portfolio variance.
    """

    minimum_return: Optional[float] = None


@dataclass
class OptimizationResult:
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


class PortfolioOptimization(PortfolioElementaryMetrics):
    """Optimize portfolio weights using mean-variance criteria."""

    @staticmethod
    def _minimum_variance_objective(
        weights: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> float:
        return PortfolioElementaryMetrics._portfolio_variance_from_inputs(
            weights,
            covariance_matrix,
        )

    @classmethod
    def _negative_sharpe_objective(
        cls,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_free_rate: float,
    ) -> float:
        volatility = cls._portfolio_volatility_from_inputs(weights, covariance_matrix)
        if np.isclose(volatility, 0.0):
            return float("inf")
        expected_return = cls._portfolio_return_from_inputs(weights, expected_returns)
        sharpe_ratio = (expected_return - risk_free_rate) / volatility
        return float(-sharpe_ratio)

    def _resolve_bounds(
        self,
        config: OptimizationConfig,
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
        config: OptimizationConfig,
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
        config: OptimizationConfig,
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
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_free_rate: float,
    ) -> OptimizationResult:
        optimal_weights = np.asarray(solution.x, dtype=float)
        portfolio_return = self._portfolio_return_from_inputs(
            optimal_weights,
            expected_returns,
        )
        portfolio_variance = max(
            self._portfolio_variance_from_inputs(optimal_weights, covariance_matrix),
            0.0,
        )
        portfolio_volatility = self._portfolio_volatility_from_inputs(
            optimal_weights,
            covariance_matrix,
        )

        if np.isclose(portfolio_volatility, 0.0):
            portfolio_sharpe = float("nan")
        else:
            portfolio_sharpe = float(
                (portfolio_return - risk_free_rate) / portfolio_volatility
            )

        objective_value = (
            portfolio_variance
            if objective == "minimum_variance"
            else portfolio_sharpe
        )

        weights_by_ticker = pd.Series(
            optimal_weights,
            index=self._assets_order(),
            name="weight",
        )

        return OptimizationResult(
            objective=objective,
            success=bool(solution.success),
            status=int(solution.status),
            message=str(solution.message),
            weights=optimal_weights,
            weights_by_ticker=weights_by_ticker,
            expected_return=float(portfolio_return),
            variance=float(portfolio_variance),
            volatility=float(portfolio_volatility),
            sharpe=float(portfolio_sharpe),
            objective_value=float(objective_value),
            iterations=int(getattr(solution, "nit", 0)),
        )

    def optimize_minimum_variance(
        self,
        config: Optional[MinimumVarianceConfig] = None,
    ) -> OptimizationResult:
        """
        Minimize the annualized portfolio variance.

        When `minimum_return` is provided, the routine also enforces
        `portfolio_return >= minimum_return`.
        """
        if config is None:
            config = MinimumVarianceConfig()

        covariance_matrix = self._annual_covariance_matrix()
        bounds = self._resolve_bounds(config, len(self._assets_order()))
        constraints = [self._sum_weights_constraint()]
        expected_returns = self._annual_returns_vector()

        if config.minimum_return is not None:
            constraints.append(
                self._minimum_return_constraint(
                    expected_returns,
                    float(config.minimum_return),
                )
            )

        def objective_function(
            weights: np.ndarray,
            cov: np.ndarray = covariance_matrix,
        ) -> float:
            return self._minimum_variance_objective(weights, cov)

        solution = self._solve(
            objective_function=objective_function,
            config=config,
            constraints=constraints,
            bounds=bounds,
        )

        return self._build_result(
            objective="minimum_variance",
            solution=solution,
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            risk_free_rate=config.risk_free_rate,
        )

    def optimize_maximum_sharpe(
        self,
        config: Optional[OptimizationConfig] = None,
    ) -> OptimizationResult:
        """Maximize the annualized Sharpe ratio of the portfolio."""
        if config is None:
            config = OptimizationConfig()

        expected_returns = self._annual_returns_vector()
        covariance_matrix = self._annual_covariance_matrix()
        bounds = self._resolve_bounds(config, len(self._assets_order()))

        def objective_function(
            weights: np.ndarray,
            mu: np.ndarray = expected_returns,
            cov: np.ndarray = covariance_matrix,
            risk_free_rate: float = config.risk_free_rate,
        ) -> float:
            return self._negative_sharpe_objective(weights, mu, cov, risk_free_rate)

        solution = self._solve(
            objective_function=objective_function,
            config=config,
            constraints=[self._sum_weights_constraint()],
            bounds=bounds,
        )

        return self._build_result(
            objective="maximum_sharpe",
            solution=solution,
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            risk_free_rate=config.risk_free_rate,
        )


if __name__ == "__main__":
    pass
