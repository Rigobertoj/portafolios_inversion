"""Mean-variance optimization implemented on top of the new package layout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..portfolio.metrics_basic import PortfolioBasicMetrics
from ..portfolio.portfolio import Portfolio
from ..research.assets_research import AssetsResearch
from .configs import MinimumVarianceConfig, OptimizationConfig
from .results import OptimizationResult


class PortfolioOptimization(AssetsResearch):
    """
    Backward-compatible mean-variance optimizer built on the new research layer.

    This class keeps the familiar constructor used across notebooks and legacy
    code while sourcing prices and returns from `src.research.AssetsResearch`.
    The optimization formulas are implemented locally so the new optimization
    package no longer depends on the legacy asset-allocation backend.
    """

    def __init__(
        self,
        tickers: Iterable[str],
        start: str,
        end: Optional[str] = None,
        price_field: str = "Close",
        weight: Optional[Iterable[float]] = None,
    ) -> None:
        super().__init__(
            tickers=tickers,
            start=start,
            end=end,
            price_field=price_field,
        )
        if weight is None:
            raise ValueError("weight must be provided.")
        self._set_weight(weight)

    def _get_weight(self) -> np.ndarray:
        return self.__weight.copy()

    def _set_weight(self, value: Iterable[float]) -> None:
        self.__weight = self._validate_weight(value)

    @staticmethod
    def _normalize_weight(weight: Iterable[float]) -> np.ndarray:
        vector = np.asarray(weight, dtype=float)
        if vector.ndim == 0:
            return vector.reshape(1)
        return vector

    def _validate_weight(self, weight: Iterable[float]) -> np.ndarray:
        vector = self._normalize_weight(weight)
        if vector.ndim != 1:
            raise ValueError("weight must be a 1D iterable.")
        if len(vector) != len(self.tickers):
            raise ValueError("weight length must match tickers length.")
        if not np.isclose(vector.sum(), 1.0):
            raise ValueError("the sum of weight must be equal to 1.")
        return vector

    def _resolve_weight(self, weight: Optional[Iterable[float]] = None) -> np.ndarray:
        if weight is None:
            return self.weight
        return self._validate_weight(weight)

    @property
    def weight(self) -> np.ndarray:
        return self._get_weight()

    @weight.setter
    def weight(self, value: Iterable[float]) -> None:
        self._set_weight(value)

    def _assets_order(self) -> list[str]:
        return list(self.tickers)

    def _build_portfolio(
        self,
        weight: Optional[Iterable[float]] = None,
    ) -> Portfolio:
        return Portfolio.from_research(
            self,
            weights=self._resolve_weight(weight),
            tickers=self._assets_order(),
            name="portfolio",
        )

    def _basic_metrics(
        self,
        weight: Optional[Iterable[float]] = None,
    ) -> PortfolioBasicMetrics:
        return PortfolioBasicMetrics(portfolio=self._build_portfolio(weight))

    def _annual_returns_vector(self) -> np.ndarray:
        asset_order = self._assets_order()
        annual_returns = self.annual_return(asset_order).loc[asset_order]
        return annual_returns.to_numpy(dtype=float)

    def _annual_covariance_matrix(self) -> np.ndarray:
        asset_order = self._assets_order()
        covariance_matrix = self.covariance(asset_order).loc[asset_order, asset_order]
        return covariance_matrix.to_numpy(dtype=float) * 252.0

    @staticmethod
    def _portfolio_return_from_inputs(
        weight: np.ndarray,
        expected_returns: np.ndarray,
    ) -> float:
        return float(weight @ expected_returns)

    @staticmethod
    def _portfolio_variance_from_inputs(
        weight: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> float:
        return float(weight.T @ covariance_matrix @ weight)

    @classmethod
    def _portfolio_volatility_from_inputs(
        cls,
        weight: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> float:
        variance = max(cls._portfolio_variance_from_inputs(weight, covariance_matrix), 0.0)
        return float(np.sqrt(variance))

    def portfolio_path(self, weight: Optional[Iterable[float]] = None) -> pd.Series:
        return self._basic_metrics(weight).portfolio_path()

    def portfolio_annual_return(self, weight: Optional[Iterable[float]] = None) -> float:
        return self._basic_metrics(weight).portfolio_annual_return()

    def portfolio_variance(self, weight: Optional[Iterable[float]] = None) -> float:
        return self._basic_metrics(weight).portfolio_variance()

    def portfolio_annual_volatility(
        self,
        weight: Optional[Iterable[float]] = None,
    ) -> float:
        return self._basic_metrics(weight).portfolio_annual_volatility()

    def portfolio_variance_coeficience(
        self,
        weight: Optional[Iterable[float]] = None,
    ) -> float:
        return self._basic_metrics(weight).portfolio_variance_coeficience()

    def portfolio_sharpe_ratio(
        self,
        free_rate: float,
        weight: Optional[Iterable[float]] = None,
    ) -> float:
        return self._basic_metrics(weight).portfolio_sharpe_ratio(free_rate)

    @staticmethod
    def _minimum_variance_objective(
        weights: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> float:
        return PortfolioOptimization._portfolio_variance_from_inputs(
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

        if solution.success:
            self.weight = optimal_weights

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


@dataclass
class MeanVarianceOptimizer:
    """Optimize a composed `Portfolio` through the migrated mean-variance solver."""

    portfolio: Portfolio

    def _build_optimizer(self) -> PortfolioOptimization:
        optimizer = PortfolioOptimization(
            tickers=self.portfolio.tickers,
            start=self.portfolio.start,
            end=self.portfolio.end,
            price_field=self.portfolio.price_field,
            weight=self.portfolio.weight,
        )
        optimizer._set_prices_cache(self.portfolio.asset_prices())
        optimizer._set_returns_cache(self.portfolio.asset_returns())
        return optimizer

    def _apply_optimized_weights(self, result: OptimizationResult) -> None:
        if result.success:
            self.portfolio.update_weights(result.weights)

    def optimize_minimum_variance(
        self,
        config: Optional[OptimizationConfig] = None,
    ) -> OptimizationResult:
        active_config = MinimumVarianceConfig() if config is None else config
        result = self._build_optimizer().optimize_minimum_variance(config=active_config)
        self._apply_optimized_weights(result)
        return result

    def optimize_maximum_sharpe(
        self,
        config: Optional[OptimizationConfig] = None,
    ) -> OptimizationResult:
        active_config = OptimizationConfig() if config is None else config
        result = self._build_optimizer().optimize_maximum_sharpe(config=active_config)
        self._apply_optimized_weights(result)
        return result


__all__ = [
    "MeanVarianceOptimizer",
    "MinimumVarianceConfig",
    "OptimizationConfig",
    "OptimizationResult",
    "PortfolioOptimization",
]
