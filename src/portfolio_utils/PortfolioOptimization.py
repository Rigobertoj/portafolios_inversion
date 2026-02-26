from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy.optimize import minimize

try:
    from .PortfolioElementaryAnalysis import PortfolioElementaryAnalysis
except ImportError:
    from PortfolioElementaryAnalysis import PortfolioElementaryAnalysis


@dataclass
class OptimizationConfig:
    objective: str = "max_sharpe"
    expected_return_model: str = "historical"
    risk_free_rate: float = 0.0
    market_expected_return: Optional[float] = None
    target_return: Optional[float] = None
    annualize_covariance: bool = True
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
class OptimizationResult:
    objective: str
    success: bool
    status: int
    message: str
    weights: np.ndarray
    weights_by_ticker: pd.Series
    expected_return: float
    volatility: float
    sharpe: float
    objective_value: float
    iterations: int


class PortfolioOptimization(PortfolioElementaryAnalysis):
    def _assets_order(self) -> list[str]:
        return self.get_returns().columns.tolist()

    def _expected_returns_vector(self, config: OptimizationConfig) -> np.ndarray:
        asset_order = self._assets_order()
        if config.expected_return_model == "historical":
            expected_returns = self.annual_return(asset_order)
        elif config.expected_return_model == "capm_assets":
            expected_returns = self.assets_capm_expected_return(
                risk_free_rate=config.risk_free_rate,
                market_expected_return=config.market_expected_return,
            )
        else:
            raise ValueError(
                "expected_return_model must be 'historical' or 'capm_assets'."
            )
        return expected_returns.loc[asset_order].to_numpy(dtype=float)

    def _covariance_matrix(self, config: OptimizationConfig) -> np.ndarray:
        asset_order = self._assets_order()
        covariance_matrix = self.covariance(asset_order).loc[asset_order, asset_order]
        covariance_values = covariance_matrix.to_numpy(dtype=float)
        if config.annualize_covariance:
            covariance_values = covariance_values * 252.0
        return covariance_values

    @staticmethod
    def _portfolio_return(weights: np.ndarray, expected_returns: np.ndarray) -> float:
        return float(weights @ expected_returns)

    @staticmethod
    def _portfolio_volatility(weights: np.ndarray, covariance_matrix: np.ndarray) -> float:
        variance = float(weights.T @ covariance_matrix @ weights)
        variance = max(variance, 0.0)
        return float(np.sqrt(variance))

    @staticmethod
    def _min_variance_objective(
        weights: np.ndarray, covariance_matrix: np.ndarray
    ) -> float:
        return float(weights.T @ covariance_matrix @ weights)

    @classmethod
    def _negative_sharpe_objective(
        cls,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_free_rate: float,
    ) -> float:
        portfolio_volatility = cls._portfolio_volatility(weights, covariance_matrix)
        if np.isclose(portfolio_volatility, 0.0):
            return float("inf")
        portfolio_return = cls._portfolio_return(weights, expected_returns)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return float(-sharpe_ratio)

    def _resolve_bounds(
        self,
        config: OptimizationConfig,
        n_assets: int,
    ) -> Sequence[Tuple[float, float]]:
        if config.bounds is not None:
            if len(config.bounds) != n_assets:
                raise ValueError("bounds length must match number of assets.")
            return config.bounds
        if config.allow_short:
            return [(-1.0, 1.0)] * n_assets
        return [(0.0, 1.0)] * n_assets

    def _resolve_initial_weights(
        self,
        config: OptimizationConfig,
        n_assets: int,
    ) -> np.ndarray:
        if config.initial_weights is None:
            return np.ones(n_assets, dtype=float) / n_assets
        initial_weights = np.asarray(config.initial_weights, dtype=float).reshape(-1)
        if len(initial_weights) != n_assets:
            raise ValueError("initial_weights length must match number of assets.")
        if not np.isclose(initial_weights.sum(), 1.0):
            raise ValueError("initial_weights must sum 1.")
        return initial_weights

    def optimize(self, config: Optional[OptimizationConfig] = None) -> OptimizationResult:
        if config is None:
            config = OptimizationConfig()

        expected_returns = self._expected_returns_vector(config)
        covariance_matrix = self._covariance_matrix(config)
        tickers = self._assets_order()
        n_assets = len(tickers)

        initial_weights = self._resolve_initial_weights(config, n_assets)
        bounds = self._resolve_bounds(config, n_assets)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        if config.objective == "min_variance":
            def objective_function(w, cov=covariance_matrix):
                return (self._min_variance_objective(w, cov))
        elif config.objective == "max_sharpe":
            def objective_function(w, mu=expected_returns, cov=covariance_matrix, rf=config.risk_free_rate):
                return (self._negative_sharpe_objective(
                                w, mu, cov, rf
                            ))
        elif config.objective == "target_return":
            if config.target_return is None:
                raise ValueError(
                    "target_return must be provided when objective='target_return'."
                )
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda w, mu=expected_returns, target=config.target_return: (
                        w @ mu
                    )
                    - target,
                }
            )
            def objective_function(w, cov=covariance_matrix):
                return (self._min_variance_objective(w, cov))
        else:
            raise ValueError(
                "objective must be 'min_variance', 'max_sharpe', or 'target_return'."
            )

        solution = minimize(
            fun=objective_function,
            x0=initial_weights,
            method=config.solver_method,
            bounds=bounds,
            constraints=constraints,
            options=config.solver_options,
        )

        optimal_weights = np.asarray(solution.x, dtype=float)
        portfolio_return = self._portfolio_return(optimal_weights, expected_returns)
        portfolio_volatility = self._portfolio_volatility(
            optimal_weights, covariance_matrix
        )

        if np.isclose(portfolio_volatility, 0.0):
            portfolio_sharpe = float("nan")
        else:
            portfolio_sharpe = float(
                (portfolio_return - config.risk_free_rate) / portfolio_volatility
            )

        objective_value = float(objective_function(optimal_weights))
        weights_by_ticker = pd.Series(optimal_weights, index=tickers, name="weight")

        iterations = int(getattr(solution, "nit", 0))

        return OptimizationResult(
            objective=config.objective,
            success=bool(solution.success),
            status=int(solution.status),
            message=str(solution.message),
            weights=optimal_weights,
            weights_by_ticker=weights_by_ticker,
            expected_return=float(portfolio_return),
            volatility=float(portfolio_volatility),
            sharpe=float(portfolio_sharpe),
            objective_value=objective_value,
            iterations=iterations,
        )

    def optimize_and_set_weights(
        self, config: Optional[OptimizationConfig] = None
    ) -> OptimizationResult:
        result = self.optimize(config=config)
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        self.weight = result.weights
        return result

def _main_():
    tickets = ["BOH", "AAPL", "JPM"]
    start_date = "2025-01-01"
    end_date = "2026-02-20"
    
    weight = np.ones(len(tickets)) / len(tickets)
    
    benchmark = "^GSPC"
    
    opt = PortfolioOptimization(
        tickers=tickets,
        start=start_date,
        weight=weight,
        benchmark=benchmark
    )
    
    cfg = OptimizationConfig()
    
    return


if __name__ == "__main__":
    pass