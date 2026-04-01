"""Post-modern optimization implemented on top of the new package layout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..portfolio.portfolio import Portfolio
from .configs import (
    MaximumOmegaConfig,
    MinimumSemivarianceConfig,
    PostModernOptimizationConfig,
)
from .mean_variance import PortfolioOptimization
from .results import PostModernOptimizationResult


class PortfolioOptimizationPostModern(PortfolioOptimization):
    """
    Backward-compatible downside optimizer built on the new optimization stack.

    The class preserves the historical constructor used in notebooks while
    moving the real optimization logic into `src.optimization`.
    """

    def __init__(
        self,
        tickers,
        start,
        end=None,
        price_field="Close",
        weight=None,
    ) -> None:
        super().__init__(
            tickers=tickers,
            start=start,
            end=end,
            price_field=price_field,
            weight=weight,
        )
        self._reset_post_modern_cache()

    def _reset_cache(self) -> None:
        super()._reset_cache()
        if hasattr(self, "_PortfolioOptimizationPostModern__returns_below_cache"):
            self._reset_post_modern_cache()

    def _reset_post_modern_cache(self) -> None:
        self.returns_down = None
        self.returns_up = None
        self.shortfall_risk = None
        self.upside_potential = None
        self.__active_threshold = 0.0
        self.__returns_below_cache = {}
        self.__returns_above_cache = {}
        self.__downside_risk_cache = {}
        self.__upside_risk_cache = {}
        self.__semivariance_cache = {}
        self.__omega_cache = {}

    @staticmethod
    def _normalize_threshold(value: Optional[float]) -> float:
        if value is None:
            raise ValueError("threshold cannot be None in this context.")

        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("threshold must be a numeric value.") from exc

    def _resolve_threshold(self, threshold: Optional[float]) -> float:
        if threshold is None:
            return self.__active_threshold
        return self._normalize_threshold(threshold)

    @staticmethod
    def _normalize_benchmark_returns(
        benchmark_returns: pd.Series | pd.DataFrame,
    ) -> pd.Series:
        if isinstance(benchmark_returns, pd.Series):
            benchmark = benchmark_returns.sort_index().dropna().copy()
        elif isinstance(benchmark_returns, pd.DataFrame):
            cleaned = benchmark_returns.sort_index().dropna().copy()
            if cleaned.shape[1] != 1:
                raise ValueError("benchmark_returns must contain exactly one column.")
            benchmark = cleaned.iloc[:, 0]
        else:
            raise TypeError("benchmark_returns must be a pandas Series or DataFrame.")

        benchmark = pd.to_numeric(benchmark, errors="coerce").dropna()
        if benchmark.empty:
            raise ValueError("benchmark_returns must contain at least one observation.")
        return benchmark

    def _reference_adjusted_returns(
        self,
        threshold: float,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> pd.DataFrame:
        returns = self.get_returns(self._assets_order())
        if benchmark_returns is None:
            return returns - threshold

        benchmark = self._normalize_benchmark_returns(benchmark_returns)
        aligned_returns, aligned_benchmark = returns.align(benchmark, join="inner", axis=0)
        if len(aligned_returns) < 2:
            raise ValueError(
                "benchmark_returns must overlap asset returns with at least two observations."
            )

        reference = aligned_benchmark + threshold
        return aligned_returns.sub(reference, axis=0)

    def returns_below(
        self,
        threshold: float = 0.0,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> pd.DataFrame:
        normalized_threshold = self._normalize_threshold(threshold)
        filtered_returns = None
        if benchmark_returns is None:
            filtered_returns = self.__returns_below_cache.get(normalized_threshold)

        if filtered_returns is None:
            deviations = self._reference_adjusted_returns(
                normalized_threshold,
                benchmark_returns=benchmark_returns,
            )
            filtered_returns = deviations.where(deviations < 0.0, 0.0)
            if benchmark_returns is None:
                self.__returns_below_cache[normalized_threshold] = filtered_returns

        self.__active_threshold = normalized_threshold
        self.returns_down = filtered_returns.copy()

        cached_risk = (
            None
            if benchmark_returns is not None
            else self.__downside_risk_cache.get(normalized_threshold)
        )
        self.shortfall_risk = (
            None if cached_risk is None else cached_risk.to_numpy(dtype=float)
        )

        return filtered_returns.copy()

    def returns_above(
        self,
        threshold: float = 0.0,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> pd.DataFrame:
        normalized_threshold = self._normalize_threshold(threshold)
        filtered_returns = None
        if benchmark_returns is None:
            filtered_returns = self.__returns_above_cache.get(normalized_threshold)

        if filtered_returns is None:
            deviations = self._reference_adjusted_returns(
                normalized_threshold,
                benchmark_returns=benchmark_returns,
            )
            filtered_returns = deviations.where(deviations > 0.0, 0.0)
            if benchmark_returns is None:
                self.__returns_above_cache[normalized_threshold] = filtered_returns

        self.__active_threshold = normalized_threshold
        self.returns_up = filtered_returns.copy()

        cached_risk = (
            None
            if benchmark_returns is not None
            else self.__upside_risk_cache.get(normalized_threshold)
        )
        self.upside_potential = (
            None if cached_risk is None else cached_risk.to_numpy(dtype=float)
        )

        return filtered_returns.copy()

    def _downside_risk_series(
        self,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> pd.Series:
        normalized_threshold = self._resolve_threshold(threshold)
        downside_risk = None
        if benchmark_returns is None:
            downside_risk = self.__downside_risk_cache.get(normalized_threshold)

        if downside_risk is None:
            filtered_returns = self.returns_below(
                normalized_threshold,
                benchmark_returns=benchmark_returns,
            )
            downside_risk = filtered_returns.std() * np.sqrt(252.0)
            downside_risk = downside_risk.loc[self._assets_order()]
            if benchmark_returns is None:
                self.__downside_risk_cache[normalized_threshold] = downside_risk

        self.__active_threshold = normalized_threshold
        self.shortfall_risk = downside_risk.to_numpy(dtype=float)
        return downside_risk.copy()

    def downside_risk(
        self,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> np.ndarray:
        return self._downside_risk_series(
            threshold,
            benchmark_returns=benchmark_returns,
        ).to_numpy(dtype=float)

    def _upside_risk_series(
        self,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> pd.Series:
        normalized_threshold = self._resolve_threshold(threshold)
        upside_risk = None
        if benchmark_returns is None:
            upside_risk = self.__upside_risk_cache.get(normalized_threshold)

        if upside_risk is None:
            filtered_returns = self.returns_above(
                normalized_threshold,
                benchmark_returns=benchmark_returns,
            )
            upside_risk = filtered_returns.std() * np.sqrt(252.0)
            upside_risk = upside_risk.loc[self._assets_order()]
            if benchmark_returns is None:
                self.__upside_risk_cache[normalized_threshold] = upside_risk

        self.__active_threshold = normalized_threshold
        self.upside_potential = upside_risk.to_numpy(dtype=float)
        return upside_risk.copy()

    def upside_risk(
        self,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> np.ndarray:
        return self._upside_risk_series(
            threshold,
            benchmark_returns=benchmark_returns,
        ).to_numpy(dtype=float)

    def semivariance_matrix(
        self,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> pd.DataFrame:
        normalized_threshold = self._resolve_threshold(threshold)
        semivariance = None
        if benchmark_returns is None:
            semivariance = self.__semivariance_cache.get(normalized_threshold)

        if semivariance is None:
            downside_risk = self._downside_risk_series(
                normalized_threshold,
                benchmark_returns=benchmark_returns,
            )
            vector = downside_risk.to_numpy(dtype=float)
            semivariance = pd.DataFrame(
                np.outer(vector, vector),
                index=downside_risk.index,
                columns=downside_risk.index,
            )
            correlation_returns = self.get_returns(self._assets_order())
            if benchmark_returns is not None:
                deviations = self._reference_adjusted_returns(
                    normalized_threshold,
                    benchmark_returns=benchmark_returns,
                )
                correlation_returns = correlation_returns.loc[deviations.index]

            correlation = correlation_returns.corr().loc[
                downside_risk.index,
                downside_risk.index,
            ]
            semivariance = semivariance * correlation
            if benchmark_returns is None:
                self.__semivariance_cache[normalized_threshold] = semivariance

        self.__active_threshold = normalized_threshold
        self.returns_down = self.returns_below(
            normalized_threshold,
            benchmark_returns=benchmark_returns,
        )
        self.shortfall_risk = self.downside_risk(
            normalized_threshold,
            benchmark_returns=benchmark_returns,
        )
        return semivariance.copy()

    def portfolio_semivariance(
        self,
        weight: Optional[Sequence[float]] = None,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> float:
        vector = self._resolve_weight(weight)
        semivariance = self.semivariance_matrix(
            threshold,
            benchmark_returns=benchmark_returns,
        ).to_numpy(dtype=float)
        value = self._portfolio_variance_from_inputs(vector, semivariance)
        return max(value, 0.0)

    def portfolio_downside_risk(
        self,
        weight: Optional[Sequence[float]] = None,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> float:
        semivariance = self.portfolio_semivariance(
            weight=weight,
            threshold=threshold,
            benchmark_returns=benchmark_returns,
        )
        return float(np.sqrt(semivariance))

    def asset_omega_ratio(
        self,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> pd.Series:
        normalized_threshold = self._resolve_threshold(threshold)
        omega = None
        if benchmark_returns is None:
            omega = self.__omega_cache.get(normalized_threshold)

        if omega is None:
            upside = self._upside_risk_series(
                normalized_threshold,
                benchmark_returns=benchmark_returns,
            )
            downside = self._downside_risk_series(
                normalized_threshold,
                benchmark_returns=benchmark_returns,
            )
            omega = upside / downside.replace(0.0, np.nan)
            omega = omega.loc[self._assets_order()]
            if benchmark_returns is None:
                self.__omega_cache[normalized_threshold] = omega

        self.__active_threshold = normalized_threshold
        return omega.copy()

    def portfolio_omega_ratio(
        self,
        weight: Optional[Sequence[float]] = None,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> float:
        vector = self._resolve_weight(weight)
        omega = self.asset_omega_ratio(
            threshold,
            benchmark_returns=benchmark_returns,
        ).to_numpy(dtype=float)
        if not np.isfinite(omega).all():
            raise ValueError("omega ratio is undefined for assets with zero downside risk.")
        return float(vector @ omega)

    def semivarianza_down(
        self,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> pd.DataFrame:
        return self.semivariance_matrix(
            threshold=threshold,
            benchmark_returns=benchmark_returns,
        )

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

    @staticmethod
    def _minimum_semivariance_objective(
        weights: np.ndarray,
        semivariance_matrix: np.ndarray,
    ) -> float:
        return PortfolioOptimizationPostModern._portfolio_variance_from_inputs(
            weights,
            semivariance_matrix,
        )

    @staticmethod
    def _negative_omega_objective(
        weights: np.ndarray,
        omega_vector: np.ndarray,
    ) -> float:
        return float(-(weights @ omega_vector))

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


@dataclass
class PostModernOptimizer:
    """Optimize a composed `Portfolio` through the migrated downside solver."""

    portfolio: Portfolio

    def _build_optimizer(self) -> PortfolioOptimizationPostModern:
        optimizer = PortfolioOptimizationPostModern(
            tickers=self.portfolio.tickers,
            start=self.portfolio.start,
            end=self.portfolio.end,
            price_field=self.portfolio.price_field,
            weight=self.portfolio.weight,
        )
        optimizer._set_prices_cache(self.portfolio.asset_prices())
        optimizer._set_returns_cache(self.portfolio.asset_returns())
        return optimizer

    def _apply_optimized_weights(self, result: PostModernOptimizationResult) -> None:
        if result.success:
            self.portfolio.update_weights(result.weights)

    def optimize_minimum_semivariance(
        self,
        config: Optional[PostModernOptimizationConfig] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> PostModernOptimizationResult:
        active_config = MinimumSemivarianceConfig() if config is None else config
        result = self._build_optimizer().optimize_minimum_semivariance(
            config=active_config,
            benchmark_returns=benchmark_returns,
        )
        self._apply_optimized_weights(result)
        return result

    def optimize_maximum_omega(
        self,
        config: Optional[PostModernOptimizationConfig] = None,
    ) -> PostModernOptimizationResult:
        active_config = MaximumOmegaConfig() if config is None else config
        result = self._build_optimizer().optimize_maximum_omega(config=active_config)
        self._apply_optimized_weights(result)
        return result


PortfolioOptimizationPostMordern = PortfolioOptimizationPostModern


__all__ = [
    "MaximumOmegaConfig",
    "MinimumSemivarianceConfig",
    "PortfolioOptimizationPostMordern",
    "PortfolioOptimizationPostModern",
    "PostModernOptimizationConfig",
    "PostModernOptimizationResult",
    "PostModernOptimizer",
]
