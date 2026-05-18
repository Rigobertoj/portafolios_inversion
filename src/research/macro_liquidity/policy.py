"""Methodology policies for macro/liquidity research runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class DataPolicy:
    """Data timing, frequency, and audit settings."""

    data_mode: str = "revised"
    real_time_safe: bool = False
    canonical_frequency: str = "M"
    rebalance_frequency: str = "Q"
    quarterly_interpolation: str = "linear"
    publication_lag_days_by_frequency: Mapping[str, int] = field(
        default_factory=lambda: {
            "D": 1,
            "W": 3,
            "M": 15,
            "Q": 45,
            "A": 90,
            "Y": 90,
        }
    )

    def lag_days_for_frequency(self, frequency: str) -> int:
        """Return the publication lag used by the feature builder."""

        return int(self.publication_lag_days_by_frequency.get(frequency.upper(), 0))


@dataclass(frozen=True)
class ValidationPolicy:
    """Rules that decide whether an indicator is useful enough to trust."""

    min_history_months: int = 24
    max_missing_ratio: float = 0.35
    min_signal_stability: float = 0.45
    candidate_weight_multiplier: float = 0.50


@dataclass(frozen=True)
class ConfidencePolicy:
    """Weights used to convert evidence quality into a confidence score."""

    coverage_weight: float = 0.35
    agreement_weight: float = 0.25
    stability_weight: float = 0.20
    freshness_weight: float = 0.10
    narrative_weight: float = 0.10
    contradiction_threshold: float = 0.50


@dataclass(frozen=True)
class ForecastPolicy:
    """Forecast horizons and model governance settings."""

    horizons_months: tuple[int, ...] = (3, 12, 6)
    base_horizon_months: int = 3
    strategic_horizon_months: int = 12
    fallback_horizon_months: int = 6
    benchmark_model: str = "ar1"
    diagnostic_benchmark_models: tuple[str, ...] = ("mean", "naive", "drift", "ar1")
    candidate_models: tuple[str, ...] = (
        "mean_reversion",
        "ets",
        "auto_sarima",
        "bootstrap_ar1",
    )
    selection_metric: str = "rmse"
    min_improvement_over_benchmark: float = 0.02
    minimum_train_months: int = 24
    rolling_initial_train_months: int = 36
    rolling_step_months: int = 1
    rolling_forecast_horizon: int = 3
    max_cv_folds: int = 24
    ewma_lambda: float = 0.94
    confidence_interval_z: float = 1.64
    bootstrap_iterations: int = 250
    bootstrap_random_seed: int = 42
    sarima_p_values: tuple[int, ...] = (0, 1, 2)
    sarima_d_values: tuple[int, ...] = (0, 1)
    sarima_q_values: tuple[int, ...] = (0, 1, 2)
    sarima_seasonal_periods: tuple[int, ...] = (12,)
    sarima_max_models: int = 12
    sarima_maxiter: int = 50


@dataclass(frozen=True)
class ScenarioPolicy:
    """Rules that turn forecasts into base/upside/downside scenarios."""

    base_floor: float = 0.40
    base_confidence_slope: float = 0.30
    downside_base: float = 0.35
    downside_confidence_penalty: float = 0.15
    liquidity_stress_penalty: float = 0.10


@dataclass(frozen=True)
class AuditPolicy:
    """Research-run storage settings."""

    run_root: str = "data/research_runs"
    write_artifacts: bool = True


@dataclass(frozen=True)
class MacroLiquidityPolicy:
    """Top-level policy bundle for the macro/liquidity workflow."""

    data: DataPolicy = field(default_factory=DataPolicy)
    validation: ValidationPolicy = field(default_factory=ValidationPolicy)
    confidence: ConfidencePolicy = field(default_factory=ConfidencePolicy)
    forecast: ForecastPolicy = field(default_factory=ForecastPolicy)
    scenario: ScenarioPolicy = field(default_factory=ScenarioPolicy)
    audit: AuditPolicy = field(default_factory=AuditPolicy)

    def methodology_summary(self) -> dict[str, object]:
        """Return a report-friendly summary of key methodology choices."""

        return {
            "data_mode": self.data.data_mode,
            "real_time_safe": self.data.real_time_safe,
            "canonical_frequency": self.data.canonical_frequency,
            "rebalance_frequency": self.data.rebalance_frequency,
            "quarterly_interpolation": self.data.quarterly_interpolation,
            "forecast_horizons_months": self.forecast.horizons_months,
            "base_horizon_months": self.forecast.base_horizon_months,
            "strategic_horizon_months": self.forecast.strategic_horizon_months,
            "fallback_horizon_months": self.forecast.fallback_horizon_months,
            "benchmark_model": self.forecast.benchmark_model,
            "diagnostic_benchmark_models": self.forecast.diagnostic_benchmark_models,
            "candidate_models": self.forecast.candidate_models,
            "selection_metric": self.forecast.selection_metric,
            "min_improvement_over_benchmark": self.forecast.min_improvement_over_benchmark,
            "rolling_initial_train_months": self.forecast.rolling_initial_train_months,
            "rolling_forecast_horizon": self.forecast.rolling_forecast_horizon,
            "run_root": self.audit.run_root,
        }
