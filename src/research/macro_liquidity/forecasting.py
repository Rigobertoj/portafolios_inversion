"""Econometric forecasting for macro/liquidity scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .model_selection import TimeSeriesModelSelectionResult, TimeSeriesModelSelector
from .policy import MacroLiquidityPolicy


def _safe_float(value: object, default: float = np.nan) -> float:
    return float(value) if pd.notna(value) else default


def _acf(values: pd.Series, lag: int) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= lag:
        return np.nan
    return float(clean.autocorr(lag=lag))


def _ewma_volatility(values: pd.Series, decay: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    variance = 0.0
    for value in clean.to_numpy(dtype=float):
        variance = decay * variance + (1.0 - decay) * value * value
    return float(np.sqrt(variance))


def _information_criteria(residuals: pd.Series, parameters: int) -> tuple[float, float]:
    clean = pd.to_numeric(residuals, errors="coerce").dropna()
    nobs = len(clean)
    if nobs == 0:
        return np.nan, np.nan
    sigma2 = float(np.mean(np.square(clean.to_numpy(dtype=float))))
    if sigma2 <= 0:
        return np.nan, np.nan
    log_likelihood = -0.5 * nobs * (np.log(2.0 * np.pi) + np.log(sigma2) + 1.0)
    aic = 2.0 * parameters - 2.0 * log_likelihood
    bic = np.log(nobs) * parameters - 2.0 * log_likelihood
    return float(aic), float(bic)


@dataclass(frozen=True)
class ForecastResult:
    """Forecast table and diagnostics for macro/liquidity scores."""

    forecasts: pd.DataFrame
    diagnostics: pd.DataFrame
    model_reports: pd.DataFrame
    rolling_backtests: pd.DataFrame
    methodology: dict[str, object]

    def forecast_wide(self, value_column: str = "forecast") -> pd.DataFrame:
        """Return forecasts in one row per horizon with target columns."""

        if self.forecasts.empty:
            return pd.DataFrame()
        return (
            self.forecasts.pivot_table(
                index="horizon_months",
                columns="target",
                values=value_column,
                aggfunc="first",
            )
            .sort_index()
            .reset_index()
        )


class MacroLiquidityForecaster:
    """Forecast block and engine scores with benchmarked model selection."""

    def __init__(self, policy: Optional[MacroLiquidityPolicy] = None) -> None:
        self.policy = policy or MacroLiquidityPolicy()

    def _diagnostics_for(
        self,
        target: str,
        series: pd.Series,
        selection: TimeSeriesModelSelectionResult,
    ) -> dict[str, object]:
        """Build one selected-model diagnostic row for a target series."""

        residuals = selection.final_fit.residuals
        fitted = selection.final_fit.fitted
        clean = pd.to_numeric(series, errors="coerce").dropna()
        comparable = clean.loc[fitted.index] if len(fitted) else pd.Series(dtype=float)
        if len(comparable):
            errors = comparable - fitted
            rmse = float(np.sqrt(np.mean(np.square(errors))))
            mae = float(np.mean(np.abs(errors)))
        else:
            rmse = np.nan
            mae = np.nan
        reports = selection.model_reports
        selected_report = reports[reports["model"] == selection.selected_model]
        rolling_rmse = (
            _safe_float(selected_report["rmse"].iloc[0])
            if not selected_report.empty
            else np.nan
        )
        rolling_mae = (
            _safe_float(selected_report["mae"].iloc[0])
            if not selected_report.empty
            else np.nan
        )
        directional_accuracy = (
            _safe_float(selected_report["directional_accuracy"].iloc[0])
            if not selected_report.empty
            else np.nan
        )
        benchmark_rmse = (
            _safe_float(selected_report["benchmark_rmse"].iloc[0])
            if not selected_report.empty and "benchmark_rmse" in selected_report
            else np.nan
        )
        improvement = (
            _safe_float(selected_report["improvement_vs_benchmark"].iloc[0])
            if not selected_report.empty and "improvement_vs_benchmark" in selected_report
            else np.nan
        )
        aic = _safe_float(selection.final_fit.aic)
        bic = _safe_float(selection.final_fit.bic)
        if pd.isna(aic) or pd.isna(bic):
            aic, bic = _information_criteria(
                residuals,
                parameters=max(int(selection.final_fit.parameter_count), 1),
            )
        return {
            "target": target,
            "model": selection.selected_model,
            "benchmark_model": selection.benchmark_model,
            "accepted": bool(selection.accepted),
            "acceptance_reason": selection.acceptance_reason,
            "observations": int(len(clean)),
            "rolling_rmse": rolling_rmse,
            "rolling_mae": rolling_mae,
            "directional_accuracy": directional_accuracy,
            "benchmark_rmse": benchmark_rmse,
            "improvement_vs_benchmark": improvement,
            "rmse": rmse,
            "mae": mae,
            "residual_std": _safe_float(pd.to_numeric(residuals, errors="coerce").std(ddof=0)),
            "ewma_residual_volatility": _ewma_volatility(residuals, self.policy.forecast.ewma_lambda),
            "aic": aic,
            "bic": bic,
            "acf_lag_1": _acf(clean, 1),
            "acf_lag_3": _acf(clean, 3),
            "acf_lag_6": _acf(clean, 6),
            "acf_lag_12": _acf(clean, 12),
        }

    def fit_predict(self, score_frame: pd.DataFrame) -> ForecastResult:
        """Select and fit forecast models for all score columns in `score_frame`."""

        if score_frame.empty:
            raise ValueError("score_frame is empty; cannot forecast macro/liquidity scores.")

        ordered = score_frame.sort_values("date").set_index("date")
        targets = [
            column
            for column in ordered.columns
            if column.endswith("_score") and pd.api.types.is_numeric_dtype(ordered[column])
        ]
        if not targets:
            raise ValueError("score_frame does not contain numeric *_score columns.")

        forecast_rows = []
        diagnostic_rows = []
        model_report_frames = []
        rolling_backtest_frames = []
        latest_date = pd.Timestamp(ordered.index.max())
        selector = TimeSeriesModelSelector(self.policy.forecast)
        for target in targets:
            series = ordered[target]
            selection = selector.select(
                target=target,
                series=series,
                horizons=tuple(int(h) for h in self.policy.forecast.horizons_months),
            )
            diagnostic = self._diagnostics_for(target, series, selection)
            diagnostic_rows.append(diagnostic)
            model_report_frames.append(selection.model_reports)
            rolling_backtest_frames.append(selection.rolling_backtests)
            residual_vol = diagnostic["ewma_residual_volatility"]
            if pd.isna(residual_vol):
                residual_vol = diagnostic["residual_std"]
            if pd.isna(residual_vol):
                residual_vol = 0.75
            for horizon in self.policy.forecast.horizons_months:
                forecast_value = float(selection.forecasts[int(horizon)])
                interval = self.policy.forecast.confidence_interval_z * float(residual_vol) * np.sqrt(horizon)
                forecast_rows.append(
                    {
                        "as_of": latest_date,
                        "target": target,
                        "horizon_months": int(horizon),
                        "forecast_date": latest_date + pd.DateOffset(months=int(horizon)),
                        "forecast": float(forecast_value),
                        "lower": float(forecast_value - interval),
                        "upper": float(forecast_value + interval),
                        "model": selection.selected_model,
                        "benchmark_model": self.policy.forecast.benchmark_model,
                        "accepted_candidate": bool(selection.accepted),
                        "acceptance_reason": selection.acceptance_reason,
                    }
                )

        forecasts = pd.DataFrame(forecast_rows).sort_values(["horizon_months", "target"])
        diagnostics = pd.DataFrame(diagnostic_rows).sort_values("target")
        model_reports = (
            pd.concat(model_report_frames, ignore_index=True)
            if model_report_frames
            else pd.DataFrame()
        )
        rolling_backtests = (
            pd.concat(rolling_backtest_frames, ignore_index=True)
            if rolling_backtest_frames
            else pd.DataFrame()
        )
        return ForecastResult(
            forecasts=forecasts,
            diagnostics=diagnostics,
            model_reports=model_reports,
            rolling_backtests=rolling_backtests,
            methodology=self.policy.methodology_summary(),
        )
