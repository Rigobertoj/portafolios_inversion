"""Auditable time-series model selection for macro/liquidity forecasts."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import numpy as np
import pandas as pd

from .policy import ForecastPolicy


def _safe_float(value: object, default: float = np.nan) -> float:
    return float(value) if pd.notna(value) else default


def _clean_series(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if clean.empty:
        raise ValueError("series must contain at least one non-null value.")
    return clean


def _information_criteria(
    residuals: pd.Series,
    parameters: int,
) -> tuple[float, float]:
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


def _aligned_residuals(actual: pd.Series, fitted: pd.Series) -> pd.Series:
    if fitted.empty:
        return pd.Series(dtype=float)
    comparable = actual.loc[fitted.index]
    return (comparable - fitted).dropna()


@dataclass(frozen=True)
class TimeSeriesFit:
    """Final fit and forecast output from one candidate model."""

    model: str
    forecasts: Mapping[int, float]
    fitted: pd.Series
    residuals: pd.Series
    parameters: Mapping[str, object]
    aic: float = np.nan
    bic: float = np.nan
    parameter_count: int = 0


@dataclass(frozen=True)
class TimeSeriesModelSelectionResult:
    """Selected model, evidence table, and final forecasts for one target."""

    target: str
    selected_model: str
    benchmark_model: str
    accepted: bool
    acceptance_reason: str
    forecasts: Mapping[int, float]
    final_fit: TimeSeriesFit
    model_reports: pd.DataFrame
    rolling_backtests: pd.DataFrame


class TimeSeriesModelSelector:
    """Select the best forecasting model against an explicit benchmark."""

    def __init__(self, policy: Optional[ForecastPolicy] = None) -> None:
        self.policy = policy or ForecastPolicy()

    @staticmethod
    def _unique_models(models: tuple[str, ...]) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered = []
        for model in models:
            key = str(model).strip().lower()
            if key and key not in seen:
                seen.add(key)
                ordered.append(key)
        return tuple(ordered)

    def _models_to_evaluate(self) -> tuple[str, ...]:
        benchmark = (self.policy.benchmark_model,)
        return self._unique_models(
            tuple(self.policy.diagnostic_benchmark_models)
            + benchmark
            + tuple(self.policy.candidate_models)
        )

    def _model_role(self, model: str) -> str:
        benchmarks = set(self._unique_models(tuple(self.policy.diagnostic_benchmark_models)))
        if model == self.policy.benchmark_model or model in benchmarks:
            return "benchmark"
        return "candidate"

    @staticmethod
    def _fit_mean(series: pd.Series, horizons: tuple[int, ...]) -> TimeSeriesFit:
        clean = _clean_series(series)
        mean_value = float(clean.mean())
        fitted = pd.Series(mean_value, index=clean.index)
        residuals = _aligned_residuals(clean, fitted)
        aic, bic = _information_criteria(residuals, parameters=1)
        return TimeSeriesFit(
            model="mean",
            forecasts={int(h): mean_value for h in horizons},
            fitted=fitted,
            residuals=residuals,
            parameters={"mean": mean_value},
            aic=aic,
            bic=bic,
            parameter_count=1,
        )

    @staticmethod
    def _fit_naive(series: pd.Series, horizons: tuple[int, ...]) -> TimeSeriesFit:
        clean = _clean_series(series)
        last = float(clean.iloc[-1])
        if len(clean) > 1:
            fitted = clean.shift(1).dropna()
            residuals = _aligned_residuals(clean, fitted)
        else:
            fitted = pd.Series(dtype=float)
            residuals = pd.Series(dtype=float)
        aic, bic = _information_criteria(residuals, parameters=1)
        return TimeSeriesFit(
            model="naive",
            forecasts={int(h): last for h in horizons},
            fitted=fitted,
            residuals=residuals,
            parameters={"last": last},
            aic=aic,
            bic=bic,
            parameter_count=1,
        )

    @staticmethod
    def _fit_drift(series: pd.Series, horizons: tuple[int, ...]) -> TimeSeriesFit:
        clean = _clean_series(series)
        last = float(clean.iloc[-1])
        if len(clean) > 1:
            slope = float((clean.iloc[-1] - clean.iloc[0]) / (len(clean) - 1))
            fitted = clean.shift(1).dropna() + slope
            residuals = _aligned_residuals(clean, fitted)
        else:
            slope = 0.0
            fitted = pd.Series(dtype=float)
            residuals = pd.Series(dtype=float)
        aic, bic = _information_criteria(residuals, parameters=2)
        return TimeSeriesFit(
            model="drift",
            forecasts={int(h): float(last + slope * int(h)) for h in horizons},
            fitted=fitted,
            residuals=residuals,
            parameters={"last": last, "slope": slope},
            aic=aic,
            bic=bic,
            parameter_count=2,
        )

    @staticmethod
    def _ar1_parameters(series: pd.Series) -> dict[str, object]:
        clean = _clean_series(series)
        if len(clean) < 3:
            return {
                "intercept": 0.0,
                "phi": 1.0,
                "last": float(clean.iloc[-1]),
                "fitted": pd.Series(dtype=float),
                "residuals": pd.Series(dtype=float),
            }

        y = clean.iloc[1:].to_numpy(dtype=float)
        x = clean.iloc[:-1].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        intercept, phi = np.linalg.lstsq(design, y, rcond=None)[0]
        fitted_values = design @ np.array([intercept, phi])
        fitted = pd.Series(fitted_values, index=clean.index[1:])
        residuals = _aligned_residuals(clean, fitted)
        return {
            "intercept": float(intercept),
            "phi": float(phi),
            "last": float(clean.iloc[-1]),
            "fitted": fitted,
            "residuals": residuals,
        }

    @staticmethod
    def _recursive_ar1(intercept: float, phi: float, last: float, horizon: int) -> float:
        value = float(last)
        for _ in range(max(int(horizon), 1)):
            value = intercept + phi * value
        return float(value)

    def _fit_ar1(self, series: pd.Series, horizons: tuple[int, ...]) -> TimeSeriesFit:
        fit = self._ar1_parameters(series)
        forecasts = {
            int(h): self._recursive_ar1(
                _safe_float(fit["intercept"], 0.0),
                _safe_float(fit["phi"], 1.0),
                _safe_float(fit["last"], 0.0),
                int(h),
            )
            for h in horizons
        }
        residuals = fit["residuals"]
        aic, bic = _information_criteria(residuals, parameters=2)
        return TimeSeriesFit(
            model="ar1",
            forecasts=forecasts,
            fitted=fit["fitted"],
            residuals=residuals,
            parameters={
                "intercept": _safe_float(fit["intercept"]),
                "phi": _safe_float(fit["phi"]),
                "last": _safe_float(fit["last"]),
            },
            aic=aic,
            bic=bic,
            parameter_count=2,
        )

    @staticmethod
    def _fit_mean_reversion(series: pd.Series, horizons: tuple[int, ...]) -> TimeSeriesFit:
        clean = _clean_series(series)
        mean_value = float(clean.mean())
        if len(clean) > 2:
            lagged = clean.iloc[:-1] - mean_value
            current = clean.iloc[1:] - mean_value
            denominator = float(np.dot(lagged, lagged))
            phi = float(np.dot(lagged, current) / denominator) if denominator else 0.0
            phi = float(np.clip(phi, 0.0, 0.98))
            fitted = mean_value + phi * (clean.shift(1).dropna() - mean_value)
            residuals = _aligned_residuals(clean, fitted)
        else:
            phi = 0.0
            fitted = pd.Series(dtype=float)
            residuals = pd.Series(dtype=float)
        last = float(clean.iloc[-1])
        forecasts = {
            int(h): float(mean_value + (phi ** int(h)) * (last - mean_value))
            for h in horizons
        }
        aic, bic = _information_criteria(residuals, parameters=2)
        return TimeSeriesFit(
            model="mean_reversion",
            forecasts=forecasts,
            fitted=fitted,
            residuals=residuals,
            parameters={"long_run_mean": mean_value, "phi": phi, "last": last},
            aic=aic,
            bic=bic,
            parameter_count=2,
        )

    def _fit_bootstrap_ar1(self, series: pd.Series, horizons: tuple[int, ...]) -> TimeSeriesFit:
        base = self._fit_ar1(series, horizons)
        residuals = pd.to_numeric(base.residuals, errors="coerce").dropna()
        if residuals.empty or self.policy.bootstrap_iterations <= 0:
            return TimeSeriesFit(
                model="bootstrap_ar1",
                forecasts=dict(base.forecasts),
                fitted=base.fitted,
                residuals=base.residuals,
                parameters={
                    **dict(base.parameters),
                    "bootstrap_iterations": 0,
                    "bootstrap_used": False,
                },
                aic=base.aic,
                bic=base.bic,
                parameter_count=base.parameter_count,
            )

        seed = int(self.policy.bootstrap_random_seed + len(residuals))
        rng = np.random.default_rng(seed)
        max_horizon = max(int(h) for h in horizons)
        paths = np.zeros((int(self.policy.bootstrap_iterations), max_horizon), dtype=float)
        intercept = float(base.parameters.get("intercept", 0.0))
        phi = float(base.parameters.get("phi", 1.0))
        residual_values = residuals.to_numpy(dtype=float)
        for path in range(paths.shape[0]):
            value = float(base.parameters.get("last", 0.0))
            shocks = rng.choice(residual_values, size=max_horizon, replace=True)
            for step in range(max_horizon):
                value = intercept + phi * value + float(shocks[step])
                paths[path, step] = value

        forecasts = {int(h): float(paths[:, int(h) - 1].mean()) for h in horizons}
        return TimeSeriesFit(
            model="bootstrap_ar1",
            forecasts=forecasts,
            fitted=base.fitted,
            residuals=base.residuals,
            parameters={
                **dict(base.parameters),
                "bootstrap_iterations": int(self.policy.bootstrap_iterations),
                "bootstrap_used": True,
            },
            aic=base.aic,
            bic=base.bic,
            parameter_count=base.parameter_count,
        )

    @staticmethod
    def _fit_ets(series: pd.Series, horizons: tuple[int, ...]) -> TimeSeriesFit:
        clean = _clean_series(series)
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("statsmodels is required for ETS.") from exc

        indexed = pd.Series(clean.to_numpy(dtype=float), index=pd.RangeIndex(len(clean)))
        trend = "add" if len(indexed) >= 4 else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ExponentialSmoothing(
                indexed,
                trend=trend,
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)
        max_horizon = max(int(h) for h in horizons)
        forecast_values = result.forecast(max_horizon)
        fitted = pd.Series(result.fittedvalues.to_numpy(dtype=float), index=clean.index)
        residuals = _aligned_residuals(clean, fitted)
        return TimeSeriesFit(
            model="ets",
            forecasts={int(h): float(forecast_values.iloc[int(h) - 1]) for h in horizons},
            fitted=fitted,
            residuals=residuals,
            parameters={"trend": trend or "none"},
            aic=_safe_float(getattr(result, "aic", np.nan)),
            bic=_safe_float(getattr(result, "bic", np.nan)),
            parameter_count=3 if trend else 2,
        )

    def _sarima_orders(self, nobs: int) -> list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]:
        orders = []
        for p in self.policy.sarima_p_values:
            for d in self.policy.sarima_d_values:
                for q in self.policy.sarima_q_values:
                    orders.append(((int(p), int(d), int(q)), (0, 0, 0, 0)))
                    for period in self.policy.sarima_seasonal_periods:
                        period = int(period)
                        if period > 1 and nobs >= period * 2:
                            orders.append(((int(p), int(d), int(q)), (1, 0, 0, period)))
        return orders[: max(int(self.policy.sarima_max_models), 1)]

    def _fit_auto_sarima(self, series: pd.Series, horizons: tuple[int, ...]) -> TimeSeriesFit:
        clean = _clean_series(series)
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("statsmodels is required for Auto SARIMA.") from exc

        values = pd.Series(clean.to_numpy(dtype=float), index=pd.RangeIndex(len(clean)))
        best = None
        best_metadata: dict[str, object] = {}
        for order, seasonal_order in self._sarima_orders(len(values)):
            trend = "c" if order[1] == 0 and seasonal_order[1] == 0 else "n"
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = SARIMAX(
                        values,
                        order=order,
                        seasonal_order=seasonal_order,
                        trend=trend,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, maxiter=int(self.policy.sarima_maxiter))
            except Exception:
                continue
            if best is None or float(result.aic) < float(best.aic):
                best = result
                best_metadata = {
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "trend": trend,
                }

        if best is None:
            raise ValueError("Auto SARIMA could not fit any configured order.")

        max_horizon = max(int(h) for h in horizons)
        forecast_values = best.forecast(steps=max_horizon)
        fitted = pd.Series(np.asarray(best.fittedvalues, dtype=float), index=clean.index)
        residuals = _aligned_residuals(clean, fitted)
        return TimeSeriesFit(
            model="auto_sarima",
            forecasts={int(h): float(forecast_values.iloc[int(h) - 1]) for h in horizons},
            fitted=fitted,
            residuals=residuals,
            parameters=best_metadata,
            aic=_safe_float(best.aic),
            bic=_safe_float(best.bic),
            parameter_count=int(len(best.params)),
        )

    def _fit_model(self, model: str, series: pd.Series, horizons: tuple[int, ...]) -> TimeSeriesFit:
        handlers: dict[str, Callable[[pd.Series, tuple[int, ...]], TimeSeriesFit]] = {
            "mean": self._fit_mean,
            "naive": self._fit_naive,
            "drift": self._fit_drift,
            "ar1": self._fit_ar1,
            "mean_reversion": self._fit_mean_reversion,
            "bootstrap_ar1": self._fit_bootstrap_ar1,
            "ets": self._fit_ets,
            "auto_sarima": self._fit_auto_sarima,
        }
        if model not in handlers:
            raise ValueError(f"Unsupported time-series model: {model}")
        return handlers[model](series, horizons)

    def _rolling_folds(self, clean: pd.Series) -> list[int]:
        horizon = int(self.policy.rolling_forecast_horizon)
        initial_required = max(
            int(self.policy.rolling_initial_train_months),
            int(self.policy.minimum_train_months),
            3,
        )
        initial = min(initial_required, len(clean) - horizon)
        if initial < 3 or len(clean) <= initial:
            return []
        step = max(int(self.policy.rolling_step_months), 1)
        fold_ends = list(range(initial, len(clean) - horizon + 1, step))
        max_folds = max(int(self.policy.max_cv_folds), 1)
        if len(fold_ends) > max_folds:
            fold_ends = fold_ends[-max_folds:]
        return fold_ends

    def _backtest_model(
        self,
        target: str,
        model: str,
        clean: pd.Series,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        folds = self._rolling_folds(clean)
        horizon = int(self.policy.rolling_forecast_horizon)
        rows = []
        errors = []
        for fold_number, train_end in enumerate(folds, start=1):
            train = clean.iloc[:train_end]
            actual_idx = train_end + horizon - 1
            actual = float(clean.iloc[actual_idx])
            actual_date = clean.index[actual_idx]
            try:
                fit = self._fit_model(model, train, (horizon,))
                prediction = float(fit.forecasts[horizon])
                last_train = float(train.iloc[-1])
                direction_correct = np.sign(prediction - last_train) == np.sign(actual - last_train)
                row = {
                    "target": target,
                    "model": model,
                    "fold": fold_number,
                    "train_observations": int(len(train)),
                    "train_end": train.index[-1],
                    "actual_date": actual_date,
                    "horizon_months": horizon,
                    "prediction": prediction,
                    "actual": actual,
                    "error": actual - prediction,
                    "absolute_error": abs(actual - prediction),
                    "squared_error": (actual - prediction) ** 2,
                    "direction_correct": bool(direction_correct),
                    "status": "ok",
                    "error_message": "",
                }
            except Exception as exc:
                errors.append(str(exc))
                row = {
                    "target": target,
                    "model": model,
                    "fold": fold_number,
                    "train_observations": int(len(train)),
                    "train_end": train.index[-1],
                    "actual_date": actual_date,
                    "horizon_months": horizon,
                    "prediction": np.nan,
                    "actual": actual,
                    "error": np.nan,
                    "absolute_error": np.nan,
                    "squared_error": np.nan,
                    "direction_correct": False,
                    "status": "failed",
                    "error_message": str(exc),
                }
            rows.append(row)

        ok = pd.DataFrame(rows)
        ok = ok[ok["status"] == "ok"] if not ok.empty else pd.DataFrame()
        if ok.empty:
            status = "skipped" if errors else "insufficient_history"
            reason = "; ".join(errors[-3:]) if errors else "not_enough_observations_for_rolling_cv"
            return (
                {
                    "target": target,
                    "model": model,
                    "role": self._model_role(model),
                    "status": status,
                    "selected": False,
                    "accepted": False,
                    "acceptance_reason": reason,
                    "observations": int(len(clean)),
                    "folds": 0,
                    "horizon_months": horizon,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "directional_accuracy": np.nan,
                    "selection_metric": self.policy.selection_metric,
                    "error": reason,
                },
                rows,
            )

        rmse = float(np.sqrt(ok["squared_error"].mean()))
        mae = float(ok["absolute_error"].mean())
        directional_accuracy = float(ok["direction_correct"].mean())
        return (
            {
                "target": target,
                "model": model,
                "role": self._model_role(model),
                "status": "ok",
                "selected": False,
                "accepted": False,
                "acceptance_reason": "evaluated",
                "observations": int(len(clean)),
                "folds": int(len(ok)),
                "horizon_months": horizon,
                "rmse": rmse,
                "mae": mae,
                "directional_accuracy": directional_accuracy,
                "selection_metric": self.policy.selection_metric,
                "error": "",
            },
            rows,
        )

    @staticmethod
    def _metric(row: pd.Series, metric_name: str) -> float:
        value = row.get(metric_name, np.nan)
        return float(value) if pd.notna(value) else np.inf

    def _select_winner(self, reports: pd.DataFrame) -> tuple[str, bool, str]:
        metric = self.policy.selection_metric
        benchmark_model = self.policy.benchmark_model
        ok_reports = reports[reports["status"] == "ok"].copy()
        if ok_reports.empty:
            return benchmark_model, False, "no_candidate_had_valid_rolling_cv"

        benchmark_rows = ok_reports[ok_reports["model"] == benchmark_model]
        if benchmark_rows.empty:
            best = ok_reports.sort_values(metric).iloc[0]
            return str(best["model"]), True, "benchmark_unavailable_selected_best_valid_model"

        benchmark_metric = self._metric(benchmark_rows.iloc[0], metric)
        candidate_rows = ok_reports[ok_reports["role"] == "candidate"].copy()
        if candidate_rows.empty or not np.isfinite(benchmark_metric):
            return benchmark_model, False, "benchmark_selected_no_valid_candidate_comparison"

        threshold = benchmark_metric * (1.0 - float(self.policy.min_improvement_over_benchmark))
        accepted = candidate_rows[candidate_rows[metric] <= threshold].copy()
        if accepted.empty:
            return benchmark_model, False, "benchmark_selected_candidates_did_not_clear_improvement_threshold"
        best_candidate = accepted.sort_values(metric).iloc[0]
        return (
            str(best_candidate["model"]),
            True,
            "candidate_selected_after_beating_benchmark_in_rolling_cv",
        )

    def select(
        self,
        target: str,
        series: pd.Series,
        horizons: tuple[int, ...],
    ) -> TimeSeriesModelSelectionResult:
        """Run rolling validation, select a model, and refit it on full history."""

        clean = _clean_series(series)
        models = self._models_to_evaluate()
        report_rows = []
        backtest_rows = []
        final_fits: dict[str, TimeSeriesFit] = {}

        for model in models:
            report, rows = self._backtest_model(target, model, clean)
            report_rows.append(report)
            backtest_rows.extend(rows)

        reports = pd.DataFrame(report_rows)
        for model in models:
            if reports.loc[reports["model"] == model, "status"].iloc[0] != "ok":
                continue
            try:
                final_fits[model] = self._fit_model(model, clean, horizons)
            except Exception as exc:
                reports.loc[reports["model"] == model, "status"] = "skipped"
                reports.loc[reports["model"] == model, "error"] = str(exc)

        selected_model, accepted, reason = self._select_winner(reports)
        if selected_model not in final_fits:
            selected_model = self.policy.benchmark_model if self.policy.benchmark_model in final_fits else "naive"
            if selected_model not in final_fits:
                final_fits[selected_model] = self._fit_naive(clean, horizons)
            accepted = False
            reason = "fallback_model_used_after_final_fit_failure"

        final_fit = final_fits[selected_model]
        benchmark_rows = reports[reports["model"] == self.policy.benchmark_model]
        benchmark_metric = (
            self._metric(benchmark_rows.iloc[0], self.policy.selection_metric)
            if not benchmark_rows.empty
            else np.nan
        )

        for model, fit in final_fits.items():
            mask = reports["model"] == model
            reports.loc[mask, "aic"] = fit.aic
            reports.loc[mask, "bic"] = fit.bic
            reports.loc[mask, "parameter_count"] = fit.parameter_count
            reports.loc[mask, "parameters"] = str(dict(fit.parameters))

        reports["benchmark_model"] = self.policy.benchmark_model
        reports["benchmark_rmse"] = (
            _safe_float(benchmark_rows.iloc[0]["rmse"]) if not benchmark_rows.empty else np.nan
        )
        can_compare = np.isfinite(benchmark_metric) and abs(benchmark_metric) > 1e-12
        reports["improvement_vs_benchmark"] = np.where(
            can_compare & reports[self.policy.selection_metric].notna(),
            (benchmark_metric - reports[self.policy.selection_metric]) / benchmark_metric,
            np.nan,
        )
        reports.loc[reports["model"] == selected_model, "selected"] = True
        reports.loc[reports["model"] == selected_model, "accepted"] = bool(accepted)
        reports.loc[reports["model"] == selected_model, "acceptance_reason"] = reason
        reports = reports.sort_values(["selected", self.policy.selection_metric], ascending=[False, True])

        return TimeSeriesModelSelectionResult(
            target=target,
            selected_model=selected_model,
            benchmark_model=self.policy.benchmark_model,
            accepted=accepted,
            acceptance_reason=reason,
            forecasts=dict(final_fit.forecasts),
            final_fit=final_fit,
            model_reports=reports.reset_index(drop=True),
            rolling_backtests=pd.DataFrame(backtest_rows),
        )


__all__ = [
    "TimeSeriesFit",
    "TimeSeriesModelSelectionResult",
    "TimeSeriesModelSelector",
]
