"""XGBoost research model for learned fundamental scoring.

The class in this module estimates nonlinear relationships between
fundamental signal features and forward returns. It does not replace the
operational scorer; instead it produces an impact report that can be converted
into `FundamentalScoreConfig` objects.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Optional

import numpy as np
import pandas as pd

from .fundamental_panel import candidate_feature_columns


EstimatorFactory = Callable[[], object]


def _rank_ic(y_true: pd.Series, y_pred: np.ndarray) -> float:
    truth = pd.to_numeric(y_true, errors="coerce")
    pred = pd.Series(y_pred, index=truth.index, dtype=float)
    valid = truth.notna() & pred.notna()
    if valid.sum() < 2:
        return np.nan
    return float(truth[valid].corr(pred[valid], method="spearman"))


def _normalized_importances(estimator: object, feature_columns: list[str]) -> pd.Series:
    values: Optional[np.ndarray] = None
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(getattr(estimator, "feature_importances_"), dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(getattr(estimator, "coef_"), dtype=float)).ravel()

    if values is None or values.shape[0] != len(feature_columns):
        values = np.zeros(len(feature_columns), dtype=float)

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    total = values.sum()
    if total > 0:
        values = values / total
    return pd.Series(values, index=feature_columns, dtype=float)


def _feature_direction(estimator: object, x: pd.DataFrame, feature: str) -> bool:
    if not hasattr(estimator, "predict"):
        return True
    predictions = pd.Series(estimator.predict(x), index=x.index, dtype=float)
    values = pd.to_numeric(x[feature], errors="coerce")
    valid = values.notna() & predictions.notna()
    if valid.sum() < 3:
        return True
    corr = values[valid].corr(predictions[valid], method="spearman")
    if pd.isna(corr):
        return True
    return bool(corr >= 0.0)


def _parse_signal_feature(feature: str) -> tuple[str, str]:
    if "__" not in feature:
        return feature, "level"
    metric, signal = feature.rsplit("__", 1)
    return metric, signal


class XGBoostFundamentalModel:
    """Fit XGBoost models and produce sector/industry feature-impact reports.

    Parameters
    ----------
    target : str, optional
        Target column. Defaults to `forward_total_return_12m` when omitted.
    group_by : str, default "sector"
        Column used to fit separate models. Use `None` for one global model.
    min_group_size : int, default 30
        Minimum complete target rows required to fit a separate group model.
    min_feature_coverage : float, default 0.50
        Minimum non-null coverage required for a feature inside a model group.
    n_splits : int, default 3
        Walk-forward splits for out-of-sample rank-IC diagnostics.
    estimator_factory : callable, optional
        Factory returning an estimator with `fit`, `predict`, and ideally
        `feature_importances_`. When omitted, `xgboost.XGBRegressor` is used.
    estimator_params : mapping, optional
        Parameters for the default XGBoost estimator.
    random_state : int, default 42
        Random state passed to the default XGBoost estimator.
    fallback_group : str, default "__global__"
        Group label used for global fallback models and configs.
    """

    def __init__(
        self,
        target: Optional[str] = None,
        group_by: Optional[str] = "sector",
        min_group_size: int = 30,
        min_feature_coverage: float = 0.50,
        n_splits: int = 3,
        estimator_factory: Optional[EstimatorFactory] = None,
        estimator_params: Optional[Mapping[str, object]] = None,
        random_state: int = 42,
        fallback_group: str = "__global__",
    ) -> None:
        self.target = target or "forward_total_return_12m"
        self.group_by = group_by
        self.min_group_size = int(min_group_size)
        self.min_feature_coverage = float(min_feature_coverage)
        self.n_splits = int(n_splits)
        self.estimator_factory = estimator_factory
        self.estimator_params = dict(estimator_params or {})
        self.random_state = int(random_state)
        self.fallback_group = fallback_group
        self.models_: dict[str, object] = {}
        self.feature_columns_by_group_: dict[str, list[str]] = {}
        self.feature_medians_by_group_: dict[str, pd.Series] = {}
        self.oos_scores_: dict[str, float] = {}
        self.impact_report_: pd.DataFrame = pd.DataFrame()
        self.training_summary_: pd.DataFrame = pd.DataFrame()

    def _default_estimator(self) -> object:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "xgboost is required to fit XGBoostFundamentalModel with the "
                "default estimator. Install xgboost or pass estimator_factory."
            ) from exc

        params = {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "objective": "reg:squarederror",
            "random_state": self.random_state,
            "n_jobs": 1,
        }
        params.update(self.estimator_params)
        return XGBRegressor(**params)

    def _new_estimator(self) -> object:
        if self.estimator_factory is not None:
            return self.estimator_factory()
        return self._default_estimator()

    def _group_frames(self, panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if self.group_by is None or self.group_by not in panel.columns:
            return {self.fallback_group: panel.copy()}

        groups: dict[str, pd.DataFrame] = {}
        grouped = panel.copy()
        grouped[self.group_by] = grouped[self.group_by].fillna(self.fallback_group).astype(str)
        for group_name, frame in grouped.groupby(self.group_by, sort=True):
            complete_targets = pd.to_numeric(frame[self.target], errors="coerce").notna().sum()
            if complete_targets >= self.min_group_size:
                groups[str(group_name)] = frame.copy()

        if not groups:
            groups[self.fallback_group] = panel.copy()
        elif self.fallback_group not in groups:
            groups[self.fallback_group] = panel.copy()
        return groups

    def _prepare_xy(
        self,
        frame: pd.DataFrame,
        feature_columns: Iterable[str],
    ) -> tuple[pd.DataFrame, pd.Series, list[str], pd.Series]:
        data = frame.copy()
        y = pd.to_numeric(data[self.target], errors="coerce")
        usable = y.notna()
        data = data.loc[usable].copy()
        y = y.loc[usable].astype(float)

        selected_features: list[str] = []
        for column in feature_columns:
            if column not in data.columns:
                continue
            values = pd.to_numeric(data[column], errors="coerce")
            if values.notna().mean() >= self.min_feature_coverage:
                selected_features.append(column)

        if not selected_features:
            return pd.DataFrame(index=data.index), y, [], pd.Series(dtype=float)

        x = data[selected_features].apply(pd.to_numeric, errors="coerce")
        medians = x.median(numeric_only=True).fillna(0.0)
        x = x.fillna(medians)
        return x.astype(float), y, selected_features, medians

    def _walk_forward_diagnostics(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        feature_columns: list[str],
    ) -> tuple[float, list[pd.Series]]:
        if self.n_splits < 2 or x.shape[0] <= self.n_splits:
            return np.nan, []

        try:
            from sklearn.model_selection import TimeSeriesSplit
        except ImportError:
            return np.nan, []

        n_splits = min(self.n_splits, x.shape[0] - 1)
        if n_splits < 2:
            return np.nan, []

        splitter = TimeSeriesSplit(n_splits=n_splits)
        scores: list[float] = []
        fold_importances: list[pd.Series] = []
        for train_idx, test_idx in splitter.split(x):
            estimator = self._new_estimator()
            estimator.fit(x.iloc[train_idx], y.iloc[train_idx])
            predictions = estimator.predict(x.iloc[test_idx])
            score = _rank_ic(y.iloc[test_idx], predictions)
            if pd.notna(score):
                scores.append(score)
            fold_importances.append(_normalized_importances(estimator, feature_columns))

        return (float(np.mean(scores)) if scores else np.nan), fold_importances

    def _impact_for_group(
        self,
        group: str,
        frame: pd.DataFrame,
        feature_columns: list[str],
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        sort_column = "available_at" if "available_at" in frame.columns else "period"
        working = frame.sort_values(sort_column).copy()
        x, y, selected_features, medians = self._prepare_xy(working, feature_columns)
        if y.empty or not selected_features:
            return [], {
                "group": group,
                "n_obs": int(y.shape[0]),
                "n_features": 0,
                "oos_score": np.nan,
                "status": "skipped",
            }

        oos_score, fold_importances = self._walk_forward_diagnostics(x, y, selected_features)
        estimator = self._new_estimator()
        estimator.fit(x, y)
        importances = _normalized_importances(estimator, selected_features)

        if fold_importances:
            fold_frame = pd.DataFrame(fold_importances).reindex(columns=selected_features).fillna(0.0)
            mean_importance = fold_frame.mean(axis=0).abs()
            std_importance = fold_frame.std(axis=0).fillna(0.0)
            stability = 1.0 - (std_importance / (mean_importance + std_importance + 1e-12))
            stability = stability.clip(lower=0.0, upper=1.0)
        else:
            stability = pd.Series(1.0, index=selected_features, dtype=float)

        self.models_[group] = estimator
        self.feature_columns_by_group_[group] = selected_features
        self.feature_medians_by_group_[group] = medians
        self.oos_scores_[group] = oos_score

        oos_multiplier = max(float(oos_score), 0.0) if pd.notna(oos_score) else 1.0
        rows: list[dict[str, object]] = []
        for feature in selected_features:
            metric, signal = _parse_signal_feature(feature)
            coverage = pd.to_numeric(working[feature], errors="coerce").notna().mean()
            importance = float(importances.get(feature, 0.0))
            feature_stability = float(stability.get(feature, 1.0))
            adjusted_impact = importance * feature_stability * oos_multiplier * float(coverage)
            rows.append(
                {
                    "group": group,
                    "feature": feature,
                    "metric": metric,
                    "signal": signal,
                    "importance": importance,
                    "higher_is_better": _feature_direction(estimator, x, feature),
                    "stability": feature_stability,
                    "oos_score": oos_score,
                    "coverage": float(coverage),
                    "adjusted_impact": adjusted_impact,
                    "n_obs": int(y.shape[0]),
                    "model": estimator.__class__.__name__,
                }
            )

        summary = {
            "group": group,
            "n_obs": int(y.shape[0]),
            "n_features": len(selected_features),
            "oos_score": oos_score,
            "status": "fit",
        }
        return rows, summary

    def fit(
        self,
        panel: pd.DataFrame,
        target: Optional[str] = None,
        feature_columns: Optional[Iterable[str]] = None,
    ) -> "XGBoostFundamentalModel":
        """Fit group-level models and build `impact_report_`."""
        if panel.empty:
            raise ValueError("panel must not be empty.")
        self.target = target or self.target
        if self.target not in panel.columns:
            raise ValueError(f"target column '{self.target}' not found in panel.")

        features = list(feature_columns) if feature_columns is not None else candidate_feature_columns(
            panel,
            min_feature_coverage=self.min_feature_coverage,
        )
        if not features:
            raise ValueError("No candidate feature columns were found.")

        self.models_.clear()
        self.feature_columns_by_group_.clear()
        self.feature_medians_by_group_.clear()
        self.oos_scores_.clear()

        impact_rows: list[dict[str, object]] = []
        summaries: list[dict[str, object]] = []
        for group, frame in self._group_frames(panel).items():
            rows, summary = self._impact_for_group(group, frame, features)
            impact_rows.extend(rows)
            summaries.append(summary)

        self.impact_report_ = pd.DataFrame(impact_rows)
        self.training_summary_ = pd.DataFrame(summaries)
        return self

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        """Predict forward returns with fitted group models."""
        if not self.models_:
            raise ValueError("Model is not fitted. Run fit(...) first.")
        if panel.empty:
            return pd.Series(dtype=float, name="predicted_forward_return")

        predictions = pd.Series(np.nan, index=panel.index, dtype=float, name="predicted_forward_return")
        group_values = (
            panel[self.group_by].fillna(self.fallback_group).astype(str)
            if self.group_by is not None and self.group_by in panel.columns
            else pd.Series(self.fallback_group, index=panel.index, dtype=str)
        )
        for group_name, group_index in group_values.groupby(group_values).groups.items():
            model_group = group_name if group_name in self.models_ else self.fallback_group
            if model_group not in self.models_:
                continue
            features = self.feature_columns_by_group_[model_group]
            medians = self.feature_medians_by_group_[model_group]
            x = panel.loc[group_index, features].apply(pd.to_numeric, errors="coerce")
            x = x.fillna(medians).fillna(0.0).astype(float)
            predictions.loc[group_index] = self.models_[model_group].predict(x)
        return predictions


__all__ = [
    "XGBoostFundamentalModel",
]
