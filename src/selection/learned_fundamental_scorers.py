"""Convert learned feature impacts into fundamental score configs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .fundamental_scorers import FundamentalScoreConfig, MetricSignalSpec, Normalizer


SignalName = Literal["level", "period_change", "yoy_change", "historical_avg_change"]
_VALID_SIGNALS = {"level", "period_change", "yoy_change", "historical_avg_change"}


def _config_name(name_prefix: str, group: str) -> str:
    clean = "".join(char.lower() if char.isalnum() else "_" for char in str(group)).strip("_")
    clean = "_".join(part for part in clean.split("_") if part)
    return f"{name_prefix}_{clean or 'global'}"


@dataclass(frozen=True)
class LearnedScoreConfigFactory:
    """Build `FundamentalScoreConfig` objects from an impact report.

    Parameters
    ----------
    max_features : int, default 20
        Maximum number of signal features kept per group.
    min_impact : float, default 0.0
        Minimum impact required before a feature can enter a config.
    impact_column : str, default "adjusted_impact"
        Column used as the primary weight source.
    fallback_to_importance : bool, default True
        If all adjusted impacts are zero, fall back to raw model importance.
    normalizer : {"percentile_rank", "robust_zscore", "target"}, default "percentile_rank"
        Normalizer assigned to learned signal specs.
    missing_component_score : float, default 0.0
        Score assigned when a selected component is missing for a company.
    name_prefix : str, default "learned"
        Prefix used for generated config names.
    """

    max_features: int = 20
    min_impact: float = 0.0
    impact_column: str = "adjusted_impact"
    fallback_to_importance: bool = True
    normalizer: Normalizer = "percentile_rank"
    missing_component_score: float = 0.0
    name_prefix: str = "learned"

    def _prepared_group(self, group_report: pd.DataFrame) -> pd.DataFrame:
        working = group_report.copy()
        if self.impact_column not in working.columns:
            raise ValueError(f"impact report must contain '{self.impact_column}'.")

        working["_impact_for_weight"] = pd.to_numeric(
            working[self.impact_column],
            errors="coerce",
        ).fillna(0.0)
        if (
            self.fallback_to_importance
            and np.isclose(float(working["_impact_for_weight"].sum()), 0.0)
            and "importance" in working.columns
        ):
            working["_impact_for_weight"] = pd.to_numeric(
                working["importance"],
                errors="coerce",
            ).fillna(0.0)

        working = working[working["_impact_for_weight"] > float(self.min_impact)]
        if "signal" in working.columns:
            working = working[working["signal"].isin(_VALID_SIGNALS)]
        else:
            working["signal"] = "level"

        if "metric" not in working.columns:
            if "feature" not in working.columns:
                raise ValueError("impact report must contain either 'metric' or 'feature'.")
            parsed = working["feature"].astype(str).str.rsplit("__", n=1, expand=True)
            working["metric"] = parsed[0]
            if parsed.shape[1] > 1:
                working["signal"] = parsed[1]

        return working.sort_values("_impact_for_weight", ascending=False).head(
            max(int(self.max_features), 1)
        )

    def from_group_report(
        self,
        group_report: pd.DataFrame,
        group: str,
    ) -> FundamentalScoreConfig:
        """Build one score config from a single-group impact report."""
        prepared = self._prepared_group(group_report)
        if prepared.empty:
            raise ValueError(f"No usable learned impacts for group '{group}'.")

        total_impact = float(prepared["_impact_for_weight"].sum())
        if np.isclose(total_impact, 0.0):
            raise ValueError(f"Total learned impact is zero for group '{group}'.")

        specs: list[MetricSignalSpec] = []
        metric_weights: dict[str, float] = {}
        higher_is_better: dict[str, bool] = {}
        for _, row in prepared.iterrows():
            weight = float(row["_impact_for_weight"]) / total_impact
            metric = str(row["metric"])
            signal = str(row["signal"])
            higher = bool(row.get("higher_is_better", True))
            specs.append(
                MetricSignalSpec(
                    metric=metric,
                    signal=signal,  # type: ignore[arg-type]
                    weight=weight,
                    higher_is_better=higher,
                    normalizer=self.normalizer,
                    category="learned",
                )
            )
            metric_weights[metric] = metric_weights.get(metric, 0.0) + weight
            higher_is_better.setdefault(metric, higher)

        return FundamentalScoreConfig(
            name=_config_name(self.name_prefix, group),
            metric_weights=metric_weights,
            higher_is_better=higher_is_better,
            signal_specs=tuple(specs),
            missing_component_score=self.missing_component_score,
        )

    def from_impact_report(
        self,
        impact_report: pd.DataFrame,
        group_column: str = "group",
    ) -> dict[str, FundamentalScoreConfig]:
        """Build score configs keyed by group name."""
        if impact_report.empty:
            raise ValueError("impact_report must not be empty.")
        if group_column not in impact_report.columns:
            return {
                "__global__": self.from_group_report(
                    impact_report,
                    group="__global__",
                )
            }

        configs: dict[str, FundamentalScoreConfig] = {}
        for group, group_report in impact_report.groupby(group_column, sort=True):
            try:
                configs[str(group)] = self.from_group_report(group_report, group=str(group))
            except ValueError:
                continue
        if not configs:
            raise ValueError("No learned score configs could be built.")
        return configs


__all__ = [
    "LearnedScoreConfigFactory",
]
