"""Feature engineering for macro/liquidity research panels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from .catalog import EconomicSeriesSpec, specs_by_name
from .policy import MacroLiquidityPolicy
from .providers import normalize_series_frame
from .transforms import build_indicator_panel


@dataclass(frozen=True)
class FeatureBuildResult:
    """Output of the feature-building stage."""

    aligned_series: pd.DataFrame
    features: pd.DataFrame
    validation: pd.DataFrame
    methodology: dict[str, object]


class FeatureBuilder:
    """Build a monthly macro/liquidity feature panel from raw API data."""

    def __init__(
        self,
        specs: Iterable[EconomicSeriesSpec],
        policy: Optional[MacroLiquidityPolicy] = None,
        min_periods: int = 6,
    ) -> None:
        self.specs = tuple(specs)
        self.spec_by_name = specs_by_name(self.specs)
        self.policy = policy or MacroLiquidityPolicy()
        self.min_periods = max(int(min_periods), 2)

    def _lagged_frame(
        self,
        frame: pd.DataFrame,
        spec: EconomicSeriesSpec,
        as_of: Optional[pd.Timestamp],
    ) -> pd.DataFrame:
        lag_days = int(
            spec.provider_params.get(
                "_release_lag_days",
                self.policy.data.lag_days_for_frequency(spec.frequency or "M"),
            )
        )
        lagged = frame.copy()
        lagged["publication_lag_days"] = lag_days
        lagged["available_date"] = lagged["date"] + pd.to_timedelta(lag_days, unit="D")
        if as_of is not None:
            lagged = lagged[lagged["available_date"] <= as_of]
        return lagged

    def _align_one(self, frame: pd.DataFrame, spec: EconomicSeriesSpec) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["date", "series", "value"])

        frequency = (spec.frequency or "M").upper()
        series = (
            frame.sort_values("date")
            .drop_duplicates("date", keep="last")
            .set_index("date")["value"]
            .astype(float)
        )

        monthly_index = pd.date_range(
            series.index.min().to_period("M").to_timestamp(how="end").normalize(),
            series.index.max().to_period("M").to_timestamp(how="end").normalize(),
            freq=pd.offsets.MonthEnd(),
        )
        monthly = series.reindex(monthly_index)
        interpolated = pd.Series(False, index=monthly.index)

        if frequency == "Q":
            before = monthly.notna()
            if self.policy.data.quarterly_interpolation == "linear":
                monthly = monthly.interpolate(method="linear").ffill()
                interpolated = monthly.notna() & ~before
            else:
                monthly = monthly.ffill()
                interpolated = monthly.notna() & ~before
        elif frequency in {"D", "W"}:
            monthly = series.resample(pd.offsets.MonthEnd()).last().reindex(monthly_index).ffill()
        else:
            monthly = series.resample(pd.offsets.MonthEnd()).last().reindex(monthly_index).ffill()

        return pd.DataFrame(
            {
                "date": monthly.index,
                "series": spec.name,
                "value": monthly.to_numpy(),
                "source_frequency": frequency,
                "target_frequency": self.policy.data.canonical_frequency,
                "interpolated": interpolated.reindex(monthly.index).fillna(False).to_numpy(),
                "data_mode": self.policy.data.data_mode,
                "real_time_safe": self.policy.data.real_time_safe,
            }
        )

    def _validate_aligned(self, aligned: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for spec in self.specs:
            group = aligned[aligned["series"] == spec.name]
            total = len(group)
            available = int(group["value"].notna().sum()) if total else 0
            missing_ratio = 1.0 if total == 0 else 1.0 - available / total
            history_months = available
            active = (
                history_months >= self.policy.validation.min_history_months
                and missing_ratio <= self.policy.validation.max_missing_ratio
            )
            rows.append(
                {
                    "series": spec.name,
                    "engine": spec.engine,
                    "block": spec.block,
                    "role": spec.provider_params.get("_role", "coincident"),
                    "history_months": history_months,
                    "missing_ratio": missing_ratio,
                    "methodology_weight": spec.weight,
                    "status": "active" if active else "candidate",
                    "effective_weight": (
                        spec.weight
                        if active
                        else spec.weight * self.policy.validation.candidate_weight_multiplier
                    ),
                    "reason": "passes_validation" if active else "insufficient_history_or_missing_data",
                }
            )
        return pd.DataFrame(rows)

    def _periods_for_target_frequency(self, spec: EconomicSeriesSpec) -> int:
        if (
            self.policy.data.canonical_frequency.upper() == "M"
            and spec.transform == "yoy_change"
            and (spec.frequency or "").upper() == "Q"
            and int(spec.periods) == 4
        ):
            return 12
        return spec.periods

    def build(
        self,
        series_frame: pd.DataFrame,
        as_of: Optional[str | pd.Timestamp] = None,
    ) -> FeatureBuildResult:
        """Return aligned monthly series, feature panel, and validation rows."""

        normalized = normalize_series_frame(series_frame)
        normalized = normalized[normalized["series"].isin(self.spec_by_name)].copy()
        as_of_ts = pd.Timestamp(as_of) if as_of is not None else None

        aligned_frames = []
        for spec in self.specs:
            raw = normalized[normalized["series"] == spec.name]
            lagged = self._lagged_frame(raw, spec, as_of_ts)
            aligned = self._align_one(lagged[["date", "series", "value"]], spec)
            if not aligned.empty:
                lag_days = int(
                    spec.provider_params.get(
                        "_release_lag_days",
                        self.policy.data.lag_days_for_frequency(spec.frequency or "M"),
                    )
                )
                aligned["publication_lag_days"] = lag_days
                aligned_frames.append(aligned)

        if aligned_frames:
            aligned_series = pd.concat(aligned_frames, ignore_index=True)
            aligned_series = normalize_series_frame(aligned_series).merge(
                pd.concat(aligned_frames, ignore_index=True).drop(columns=["value"]),
                on=["date", "series"],
                how="left",
            )
        else:
            aligned_series = pd.DataFrame(columns=["date", "series", "value"])

        validation = self._validate_aligned(aligned_series)
        active_specs = []
        validation_by_series = validation.set_index("series")
        for spec in self.specs:
            if spec.name not in validation_by_series.index:
                continue
            effective_weight = float(validation_by_series.loc[spec.name, "effective_weight"])
            active_specs.append(
                EconomicSeriesSpec(
                    name=spec.name,
                    engine=spec.engine,
                    block=spec.block,
                    higher_is_better=spec.higher_is_better,
                    provider=spec.provider,
                    provider_code=spec.provider_code,
                    frequency=self.policy.data.canonical_frequency,
                    transform=spec.transform,
                    periods=self._periods_for_target_frequency(spec),
                    weight=effective_weight,
                    required=spec.required,
                    description=spec.description,
                    provider_params=spec.provider_params,
                )
            )

        features = build_indicator_panel(aligned_series, active_specs, self.min_periods)
        if not features.empty:
            metadata_cols = [
                "date",
                "series",
                "source_frequency",
                "target_frequency",
                "interpolated",
                "publication_lag_days",
                "data_mode",
                "real_time_safe",
            ]
            metadata = aligned_series[[col for col in metadata_cols if col in aligned_series.columns]]
            features = features.merge(metadata, on=["date", "series"], how="left")

        methodology = self.policy.methodology_summary()
        methodology["as_of"] = None if as_of_ts is None else str(as_of_ts.date())
        return FeatureBuildResult(
            aligned_series=aligned_series,
            features=features,
            validation=validation,
            methodology=methodology,
        )
