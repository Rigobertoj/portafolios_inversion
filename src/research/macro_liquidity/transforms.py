"""Transform and score macro/liquidity indicator panels."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .catalog import EconomicSeriesSpec, specs_by_name
from .providers import normalize_series_frame


def _transform_values(values: pd.Series, spec: EconomicSeriesSpec) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    periods = max(int(spec.periods), 1)
    if spec.transform == "diff":
        return numeric.diff(periods=periods)
    if spec.transform in {"pct_change", "yoy_change"}:
        return numeric.pct_change(periods=periods, fill_method=None)
    return numeric


def _expanding_zscore(values: pd.Series, min_periods: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mean = numeric.expanding(min_periods=min_periods).mean()
    std = numeric.expanding(min_periods=min_periods).std(ddof=0).replace(0.0, np.nan)
    return (numeric - mean) / std


def build_indicator_panel(
    series_frame: pd.DataFrame,
    specs: Iterable[EconomicSeriesSpec],
    min_periods: int = 6,
) -> pd.DataFrame:
    """Return a long indicator panel with transformed and signed z-score values."""

    spec_by_name = specs_by_name(specs)
    panel = normalize_series_frame(series_frame)
    panel = panel[panel["series"].isin(spec_by_name)].copy()
    if panel.empty:
        return panel

    panel["engine"] = panel["series"].map(lambda name: spec_by_name[name].engine)
    panel["block"] = panel["series"].map(lambda name: spec_by_name[name].block)
    panel["higher_is_better"] = panel["series"].map(lambda name: spec_by_name[name].higher_is_better)
    panel["weight"] = panel["series"].map(lambda name: spec_by_name[name].weight)
    panel["required"] = panel["series"].map(lambda name: spec_by_name[name].required)
    panel["provider"] = panel["series"].map(lambda name: spec_by_name[name].provider)
    panel["provider_code"] = panel["series"].map(lambda name: spec_by_name[name].provider_code)

    pieces = []
    for _, group in panel.groupby("series", sort=False):
        spec = spec_by_name[str(group["series"].iloc[0])]
        enriched = group.copy()
        enriched["signal"] = _transform_values(group["value"], spec)
        enriched["zscore"] = _expanding_zscore(enriched["signal"], min_periods=min_periods)
        direction = 1.0 if spec.higher_is_better else -1.0
        enriched["signed_zscore"] = enriched["zscore"] * direction
        pieces.append(enriched)
    return pd.concat(pieces, ignore_index=True).sort_values(["date", "engine", "block", "series"])


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    """Return a weighted average ignoring missing values."""

    values_numeric = pd.to_numeric(values, errors="coerce")
    weights_numeric = pd.to_numeric(weights, errors="coerce").abs()
    valid = values_numeric.notna() & weights_numeric.notna() & ~np.isclose(weights_numeric, 0.0)
    if not valid.any():
        return np.nan
    return float((values_numeric[valid] * weights_numeric[valid]).sum() / weights_numeric[valid].sum())


def build_score_frame(indicators: pd.DataFrame) -> pd.DataFrame:
    """Aggregate indicator z-scores into block and engine scores."""

    if indicators.empty:
        return pd.DataFrame()

    block_rows = []
    for keys, group in indicators.groupby(["date", "engine", "block"], dropna=False):
        date, engine, block = keys
        block_rows.append(
            {
                "date": date,
                "engine": engine,
                "block": block,
                "score": weighted_average(group["signed_zscore"], group["weight"]),
                "available_indicators": int(group["signed_zscore"].notna().sum()),
                "total_indicators": int(group["series"].nunique()),
            }
        )
    block_scores = pd.DataFrame(block_rows)
    block_wide = (
        block_scores.pivot_table(index="date", columns="block", values="score", aggfunc="first")
        .add_suffix("_score")
    )

    engine_rows = []
    for keys, group in indicators.groupby(["date", "engine"], dropna=False):
        date, engine = keys
        engine_rows.append(
            {
                "date": date,
                "engine": engine,
                "score": weighted_average(group["signed_zscore"], group["weight"]),
                "coverage": float(group["signed_zscore"].notna().mean()),
            }
        )
    engine_scores = pd.DataFrame(engine_rows)
    engine_wide = engine_scores.pivot_table(
        index="date", columns="engine", values="score", aggfunc="first"
    ).rename(columns={"macro": "macro_score", "liquidity": "liquidity_score"})
    coverage_wide = engine_scores.pivot_table(
        index="date", columns="engine", values="coverage", aggfunc="first"
    ).rename(columns={"macro": "macro_coverage", "liquidity": "liquidity_coverage"})

    return block_wide.join(engine_wide, how="outer").join(coverage_wide, how="outer").sort_index().reset_index()
