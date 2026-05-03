"""Scoring rules for value and growth fundamental selection.

This module ranks companies by weighted cross-sectional percentiles over
fundamental metrics. Lower-is-better and higher-is-better metrics can be mixed
inside the same scoring configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FundamentalScoreConfig:
    """
    Weighted score configuration for a fundamental investing style.

    Parameters
    ----------
    name : str
        Strategy or style label.
    metric_weights : mapping of str to float
        Relative weight assigned to each metric in the composite score.
    higher_is_better : mapping of str to bool
        Direction used when ranking each metric.
    score_column : str, default "fundamental_score"
        Name of the composite score column added to ranked outputs.
    """

    name: str
    metric_weights: Mapping[str, float]
    higher_is_better: Mapping[str, bool]
    score_column: str = "fundamental_score"


@dataclass(frozen=True)
class ValueScoreConfig(FundamentalScoreConfig):
    """Default Graham/Buffett-inspired value score."""

    name: str = "value"
    metric_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "trailing_pe": 0.25,
            "price_to_book": 0.20,
            "debt_to_equity": 0.15,
            "roe": 0.15,
            "profit_margin": 0.10,
            "free_cash_flow_margin": 0.10,
            "current_ratio": 0.05,
        }
    )
    higher_is_better: Mapping[str, bool] = field(
        default_factory=lambda: {
            "trailing_pe": False,
            "price_to_book": False,
            "debt_to_equity": False,
            "roe": True,
            "profit_margin": True,
            "free_cash_flow_margin": True,
            "current_ratio": True,
        }
    )


@dataclass(frozen=True)
class GrowthScoreConfig(FundamentalScoreConfig):
    """Default growth score focused on expansion and quality."""

    name: str = "growth"
    metric_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "revenue_growth": 0.25,
            "eps_growth": 0.25,
            "earnings_growth": 0.15,
            "roe": 0.10,
            "profit_margin": 0.10,
            "operating_margin": 0.05,
            "peg_ratio": 0.05,
            "debt_to_equity": 0.05,
        }
    )
    higher_is_better: Mapping[str, bool] = field(
        default_factory=lambda: {
            "revenue_growth": True,
            "eps_growth": True,
            "earnings_growth": True,
            "roe": True,
            "profit_margin": True,
            "operating_margin": True,
            "peg_ratio": False,
            "debt_to_equity": False,
        }
    )


def config_for_strategy(strategy: str) -> FundamentalScoreConfig:
    """
    Return the default score config for a supported strategy name.

    Parameters
    ----------
    strategy : str
        Strategy name. Supported values are `"value"` and `"growth"`.

    Returns
    -------
    FundamentalScoreConfig
        Default scoring configuration for the requested strategy.
    """
    strategy_clean = str(strategy).strip().lower()
    if strategy_clean == "value":
        return ValueScoreConfig()
    if strategy_clean == "growth":
        return GrowthScoreConfig()
    raise ValueError("strategy must be either 'value' or 'growth'.")


def score_fundamentals(
    metrics: pd.DataFrame,
    config: FundamentalScoreConfig,
) -> pd.DataFrame:
    """
    Rank companies using weighted cross-sectional percentiles.

    Parameters
    ----------
    metrics : pandas.DataFrame
        Fundamental metrics table with one row per company.
    config : FundamentalScoreConfig
        Metric weights, directions, and score column name.

    Returns
    -------
    pandas.DataFrame
        Ranked table with composite score and component score columns.
    """
    if metrics.empty:
        return metrics.copy()

    scored = metrics.copy()
    score_parts: Dict[str, pd.Series] = {}
    weighted_score = pd.Series(0.0, index=scored.index, dtype=float)
    total_weight = 0.0

    for metric, weight in config.metric_weights.items():
        if metric not in scored.columns or np.isclose(float(weight), 0.0):
            continue

        values = pd.to_numeric(scored[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.dropna().empty:
            continue

        higher = bool(config.higher_is_better.get(metric, True))
        percentile = values.rank(pct=True, ascending=higher).fillna(0.0)
        component_name = f"{metric}_score"
        score_parts[component_name] = percentile
        weighted_score = weighted_score.add(percentile * float(weight), fill_value=0.0)
        total_weight += abs(float(weight))

    if np.isclose(total_weight, 0.0):
        scored[config.score_column] = 0.0
    else:
        scored[config.score_column] = weighted_score / total_weight * 100

    for column, values in score_parts.items():
        scored[column] = values

    sort_columns = [config.score_column]
    ascending = [False]
    if "market_cap" in scored.columns:
        sort_columns.append("market_cap")
        ascending.append(False)

    return scored.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def score_fundamentals_over_time(
    metric_history: pd.DataFrame,
    config: FundamentalScoreConfig,
) -> pd.DataFrame:
    """
    Score companies cross-sectionally inside each reporting period.

    Parameters
    ----------
    metric_history : pandas.DataFrame
        Long-format metrics table containing a `period` column.
    config : FundamentalScoreConfig
        Metric weights, directions, and score column name.

    Returns
    -------
    pandas.DataFrame
        Ranked history with one scored cross-section per reporting period.
    """
    if metric_history.empty:
        return metric_history.copy()
    if "period" not in metric_history.columns:
        raise ValueError("metric_history must contain a 'period' column.")

    scored_frames: list[pd.DataFrame] = []
    for period, period_metrics in metric_history.groupby("period", sort=True):
        scored = score_fundamentals(period_metrics, config)
        scored["period"] = period
        scored["strategy"] = config.name
        scored_frames.append(scored)

    return pd.concat(scored_frames, ignore_index=True)


__all__ = [
    "FundamentalScoreConfig",
    "GrowthScoreConfig",
    "ValueScoreConfig",
    "config_for_strategy",
    "score_fundamentals",
    "score_fundamentals_over_time",
]
