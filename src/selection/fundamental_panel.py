"""Learning-panel builders for statistical fundamental selection.

This module prepares the research dataset used by learned scoring models. It
keeps the operational scorer unchanged by producing feature columns with the
same `metric__signal` naming convention used by `MetricSignalSpec`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Optional

import numpy as np
import pandas as pd

from .fundamental_metrics import StatementFrequency, build_metric_history_frame
from .fundamental_scorers import (
    MetricSignalSpec,
    _change_value,
    _lag_for_frequency,
    fundamental_metric_signal_specs,
)
from .fundamental_targets import build_forward_return_targets
from .fundamentals import FundamentalData, YahooFundamentalsProvider


LearningSignal = tuple[MetricSignalSpec, ...]


def _records_by_ticker(
    records: Mapping[str, FundamentalData] | Iterable[FundamentalData],
) -> dict[str, FundamentalData]:
    if isinstance(records, Mapping):
        values = list(records.values())
    else:
        values = list(records)
    return {record.ticker.upper(): record for record in values}


def _default_learning_signal_specs() -> LearningSignal:
    signals = ("level", "period_change", "yoy_change", "historical_avg_change")
    specs = list(fundamental_metric_signal_specs(signals=signals))
    for metric in ("revenue", "eps", "net_income", "free_cash_flow"):
        for signal in signals:
            specs.append(
                MetricSignalSpec(
                    metric=metric,
                    signal=signal,
                    higher_is_better=True,
                    change_method="relative",
                    category="growth",
                )
            )

    by_name: dict[str, MetricSignalSpec] = {}
    for spec in specs:
        by_name.setdefault(spec.signal_name, spec)
    return tuple(by_name.values())


def _signal_series_for_group(
    values: pd.Series,
    spec: MetricSignalSpec,
    frequency: Optional[str],
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if spec.signal == "level":
        return numeric

    lag = 1 if spec.signal in {"period_change", "historical_avg_change"} else _lag_for_frequency(frequency)
    changes = pd.Series(np.nan, index=values.index, dtype=float)
    for position in range(lag, numeric.shape[0]):
        changes.iloc[position] = _change_value(
            numeric.iloc[position],
            numeric.iloc[position - lag],
            spec.change_method,
            spec.metric,
        )

    if spec.signal == "historical_avg_change":
        result = changes.expanding(min_periods=1).mean()
    else:
        result = changes
    return result.replace([np.inf, -np.inf], np.nan)


def add_signal_features(
    metric_history: pd.DataFrame,
    signal_specs: Optional[Iterable[MetricSignalSpec]] = None,
    frequency: StatementFrequency = "quarterly",
) -> pd.DataFrame:
    """Append `metric__signal` feature columns to a metric-history table."""
    if metric_history.empty:
        return metric_history.copy()
    required = {"ticker", "period"}
    missing = required.difference(metric_history.columns)
    if missing:
        raise ValueError(f"metric_history must contain columns: {sorted(missing)}")

    specs = tuple(signal_specs) if signal_specs is not None else _default_learning_signal_specs()
    by_name: dict[str, MetricSignalSpec] = {}
    for spec in specs:
        by_name.setdefault(spec.signal_name, spec)

    panel = metric_history.copy()
    panel["ticker"] = panel["ticker"].astype(str).str.upper()
    panel["period"] = pd.to_datetime(panel["period"])
    panel = panel.sort_values(["ticker", "period"]).reset_index(drop=True)

    for spec in by_name.values():
        if spec.metric not in panel.columns:
            continue
        feature = pd.Series(np.nan, index=panel.index, dtype=float)
        for _, group in panel.groupby("ticker", sort=False):
            feature.loc[group.index] = _signal_series_for_group(
                group[spec.metric],
                spec,
                frequency,
            )
        panel[spec.signal_name] = feature

    return panel


def add_group_metadata(
    metric_history: pd.DataFrame,
    records: Mapping[str, FundamentalData] | Iterable[FundamentalData],
) -> pd.DataFrame:
    """Attach sector and industry metadata to a metric-history table."""
    if metric_history.empty:
        return metric_history.copy()

    by_ticker = _records_by_ticker(records)
    panel = metric_history.copy()
    panel["ticker"] = panel["ticker"].astype(str).str.upper()
    sector_map = {
        ticker: record.info.get("sector")
        for ticker, record in by_ticker.items()
        if isinstance(record.info, dict)
    }
    industry_map = {
        ticker: record.info.get("industry")
        for ticker, record in by_ticker.items()
        if isinstance(record.info, dict)
    }
    if "sector" not in panel.columns:
        panel["sector"] = panel["ticker"].map(sector_map)
    else:
        panel["sector"] = panel["sector"].fillna(panel["ticker"].map(sector_map))
    if "industry" not in panel.columns:
        panel["industry"] = panel["ticker"].map(industry_map)
    else:
        panel["industry"] = panel["industry"].fillna(panel["ticker"].map(industry_map))
    return panel


def candidate_feature_columns(
    panel: pd.DataFrame,
    min_feature_coverage: float = 0.50,
) -> list[str]:
    """Return numeric, scoring-compatible feature columns from a panel."""
    if panel.empty:
        return []
    threshold = min(max(float(min_feature_coverage), 0.0), 1.0)
    columns: list[str] = []
    for column in panel.columns:
        if "__" not in column or column.startswith("forward_"):
            continue
        numeric = pd.to_numeric(panel[column], errors="coerce")
        if numeric.notna().mean() >= threshold:
            columns.append(column)
    return columns


def build_fundamental_learning_panel(
    records: Mapping[str, FundamentalData] | Iterable[FundamentalData],
    frequency: StatementFrequency = "quarterly",
    trailing_periods: int = 20,
    horizon_months: int = 12,
    reporting_lag_days: int = 60,
    signal_specs: Optional[Iterable[MetricSignalSpec]] = None,
) -> pd.DataFrame:
    """Build a point-in-time panel with features and forward-return targets."""
    by_ticker = _records_by_ticker(records)
    metric_history = build_metric_history_frame(
        by_ticker.values(),
        frequency=frequency,
        trailing_periods=trailing_periods,
    )
    if metric_history.empty:
        return metric_history

    panel = add_group_metadata(metric_history, by_ticker)
    panel = add_signal_features(panel, signal_specs=signal_specs, frequency=frequency)
    return build_forward_return_targets(
        panel,
        by_ticker,
        horizon_months=horizon_months,
        reporting_lag_days=reporting_lag_days,
    )


class FundamentalLearningPanelBuilder:
    """Fetch fundamentals and build a reusable learning panel."""

    def __init__(
        self,
        provider: Optional[YahooFundamentalsProvider] = None,
        frequency: StatementFrequency = "quarterly",
        trailing_periods: int = 20,
        horizon_months: int = 12,
        reporting_lag_days: int = 60,
        signal_specs: Optional[Iterable[MetricSignalSpec]] = None,
    ) -> None:
        self.provider = provider or YahooFundamentalsProvider()
        self.frequency = frequency
        self.trailing_periods = trailing_periods
        self.horizon_months = horizon_months
        self.reporting_lag_days = reporting_lag_days
        self.signal_specs = tuple(signal_specs) if signal_specs is not None else None
        self.raw_data: dict[str, FundamentalData] = {}
        self.panel_: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def _normalize_tickers(tickers: Iterable[str]) -> list[str]:
        if isinstance(tickers, str):
            tickers = [tickers]
        normalized = [str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()]
        if not normalized:
            raise ValueError("tickers must contain at least one symbol.")
        return list(dict.fromkeys(normalized))

    def build(self, tickers: Iterable[str]) -> pd.DataFrame:
        """Fetch data for `tickers` and return a learning panel."""
        normalized = self._normalize_tickers(tickers)
        self.raw_data = self.provider.fetch_many(normalized)
        self.panel_ = build_fundamental_learning_panel(
            self.raw_data,
            frequency=self.frequency,
            trailing_periods=self.trailing_periods,
            horizon_months=self.horizon_months,
            reporting_lag_days=self.reporting_lag_days,
            signal_specs=self.signal_specs,
        )
        return self.panel_.copy()


__all__ = [
    "FundamentalLearningPanelBuilder",
    "add_group_metadata",
    "add_signal_features",
    "build_fundamental_learning_panel",
    "candidate_feature_columns",
]
