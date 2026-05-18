"""Operational selector that applies learned fundamental score configs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Optional

import pandas as pd

from .fundamental_metrics import (
    StatementFrequency,
    build_metric_history_frame,
    build_metrics_frame,
)
from .fundamental_scorers import FundamentalScoreConfig, score_fundamentals
from .fundamentals import FundamentalData, YahooFundamentalsProvider


class LearnedFundamentalSelector:
    """Rank companies with group-specific learned score configurations.

    Learned configs are usually produced by `LearnedScoreConfigFactory` from an
    XGBoost impact report. The selector keeps the same high-level contract as
    `FundamentalSelector`: rank companies, select top rows, and expose
    auditable metrics, weights, and score components.
    """

    def __init__(
        self,
        score_configs_by_group: Mapping[str, FundamentalScoreConfig],
        provider: Optional[YahooFundamentalsProvider] = None,
        group_by: str = "sector",
        fallback_group: str = "__global__",
        default_score_config: Optional[FundamentalScoreConfig] = None,
    ) -> None:
        if not score_configs_by_group:
            raise ValueError("score_configs_by_group must not be empty.")
        self.score_configs_by_group = dict(score_configs_by_group)
        self.provider = provider or YahooFundamentalsProvider()
        self.group_by = group_by
        self.fallback_group = fallback_group
        self.default_score_config = (
            default_score_config
            or self.score_configs_by_group.get(fallback_group)
            or next(iter(self.score_configs_by_group.values()))
        )
        self.raw_data: dict[str, FundamentalData] = {}
        self.metrics_: pd.DataFrame = pd.DataFrame()
        self.metric_history_: pd.DataFrame = pd.DataFrame()
        self.ranking_: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def _normalize_tickers(tickers: Iterable[str]) -> list[str]:
        if isinstance(tickers, str):
            tickers = [tickers]
        normalized = [str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()]
        if not normalized:
            raise ValueError("tickers must contain at least one symbol.")
        return list(dict.fromkeys(normalized))

    def _config_for_group(self, group: object) -> FundamentalScoreConfig:
        group_key = str(group) if pd.notna(group) else self.fallback_group
        return (
            self.score_configs_by_group.get(group_key)
            or self.score_configs_by_group.get(self.fallback_group)
            or self.default_score_config
        )

    def collect_metrics(self, tickers: Iterable[str]) -> pd.DataFrame:
        """Fetch raw data and build current metrics for a ticker universe."""
        normalized = self._normalize_tickers(tickers)
        self.raw_data = self.provider.fetch_many(normalized)
        self.metrics_ = build_metrics_frame(self.raw_data.values())
        return self.metrics_.copy()

    def collect_metric_history(
        self,
        tickers: Iterable[str],
        frequency: StatementFrequency = "quarterly",
        trailing_periods: int = 8,
    ) -> pd.DataFrame:
        """Fetch raw data and build historical metrics for score signals."""
        normalized = self._normalize_tickers(tickers)
        self.raw_data = self.provider.fetch_many(normalized)
        self.metric_history_ = build_metric_history_frame(
            self.raw_data.values(),
            frequency=frequency,
            trailing_periods=trailing_periods,
        )
        return self.metric_history_.copy()

    def rank(
        self,
        tickers: Iterable[str],
        frequency: StatementFrequency = "quarterly",
        trailing_periods: int = 8,
    ) -> pd.DataFrame:
        """Rank tickers by the learned config assigned to each group."""
        metrics = self.collect_metrics(tickers)
        if metrics.empty:
            raise ValueError("No fundamental metrics could be built for the ticker universe.")

        self.metric_history_ = build_metric_history_frame(
            self.raw_data.values(),
            frequency=frequency,
            trailing_periods=trailing_periods,
        )

        group_values = (
            metrics[self.group_by].fillna(self.fallback_group).astype(str)
            if self.group_by in metrics.columns
            else pd.Series(self.fallback_group, index=metrics.index, dtype=str)
        )

        scored_frames: list[pd.DataFrame] = []
        for group_name, group_index in group_values.groupby(group_values).groups.items():
            group_metrics = metrics.loc[group_index].copy()
            config = self._config_for_group(group_name)
            tickers_in_group = set(group_metrics["ticker"].astype(str).str.upper())
            history = self.metric_history_
            if not history.empty and "ticker" in history.columns:
                history = history[history["ticker"].astype(str).str.upper().isin(tickers_in_group)].copy()

            scored = score_fundamentals(
                group_metrics,
                config,
                metric_history=history,
                frequency=frequency,
            )
            scored["score_group"] = str(group_name)
            scored["strategy"] = config.name
            scored["rank_in_group"] = range(1, scored.shape[0] + 1)
            scored_frames.append(scored)

        if not scored_frames:
            self.ranking_ = pd.DataFrame()
        else:
            ranking = pd.concat(scored_frames, ignore_index=True)
            sort_columns = ["fundamental_score"]
            ascending = [False]
            if "score_coverage" in ranking.columns:
                sort_columns.append("score_coverage")
                ascending.append(False)
            if "market_cap" in ranking.columns:
                sort_columns.append("market_cap")
                ascending.append(False)
            self.ranking_ = ranking.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
        return self.ranking_.copy()

    def select_top(
        self,
        ranking: Optional[pd.DataFrame] = None,
        top_k: int = 5,
        per_group: bool = False,
    ) -> pd.DataFrame:
        """Return top-ranked rows globally or inside each learned group."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")

        source = self.ranking_ if ranking is None else ranking
        if source.empty:
            raise ValueError("ranking is empty. Run rank(...) first or pass a ranking DataFrame.")
        if per_group:
            return (
                source.sort_values(["score_group", "rank_in_group"])
                .groupby("score_group", sort=True)
                .head(top_k)
                .reset_index(drop=True)
            )
        return source.head(top_k).copy().reset_index(drop=True)

    def selection_report(
        self,
        ranking: Optional[pd.DataFrame] = None,
        top_k: int = 5,
        per_group: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Build an interpretability report for a learned selection."""
        selected = self.select_top(ranking=ranking, top_k=top_k, per_group=per_group)
        summary_columns = [
            column
            for column in [
                "ticker",
                "fundamental_score",
                "score_coverage",
                "score_group",
                "rank_in_group",
                "strategy",
                "sector",
                "industry",
            ]
            if column in selected.columns
        ]
        component_columns = [
            column for column in selected.columns if column.endswith("_score") and column != "fundamental_score"
        ]
        signal_columns = [
            column
            for column in selected.columns
            if "__" in column and not column.endswith("_score")
        ]

        weight_rows: list[dict[str, object]] = []
        for group, config in self.score_configs_by_group.items():
            for spec in config.resolved_signal_specs():
                weight_rows.append(
                    {
                        "group": group,
                        "strategy": config.name,
                        "metric": spec.metric,
                        "signal": spec.signal,
                        "signal_column": spec.signal_name,
                        "weight": float(spec.weight),
                        "higher_is_better": bool(spec.higher_is_better),
                        "normalizer": spec.normalizer,
                        "category": spec.category,
                        "component_column": spec.component_name,
                    }
                )

        return {
            "selected": selected[summary_columns].copy(),
            "metric_snapshot": selected[["ticker", *signal_columns]].copy(),
            "score_weights": pd.DataFrame(weight_rows),
            "score_components": selected[["ticker", *component_columns]].copy(),
        }

    def run_pipeline(
        self,
        tickers: Iterable[str],
        top_k: int = 5,
        per_group: bool = False,
    ) -> dict[str, Any]:
        """Execute ranking, selection, and reporting end to end."""
        ranking = self.rank(tickers)
        selected = self.select_top(ranking=ranking, top_k=top_k, per_group=per_group)
        report = self.selection_report(ranking=ranking, top_k=top_k, per_group=per_group)
        return {
            "metrics": self.metrics_.copy(),
            "metric_history": self.metric_history_.copy(),
            "ranking": ranking,
            "selected": selected,
            "selected_tickers": selected["ticker"].tolist(),
            "report": report,
        }


__all__ = [
    "LearnedFundamentalSelector",
]
