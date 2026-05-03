"""High-level fundamental selector for value and growth workflows.

`FundamentalSelector` orchestrates data collection, metric construction,
scoring, top-k selection, interpretability reports, and historical metric
evolution for fundamental investing strategies.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import pandas as pd

from .fundamental_metrics import (
    StatementFrequency,
    build_metric_history_frame,
    build_metrics_frame,
)
from .fundamental_scorers import (
    FundamentalScoreConfig,
    config_for_strategy,
    score_fundamentals,
    score_fundamentals_over_time,
)
from .fundamentals import FundamentalData, YahooFundamentalsProvider


class FundamentalSelector:
    """
    Rank and select companies using Yahoo Finance fundamental data.

    Parameters
    ----------
    strategy : str, default "value"
        Strategy label used to choose the default scoring configuration.
    provider : YahooFundamentalsProvider, optional
        Data provider used to fetch raw fundamental records.
    score_config : FundamentalScoreConfig, optional
        Custom scoring configuration. If omitted, a default config is created
        from `strategy`.
    """

    def __init__(
        self,
        strategy: str = "value",
        provider: Optional[YahooFundamentalsProvider] = None,
        score_config: Optional[FundamentalScoreConfig] = None,
    ) -> None:
        self.strategy = str(strategy).strip().lower()
        self.provider = provider or YahooFundamentalsProvider()
        self.score_config = score_config or config_for_strategy(self.strategy)
        self.raw_data: Dict[str, FundamentalData] = {}
        self.metrics_: pd.DataFrame = pd.DataFrame()
        self.metric_history_: pd.DataFrame = pd.DataFrame()
        self.ranking_: pd.DataFrame = pd.DataFrame()
        self.ranking_history_: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def _normalize_tickers(tickers: Iterable[str]) -> list[str]:
        if isinstance(tickers, str):
            tickers = [tickers]
        normalized = [str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()]
        if not normalized:
            raise ValueError("tickers must contain at least one symbol.")
        return list(dict.fromkeys(normalized))

    def set_strategy(
        self,
        strategy: str,
        score_config: Optional[FundamentalScoreConfig] = None,
        clear_rankings: bool = True,
    ) -> None:
        """Update the selector strategy and its scoring configuration.

        Parameters
        ----------
        strategy : str
            Strategy name to attach to future rankings.
        score_config : FundamentalScoreConfig, optional
            Custom scoring configuration. When omitted, the default config for
            `strategy` is used.
        clear_rankings : bool, default True
            If True, discard rankings computed with the previous strategy while
            keeping downloaded raw data and calculated base metrics.
        """
        strategy_clean = str(strategy).strip().lower()
        self.score_config = score_config or config_for_strategy(strategy_clean)
        self.strategy = strategy_clean

        if clear_rankings:
            self.ranking_ = pd.DataFrame()
            self.ranking_history_ = pd.DataFrame()

    def collect_metrics(self, tickers: Iterable[str]) -> pd.DataFrame:
        """
        Download raw data and compute fundamental metrics for a ticker universe.

        Parameters
        ----------
        tickers : iterable of str
            Ticker symbols to analyze.

        Returns
        -------
        pandas.DataFrame
            Metrics table indexed by ticker.
        """
        normalized = self._normalize_tickers(tickers)
        self.raw_data = self.provider.fetch_many(normalized)
        self.metrics_ = build_metrics_frame(self.raw_data.values())
        return self.metrics_.copy()

    def collect_metric_history(
        self,
        tickers: Iterable[str],
        frequency: StatementFrequency = "quarterly",
        trailing_periods: int = 4,
    ) -> pd.DataFrame:
        """
        Collect period-by-period metrics for a ticker universe.

        Parameters
        ----------
        tickers : iterable of str
            Ticker symbols to analyze.
        frequency : {"annual", "quarterly"}, default "quarterly"
            Statement frequency used to build histories.
        trailing_periods : int, default 4
            Maximum number of recent periods returned per company.

        Returns
        -------
        pandas.DataFrame
            Long-format historical metrics table.
        """
        normalized = self._normalize_tickers(tickers)
        self.raw_data = self.provider.fetch_many(normalized)
        self.metric_history_ = build_metric_history_frame(
            self.raw_data.values(),
            frequency=frequency,
            trailing_periods=trailing_periods,
        )
        return self.metric_history_.copy()

    def rank(self, tickers: Iterable[str]) -> pd.DataFrame:
        """
        Return a ranked DataFrame for the configured fundamental strategy.

        Parameters
        ----------
        tickers : iterable of str
            Ticker symbols to rank.

        Returns
        -------
        pandas.DataFrame
            Ranking table sorted by composite fundamental score.
        """
        metrics = self.collect_metrics(tickers)
        if metrics.empty:
            raise ValueError("No fundamental metrics could be built for the ticker universe.")
        self.ranking_ = score_fundamentals(metrics, self.score_config)
        self.ranking_["strategy"] = self.score_config.name
        return self.ranking_.copy()

    def rank_over_time(
        self,
        tickers: Iterable[str],
        frequency: StatementFrequency = "quarterly",
        trailing_periods: int = 4,
    ) -> pd.DataFrame:
        """
        Return fundamental scores by reporting period.

        Parameters
        ----------
        tickers : iterable of str
            Ticker symbols to rank through time.
        frequency : {"annual", "quarterly"}, default "quarterly"
            Statement frequency used to build histories.
        trailing_periods : int, default 4
            Maximum number of recent periods returned per company.

        Returns
        -------
        pandas.DataFrame
            Long-format ranked history.
        """
        metric_history = self.collect_metric_history(
            tickers,
            frequency=frequency,
            trailing_periods=trailing_periods,
        )
        if metric_history.empty:
            raise ValueError("No historical fundamental metrics could be built.")
        self.ranking_history_ = score_fundamentals_over_time(
            metric_history,
            self.score_config,
        )
        return self.ranking_history_.copy()

    def select_top(self, ranking: Optional[pd.DataFrame] = None, top_k: int = 5) -> pd.DataFrame:
        """
        Return the top-ranked companies from an existing ranking table.

        Parameters
        ----------
        ranking : pandas.DataFrame, optional
            Ranking table. If omitted, the latest `ranking_` is used.
        top_k : int, default 5
            Number of rows to return.

        Returns
        -------
        pandas.DataFrame
            Top-ranked companies reset to a clean integer index.
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")

        source = self.ranking_ if ranking is None else ranking
        if source.empty:
            raise ValueError("ranking is empty. Run rank(...) first or pass a ranking DataFrame.")
        return source.head(top_k).copy().reset_index(drop=True)

    def selection_report(
        self,
        ranking: Optional[pd.DataFrame] = None,
        top_k: int = 5,
    ) -> Dict[str, pd.DataFrame]:
        """
        Build an interpretability report for the selected companies.

        Parameters
        ----------
        ranking : pandas.DataFrame, optional
            Ranking table. If omitted, the latest `ranking_` is used.
        top_k : int, default 5
            Number of selected companies to include.

        Returns
        -------
        dict of str to pandas.DataFrame
            Selected companies, metric snapshot, score weights, and component
            scores.
        """
        selected = self.select_top(ranking=ranking, top_k=top_k)
        metric_names = list(self.score_config.metric_weights.keys())
        available_metrics = [metric for metric in metric_names if metric in selected.columns]
        component_columns = [
            f"{metric}_score"
            for metric in available_metrics
            if f"{metric}_score" in selected.columns
        ]

        weights = pd.DataFrame(
            [
                {
                    "metric": metric,
                    "weight": float(weight),
                    "higher_is_better": bool(
                        self.score_config.higher_is_better.get(metric, True)
                    ),
                    "component_column": f"{metric}_score",
                }
                for metric, weight in self.score_config.metric_weights.items()
            ]
        )

        summary_columns = [
            column
            for column in ["ticker", self.score_config.score_column, "strategy", "sector", "industry"]
            if column in selected.columns
        ]
        metric_snapshot = selected[["ticker", *available_metrics]].copy()
        score_components = selected[["ticker", *component_columns]].copy()

        return {
            "selected": selected[summary_columns].copy(),
            "metric_snapshot": metric_snapshot,
            "score_weights": weights,
            "score_components": score_components,
        }

    def metric_evolution(
        self,
        metric: str = "fundamental_score",
        tickers: Optional[Iterable[str]] = None,
        top_k: Optional[int] = None,
        history: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Return a ticker-by-period table for one historical metric.

        Parameters
        ----------
        metric : str, default "fundamental_score"
            Metric to display through time.
        tickers : iterable of str, optional
            Explicit ticker subset. When omitted, all tickers are used unless
            `top_k` is provided.
        top_k : int, optional
            Select the top companies from the latest period before pivoting.
        history : pandas.DataFrame, optional
            Custom long-format history. When omitted, the method uses
            `ranking_history_` if available and falls back to `metric_history_`.

        Returns
        -------
        pandas.DataFrame
            Matrix with tickers as rows, periods as columns, and the requested
            metric as values.
        """
        source = history
        if source is None:
            source = self.ranking_history_ if not self.ranking_history_.empty else self.metric_history_

        if source.empty:
            raise ValueError(
                "No historical data available. Run rank_over_time(...) or "
                "collect_metric_history(...) first."
            )
        if metric not in source.columns:
            raise ValueError(f"metric '{metric}' not found in historical data.")
        if "ticker" not in source.columns or "period" not in source.columns:
            raise ValueError("history must contain 'ticker' and 'period' columns.")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be >= 1 when provided.")

        working = source.copy()
        working["ticker"] = working["ticker"].astype(str).str.upper()
        working["period"] = pd.to_datetime(working["period"])

        if tickers is not None:
            selected_tickers = set(self._normalize_tickers(tickers))
            working = working[working["ticker"].isin(selected_tickers)]
        elif top_k is not None:
            latest_period = working["period"].max()
            latest = working[working["period"] == latest_period].copy()
            sort_metric = (
                self.score_config.score_column
                if self.score_config.score_column in latest.columns
                else metric
            )
            latest[sort_metric] = pd.to_numeric(latest[sort_metric], errors="coerce")
            selected_tickers = set(
                latest.sort_values(sort_metric, ascending=False)
                .head(top_k)["ticker"]
                .tolist()
            )
            working = working[working["ticker"].isin(selected_tickers)]

        if working.empty:
            return pd.DataFrame()

        evolution = working.pivot_table(
            index="ticker",
            columns="period",
            values=metric,
            aggfunc="last",
        )
        evolution = evolution.reindex(sorted(evolution.columns), axis=1)
        if top_k is not None:
            latest_column = evolution.columns.max()
            evolution = evolution.sort_values(latest_column, ascending=False)
        return evolution

    def run_pipeline(self, tickers: Iterable[str], top_k: int = 5) -> Dict[str, Any]:
        """
        Execute the fundamental selection workflow end to end.

        Parameters
        ----------
        tickers : iterable of str
            Ticker symbols to rank and select.
        top_k : int, default 5
            Number of selected companies to return.

        Returns
        -------
        dict
            Metrics, ranking, selected rows, selected ticker list, and
            interpretability report.
        """
        ranking = self.rank(tickers)
        selected = self.select_top(ranking=ranking, top_k=top_k)
        report = self.selection_report(ranking=ranking, top_k=top_k)
        return {
            "metrics": self.metrics_.copy(),
            "ranking": ranking,
            "selected": selected,
            "selected_tickers": selected["ticker"].tolist(),
            "report": report,
        }


__all__ = [
    "FundamentalSelector",
]
