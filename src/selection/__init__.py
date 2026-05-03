"""Public API for security selection.

The package exports correlation-based portfolio selection, Yahoo Finance
fundamental-data access, metric builders, scoring configurations, and the
high-level `FundamentalSelector` workflow.
"""

from .correlation_selector import CorrelationPortfolioSelector, CorrelationSelector
from .fundamental_metrics import (
    build_fundamental_metric_history,
    build_fundamental_metrics,
    build_metric_history_frame,
    build_metrics_frame,
)
from .fundamental_scorers import (
    FundamentalScoreConfig,
    GrowthScoreConfig,
    ValueScoreConfig,
    score_fundamentals,
    score_fundamentals_over_time,
)
from .fundamental_selector import FundamentalSelector
from .fundamentals import FundamentalData, YahooFundamentalsProvider

__all__ = [
    "CorrelationPortfolioSelector",
    "CorrelationSelector",
    "FundamentalData",
    "FundamentalScoreConfig",
    "FundamentalSelector",
    "GrowthScoreConfig",
    "ValueScoreConfig",
    "YahooFundamentalsProvider",
    "build_fundamental_metric_history",
    "build_fundamental_metrics",
    "build_metric_history_frame",
    "build_metrics_frame",
    "score_fundamentals",
    "score_fundamentals_over_time",
]
