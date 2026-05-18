"""Public API for security selection.

The package exports correlation-based portfolio selection, Yahoo Finance
fundamental-data access, asset-level market research, time-series helpers,
metric builders, scoring configurations, and the high-level
`FundamentalSelector` workflow.
"""

from .arima_research import ARIMAResearch
from .assets_research import AssetsResearch
from .correlation_selector import CorrelationPortfolioSelector, CorrelationSelector
from .fundamental_metrics import (
    build_fundamental_metric_history,
    build_fundamental_metrics,
    build_metric_history_frame,
    build_metrics_frame,
)
from .fundamental_scorers import (
    FundamentalScoreConfig,
    FUNDAMENTAL_METRIC_SIGNAL_SPECS,
    GrowthScoreConfig,
    MetricSignalSpec,
    ValueScoreConfig,
    fundamental_metric_signal_specs,
    score_fundamentals,
    score_fundamentals_over_time,
)
from .fundamental_selector import FundamentalSelector
from .fundamental_panel import (
    FundamentalLearningPanelBuilder,
    add_signal_features,
    build_fundamental_learning_panel,
    candidate_feature_columns,
)
from .fundamentals import FundamentalData, YahooFundamentalsProvider
from .fundamental_targets import (
    build_forward_return_targets,
    dividends_between,
    price_at_or_after,
    target_column_names,
)
from .learned_fundamental_scorers import LearnedScoreConfigFactory
from .learned_fundamental_selector import LearnedFundamentalSelector
from .xgboost_fundamental_model import XGBoostFundamentalModel

__all__ = [
    "ARIMAResearch",
    "AssetsResearch",
    "CorrelationPortfolioSelector",
    "CorrelationSelector",
    "FundamentalData",
    "FundamentalLearningPanelBuilder",
    "FundamentalScoreConfig",
    "FundamentalSelector",
    "FUNDAMENTAL_METRIC_SIGNAL_SPECS",
    "GrowthScoreConfig",
    "LearnedFundamentalSelector",
    "LearnedScoreConfigFactory",
    "MetricSignalSpec",
    "ValueScoreConfig",
    "XGBoostFundamentalModel",
    "YahooFundamentalsProvider",
    "add_signal_features",
    "build_forward_return_targets",
    "build_fundamental_learning_panel",
    "build_fundamental_metric_history",
    "build_fundamental_metrics",
    "build_metric_history_frame",
    "build_metrics_frame",
    "candidate_feature_columns",
    "dividends_between",
    "fundamental_metric_signal_specs",
    "price_at_or_after",
    "score_fundamentals",
    "score_fundamentals_over_time",
    "target_column_names",
]
