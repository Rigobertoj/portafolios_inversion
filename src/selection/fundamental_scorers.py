"""Scoring rules for value and growth fundamental selection.

This module ranks companies by weighted, normalized fundamental signals. A
signal is a metric observed as a level, a recent change, a year-over-year
change, or an average historical change. The default normalizer is a robust
cross-sectional percentile so valuation multiples, margins, leverage ratios,
and growth rates can be combined on a common 0-100 scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Literal, Mapping, Optional

import numpy as np
import pandas as pd

SignalType = Literal["level", "period_change", "yoy_change", "historical_avg_change"]
Normalizer = Literal["percentile_rank", "robust_zscore", "target"]
ChangeMethod = Literal["relative", "difference", "auto"]


_DIFFERENCE_CHANGE_METRICS = {
    "gross_margin",
    "sga_to_sales",
    "operating_margin",
    "pretax_margin",
    "profit_margin",
    "free_cash_flow_margin",
    "roe",
    "return_on_assets",
    "return_on_total_capital",
    "return_on_invested_capital",
    "cash_flow_return_on_invested_capital",
    "current_ratio",
    "quick_ratio",
    "cash_ratio",
    "debt_to_equity",
    "total_debt_to_total_assets",
    "total_debt_to_total_capital",
    "net_debt_to_total_capital",
    "interest_coverage",
}


@dataclass(frozen=True)
class MetricSignalSpec:
    """
    Configuration for one scoring component.

    Parameters
    ----------
    metric : str
        Metric column to score.
    signal : {"level", "period_change", "yoy_change", "historical_avg_change"}
        Transformation applied before cross-sectional scoring.
    weight : float
        Relative weight in the composite score.
    higher_is_better : bool
        Ranking direction after the signal is computed.
    normalizer : {"percentile_rank", "robust_zscore", "target"}
        Method used to map raw signal values to 0-1 component scores.
    change_method : {"relative", "difference", "auto"}
        How change signals are computed from period histories.
    target, tolerance : float, optional
        Center and half-width for target-based scoring.
    category : str
        Financial category used for reporting and diagnostics.
    """

    metric: str
    signal: SignalType = "level"
    weight: float = 1.0
    higher_is_better: bool = True
    normalizer: Normalizer = "percentile_rank"
    change_method: ChangeMethod = "auto"
    target: Optional[float] = None
    tolerance: Optional[float] = None
    category: str = "general"

    @property
    def signal_name(self) -> str:
        return f"{self.metric}__{self.signal}"

    @property
    def component_name(self) -> str:
        return f"{self.signal_name}_score"


@dataclass(frozen=True)
class FundamentalScoreConfig:
    """
    Weighted score configuration for a fundamental investing style.

    Parameters
    ----------
    name : str
        Strategy or style label.
    metric_weights : mapping of str to float
        Legacy metric weights. Used directly when `signal_specs` is empty.
    higher_is_better : mapping of str to bool
        Legacy ranking direction. Used directly when `signal_specs` is empty.
    signal_specs : iterable of MetricSignalSpec
        Granular scoring components. When present, these replace the legacy
        metric-only score while preserving the public configuration contract.
    score_column : str, default "fundamental_score"
        Name of the composite score column added to ranked outputs.
    winsorize_quantiles : tuple of float, default (0.05, 0.95)
        Cross-sectional quantiles used to cap raw values before normalization.
    missing_component_score : float, default 0.0
        Component score assigned to a company when a component exists in the
        universe but is missing for that specific company.
    """

    name: str
    metric_weights: Mapping[str, float] = field(default_factory=dict)
    higher_is_better: Mapping[str, bool] = field(default_factory=dict)
    signal_specs: Iterable[MetricSignalSpec] = field(default_factory=tuple)
    score_column: str = "fundamental_score"
    winsorize_quantiles: tuple[float, float] = (0.05, 0.95)
    missing_component_score: float = 0.0

    def resolved_signal_specs(self) -> tuple[MetricSignalSpec, ...]:
        return tuple(self.signal_specs)

    def uses_historical_signals(self) -> bool:
        return any(spec.signal != "level" for spec in self.resolved_signal_specs())


def _spec(
    metric: str,
    signal: SignalType,
    weight: float,
    higher_is_better: bool,
    category: str,
    change_method: ChangeMethod = "auto",
    normalizer: Normalizer = "percentile_rank",
    target: Optional[float] = None,
    tolerance: Optional[float] = None,
) -> MetricSignalSpec:
    return MetricSignalSpec(
        metric=metric,
        signal=signal,
        weight=weight,
        higher_is_better=higher_is_better,
        normalizer=normalizer,
        change_method=change_method,
        target=target,
        tolerance=tolerance,
        category=category,
    )


_FUNDAMENTAL_METRIC_DEFINITIONS: tuple[tuple[str, str, bool, ChangeMethod], ...] = (
    ("gross_margin", "profitability", True, "difference"),
    ("sga_to_sales", "profitability", False, "difference"),
    ("operating_margin", "profitability", True, "difference"),
    ("pretax_margin", "profitability", True, "difference"),
    ("profit_margin", "profitability", True, "difference"),
    ("free_cash_flow_margin", "profitability", True, "difference"),
    ("free_cash_flow_conversion_ratio", "profitability", True, "relative"),
    ("capex_to_sales", "profitability", False, "difference"),
    ("return_on_assets", "profitability", True, "difference"),
    ("roe", "profitability", True, "difference"),
    ("return_on_common_equity", "profitability", True, "difference"),
    ("return_on_total_capital", "profitability", True, "difference"),
    ("return_on_invested_capital", "profitability", True, "difference"),
    ("cash_flow_return_on_invested_capital", "profitability", True, "difference"),
    ("price_to_sales", "valuation", False, "relative"),
    ("trailing_pe", "valuation", False, "relative"),
    ("price_to_book", "valuation", False, "relative"),
    ("price_to_tangible_book", "valuation", False, "relative"),
    ("price_to_cash_flow", "valuation", False, "relative"),
    ("price_to_free_cash_flow", "valuation", False, "relative"),
    ("dividend_yield", "valuation", True, "difference"),
    ("enterprise_value_to_ebit", "valuation", False, "relative"),
    ("enterprise_value_to_ebitda", "valuation", False, "relative"),
    ("enterprise_value_to_sales", "valuation", False, "relative"),
    ("sales_per_share", "per_share", True, "relative"),
    ("operating_income_per_share", "per_share", True, "relative"),
    ("eps_recurring", "per_share", True, "relative"),
    ("eps_basic", "per_share", True, "relative"),
    ("eps_diluted", "per_share", True, "relative"),
    ("dividends_per_share", "per_share", True, "relative"),
    ("dividend_payout_ratio", "per_share", False, "difference"),
    ("book_value_per_share", "per_share", True, "relative"),
    ("tangible_book_value_per_share", "per_share", True, "relative"),
    ("cash_flow_per_share", "per_share", True, "relative"),
    ("free_cash_flow_per_share", "per_share", True, "relative"),
    ("diluted_shares_outstanding", "per_share", False, "relative"),
    ("basic_shares_outstanding", "per_share", False, "relative"),
    ("total_shares_outstanding", "per_share", False, "relative"),
    ("cash_and_short_term_turnover", "asset_turnover_analysis", True, "relative"),
    ("receivables_turnover", "asset_turnover_analysis", True, "relative"),
    ("inventory_turnover", "asset_turnover_analysis", True, "relative"),
    ("current_assets_turnover", "asset_turnover_analysis", True, "relative"),
    ("fixed_assets_turnover", "asset_turnover_analysis", True, "relative"),
    ("total_assets_turnover", "asset_turnover_analysis", True, "relative"),
    ("asset_turnover_dupont", "dupont_analysis", True, "relative"),
    ("pretax_margin", "dupont_analysis", True, "difference"),
    ("pretax_return_on_assets", "dupont_analysis", True, "difference"),
    ("tax_rate_complement", "dupont_analysis", True, "difference"),
    ("return_on_assets_dupont", "dupont_analysis", True, "difference"),
    ("equity_multiplier", "dupont_analysis", False, "difference"),
    ("return_on_equity_dupont", "dupont_analysis", True, "difference"),
    ("earnings_retention", "dupont_analysis", True, "difference"),
    ("reinvestment_rate", "dupont_analysis", True, "difference"),
    ("ebit_return_on_assets", "dupont_analysis", True, "difference"),
    ("interest_as_percent_assets", "dupont_analysis", False, "difference"),
    ("receivables_turnover", "operating_efficiency", True, "relative"),
    ("inventory_turnover", "operating_efficiency", True, "relative"),
    ("payables_turnover", "operating_efficiency", True, "relative"),
    ("asset_turnover", "operating_efficiency", True, "relative"),
    ("working_capital_turnover", "operating_efficiency", True, "relative"),
    ("days_inventory_on_hand", "operating_cycle_days", False, "relative"),
    ("days_sales_outstanding", "operating_cycle_days", False, "relative"),
    ("operating_cycle", "operating_cycle_days", False, "relative"),
    ("days_payables_outstanding", "operating_cycle_days", True, "relative"),
    ("net_operating_cycle", "operating_cycle_days", False, "relative"),
    ("current_ratio", "liquidity", True, "difference"),
    ("quick_ratio", "liquidity", True, "difference"),
    ("cash_ratio", "liquidity", True, "difference"),
    ("cash_and_short_term_to_current_assets", "liquidity", True, "difference"),
    ("cfo_to_current_liabilities", "liquidity", True, "relative"),
    ("net_debt_to_ebitda", "coverage", False, "relative"),
    ("net_debt_to_ebitda_minus_capex", "coverage", False, "relative"),
    ("total_debt_to_ebitda", "coverage", False, "relative"),
    ("interest_coverage", "coverage", True, "relative"),
    ("ebitda_to_interest_expense", "coverage", True, "relative"),
    ("fixed_charge_coverage_ratio", "coverage", True, "relative"),
    ("cfo_to_interest_expense", "coverage", True, "relative"),
    ("cash_dividend_coverage_ratio", "coverage", True, "relative"),
    ("long_term_debt_to_ebitda", "coverage", False, "relative"),
    ("net_debt_to_ffo", "coverage", False, "relative"),
    ("long_term_debt_to_ffo", "coverage", False, "relative"),
    ("cfo_to_total_debt", "coverage", True, "relative"),
    ("ebitda_minus_capex_to_interest_expense", "coverage", True, "relative"),
    ("long_term_debt_to_total_equity", "leverage", False, "difference"),
    ("long_term_debt_to_total_capital", "leverage", False, "difference"),
    ("long_term_debt_to_total_assets", "leverage", False, "difference"),
    ("total_debt_to_total_assets", "leverage", False, "difference"),
    ("net_debt_to_total_equity", "leverage", False, "difference"),
    ("debt_to_equity", "leverage", False, "difference"),
    ("net_debt_to_total_capital", "leverage", False, "difference"),
    ("total_debt_to_total_capital", "leverage", False, "difference"),
)


def fundamental_metric_signal_specs(
    signals: Iterable[SignalType] = ("level",),
) -> tuple[MetricSignalSpec, ...]:
    """Return `MetricSignalSpec` entries for the full fundamental metric catalog."""
    return tuple(
        _spec(
            metric=metric,
            signal=signal,
            weight=1.0,
            higher_is_better=higher_is_better,
            category=category,
            change_method=change_method,
        )
        for metric, category, higher_is_better, change_method in _FUNDAMENTAL_METRIC_DEFINITIONS
        for signal in signals
    )


FUNDAMENTAL_METRIC_SIGNAL_SPECS = fundamental_metric_signal_specs()


def _value_signal_specs() -> tuple[MetricSignalSpec, ...]:
    return (
        _spec("trailing_pe", "level", 0.12, False, "valuation", "relative"),
        _spec("price_to_book", "level", 0.09, False, "valuation", "relative"),
        _spec("enterprise_value_to_ebitda", "level", 0.07, False, "valuation", "relative"),
        _spec("price_to_free_cash_flow", "level", 0.06, False, "valuation", "relative"),
        _spec("dividend_yield", "level", 0.04, True, "shareholder_return", "difference"),
        _spec("roe", "level", 0.06, True, "profitability", "difference"),
        _spec("return_on_invested_capital", "level", 0.06, True, "profitability", "difference"),
        _spec("profit_margin", "level", 0.05, True, "profitability", "difference"),
        _spec("free_cash_flow_margin", "level", 0.06, True, "cash_flow", "difference"),
        _spec("debt_to_equity", "level", 0.05, False, "leverage", "difference"),
        _spec("total_debt_to_ebitda", "level", 0.04, False, "coverage", "relative"),
        _spec("current_ratio", "level", 0.03, True, "liquidity", "difference"),
        _spec("revenue", "period_change", 0.04, True, "growth", "relative"),
        _spec("eps", "period_change", 0.04, True, "growth", "relative"),
        _spec("free_cash_flow", "period_change", 0.04, True, "cash_flow", "relative"),
        _spec("profit_margin", "period_change", 0.03, True, "profitability", "difference"),
        _spec("revenue", "historical_avg_change", 0.03, True, "growth", "relative"),
        _spec("eps", "historical_avg_change", 0.03, True, "growth", "relative"),
        _spec("free_cash_flow_margin", "historical_avg_change", 0.03, True, "cash_flow", "difference"),
        _spec("debt_to_equity", "historical_avg_change", 0.03, False, "leverage", "difference"),
    )


def _growth_signal_specs() -> tuple[MetricSignalSpec, ...]:
    return (
        _spec("revenue", "period_change", 0.09, True, "growth", "relative"),
        _spec("revenue", "yoy_change", 0.09, True, "growth", "relative"),
        _spec("eps", "period_change", 0.09, True, "growth", "relative"),
        _spec("eps", "yoy_change", 0.09, True, "growth", "relative"),
        _spec("net_income", "period_change", 0.05, True, "growth", "relative"),
        _spec("free_cash_flow", "period_change", 0.05, True, "cash_flow", "relative"),
        _spec("revenue", "historical_avg_change", 0.07, True, "growth", "relative"),
        _spec("eps", "historical_avg_change", 0.07, True, "growth", "relative"),
        _spec("free_cash_flow", "historical_avg_change", 0.05, True, "cash_flow", "relative"),
        _spec("gross_margin", "level", 0.04, True, "profitability", "difference"),
        _spec("operating_margin", "level", 0.05, True, "profitability", "difference"),
        _spec("roe", "level", 0.05, True, "profitability", "difference"),
        _spec("return_on_invested_capital", "level", 0.05, True, "profitability", "difference"),
        _spec("operating_margin", "historical_avg_change", 0.04, True, "profitability", "difference"),
        _spec("peg_ratio", "level", 0.04, False, "valuation", "relative"),
        _spec("trailing_pe", "level", 0.03, False, "valuation", "relative"),
        _spec("debt_to_equity", "level", 0.03, False, "leverage", "difference"),
        _spec("interest_coverage", "level", 0.02, True, "coverage", "relative"),
    )


@dataclass(frozen=True)
class ValueScoreConfig(FundamentalScoreConfig):
    """Default value score with valuation, quality, cash-flow, and trend signals."""

    name: str = "value"
    metric_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "trailing_pe": 0.12,
            "price_to_book": 0.09,
            "enterprise_value_to_ebitda": 0.07,
            "price_to_free_cash_flow": 0.06,
            "dividend_yield": 0.04,
            "roe": 0.06,
            "return_on_invested_capital": 0.06,
            "profit_margin": 0.08,
            "free_cash_flow_margin": 0.09,
            "debt_to_equity": 0.08,
            "total_debt_to_ebitda": 0.04,
            "current_ratio": 0.03,
            "revenue": 0.07,
            "eps": 0.07,
            "free_cash_flow": 0.04,
        }
    )
    higher_is_better: Mapping[str, bool] = field(
        default_factory=lambda: {
            "trailing_pe": False,
            "price_to_book": False,
            "enterprise_value_to_ebitda": False,
            "price_to_free_cash_flow": False,
            "dividend_yield": True,
            "roe": True,
            "return_on_invested_capital": True,
            "profit_margin": True,
            "free_cash_flow_margin": True,
            "debt_to_equity": False,
            "total_debt_to_ebitda": False,
            "current_ratio": True,
            "revenue": True,
            "eps": True,
            "free_cash_flow": True,
        }
    )
    signal_specs: Iterable[MetricSignalSpec] = field(default_factory=_value_signal_specs)


@dataclass(frozen=True)
class GrowthScoreConfig(FundamentalScoreConfig):
    """Default growth score focused on expansion, quality, and valuation discipline."""

    name: str = "growth"
    metric_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "revenue": 0.25,
            "eps": 0.25,
            "net_income": 0.05,
            "free_cash_flow": 0.10,
            "gross_margin": 0.04,
            "operating_margin": 0.09,
            "roe": 0.05,
            "return_on_invested_capital": 0.05,
            "peg_ratio": 0.04,
            "trailing_pe": 0.03,
            "debt_to_equity": 0.03,
            "interest_coverage": 0.02,
            "revenue_growth": 0.00,
            "eps_growth": 0.00,
            "earnings_growth": 0.00,
        }
    )
    higher_is_better: Mapping[str, bool] = field(
        default_factory=lambda: {
            "revenue": True,
            "eps": True,
            "net_income": True,
            "free_cash_flow": True,
            "gross_margin": True,
            "operating_margin": True,
            "roe": True,
            "return_on_invested_capital": True,
            "peg_ratio": False,
            "trailing_pe": False,
            "debt_to_equity": False,
            "interest_coverage": True,
            "revenue_growth": True,
            "eps_growth": True,
            "earnings_growth": True,
        }
    )
    signal_specs: Iterable[MetricSignalSpec] = field(default_factory=_growth_signal_specs)


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


def _ticker_series(frame: pd.DataFrame) -> pd.Series:
    if "ticker" in frame.columns:
        return frame["ticker"].astype(str).str.upper()
    return pd.Series(frame.index.astype(str).str.upper(), index=frame.index)


def _winsorized(values: pd.Series, quantiles: tuple[float, float]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = numeric.dropna()
    if valid.empty:
        return numeric

    lower_q, upper_q = quantiles
    lower_q = min(max(float(lower_q), 0.0), 1.0)
    upper_q = min(max(float(upper_q), 0.0), 1.0)
    if lower_q > upper_q:
        lower_q, upper_q = upper_q, lower_q

    lower = valid.quantile(lower_q)
    upper = valid.quantile(upper_q)
    return numeric.clip(lower=lower, upper=upper)


def _target_component(values: pd.Series, spec: MetricSignalSpec) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if spec.target is None or spec.tolerance is None or np.isclose(float(spec.tolerance), 0.0):
        return pd.Series(np.nan, index=values.index, dtype=float)

    distance = (numeric - float(spec.target)).abs() / abs(float(spec.tolerance))
    return (1.0 - distance).clip(lower=0.0, upper=1.0)


def _robust_zscore_component(values: pd.Series, spec: MetricSignalSpec) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=values.index, dtype=float)
    if valid.shape[0] == 1:
        return numeric.notna().astype(float)

    median = valid.median()
    iqr = valid.quantile(0.75) - valid.quantile(0.25)
    scale = iqr if not np.isclose(float(iqr), 0.0) else valid.std(ddof=0)
    if pd.isna(scale) or np.isclose(float(scale), 0.0):
        return numeric.rank(pct=True, ascending=spec.higher_is_better)

    zscore = (numeric - median) / scale
    if not spec.higher_is_better:
        zscore = -zscore
    return 1.0 / (1.0 + np.exp(-zscore))


def _component_score(
    values: pd.Series,
    spec: MetricSignalSpec,
    config: FundamentalScoreConfig,
) -> pd.Series:
    if spec.normalizer == "target":
        return _target_component(values, spec)
    if spec.normalizer == "robust_zscore":
        return _robust_zscore_component(values, spec)

    winsorized = _winsorized(values, config.winsorize_quantiles)
    return winsorized.rank(pct=True, ascending=spec.higher_is_better)


def _lag_for_frequency(frequency: Optional[str]) -> int:
    frequency_clean = str(frequency or "").strip().lower()
    if frequency_clean == "monthly":
        return 12
    if frequency_clean == "quarterly":
        return 4
    return 1


def _change_value(current: float, previous: float, method: ChangeMethod, metric: str) -> float:
    if pd.isna(current) or pd.isna(previous):
        return np.nan

    change_method = method
    if change_method == "auto":
        change_method = "difference" if metric in _DIFFERENCE_CHANGE_METRICS else "relative"

    current_float = float(current)
    previous_float = float(previous)
    if change_method == "difference":
        return current_float - previous_float
    if np.isclose(previous_float, 0.0):
        return np.nan
    return (current_float - previous_float) / abs(previous_float)


def _history_signal_for_group(
    group: pd.DataFrame,
    spec: MetricSignalSpec,
    frequency: Optional[str],
) -> float:
    values = pd.to_numeric(group[spec.metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.dropna()
    if values.empty:
        return np.nan

    if spec.signal == "level":
        return float(values.iloc[-1])

    lag = 1 if spec.signal in {"period_change", "historical_avg_change"} else _lag_for_frequency(frequency)
    if spec.signal == "historical_avg_change":
        changes = [
            _change_value(values.iloc[position], values.iloc[position - 1], spec.change_method, spec.metric)
            for position in range(1, values.shape[0])
        ]
        changes = pd.Series(changes, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        return float(changes.mean()) if not changes.empty else np.nan

    if values.shape[0] <= lag:
        return np.nan
    return _change_value(values.iloc[-1], values.iloc[-1 - lag], spec.change_method, spec.metric)


def _signal_from_history(
    metric_history: pd.DataFrame,
    tickers: pd.Series,
    spec: MetricSignalSpec,
    frequency: Optional[str],
) -> pd.Series:
    empty = pd.Series(np.nan, index=tickers.index, dtype=float)
    required = {"ticker", "period", spec.metric}
    if metric_history is None or metric_history.empty or not required.issubset(metric_history.columns):
        return empty

    history = metric_history[list(required)].copy()
    history["ticker"] = history["ticker"].astype(str).str.upper()
    history["period"] = pd.to_datetime(history["period"])
    history = history.sort_values(["ticker", "period"])

    values_by_ticker = {
        ticker: _history_signal_for_group(group, spec, frequency)
        for ticker, group in history.groupby("ticker", sort=False)
    }
    return tickers.map(values_by_ticker).astype(float)


def _signal_values(
    metrics: pd.DataFrame,
    metric_history: Optional[pd.DataFrame],
    spec: MetricSignalSpec,
    frequency: Optional[str],
) -> pd.Series:
    if spec.signal == "level" and spec.metric in metrics.columns:
        return pd.to_numeric(metrics[spec.metric], errors="coerce").replace([np.inf, -np.inf], np.nan)

    tickers = _ticker_series(metrics)
    return _signal_from_history(metric_history, tickers, spec, frequency)


def _score_legacy_metrics(
    metrics: pd.DataFrame,
    config: FundamentalScoreConfig,
) -> pd.DataFrame:
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
        weighted_score = weighted_score.add(percentile * abs(float(weight)), fill_value=0.0)
        total_weight += abs(float(weight))

    if np.isclose(total_weight, 0.0):
        scored[config.score_column] = 0.0
    else:
        scored[config.score_column] = weighted_score / total_weight * 100

    for column, values in score_parts.items():
        scored[column] = values

    scored["score_coverage"] = 1.0 if not np.isclose(total_weight, 0.0) else 0.0
    return scored


def _score_signal_metrics(
    metrics: pd.DataFrame,
    config: FundamentalScoreConfig,
    metric_history: Optional[pd.DataFrame],
    frequency: Optional[str],
) -> pd.DataFrame:
    scored = metrics.copy()
    weighted_score = pd.Series(0.0, index=scored.index, dtype=float)
    available_weight_by_row = pd.Series(0.0, index=scored.index, dtype=float)
    total_weight = 0.0

    for spec in config.resolved_signal_specs():
        weight = abs(float(spec.weight))
        if np.isclose(weight, 0.0):
            continue

        values = _signal_values(scored, metric_history, spec, frequency)
        if values.dropna().empty:
            continue

        scored[spec.signal_name] = values
        component = _component_score(values, spec, config)
        component_filled = component.fillna(float(config.missing_component_score))
        scored[spec.component_name] = component_filled

        weighted_score = weighted_score.add(component_filled * weight, fill_value=0.0)
        available_weight_by_row = available_weight_by_row.add(values.notna().astype(float) * weight, fill_value=0.0)
        total_weight += weight

    if np.isclose(total_weight, 0.0):
        scored[config.score_column] = 0.0
        scored["score_coverage"] = 0.0
    else:
        scored[config.score_column] = weighted_score / total_weight * 100
        scored["score_coverage"] = available_weight_by_row / total_weight

    return scored


def _sort_scored(scored: pd.DataFrame, score_column: str) -> pd.DataFrame:
    sort_columns = [score_column]
    ascending = [False]
    if "score_coverage" in scored.columns:
        sort_columns.append("score_coverage")
        ascending.append(False)
    if "market_cap" in scored.columns:
        sort_columns.append("market_cap")
        ascending.append(False)

    return scored.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def score_fundamentals(
    metrics: pd.DataFrame,
    config: FundamentalScoreConfig,
    metric_history: Optional[pd.DataFrame] = None,
    frequency: Optional[str] = None,
) -> pd.DataFrame:
    """
    Rank companies using weighted, normalized fundamental components.

    Parameters
    ----------
    metrics : pandas.DataFrame
        Fundamental metrics table with one row per company.
    config : FundamentalScoreConfig
        Metric weights, directions, score column, and optional signal specs.
    metric_history : pandas.DataFrame, optional
        Long-format period history used for change and historical signals.
    frequency : str, optional
        Reporting frequency used to infer year-over-year lags.

    Returns
    -------
    pandas.DataFrame
        Ranked table with composite score, component score columns, and coverage.
    """
    if metrics.empty:
        return metrics.copy()

    if config.resolved_signal_specs():
        scored = _score_signal_metrics(metrics, config, metric_history, frequency)
    else:
        scored = _score_legacy_metrics(metrics, config)
    return _sort_scored(scored, config.score_column)


def score_fundamentals_over_time(
    metric_history: pd.DataFrame,
    config: FundamentalScoreConfig,
    frequency: Optional[str] = None,
) -> pd.DataFrame:
    """
    Score companies cross-sectionally inside each reporting period.

    Historical signal configs are evaluated with only the data available up to
    each period, avoiding look-ahead in period-by-period score histories.
    """
    if metric_history.empty:
        return metric_history.copy()
    if "period" not in metric_history.columns:
        raise ValueError("metric_history must contain a 'period' column.")

    history = metric_history.copy()
    history["period"] = pd.to_datetime(history["period"])

    scored_frames: list[pd.DataFrame] = []
    for period in sorted(history["period"].dropna().unique()):
        history_to_period = history[history["period"] <= period].copy()
        period_metrics = history_to_period[history_to_period["period"] == period].copy()
        scored = score_fundamentals(
            period_metrics,
            config,
            metric_history=history_to_period,
            frequency=frequency,
        )
        scored["period"] = period
        scored["strategy"] = config.name
        scored_frames.append(scored)

    return pd.concat(scored_frames, ignore_index=True)


__all__ = [
    "FundamentalScoreConfig",
    "FUNDAMENTAL_METRIC_SIGNAL_SPECS",
    "GrowthScoreConfig",
    "MetricSignalSpec",
    "ValueScoreConfig",
    "config_for_strategy",
    "fundamental_metric_signal_specs",
    "score_fundamentals",
    "score_fundamentals_over_time",
]
