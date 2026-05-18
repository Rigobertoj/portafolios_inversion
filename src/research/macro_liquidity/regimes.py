"""Current macro/liquidity regime analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .catalog import EconomicSeriesSpec, default_us_macro_liquidity_catalog
from .transforms import build_indicator_panel, build_score_frame


def _score(row: pd.Series, column: str, fallback: Optional[str] = None) -> float:
    value = row.get(column, np.nan)
    if pd.isna(value) and fallback is not None:
        value = row.get(fallback, np.nan)
    return float(value) if pd.notna(value) else np.nan


def classify_liquidity_regime(score: float) -> str:
    """Classify liquidity with the methodology thresholds."""

    if pd.isna(score):
        return "unknown"
    if score >= 1.0:
        return "expansiva"
    if score >= 0.5:
        return "moderadamente_expansiva"
    if score > -0.5:
        return "neutral"
    if score > -1.0:
        return "restrictiva"
    return "estresada"


def classify_macro_regime(row: pd.Series) -> str:
    """Classify macro conditions from growth, inflation, labor, and policy."""

    growth = _score(row, "growth_score", "macro_score")
    inflation = _score(row, "inflation_score")
    labor = _score(row, "labor_score")
    policy = _score(row, "policy_score")

    if pd.isna(growth):
        return "unknown"
    inflation = 0.0 if pd.isna(inflation) else inflation
    labor = 0.0 if pd.isna(labor) else labor
    policy = 0.0 if pd.isna(policy) else policy

    if growth <= -0.75 and inflation >= 0.0:
        return "recesion_desinflacionaria"
    if growth <= -0.35 and inflation < -0.20:
        return "estanflacion"
    if growth >= 0.60 and inflation < -0.20:
        return "reaceleracion_inflacionaria" if policy < 0.0 else "sobrecalentamiento"
    if growth >= 0.50 and inflation >= -0.20 and labor >= -0.50:
        return "expansion_desinflacionaria"
    if growth < 0.50 and inflation >= -0.20:
        return "desaceleracion_ordenada"
    if growth < 0.50 and inflation < -0.20:
        return "estanflacion"
    return "neutral"


@dataclass(frozen=True)
class CurrentRegimeSnapshot:
    """Point-in-time macro/liquidity diagnosis."""

    date: pd.Timestamp
    macro_regime: str
    liquidity_regime: str
    macro_score: float
    liquidity_score: float
    confidence: float
    macro_coverage: float
    liquidity_coverage: float

    def label(self) -> str:
        """Return a compact integrated-regime label."""

        return f"{self.macro_regime} + {self.liquidity_regime}"


@dataclass(frozen=True)
class MacroLiquidityResult:
    """Output bundle for current-regime analysis."""

    indicators: pd.DataFrame
    scores: pd.DataFrame
    current: CurrentRegimeSnapshot


class MacroLiquidityResearch:
    """Diagnose the current U.S. macro and liquidity regime."""

    def __init__(
        self,
        series_specs: Optional[Iterable[EconomicSeriesSpec]] = None,
        min_periods: int = 6,
    ) -> None:
        self.series_specs = tuple(series_specs or default_us_macro_liquidity_catalog())
        self.min_periods = max(int(min_periods), 2)

    @staticmethod
    def _snapshot_from_scores(scores: pd.DataFrame) -> CurrentRegimeSnapshot:
        if scores.empty:
            raise ValueError("scores are empty; cannot build a current regime snapshot.")
        latest = scores.sort_values("date").iloc[-1]
        macro_regime = classify_macro_regime(latest)
        liquidity_regime = classify_liquidity_regime(_score(latest, "liquidity_score"))
        macro_coverage = _score(latest, "macro_coverage")
        liquidity_coverage = _score(latest, "liquidity_coverage")
        coverage_values = [value for value in (macro_coverage, liquidity_coverage) if pd.notna(value)]
        confidence = float(np.mean(coverage_values)) if coverage_values else 0.0
        return CurrentRegimeSnapshot(
            date=pd.Timestamp(latest["date"]),
            macro_regime=macro_regime,
            liquidity_regime=liquidity_regime,
            macro_score=_score(latest, "macro_score"),
            liquidity_score=_score(latest, "liquidity_score"),
            confidence=confidence,
            macro_coverage=macro_coverage,
            liquidity_coverage=liquidity_coverage,
        )

    def analyze_current(self, series_frame: pd.DataFrame) -> MacroLiquidityResult:
        """Build indicator scores and return the latest current-regime snapshot."""

        indicators = build_indicator_panel(series_frame, self.series_specs, self.min_periods)
        scores = build_score_frame(indicators)
        current = self._snapshot_from_scores(scores)
        return MacroLiquidityResult(indicators=indicators, scores=scores, current=current)
