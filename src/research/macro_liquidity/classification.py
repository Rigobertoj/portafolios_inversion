"""Regime classification from nowcast and forecast outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .forecasting import ForecastResult
from .nowcast import NowcastResult
from .policy import MacroLiquidityPolicy
from .regimes import CurrentRegimeSnapshot, classify_liquidity_regime, classify_macro_regime


def _row_for_horizon(forecast: ForecastResult, horizon: int, value_column: str) -> pd.Series:
    wide = forecast.forecast_wide(value_column=value_column)
    row = wide[wide["horizon_months"] == horizon]
    if row.empty:
        raise ValueError(f"forecast does not contain horizon {horizon}.")
    return row.iloc[0]


def _macro_probability(row: pd.Series, regime: str) -> float:
    macro_score = float(row.get("macro_score", 0.0))
    growth_score = float(row.get("growth_score", macro_score))
    inflation_score = float(row.get("inflation_score", 0.0))
    labor_score = float(row.get("labor_score", 0.0))
    if regime == "expansion_desinflacionaria":
        raw = 0.45 + 0.20 * max(growth_score, 0.0) + 0.15 * max(labor_score, 0.0)
    elif regime == "desaceleracion_ordenada":
        raw = 0.45 + 0.20 * max(0.5 - growth_score, 0.0) + 0.10 * max(inflation_score, 0.0)
    elif regime == "estanflacion":
        raw = 0.35 + 0.25 * max(-growth_score, 0.0) + 0.20 * max(-inflation_score, 0.0)
    elif regime == "recesion_desinflacionaria":
        raw = 0.35 + 0.30 * max(-growth_score, 0.0)
    else:
        raw = 0.30 + 0.10 * abs(macro_score)
    return float(min(0.95, max(0.05, raw)))


def _liquidity_probability(score: float, regime: str) -> float:
    centers = {
        "expansiva": 1.25,
        "moderadamente_expansiva": 0.75,
        "neutral": 0.0,
        "restrictiva": -0.75,
        "estresada": -1.25,
    }
    center = centers.get(regime, 0.0)
    distance = abs(float(score) - center)
    return float(min(0.95, max(0.05, 1.0 / (1.0 + distance))))


@dataclass(frozen=True)
class RegimeView:
    """Current and expected regime classification."""

    current: CurrentRegimeSnapshot
    expected: CurrentRegimeSnapshot
    strategic: CurrentRegimeSnapshot
    fallback: CurrentRegimeSnapshot
    downside: CurrentRegimeSnapshot
    upside: CurrentRegimeSnapshot
    probabilities: pd.DataFrame
    transition: str
    confidence: float
    methodology: dict[str, object]


class RegimeClassifier:
    """Classify current and forward macro/liquidity regimes."""

    def __init__(self, policy: Optional[MacroLiquidityPolicy] = None) -> None:
        self.policy = policy or MacroLiquidityPolicy()

    @staticmethod
    def _snapshot_from_forecast_row(
        row: pd.Series,
        value_column_label: str,
        confidence: float,
    ) -> CurrentRegimeSnapshot:
        macro_regime = classify_macro_regime(row)
        liquidity_score = float(row.get("liquidity_score", np.nan))
        return CurrentRegimeSnapshot(
            date=pd.Timestamp(row.get("forecast_date", pd.Timestamp.today().normalize())),
            macro_regime=macro_regime,
            liquidity_regime=classify_liquidity_regime(liquidity_score),
            macro_score=float(row.get("macro_score", np.nan)),
            liquidity_score=liquidity_score,
            confidence=float(confidence),
            macro_coverage=1.0,
            liquidity_coverage=1.0,
        )

    def _snapshot_for(
        self,
        forecast: ForecastResult,
        horizon: int,
        value_column: str,
        confidence: float,
    ) -> CurrentRegimeSnapshot:
        row = _row_for_horizon(forecast, horizon, value_column)
        forecast_date = forecast.forecasts[forecast.forecasts["horizon_months"] == horizon][
            "forecast_date"
        ].iloc[0]
        row = row.copy()
        row["forecast_date"] = forecast_date
        return self._snapshot_from_forecast_row(row, value_column, confidence)

    def classify(self, nowcast: NowcastResult, forecast: ForecastResult) -> RegimeView:
        """Return current, base-horizon, and stress-boundary regime labels."""

        base_horizon = self.policy.forecast.base_horizon_months
        strategic_horizon = self.policy.forecast.strategic_horizon_months
        fallback_horizon = self.policy.forecast.fallback_horizon_months

        diagnostics = forecast.diagnostics
        confidence_penalty = float(diagnostics["rmse"].fillna(0.75).mean())
        forecast_confidence = float(max(0.05, min(1.0, nowcast.current.confidence / (1.0 + confidence_penalty))))

        expected = self._snapshot_for(forecast, base_horizon, "forecast", forecast_confidence)
        strategic = self._snapshot_for(forecast, strategic_horizon, "forecast", forecast_confidence)
        fallback = self._snapshot_for(forecast, fallback_horizon, "forecast", forecast_confidence)
        downside = self._snapshot_for(forecast, base_horizon, "lower", forecast_confidence * 0.80)
        upside = self._snapshot_for(forecast, base_horizon, "upper", forecast_confidence * 0.80)

        base_row = _row_for_horizon(forecast, base_horizon, "forecast")
        probabilities = pd.DataFrame(
            [
                {
                    "horizon_months": base_horizon,
                    "macro_regime": expected.macro_regime,
                    "liquidity_regime": expected.liquidity_regime,
                    "macro_probability": _macro_probability(base_row, expected.macro_regime),
                    "liquidity_probability": _liquidity_probability(
                        expected.liquidity_score, expected.liquidity_regime
                    ),
                    "confidence": forecast_confidence,
                }
            ]
        )
        transition = (
            f"{nowcast.current.macro_regime}+{nowcast.current.liquidity_regime}"
            f" -> {expected.macro_regime}+{expected.liquidity_regime}"
        )
        return RegimeView(
            current=nowcast.current,
            expected=expected,
            strategic=strategic,
            fallback=fallback,
            downside=downside,
            upside=upside,
            probabilities=probabilities,
            transition=transition,
            confidence=forecast_confidence,
            methodology=self.policy.methodology_summary(),
        )

