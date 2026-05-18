"""Scenario construction from classified macro/liquidity regimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from .classification import RegimeView
from .forecast import ForecastScenario, RegimeForecast
from .policy import MacroLiquidityPolicy


def _normalize_probabilities(scenarios: Iterable[ForecastScenario]) -> tuple[ForecastScenario, ...]:
    scenario_tuple = tuple(scenarios)
    total = sum(max(scenario.probability, 0.0) for scenario in scenario_tuple)
    if total <= 0:
        equal = 1.0 / len(scenario_tuple)
        return tuple(
            ForecastScenario(
                name=scenario.name,
                macro_regime=scenario.macro_regime,
                liquidity_regime=scenario.liquidity_regime,
                probability=equal,
                horizon_months=scenario.horizon_months,
                confidence=scenario.confidence,
                rationale=scenario.rationale,
                signals=scenario.signals,
            )
            for scenario in scenario_tuple
        )
    return tuple(
        ForecastScenario(
            name=scenario.name,
            macro_regime=scenario.macro_regime,
            liquidity_regime=scenario.liquidity_regime,
            probability=max(scenario.probability, 0.0) / total,
            horizon_months=scenario.horizon_months,
            confidence=scenario.confidence,
            rationale=scenario.rationale,
            signals=scenario.signals,
        )
        for scenario in scenario_tuple
    )


@dataclass(frozen=True)
class ScenarioOverride:
    """Human override for an auditable scenario adjustment."""

    scenario_name: str
    probability: Optional[float] = None
    macro_regime: Optional[str] = None
    liquidity_regime: Optional[str] = None
    rationale: str = ""
    approved_by: str = "research_owner"


@dataclass(frozen=True)
class ScenarioBuildResult:
    """Scenario forecast plus audit trail."""

    forecast: RegimeForecast
    scenario_table: pd.DataFrame
    override_log: pd.DataFrame
    methodology: dict[str, object]


class ScenarioEngine:
    """Convert model regimes into actionable base/upside/downside scenarios."""

    def __init__(self, policy: Optional[MacroLiquidityPolicy] = None) -> None:
        self.policy = policy or MacroLiquidityPolicy()

    def _base_probabilities(self, regime_view: RegimeView) -> tuple[float, float, float]:
        confidence = float(regime_view.confidence)
        base = self.policy.scenario.base_floor + self.policy.scenario.base_confidence_slope * confidence
        downside = self.policy.scenario.downside_base - self.policy.scenario.downside_confidence_penalty * confidence
        if regime_view.current.liquidity_regime in {"restrictiva", "estresada"}:
            downside += self.policy.scenario.liquidity_stress_penalty
        upside = max(0.05, 1.0 - base - downside)
        total = base + downside + upside
        return base / total, upside / total, downside / total

    @staticmethod
    def _apply_overrides(
        scenarios: tuple[ForecastScenario, ...],
        overrides: tuple[ScenarioOverride, ...],
    ) -> tuple[ForecastScenario, ...]:
        override_by_name = {override.scenario_name: override for override in overrides}
        adjusted = []
        for scenario in scenarios:
            override = override_by_name.get(scenario.name)
            if override is None:
                adjusted.append(scenario)
                continue
            adjusted.append(
                ForecastScenario(
                    name=scenario.name,
                    macro_regime=override.macro_regime or scenario.macro_regime,
                    liquidity_regime=override.liquidity_regime or scenario.liquidity_regime,
                    probability=(
                        scenario.probability
                        if override.probability is None
                        else float(override.probability)
                    ),
                    horizon_months=scenario.horizon_months,
                    confidence=scenario.confidence,
                    rationale=(
                        f"{scenario.rationale} Override: {override.rationale}".strip()
                    ),
                    signals=scenario.signals,
                )
            )
        return tuple(adjusted)

    @staticmethod
    def _override_log(overrides: tuple[ScenarioOverride, ...]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "scenario_name": override.scenario_name,
                    "probability": override.probability,
                    "macro_regime": override.macro_regime,
                    "liquidity_regime": override.liquidity_regime,
                    "rationale": override.rationale,
                    "approved_by": override.approved_by,
                }
                for override in overrides
            ],
            columns=[
                "scenario_name",
                "probability",
                "macro_regime",
                "liquidity_regime",
                "rationale",
                "approved_by",
            ],
        )

    def build(
        self,
        regime_view: RegimeView,
        overrides: Optional[Iterable[ScenarioOverride]] = None,
    ) -> ScenarioBuildResult:
        """Return base/upside/downside scenarios with optional human overrides."""

        base_p, upside_p, downside_p = self._base_probabilities(regime_view)
        horizon = self.policy.forecast.base_horizon_months
        scenarios = (
            ForecastScenario(
                name="base",
                macro_regime=regime_view.expected.macro_regime,
                liquidity_regime=regime_view.expected.liquidity_regime,
                probability=base_p,
                horizon_months=horizon,
                confidence=regime_view.confidence,
                rationale="Highest-probability model-implied regime at the base rebalance horizon.",
                signals={
                    "macro_score": regime_view.expected.macro_score,
                    "liquidity_score": regime_view.expected.liquidity_score,
                },
            ),
            ForecastScenario(
                name="upside",
                macro_regime=regime_view.upside.macro_regime,
                liquidity_regime=regime_view.upside.liquidity_regime,
                probability=upside_p,
                horizon_months=horizon,
                confidence=regime_view.confidence * 0.80,
                rationale="Upper-bound forecast path with better macro/liquidity mix.",
                signals={
                    "macro_score": regime_view.upside.macro_score,
                    "liquidity_score": regime_view.upside.liquidity_score,
                },
            ),
            ForecastScenario(
                name="downside",
                macro_regime=regime_view.downside.macro_regime,
                liquidity_regime=regime_view.downside.liquidity_regime,
                probability=downside_p,
                horizon_months=horizon,
                confidence=regime_view.confidence * 0.80,
                rationale="Lower-bound forecast path with weaker macro/liquidity mix.",
                signals={
                    "macro_score": regime_view.downside.macro_score,
                    "liquidity_score": regime_view.downside.liquidity_score,
                },
            ),
        )
        override_tuple = tuple(overrides or ())
        scenarios = self._apply_overrides(scenarios, override_tuple)
        scenarios = _normalize_probabilities(scenarios)
        forecast = RegimeForecast(scenarios=scenarios, method="model_ar1_plus_house_view")
        return ScenarioBuildResult(
            forecast=forecast,
            scenario_table=forecast.scenario_table(),
            override_log=self._override_log(override_tuple),
            methodology=self.policy.methodology_summary(),
        )

