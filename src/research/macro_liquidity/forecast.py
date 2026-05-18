"""In-house regime forecast objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class ForecastScenario:
    """One expected macro/liquidity scenario for a forward horizon."""

    name: str
    macro_regime: str
    liquidity_regime: str
    probability: float
    horizon_months: int = 6
    confidence: float = 0.50
    rationale: str = ""
    signals: Mapping[str, float] = field(default_factory=dict)

    def normalized_probability(self, total: float) -> float:
        """Return probability normalized by scenario total."""

        if total <= 0:
            return 0.0
        return max(float(self.probability), 0.0) / total


@dataclass(frozen=True)
class RegimeForecast:
    """Set of in-house forecast scenarios."""

    scenarios: tuple[ForecastScenario, ...]
    method: str = "house_view"
    as_of: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.today().normalize())

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ValueError("RegimeForecast requires at least one scenario.")

    @property
    def base_case(self) -> ForecastScenario:
        """Return the highest-probability scenario."""

        return max(self.scenarios, key=lambda scenario: scenario.probability)

    @property
    def expected_macro_regime(self) -> str:
        """Return the base-case expected macro regime."""

        return self.base_case.macro_regime

    @property
    def expected_liquidity_regime(self) -> str:
        """Return the base-case expected liquidity regime."""

        return self.base_case.liquidity_regime

    @property
    def horizon_months(self) -> int:
        """Return the base-case forecast horizon."""

        return self.base_case.horizon_months

    @property
    def confidence(self) -> float:
        """Return probability-weighted forecast confidence."""

        total = sum(max(scenario.probability, 0.0) for scenario in self.scenarios)
        if total <= 0:
            return 0.0
        return float(
            sum(
                scenario.confidence * scenario.normalized_probability(total)
                for scenario in self.scenarios
            )
        )

    def scenario_table(self) -> pd.DataFrame:
        """Return forecast scenarios as a DataFrame."""

        return pd.DataFrame(
            [
                {
                    "name": scenario.name,
                    "macro_regime": scenario.macro_regime,
                    "liquidity_regime": scenario.liquidity_regime,
                    "probability": scenario.probability,
                    "horizon_months": scenario.horizon_months,
                    "confidence": scenario.confidence,
                    "rationale": scenario.rationale,
                }
                for scenario in self.scenarios
            ]
        )
