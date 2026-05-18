"""Selection context produced by research and client-policy overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class SelectionContext:
    """Input object that tells `selection` how to tilt the investable universe."""

    current_macro_regime: str
    current_liquidity_regime: str
    expected_macro_regime: str
    expected_liquidity_regime: str
    base_currency: str
    risk_profile: str
    horizon_years: float
    asset_class_overweights: tuple[str, ...]
    asset_class_underweights: tuple[str, ...]
    preferred_styles: tuple[str, ...]
    excluded_sectors: tuple[str, ...]
    eligible_assets: tuple[str, ...]
    risk_controls: tuple[str, ...]
    score_tilts: Mapping[str, float]
    rationale: str

    def as_dict(self) -> dict[str, object]:
        """Return a plain dictionary for notebooks and reports."""

        return {
            "current_macro_regime": self.current_macro_regime,
            "current_liquidity_regime": self.current_liquidity_regime,
            "expected_macro_regime": self.expected_macro_regime,
            "expected_liquidity_regime": self.expected_liquidity_regime,
            "base_currency": self.base_currency,
            "risk_profile": self.risk_profile,
            "horizon_years": self.horizon_years,
            "asset_class_overweights": self.asset_class_overweights,
            "asset_class_underweights": self.asset_class_underweights,
            "preferred_styles": self.preferred_styles,
            "excluded_sectors": self.excluded_sectors,
            "eligible_assets": self.eligible_assets,
            "risk_controls": self.risk_controls,
            "score_tilts": dict(self.score_tilts),
            "rationale": self.rationale,
        }

    def to_frame(self) -> pd.DataFrame:
        """Return a compact one-row DataFrame for notebook display."""

        row = self.as_dict()
        for key in (
            "asset_class_overweights",
            "asset_class_underweights",
            "preferred_styles",
            "excluded_sectors",
            "eligible_assets",
            "risk_controls",
        ):
            row[key] = ", ".join(row[key])
        row["score_tilts"] = ", ".join(
            f"{name}: {tilt:+.2f}" for name, tilt in self.score_tilts.items()
        )
        return pd.DataFrame([row])
