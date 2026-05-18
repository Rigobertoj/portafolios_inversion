"""Map current and expected regimes into an asset-class view."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .forecast import RegimeForecast
from .regimes import CurrentRegimeSnapshot


def _macro_bucket(regime: str) -> str:
    mapping = {
        "expansion_desinflacionaria": "growth_low_inflation",
        "sobrecalentamiento": "growth_high_inflation",
        "desaceleracion_ordenada": "slowdown_low_inflation",
        "estanflacion": "slowdown_high_inflation",
        "recesion_desinflacionaria": "recession_disinflation",
        "reaceleracion_inflacionaria": "reacceleration_inflation",
    }
    return mapping.get(regime, "slowdown_low_inflation")


def _liquidity_bucket(regime: str) -> str:
    if regime in {"expansiva", "moderadamente_expansiva"}:
        return "expansive"
    if regime == "neutral":
        return "neutral"
    if regime == "estresada":
        return "stressed"
    return "restrictive"


_STANCE_MATRIX = {
    ("growth_low_inflation", "expansive"): "risk_on_amplio",
    ("growth_low_inflation", "neutral"): "risk_on_selectivo",
    ("growth_low_inflation", "restrictive"): "quality_equity",
    ("growth_low_inflation", "stressed"): "cuidado_con_valuacion",
    ("growth_high_inflation", "expansive"): "commodities_value",
    ("growth_high_inflation", "neutral"): "value_ciclicos",
    ("growth_high_inflation", "restrictive"): "short_duration",
    ("growth_high_inflation", "stressed"): "alta_volatilidad",
    ("slowdown_low_inflation", "expansive"): "growth_quality_bonos",
    ("slowdown_low_inflation", "neutral"): "bonos_defensivos",
    ("slowdown_low_inflation", "restrictive"): "quality_cash",
    ("slowdown_low_inflation", "stressed"): "treasuries_cash",
    ("slowdown_high_inflation", "expansive"): "oro_energia",
    ("slowdown_high_inflation", "neutral"): "defensivos",
    ("slowdown_high_inflation", "restrictive"): "cash_tbills",
    ("slowdown_high_inflation", "stressed"): "maxima_cautela",
    ("recession_disinflation", "expansive"): "duration_larga",
    ("recession_disinflation", "neutral"): "treasuries",
    ("recession_disinflation", "restrictive"): "treasuries_cash",
    ("recession_disinflation", "stressed"): "defensa_total",
    ("reacceleration_inflation", "expansive"): "ciclicos_commodities",
    ("reacceleration_inflation", "neutral"): "value",
    ("reacceleration_inflation", "restrictive"): "evitar_duration_larga",
    ("reacceleration_inflation", "stressed"): "riesgo_alto",
}


_STANCE_DETAILS = {
    "risk_on_amplio": {
        "overweights": ("equity", "growth_equity", "small_caps", "cyclicals"),
        "underweights": ("cash",),
        "styles": ("growth", "high_beta", "cyclical_recovery"),
        "risk_controls": ("avoid_excess_concentration",),
    },
    "risk_on_selectivo": {
        "overweights": ("quality_growth", "profitable_cyclicals"),
        "underweights": ("unprofitable_growth",),
        "styles": ("quality_growth", "profitability"),
        "risk_controls": ("valuation_discipline",),
    },
    "quality_equity": {
        "overweights": ("quality_equity", "large_caps"),
        "underweights": ("speculative_growth", "weak_balance_sheets"),
        "styles": ("quality", "profitability", "low_leverage"),
        "risk_controls": ("liquidity_filter", "refinancing_risk_filter"),
    },
    "quality_cash": {
        "overweights": ("quality_equity", "cash_like", "short_duration_bonds"),
        "underweights": ("high_beta_equity", "high_yield", "illiquid_credit"),
        "styles": ("quality", "profitability", "free_cash_flow", "low_leverage"),
        "risk_controls": ("liquidity_filter", "drawdown_budget"),
    },
    "bonos_defensivos": {
        "overweights": ("defensive_equity", "investment_grade", "intermediate_bonds"),
        "underweights": ("high_beta_equity",),
        "styles": ("low_volatility", "dividend_quality", "profitability"),
        "risk_controls": ("duration_review",),
    },
    "cash_tbills": {
        "overweights": ("cash_like", "short_duration_bonds"),
        "underweights": ("long_duration_bonds", "high_yield", "high_beta_equity"),
        "styles": ("balance_sheet_quality", "low_leverage"),
        "risk_controls": ("preserve_liquidity",),
    },
    "defensa_total": {
        "overweights": ("cash_like", "treasuries", "defensive_equity"),
        "underweights": ("high_yield", "small_caps", "cyclicals"),
        "styles": ("minimum_volatility", "quality", "low_leverage"),
        "risk_controls": ("maximum_drawdown_control",),
    },
    "treasuries_cash": {
        "overweights": ("treasuries", "cash_like", "quality_equity"),
        "underweights": ("high_yield", "illiquid_credit"),
        "styles": ("quality", "defensive"),
        "risk_controls": ("liquidity_filter",),
    },
    "short_duration": {
        "overweights": ("short_duration_bonds", "cash_like", "value_equity"),
        "underweights": ("long_duration_bonds", "expensive_growth"),
        "styles": ("value", "pricing_power", "low_duration_equity"),
        "risk_controls": ("inflation_sensitivity_review",),
    },
    "commodities_value": {
        "overweights": ("value_equity", "materials", "energy_substitutes"),
        "underweights": ("long_duration_bonds",),
        "styles": ("value", "pricing_power"),
        "risk_controls": ("ips_exclusion_review",),
    },
    "defensivos": {
        "overweights": ("defensive_equity", "cash_like"),
        "underweights": ("cyclicals", "high_yield"),
        "styles": ("low_volatility", "quality"),
        "risk_controls": ("liquidity_filter",),
    },
}


def _stance_for(macro_regime: str, liquidity_regime: str) -> str:
    key = (_macro_bucket(macro_regime), _liquidity_bucket(liquidity_regime))
    return _STANCE_MATRIX.get(key, "quality_cash")


@dataclass(frozen=True)
class AssetClassView:
    """Actionable top-down view before client policy overlay."""

    current_stance: str
    expected_stance: str
    final_stance: str
    current_weight: float
    expected_weight: float
    overweights: tuple[str, ...]
    underweights: tuple[str, ...]
    preferred_styles: tuple[str, ...]
    risk_controls: tuple[str, ...]
    rationale: str

    def to_frame(self) -> pd.DataFrame:
        """Return a compact table representation."""

        return pd.DataFrame(
            [
                {
                    "current_stance": self.current_stance,
                    "expected_stance": self.expected_stance,
                    "final_stance": self.final_stance,
                    "current_weight": self.current_weight,
                    "expected_weight": self.expected_weight,
                    "overweights": ", ".join(self.overweights),
                    "underweights": ", ".join(self.underweights),
                    "preferred_styles": ", ".join(self.preferred_styles),
                    "risk_controls": ", ".join(self.risk_controls),
                    "rationale": self.rationale,
                }
            ]
        )


def _details_for(stance: str) -> dict[str, tuple[str, ...]]:
    return _STANCE_DETAILS.get(
        stance,
        {
            "overweights": ("quality_equity", "cash_like"),
            "underweights": ("high_beta_equity",),
            "styles": ("quality", "low_leverage"),
            "risk_controls": ("liquidity_filter",),
        },
    )


def build_asset_class_view(
    current: CurrentRegimeSnapshot,
    forecast: RegimeForecast,
    current_weight: float = 0.40,
    expected_weight: float = 0.60,
    liquidity_stress_veto: bool = True,
) -> AssetClassView:
    """Combine current and expected regimes into a top-down asset view."""

    current_stance = _stance_for(current.macro_regime, current.liquidity_regime)
    expected_stance = _stance_for(
        forecast.expected_macro_regime,
        forecast.expected_liquidity_regime,
    )
    if liquidity_stress_veto and current.liquidity_regime == "estresada":
        final_stance = current_stance
        rationale = "Current liquidity stress overrides the forward-looking base case."
    else:
        final_stance = expected_stance if expected_weight >= current_weight else current_stance
        rationale = "Forward-looking regime receives the larger tactical weight."

    details = _details_for(final_stance)
    return AssetClassView(
        current_stance=current_stance,
        expected_stance=expected_stance,
        final_stance=final_stance,
        current_weight=float(current_weight),
        expected_weight=float(expected_weight),
        overweights=tuple(details["overweights"]),
        underweights=tuple(details["underweights"]),
        preferred_styles=tuple(details["styles"]),
        risk_controls=tuple(details["risk_controls"]),
        rationale=rationale,
    )
