"""Map macro/liquidity research and client policy into selection inputs."""

from __future__ import annotations

from .selection_context import SelectionContext
from ..client_policy.profiles import ClientPolicy
from ..research.macro_liquidity.asset_view import AssetClassView
from ..research.macro_liquidity.forecast import RegimeForecast
from ..research.macro_liquidity.regimes import CurrentRegimeSnapshot


def _score_tilts(styles: tuple[str, ...], risk_controls: tuple[str, ...]) -> dict[str, float]:
    tilts = {
        "valuation": 0.0,
        "growth": 0.0,
        "profitability": 0.0,
        "cash_flow": 0.0,
        "leverage": 0.0,
        "liquidity": 0.0,
        "volatility": 0.0,
    }
    if "quality" in styles or "profitability" in styles:
        tilts["profitability"] += 0.15
        tilts["cash_flow"] += 0.10
    if "free_cash_flow" in styles:
        tilts["cash_flow"] += 0.15
    if "low_leverage" in styles or "balance_sheet_quality" in styles:
        tilts["leverage"] += 0.15
    if "growth" in styles or "quality_growth" in styles:
        tilts["growth"] += 0.10
    if "value" in styles:
        tilts["valuation"] += 0.10
    if "low_volatility" in styles or "minimum_volatility" in styles:
        tilts["volatility"] += 0.15
    if "liquidity_filter" in risk_controls or "preserve_liquidity" in risk_controls:
        tilts["liquidity"] += 0.10
    return {key: value for key, value in tilts.items() if value != 0.0}


def _client_risk_controls(policy: ClientPolicy) -> tuple[str, ...]:
    controls = []
    if policy.profile.base_currency.upper() != "USD":
        controls.append("currency_risk_review")
    if policy.profile.max_drawdown is not None:
        controls.append(f"max_drawdown_{int(policy.profile.max_drawdown * 100)}pct")
    if policy.profile.contribution_style == "periodic":
        controls.append("periodic_contribution_rebalancing")
    if policy.profile.liquidity_need in {"reasonable", "high"}:
        controls.append("market_liquidity_requirement")
    return tuple(controls)


def build_selection_context(
    current: CurrentRegimeSnapshot,
    forecast: RegimeForecast,
    asset_view: AssetClassView,
    policy: ClientPolicy,
) -> SelectionContext:
    """Build the final object consumed by security or fund selection."""

    risk_controls = tuple(
        dict.fromkeys(asset_view.risk_controls + _client_risk_controls(policy))
    )
    score_tilts = _score_tilts(asset_view.preferred_styles, risk_controls)
    rationale = (
        f"Current regime is {current.label()}; expected base case is "
        f"{forecast.expected_macro_regime} + {forecast.expected_liquidity_regime}. "
        f"Client overlay uses {policy.profile.base_currency} as base currency, "
        f"{policy.profile.risk_profile} risk, and IPS exclusions."
    )
    return SelectionContext(
        current_macro_regime=current.macro_regime,
        current_liquidity_regime=current.liquidity_regime,
        expected_macro_regime=forecast.expected_macro_regime,
        expected_liquidity_regime=forecast.expected_liquidity_regime,
        base_currency=policy.profile.base_currency,
        risk_profile=policy.profile.risk_profile,
        horizon_years=policy.profile.horizon_years,
        asset_class_overweights=asset_view.overweights,
        asset_class_underweights=asset_view.underweights,
        preferred_styles=asset_view.preferred_styles,
        excluded_sectors=policy.mandate.excluded_sectors,
        eligible_assets=policy.mandate.eligible_assets,
        risk_controls=risk_controls,
        score_tilts=score_tilts,
        rationale=rationale,
    )
