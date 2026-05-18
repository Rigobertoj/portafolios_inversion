"""Reusable client profile and investment mandate definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ClientProfile:
    """Stable investor characteristics independent from market research."""

    client_type: str
    base_currency: str
    horizon_years: float
    risk_profile: str
    max_drawdown: Optional[float] = None
    liquidity_need: str = "normal"
    contribution_style: str = "lump_sum"


@dataclass(frozen=True)
class InvestmentMandate:
    """Investable-universe and policy constraints from the IPS."""

    target_return_range: Optional[tuple[float, float]] = None
    eligible_assets: tuple[str, ...] = ()
    excluded_sectors: tuple[str, ...] = ()
    eligible_regions: tuple[str, ...] = ()
    benchmark_currency: str = ""
    notes: str = ""


@dataclass(frozen=True)
class ClientPolicy:
    """Client profile plus investable mandate."""

    profile: ClientProfile
    mandate: InvestmentMandate


def mexican_moderate_aggressive_growth_policy() -> ClientPolicy:
    """Return the IPS-aligned policy for the current target client."""

    return ClientPolicy(
        profile=ClientProfile(
            client_type="individual_mexican_employee",
            base_currency="MXN",
            horizon_years=10.0,
            risk_profile="moderate_aggressive",
            max_drawdown=0.20,
            liquidity_need="reasonable",
            contribution_style="periodic",
        ),
        mandate=InvestmentMandate(
            target_return_range=(0.13, 0.18),
            eligible_assets=(
                "public_equity",
                "etf",
                "government_bonds",
                "investment_grade_credit",
                "reit",
                "listed_infrastructure",
                "cash_like",
            ),
            excluded_sectors=("tobacco", "weapons", "gambling", "fossil_fuels"),
            eligible_regions=("US", "MX", "EM"),
            benchmark_currency="MXN",
            notes="Periodic-contribution growth mandate with reasonable liquidity.",
        ),
    )
