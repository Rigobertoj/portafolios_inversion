"""Indicator catalog for U.S. macro and liquidity research."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping

Engine = Literal["macro", "liquidity"]
Transform = Literal["level", "diff", "pct_change", "yoy_change"]


@dataclass(frozen=True)
class EconomicSeriesSpec:
    """Metadata needed to request, transform, and score one economic series."""

    name: str
    engine: Engine
    block: str
    higher_is_better: bool
    provider: str = ""
    provider_code: str = ""
    frequency: str = ""
    transform: Transform = "level"
    periods: int = 1
    weight: float = 1.0
    required: bool = False
    description: str = ""
    provider_params: Mapping[str, Any] = field(default_factory=dict)


def default_us_macro_liquidity_catalog() -> tuple[EconomicSeriesSpec, ...]:
    """Return a first-pass catalog aligned with the methodology document."""

    return (
        EconomicSeriesSpec("fred_real_gdp", "macro", "growth", True, "FRED", "GDPC1", "Q", "yoy_change", 4, 1.2, True),
        EconomicSeriesSpec("fred_industrial_production", "macro", "growth", True, "FRED", "INDPRO", "M", "yoy_change", 12),
        EconomicSeriesSpec("fred_retail_sales", "macro", "growth", True, "FRED", "RSXFS", "M", "yoy_change", 12),
        EconomicSeriesSpec("fred_real_pce", "macro", "consumer", True, "FRED", "PCEC96", "M", "yoy_change", 12),
        EconomicSeriesSpec("fred_personal_income", "macro", "consumer", True, "FRED", "PI", "M", "yoy_change", 12),
        EconomicSeriesSpec("fred_saving_rate", "macro", "consumer", True, "FRED", "PSAVERT", "M"),
        EconomicSeriesSpec("fred_cpi", "macro", "inflation", False, "FRED", "CPIAUCSL", "M", "yoy_change", 12, 1.2, True),
        EconomicSeriesSpec("fred_core_cpi", "macro", "inflation", False, "FRED", "CPILFESL", "M", "yoy_change", 12),
        EconomicSeriesSpec("fred_core_pce", "macro", "inflation", False, "FRED", "PCEPILFE", "M", "yoy_change", 12),
        EconomicSeriesSpec("fred_unemployment_rate", "macro", "labor", False, "FRED", "UNRATE", "M", "level", 1, 1.2, True),
        EconomicSeriesSpec("fred_nonfarm_payrolls", "macro", "labor", True, "FRED", "PAYEMS", "M", "yoy_change", 12),
        EconomicSeriesSpec("fred_initial_claims", "macro", "labor", False, "FRED", "ICSA", "W"),
        EconomicSeriesSpec("fred_fed_funds_rate", "macro", "policy", False, "FRED", "FEDFUNDS", "M"),
        EconomicSeriesSpec("fred_two_year_yield", "macro", "policy", False, "FRED", "DGS2", "D"),
        EconomicSeriesSpec("fred_ten_year_yield", "macro", "policy", False, "FRED", "DGS10", "D"),
        EconomicSeriesSpec("fred_yield_curve_10y_3m", "macro", "policy", True, "FRED", "T10Y3M", "D"),
        EconomicSeriesSpec("fred_fed_assets", "liquidity", "fed_liquidity", True, "FRED", "WALCL", "W", "yoy_change", 52),
        EconomicSeriesSpec("fred_reserve_balances", "liquidity", "fed_liquidity", True, "FRED", "WRESBAL", "W", "yoy_change", 52, 1.2, True),
        EconomicSeriesSpec("fred_reverse_repo", "liquidity", "fed_liquidity", False, "FRED", "RRPONTSYD", "D"),
        EconomicSeriesSpec("fred_tga", "liquidity", "treasury_liquidity", False, "FRED", "WTREGEN", "W"),
        EconomicSeriesSpec("fred_sofr", "liquidity", "funding", False, "FRED", "SOFR", "D"),
        EconomicSeriesSpec("fred_nfci", "liquidity", "financial_conditions", False, "FRED", "NFCI", "W", "level", 1, 1.2, True),
        EconomicSeriesSpec("fred_kcfsi", "liquidity", "financial_conditions", False, "FRED", "KCFSI", "M"),
        EconomicSeriesSpec("fred_hy_oas", "liquidity", "financial_conditions", False, "FRED", "BAMLH0A0HYM2", "D"),
        EconomicSeriesSpec("fred_bank_credit", "liquidity", "credit", True, "FRED", "TOTBKCR", "W", "yoy_change", 52),
        EconomicSeriesSpec("fred_ci_loans", "liquidity", "credit", True, "FRED", "BUSLOANS", "W", "yoy_change", 52),
        EconomicSeriesSpec("fred_broad_dollar", "liquidity", "global_dollar_liquidity", False, "FRED", "DTWEXBGS", "D"),
        EconomicSeriesSpec("banxico_usd_mxn", "liquidity", "client_currency", False, "BANXICO", "SF63528", "D"),
    )



def specs_by_name(specs: Iterable[EconomicSeriesSpec]) -> dict[str, EconomicSeriesSpec]:
    """Return a lookup keyed by series name."""

    return {spec.name: spec for spec in specs}
