"""Client policy objects that stay separate from economic research."""

from .profiles import (
    ClientPolicy,
    ClientProfile,
    InvestmentMandate,
    mexican_moderate_aggressive_growth_policy,
)

__all__ = [
    "ClientPolicy",
    "ClientProfile",
    "InvestmentMandate",
    "mexican_moderate_aggressive_growth_policy",
]
