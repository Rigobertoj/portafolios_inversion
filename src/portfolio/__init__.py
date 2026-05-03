"""Public portfolio API built around the new portfolio package layout."""

from importlib import import_module
from typing import TYPE_CHECKING

from .benchmark_analysis import (
    PortfolioBenchmarkAnalysis,
)
from .metrics_basic import (
    PortfolioBasicMetrics,
)
from .metrics_downside import (
    PortfolioDownsideMetrics,
)
from .performance_analysis import PortfolioPerformanceAnalysis
from .portfolio import Portfolio

if TYPE_CHECKING:
    from .legacy_adapters import (
        PortfolioElementaryAnalysis,
        PortfolioElementaryMetrics,
        PortfolioPostModernMetrics,
    )

_LEGACY_EXPORTS = {
    "PortfolioElementaryAnalysis",
    "PortfolioElementaryMetrics",
    "PortfolioPostModernMetrics",
}

__all__ = [
    "Portfolio",
    "PortfolioBasicMetrics",
    "PortfolioBenchmarkAnalysis",
    "PortfolioDownsideMetrics",
    "PortfolioElementaryAnalysis",
    "PortfolioElementaryMetrics",
    "PortfolioPerformanceAnalysis",
    "PortfolioPostModernMetrics",
]


def __getattr__(name: str):
    if name in _LEGACY_EXPORTS:
        module = import_module(".legacy_adapters", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
