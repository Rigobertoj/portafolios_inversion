"""Selection layer entrypoint for correlation-based portfolio filtering."""

from ..security_selection.CorrelationPortfolioSelector import CorrelationPortfolioSelector

CorrelationSelector = CorrelationPortfolioSelector

__all__ = [
    "CorrelationPortfolioSelector",
    "CorrelationSelector",
]
