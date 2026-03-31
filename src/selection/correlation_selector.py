"""Selection layer entrypoint for correlation-based portfolio filtering."""

from ..security_selection.correlation_portfolio_selector import (
    CorrelationPortfolioSelector,
)

CorrelationSelector = CorrelationPortfolioSelector

__all__ = [
    "CorrelationPortfolioSelector",
    "CorrelationSelector",
]
