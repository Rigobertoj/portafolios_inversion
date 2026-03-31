"""Portfolio basic metrics aliases over the current metrics implementation."""

from ..asset_allocation.PortfolioElementaryMetrics import PortfolioElementaryMetrics

PortfolioBasicMetrics = PortfolioElementaryMetrics

__all__ = [
    "PortfolioBasicMetrics",
    "PortfolioElementaryMetrics",
]
