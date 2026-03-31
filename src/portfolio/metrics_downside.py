"""Portfolio downside metrics aliases over the current post-modern layer."""

from ..asset_allocation.PortfolioPostModernMetrics import PortfolioPostModernMetrics

PortfolioDownsideMetrics = PortfolioPostModernMetrics

__all__ = [
    "PortfolioDownsideMetrics",
    "PortfolioPostModernMetrics",
]
