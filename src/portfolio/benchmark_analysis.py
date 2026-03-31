"""Portfolio benchmark analysis aliases over the current analysis layer."""

from ..asset_allocation.PortfolioElementaryAnalysis import PortfolioElementaryAnalysis

PortfolioBenchmarkAnalysis = PortfolioElementaryAnalysis

__all__ = [
    "PortfolioBenchmarkAnalysis",
    "PortfolioElementaryAnalysis",
]
