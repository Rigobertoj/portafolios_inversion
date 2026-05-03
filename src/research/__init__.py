"""Public API for market and time-series research.

The package exports market-data research helpers for asset prices and returns,
plus ARIMA modeling utilities for compact time-series reports.
"""

from .arima_research import ARIMAResearch
from .assets_research import AssetsResearch

__all__ = [
    "ARIMAResearch",
    "AssetsResearch",
]
