"""Deprecated compatibility wrapper for selection time-series research.

The implementation lives in `src.selection.arima_research`. This module keeps
older imports such as `from src.research import ARIMAResearch` working while
the project moves asset-selection forecasting helpers into the selection layer.
"""

from __future__ import annotations

from ..selection.arima_research import ARIMAResearch

__all__ = [
    "ARIMAResearch",
]
