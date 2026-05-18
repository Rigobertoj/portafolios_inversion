"""Deprecated compatibility wrapper for asset selection research.

The implementation lives in `src.selection.assets_research`. This module keeps
older imports such as `from src.research import AssetsResearch` working while
the project moves asset-level market data helpers into the selection layer.
"""

from __future__ import annotations

from ..selection.assets_research import AssetsResearch, yf

__all__ = [
    "AssetsResearch",
    "yf",
]
