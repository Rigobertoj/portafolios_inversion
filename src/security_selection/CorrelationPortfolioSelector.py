"""Compatibility wrapper for the migrated correlation selector."""

from __future__ import annotations

import sys

from ..selection.correlation_selector import CorrelationPortfolioSelector


_parent_package = sys.modules.get(__package__)
if _parent_package is not None:
    _parent_package.CorrelationPortfolioSelector = CorrelationPortfolioSelector


__all__ = [
    "CorrelationPortfolioSelector",
]
