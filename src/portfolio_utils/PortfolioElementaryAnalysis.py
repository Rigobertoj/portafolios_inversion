from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Iterable, Optional

import numpy as np

try:
    from .AssetsResearch import AssetsResearch
    from .PortfolioElementaryMetrics import PortfolioElementaryMetrics
except ImportError:
    from AssetsResearch import AssetsResearch
    from PortfolioElementaryAnalysis import PortfolioElementaryMetrics
    
    
@dataclass
class PortfolioElementaryAnalysis(PortfolioElementaryMetrics):
    def __post_init__(self, tickers, start, end, price_field, weight):
        return super().__post_init__(tickers, start, end, price_field, weight)
    
def _main_():
    return

if __name__ == "__main__":
    _main_()