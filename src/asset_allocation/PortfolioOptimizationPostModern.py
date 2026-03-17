from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy.optimize import minimize

try:
    from .PortfolioElementaryAnalysis import PortfolioElementaryMetrics
    from .PortfolioPostModernMetrics import PortfolioPostModernMetrics
except ImportError:
    from PortfolioElementaryAnalysis import PortfolioElementaryMetrics
    from PortfolioPostModernMetrics import PortfolioPostModernMetrics


class PortfolioOptimizationPostMordern(PortfolioPostModernMetrics):
    def __init__(self, 
        tickers: Iterable[str],
        start: str,
        end: Optional[str],
        price_field: str = "Close",
        initial_weight: Iterable[float] = None):
        
        if initial_weight is None:
            initial_weight = np.ones(len(tickers)) / len(tickers)
        
        PortfolioElementaryMetrics(
            tickers=tickers, 
            start=start, 
            end=end, 
            weight=initial_weight, 
            price_field=price_field
            )


    def __eq__(self, value):
        pass

def _main_():
    return


if __name__ == "__main__":
    pd.DataFrame()
    pass