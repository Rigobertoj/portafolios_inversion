from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import yfinance as yf

from .AssetsResearch import AssetsResearch


@dataclass
class PortfolioElementaryMetrics(AssetsResearch):
    
    weight : Iterable[float]
    
    
    def __post_init__(self):
        
        return super().__post_init__()