"""Backward-compatible bridge to the new backtesting package."""

from ..backtesting.engine_static import BackTestingStrategy, Backtester
from ..backtesting.results import (
    BacktestConfig,
    BacktestResult,
    BacktestStrategyResult,
    StrategyAllocation,
)
from ..backtesting.strategies import (
    AllocationStrategy,
    MeanVarianceStrategy,
    PostModernStrategy,
)

__all__ = [
    "AllocationStrategy",
    "BacktestConfig",
    "BacktestResult",
    "BacktestStrategyResult",
    "Backtester",
    "BackTestingStrategy",
    "MeanVarianceStrategy",
    "PostModernStrategy",
    "StrategyAllocation",
]
