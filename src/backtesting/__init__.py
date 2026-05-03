"""Public API for portfolio backtesting.

The package exposes two engines:

`Backtester`
    Static engine that estimates weights once and simulates a fixed allocation.
`DynamicBacktester`
    Dynamic engine that recalculates weights on scheduled rebalance dates.

It also exports configuration/result dataclasses and strategy adapters so users
can build complete backtesting workflows from the package root.
"""

from .engine_dynamic import DynamicBacktestEngine, DynamicBacktester
from .engine_static import BackTestingStrategy, Backtester, StaticBacktestEngine
from .results import (
    BacktestConfig,
    BacktestResult,
    BacktestStrategyResult,
    DynamicBacktestResult,
    DynamicBacktestStrategyResult,
    RebalanceConfig,
    StrategyAllocation,
)
from .strategies import AllocationStrategy, MeanVarianceStrategy, PostModernStrategy

__all__ = [
    "AllocationStrategy",
    "BacktestConfig",
    "BacktestResult",
    "BacktestStrategyResult",
    "Backtester",
    "BackTestingStrategy",
    "DynamicBacktestEngine",
    "DynamicBacktester",
    "DynamicBacktestResult",
    "DynamicBacktestStrategyResult",
    "MeanVarianceStrategy",
    "PostModernStrategy",
    "RebalanceConfig",
    "StaticBacktestEngine",
    "StrategyAllocation",
]
