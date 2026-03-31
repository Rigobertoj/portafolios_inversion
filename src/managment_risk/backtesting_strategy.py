"""Snake-case compatibility wrapper for the legacy backtesting bridge."""

from .BackTestingStrategy import (
    AllocationStrategy,
    BacktestConfig,
    BacktestResult,
    BacktestStrategyResult,
    Backtester,
    BackTestingStrategy,
    MeanVarianceStrategy,
    PostModernStrategy,
    StrategyAllocation,
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
