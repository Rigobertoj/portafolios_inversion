"""Backtesting strategy aliases over the current strategy implementation."""

from ..managment_risk.BackTestingStrategy import (
    AllocationStrategy,
    MeanVarianceStrategy,
    PostModernStrategy,
)

__all__ = [
    "AllocationStrategy",
    "MeanVarianceStrategy",
    "PostModernStrategy",
]
