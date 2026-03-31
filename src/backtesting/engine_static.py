"""Static backtesting aliases over the current backtesting engine."""

from ..managment_risk.BackTestingStrategy import BackTestingStrategy, Backtester

StaticBacktestEngine = Backtester

__all__ = [
    "Backtester",
    "BackTestingStrategy",
    "StaticBacktestEngine",
]
