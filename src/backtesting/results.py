"""Dataclasses used by the composition-based backtesting layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    """
    Global configuration shared by the backtesting engine.

    The validation rules intentionally match the previous implementation so
    notebooks and tests keep the same behavior while the engine is migrated to
    the new package layout.
    """

    tickers: Sequence[str]
    initial_capital: float
    optimization_start: str
    backtest_start: str
    end: Optional[str] = None
    price_field: str = "Close"
    benchmark_ticker: Optional[str] = None
    benchmark_label: Optional[str] = None
    reuse_optimization_window: bool = False
    risk_free_rate: float = 0.0
    trading_days: int = 252

    def __post_init__(self) -> None:
        tickers = list(self.tickers)
        if not tickers:
            raise ValueError("tickers must contain at least one symbol.")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be greater than zero.")
        if self.trading_days <= 0:
            raise ValueError("trading_days must be greater than zero.")

        optimization_start = pd.Timestamp(self.optimization_start)
        backtest_start = pd.Timestamp(self.backtest_start)
        if self.reuse_optimization_window:
            if backtest_start != optimization_start:
                raise ValueError(
                    "backtest_start must match optimization_start when "
                    "reuse_optimization_window is enabled."
                )
        elif backtest_start <= optimization_start:
            raise ValueError("backtest_start must be later than optimization_start.")

        if self.end is not None:
            end = pd.Timestamp(self.end)
            minimum_end = (
                optimization_start
                if self.reuse_optimization_window
                else backtest_start
            )
            if end <= minimum_end:
                message = (
                    "end must be later than optimization_start when "
                    "reuse_optimization_window is enabled."
                    if self.reuse_optimization_window
                    else "end must be later than backtest_start."
                )
                raise ValueError(message)

        benchmark_label = self.benchmark_label or self.benchmark_ticker or "Benchmark"

        object.__setattr__(self, "tickers", tickers)
        object.__setattr__(self, "benchmark_label", benchmark_label)


@dataclass
class StrategyAllocation:
    """Optimization output consumed by the backtester."""

    name: str
    weights: np.ndarray
    weights_by_ticker: pd.Series
    optimization_result: Any


@dataclass
class BacktestStrategyResult:
    """Per-strategy result returned by the backtesting engine."""

    name: str
    weights: np.ndarray
    weights_by_ticker: pd.Series
    optimization_result: Any
    portfolio_returns: pd.Series
    evolution: pd.Series


@dataclass
class BacktestResult:
    """Full result returned by the backtester."""

    config: BacktestConfig
    prices_optimization: pd.DataFrame
    prices_backtest: pd.DataFrame
    strategy_results: Dict[str, BacktestStrategyResult]
    returns: pd.DataFrame
    evolution: pd.DataFrame
    metrics: pd.DataFrame


__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "BacktestStrategyResult",
    "StrategyAllocation",
]
