"""Configuration and result containers for the backtesting package.

This module defines the small dataclasses exchanged by the static and dynamic
backtesting engines. They keep configuration, per-strategy allocations, return
series, wealth paths, metrics, and rebalance diagnostics in explicit containers
instead of passing loose dictionaries between modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    """
    Global configuration shared by backtesting engines.

    The validation rules intentionally match the previous implementation so
    notebooks and tests keep the same behavior while the engine is migrated to
    the new package layout.

    Parameters
    ----------
    tickers : sequence of str
        Asset symbols to include in the portfolio. The order is preserved and is
        used to align price columns and optimization weights.
    initial_capital : float
        Starting portfolio value used to build wealth evolution series.
    optimization_start : str
        First date available to the optimization window.
    backtest_start : str
        First date used for the out-of-sample backtest window when
        `reuse_optimization_window` is false.
    end : str, optional
        Exclusive end date for optimization, backtest, and downloaded price
        windows. If omitted, downstream data providers decide the effective end.
    price_field : str, default "Close"
        Price column requested from the data provider when prices are downloaded.
    benchmark_ticker : str, optional
        Passive benchmark symbol downloaded by the engine when
        `benchmark_prices` is not supplied to `run`.
    benchmark_label : str, optional
        Display label used for benchmark returns, evolution, and metrics.
    reuse_optimization_window : bool, default False
        If true, the optimization and backtest windows use the same date range.
        In this mode `backtest_start` must equal `optimization_start`.
    risk_free_rate : float, default 0.0
        Annual risk-free rate used in Sharpe ratio calculations.
    trading_days : int, default 252
        Number of trading days used to annualize expected returns, volatility,
        downside, and upside measures.

    Raises
    ------
    ValueError
        If no tickers are provided, capital is non-positive, `trading_days` is
        non-positive, or the configured dates cannot define a valid window.
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


@dataclass(frozen=True)
class RebalanceConfig:
    """
    Configuration for dynamic backtests with periodic rebalancing.

    Parameters
    ----------
    lookback_months : int, default 12
        Number of months used in each rolling training window when
        `expanding_window` is false.
    rebalance_months : int, default 1
        Number of months between scheduled rebalance anchors.
    expanding_window : bool, default False
        If true, every rebalance trains from `BacktestConfig.optimization_start`
        through the rebalance date. If false, each training window uses the
        configured rolling lookback.
    transaction_cost : float, default 0.0
        Proportional transaction cost charged on turnover at each rebalance.

    Raises
    ------
    ValueError
        If `lookback_months` or `rebalance_months` is non-positive, or if
        `transaction_cost` is negative.
    """

    lookback_months: int = 12
    rebalance_months: int = 1
    expanding_window: bool = False
    transaction_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.lookback_months <= 0:
            raise ValueError("lookback_months must be greater than zero.")
        if self.rebalance_months <= 0:
            raise ValueError("rebalance_months must be greater than zero.")
        if self.transaction_cost < 0:
            raise ValueError("transaction_cost must be greater than or equal to zero.")


@dataclass
class StrategyAllocation:
    """
    Optimization output consumed by backtesting engines.

    Parameters
    ----------
    name : str
        Strategy label used in result dictionaries and output columns.
    weights : numpy.ndarray
        Numeric vector of portfolio weights aligned with the engine's ticker
        order.
    weights_by_ticker : pandas.Series
        Same allocation indexed by ticker for readable reporting.
    optimization_result : Any
        Raw result returned by the underlying optimizer.
    """

    name: str
    weights: np.ndarray
    weights_by_ticker: pd.Series
    optimization_result: Any


@dataclass
class BacktestStrategyResult:
    """
    Per-strategy result returned by the static backtesting engine.

    Parameters
    ----------
    name : str
        Strategy label used in result dictionaries and output columns.
    weights : numpy.ndarray
        Final optimized weights used for the full static backtest window.
    weights_by_ticker : pandas.Series
        Final optimized weights indexed by ticker.
    optimization_result : Any
        Raw optimizer output produced during the in-sample step.
    portfolio_returns : pandas.Series
        Simulated portfolio returns over the backtest window.
    evolution : pandas.Series
        Simulated wealth path generated from `initial_capital`.
    """

    name: str
    weights: np.ndarray
    weights_by_ticker: pd.Series
    optimization_result: Any
    portfolio_returns: pd.Series
    evolution: pd.Series


@dataclass
class BacktestResult:
    """
    Full result returned by the static backtesting engine.

    Parameters
    ----------
    config : BacktestConfig
        Configuration used to run the backtest.
    prices_optimization : pandas.DataFrame
        Price window used to estimate static strategy weights.
    prices_backtest : pandas.DataFrame
        Price window used to simulate out-of-sample performance.
    strategy_results : dict of str to BacktestStrategyResult
        Per-strategy detailed outputs keyed by strategy name.
    returns : pandas.DataFrame
        Strategy and optional benchmark return series aligned by date.
    evolution : pandas.DataFrame
        Strategy and optional benchmark wealth paths aligned by date.
    metrics : pandas.DataFrame
        Summary metrics table computed from `returns` and `evolution`.
    """

    config: BacktestConfig
    prices_optimization: pd.DataFrame
    prices_backtest: pd.DataFrame
    strategy_results: Dict[str, BacktestStrategyResult]
    returns: pd.DataFrame
    evolution: pd.DataFrame
    metrics: pd.DataFrame


@dataclass
class DynamicBacktestStrategyResult:
    """
    Per-strategy result returned by the dynamic backtesting engine.

    Parameters
    ----------
    name : str
        Strategy label used in result dictionaries and output columns.
    allocations : dict of pandas.Timestamp to StrategyAllocation
        Optimization outputs keyed by rebalance date.
    portfolio_returns : pandas.Series
        Simulated portfolio returns across all rebalance segments.
    evolution : pandas.Series
        Simulated wealth path after returns and transaction costs.
    weights_history : pandas.DataFrame
        Rebalance-date table of strategy weights by ticker.
    turnover : pandas.Series
        Turnover charged at each rebalance date.
    transaction_costs : pandas.Series
        Currency amount subtracted from portfolio value at each rebalance.
    """

    name: str
    allocations: Dict[pd.Timestamp, StrategyAllocation]
    portfolio_returns: pd.Series
    evolution: pd.Series
    weights_history: pd.DataFrame
    turnover: pd.Series
    transaction_costs: pd.Series


@dataclass
class DynamicBacktestResult:
    """
    Full result returned by the dynamic backtesting engine.

    Parameters
    ----------
    config : BacktestConfig
        Global backtest configuration used for the run.
    rebalance_config : RebalanceConfig
        Dynamic rebalancing configuration used for the run.
    prices : pandas.DataFrame
        Full prepared price table available to dynamic training windows.
    prices_backtest : pandas.DataFrame
        Price window used to generate rebalance dates and simulated returns.
    strategy_results : dict of str to DynamicBacktestStrategyResult
        Per-strategy dynamic outputs keyed by strategy name.
    returns : pandas.DataFrame
        Strategy and optional benchmark return series aligned by date.
    evolution : pandas.DataFrame
        Strategy and optional benchmark wealth paths aligned by date.
    metrics : pandas.DataFrame
        Summary metrics table computed from `returns` and `evolution`.
    weights_history : pandas.DataFrame
        Combined rebalance-date weight table for all strategies.
    turnover : pandas.DataFrame
        Turnover table with one column per strategy.
    transaction_costs : pandas.DataFrame
        Transaction-cost table with one column per strategy.
    """

    config: BacktestConfig
    rebalance_config: RebalanceConfig
    prices: pd.DataFrame
    prices_backtest: pd.DataFrame
    strategy_results: Dict[str, DynamicBacktestStrategyResult]
    returns: pd.DataFrame
    evolution: pd.DataFrame
    metrics: pd.DataFrame
    weights_history: pd.DataFrame
    turnover: pd.DataFrame
    transaction_costs: pd.DataFrame


__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "BacktestStrategyResult",
    "DynamicBacktestResult",
    "DynamicBacktestStrategyResult",
    "RebalanceConfig",
    "StrategyAllocation",
]
