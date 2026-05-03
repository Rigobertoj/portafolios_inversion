"""Static backtesting engine for allocation strategies.

The static engine estimates strategy weights once on an optimization window and
then simulates those fixed weights over a backtest window. It also prepares
optional benchmark series and computes the shared performance metrics table.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ..research.assets_research import AssetsResearch
from ._helpers import normalize_benchmark_prices, normalize_prices, slice_time_window
from .results import (
    BacktestConfig,
    BacktestResult,
    BacktestStrategyResult,
    StrategyAllocation,
)
from .strategies import AllocationStrategy


class Backtester:
    """
    Single-split backtesting engine for portfolio allocation strategies.

    The implementation preserves the previous behavior while relocating the
    engine to the dedicated `backtesting` package so future growth can happen
    outside the old `managment_risk` module.

    Parameters
    ----------
    config : BacktestConfig
        Global backtest configuration. It defines the asset universe, initial
        capital, optimization window, backtest window, benchmark metadata,
        annualization convention, and risk-free rate.

    Attributes
    ----------
    config : BacktestConfig
        Configuration used by all data preparation, optimization, simulation,
        benchmark, and metrics steps.

    Notes
    -----
    A static backtest uses a single allocation per strategy. The engine
    optimizes each strategy on `prices_optimization`, applies the resulting
    weights unchanged across `prices_backtest`, and compounds daily portfolio
    returns from `initial_capital`.

    Examples
    --------
    >>> config = BacktestConfig(
    ...     tickers=["AAPL", "MSFT"],
    ...     initial_capital=100_000,
    ...     optimization_start="2020-01-01",
    ...     backtest_start="2021-01-01",
    ...     end="2022-01-01",
    ... )
    >>> engine = Backtester(config)
    >>> strategy = MeanVarianceStrategy("minimum_variance")
    >>> result = engine.run(strategy, prices=prices)
    >>> result.metrics
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def _backtest_window_start(self) -> str:
        """Return the first date used for the simulated evaluation window."""
        if self.config.reuse_optimization_window:
            return self.config.optimization_start
        return self.config.backtest_start

    def _optimization_window_end(self) -> Optional[str]:
        """Return the exclusive end date of the optimization window."""
        if self.config.reuse_optimization_window:
            return self.config.end
        return self.config.backtest_start

    def _load_prices(self) -> pd.DataFrame:
        """Download and return the full asset price table for the backtest."""
        research = AssetsResearch(
            tickers=self.config.tickers,
            start=self.config.optimization_start,
            end=self.config.end,
            price_field=self.config.price_field,
        )
        return research.get_prices(self.config.tickers)

    def _prepare_prices(self, prices: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Return validated prices aligned with the configured ticker order.

        Parameters
        ----------
        prices : pandas.DataFrame, optional
            User-supplied asset prices. If omitted, prices are downloaded with
            `_load_prices`.

        Returns
        -------
        pandas.DataFrame
            Clean price table sliced to the configured global window and ordered
            by `config.tickers`.
        """
        if prices is None:
            return normalize_prices(self._load_prices())

        normalized = normalize_prices(prices)
        missing = [ticker for ticker in self.config.tickers if ticker not in normalized.columns]
        if missing:
            raise ValueError(f"prices is missing configured tickers: {missing}")

        ordered = normalized[self.config.tickers]
        prepared = slice_time_window(
            ordered,
            start=self.config.optimization_start,
            end=self.config.end,
        )
        if prepared.empty:
            raise ValueError(
                "prices does not contain observations inside the configured window."
            )
        return prepared

    def _split_prices(self, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the full price table into optimization and backtest windows.

        Parameters
        ----------
        prices : pandas.DataFrame
            Prepared asset prices covering the configured date range.

        Returns
        -------
        tuple of pandas.DataFrame
            `(prices_optimization, prices_backtest)` used for weight estimation
            and fixed-weight simulation.
        """
        if self.config.reuse_optimization_window:
            prices_window = slice_time_window(
                prices,
                start=self.config.optimization_start,
                end=self.config.end,
            )
            prices_optimization = prices_window.copy()
            prices_backtest = prices_window.copy()
        else:
            prices_optimization = slice_time_window(
                prices,
                start=self.config.optimization_start,
                end=self.config.backtest_start,
            )
            prices_backtest = slice_time_window(
                prices,
                start=self._backtest_window_start(),
                end=self.config.end,
            )

        if len(prices_optimization) < 2:
            raise ValueError("optimization window must contain at least two price rows.")
        if len(prices_backtest) < 2:
            raise ValueError("backtest window must contain at least two price rows.")

        return prices_optimization, prices_backtest

    def _resolve_strategies(
        self,
        strategies: AllocationStrategy | Sequence[AllocationStrategy],
    ) -> list[AllocationStrategy]:
        """
        Normalize the strategy input into a validated list.

        Parameters
        ----------
        strategies : AllocationStrategy or sequence of AllocationStrategy
            Strategy or strategies passed to `run`.

        Returns
        -------
        list of AllocationStrategy
            Non-empty list of strategies with unique names.
        """
        if isinstance(strategies, AllocationStrategy):
            resolved = [strategies]
        else:
            resolved = list(strategies)

        if not resolved:
            raise ValueError("strategies must contain at least one strategy.")

        names = [strategy.name for strategy in resolved]
        if len(set(names)) != len(names):
            raise ValueError("strategy names must be unique within a backtest run.")

        return resolved

    def _build_strategy_result(
        self,
        allocation: StrategyAllocation,
        prices_backtest: pd.DataFrame,
    ) -> BacktestStrategyResult:
        """
        Simulate portfolio returns and wealth path for one allocation.

        Parameters
        ----------
        allocation : StrategyAllocation
            Optimized static allocation returned by a strategy.
        prices_backtest : pandas.DataFrame
            Price window used for fixed-weight simulation.

        Returns
        -------
        BacktestStrategyResult
            Per-strategy output with returns, wealth evolution, and optimizer
            metadata.
        """
        returns_backtest = prices_backtest.pct_change().dropna()
        portfolio_returns = (returns_backtest @ allocation.weights).rename(allocation.name)
        evolution = (
            self.config.initial_capital * (1.0 + portfolio_returns).cumprod()
        ).rename(allocation.name)

        return BacktestStrategyResult(
            name=allocation.name,
            weights=allocation.weights,
            weights_by_ticker=allocation.weights_by_ticker,
            optimization_result=allocation.optimization_result,
            portfolio_returns=portfolio_returns,
            evolution=evolution,
        )

    def _load_benchmark_prices(self) -> pd.Series:
        """Download and normalize the passive benchmark price series."""
        if self.config.benchmark_ticker is None:
            raise ValueError("benchmark_ticker is not configured.")

        research = AssetsResearch(
            tickers=[self.config.benchmark_ticker],
            start=self._backtest_window_start(),
            end=self.config.end,
            price_field=self.config.price_field,
        )
        benchmark_prices = research.get_prices(self.config.benchmark_ticker)
        return normalize_benchmark_prices(
            benchmark_prices,
            label=str(self.config.benchmark_label),
        )

    def _prepare_optimization_benchmark_prices(
        self,
        optimization_benchmark_prices: Optional[pd.Series | pd.DataFrame] = None,
    ) -> Optional[pd.Series]:
        """
        Validate and slice the optimization benchmark to the in-sample window.

        Parameters
        ----------
        optimization_benchmark_prices : pandas.Series or pandas.DataFrame, optional
            Benchmark prices supplied to strategies during optimization.

        Returns
        -------
        pandas.Series or None
            Clean benchmark prices aligned to the optimization window, or `None`
            when no benchmark was supplied.
        """
        if optimization_benchmark_prices is None:
            return None

        label = "Optimization Benchmark"
        if isinstance(optimization_benchmark_prices, pd.Series):
            label = str(optimization_benchmark_prices.name or label)
        elif (
            isinstance(optimization_benchmark_prices, pd.DataFrame)
            and optimization_benchmark_prices.shape[1] == 1
        ):
            label = str(optimization_benchmark_prices.columns[0] or label)

        benchmark = normalize_benchmark_prices(
            optimization_benchmark_prices,
            label=label,
        )
        benchmark = slice_time_window(
            benchmark,
            start=self.config.optimization_start,
            end=self._optimization_window_end(),
        )
        if len(benchmark) < 2:
            raise ValueError(
                "optimization benchmark window must contain at least two price rows."
            )
        return benchmark

    def _build_benchmark_result(
        self,
        benchmark_prices: Optional[pd.Series | pd.DataFrame] = None,
    ) -> tuple[Optional[pd.Series], Optional[pd.Series]]:
        """
        Build benchmark return and wealth series when a benchmark is available.

        Parameters
        ----------
        benchmark_prices : pandas.Series or pandas.DataFrame, optional
            User-supplied benchmark prices. If omitted and `benchmark_ticker` is
            configured, prices are downloaded with `_load_benchmark_prices`.

        Returns
        -------
        tuple of pandas.Series or None
            `(benchmark_returns, benchmark_evolution)` when a benchmark is
            available; otherwise `(None, None)`.
        """
        if benchmark_prices is None and self.config.benchmark_ticker is None:
            return None, None

        if benchmark_prices is None:
            benchmark = self._load_benchmark_prices()
        else:
            benchmark = normalize_benchmark_prices(
                benchmark_prices,
                label=str(self.config.benchmark_label),
            )

        benchmark = slice_time_window(
            benchmark,
            start=self._backtest_window_start(),
            end=self.config.end,
        )
        if len(benchmark) < 2:
            raise ValueError("benchmark window must contain at least two price rows.")

        benchmark_returns = benchmark.pct_change().dropna().rename(benchmark.name)
        benchmark_evolution = (
            self.config.initial_capital * (1.0 + benchmark_returns).cumprod()
        ).rename(benchmark.name)
        return benchmark_returns, benchmark_evolution

    def _compute_metrics(
        self,
        returns: pd.DataFrame,
        evolution: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute the summary metrics table used in the backtest output.

        Parameters
        ----------
        returns : pandas.DataFrame
            Strategy and optional benchmark return series.
        evolution : pandas.DataFrame
            Strategy and optional benchmark wealth paths.

        Returns
        -------
        pandas.DataFrame
            Metrics table with expected return, effective return, volatility,
            Sharpe ratio, downside, upside, and Omega ratio.
        """
        expected_return = returns.mean() * self.config.trading_days
        effective_return = evolution.iloc[-1] / self.config.initial_capital - 1.0
        volatility = returns.std() * np.sqrt(self.config.trading_days)
        sharpe = (expected_return - self.config.risk_free_rate) / volatility.replace(0.0, np.nan)

        downside = returns.where(returns < 0.0, 0.0).std() * np.sqrt(self.config.trading_days)
        upside = returns.where(returns > 0.0, 0.0).std() * np.sqrt(self.config.trading_days)
        omega = upside / downside.replace(0.0, np.nan)

        return pd.DataFrame(
            {
                "Rend Esperado": expected_return,
                "Rend Efectivo": effective_return,
                "Volatilidad": volatility,
                "Sharpe": sharpe,
                "Downside": downside,
                "Upside": upside,
                "Omega": omega,
            }
        ).T

    def run(
        self,
        strategies: AllocationStrategy | Sequence[AllocationStrategy],
        *,
        prices: Optional[pd.DataFrame] = None,
        benchmark_prices: Optional[pd.Series | pd.DataFrame] = None,
        optimization_benchmark_prices: Optional[pd.Series | pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Execute a static backtest for one or more allocation strategies.

        Parameters
        ----------
        strategies : AllocationStrategy or sequence of AllocationStrategy
            Strategy, or strategies, to evaluate. Each strategy must expose a
            unique `name` and implement `optimize(prices,
            optimization_benchmark_prices=None)`.
        prices : pandas.DataFrame, optional
            Price history for the configured tickers. Columns must include all
            tickers in `config.tickers`; extra columns are ignored. When omitted,
            prices are downloaded with the configuration stored in `config`.
        benchmark_prices : pandas.Series or pandas.DataFrame, optional
            Passive benchmark prices used only for comparison in the output
            returns, evolution, and metrics tables.
        optimization_benchmark_prices : pandas.Series or pandas.DataFrame, optional
            Benchmark prices passed to strategies during the optimization window
            when the strategy requires a reference benchmark.

        Returns
        -------
        BacktestResult
            Full static backtest result. It contains the optimization prices,
            backtest prices, per-strategy results, return series, wealth
            evolution, and summary metrics.

        Raises
        ------
        ValueError
            If price data cannot produce valid optimization or backtest windows,
            a benchmark window is too short, required tickers are missing, no
            strategies are supplied, or strategy names are duplicated.
        RuntimeError
            If one of the supplied strategies fails during optimization.
        """
        resolved_strategies = self._resolve_strategies(strategies)
        full_prices = self._prepare_prices(prices)
        prices_optimization, prices_backtest = self._split_prices(full_prices)
        optimization_benchmark = self._prepare_optimization_benchmark_prices(
            optimization_benchmark_prices=optimization_benchmark_prices,
        )

        strategy_results: Dict[str, BacktestStrategyResult] = {}
        returns_data: Dict[str, pd.Series] = {}
        evolution_data: Dict[str, pd.Series] = {}

        for strategy in resolved_strategies:
            allocation = strategy.optimize(
                prices_optimization,
                optimization_benchmark_prices=optimization_benchmark,
            )
            result = self._build_strategy_result(allocation, prices_backtest)
            strategy_results[result.name] = result
            returns_data[result.name] = result.portfolio_returns
            evolution_data[result.name] = result.evolution

        benchmark_returns, benchmark_evolution = self._build_benchmark_result(
            benchmark_prices=benchmark_prices,
        )
        if benchmark_returns is not None and benchmark_evolution is not None:
            returns_data[benchmark_returns.name] = benchmark_returns
            evolution_data[benchmark_evolution.name] = benchmark_evolution

        returns = pd.DataFrame(returns_data)
        evolution = pd.DataFrame(evolution_data)
        metrics = self._compute_metrics(returns=returns, evolution=evolution)

        return BacktestResult(
            config=self.config,
            prices_optimization=prices_optimization,
            prices_backtest=prices_backtest,
            strategy_results=strategy_results,
            returns=returns,
            evolution=evolution,
            metrics=metrics,
        )


BackTestingStrategy = Backtester
StaticBacktestEngine = Backtester


__all__ = [
    "Backtester",
    "BackTestingStrategy",
    "StaticBacktestEngine",
]
