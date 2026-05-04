"""Dynamic backtesting engine with periodic strategy rebalancing.

The dynamic engine recalculates strategy allocations through time. Each
rebalance builds a fresh training window, asks the strategy for new weights,
charges turnover-based transaction costs, and simulates performance until the
next rebalance date.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ._helpers import normalize_benchmark_prices, slice_time_window
from .engine_static import Backtester
from .results import (
    BacktestConfig,
    DynamicBacktestResult,
    DynamicBacktestStrategyResult,
    RebalanceConfig,
    StrategyAllocation,
)
from .strategies import AllocationStrategy


class DynamicBacktester(Backtester):
    """
    Dynamic backtesting engine with periodic portfolio rebalancing.

    `DynamicBacktester` evaluates one or more allocation strategies over a
    backtest window and recalculates their weights at scheduled rebalance dates.
    At each rebalance date, the engine builds a training window, calls
    `strategy.optimize(...)`, applies transaction costs, and simulates the
    portfolio value until the next rebalance.

    Parameters
    ----------
    config : BacktestConfig
        Global backtest configuration. It defines the asset universe, initial
        capital, optimization and evaluation dates, benchmark metadata, risk-free
        rate, and annualization convention.
    rebalance_config : RebalanceConfig
        Dynamic rebalancing configuration. It controls the lookback window,
        rebalance frequency, expanding-window behavior, and transaction cost
        rate applied to turnover.

    Attributes
    ----------
    config : BacktestConfig
        Backtest configuration inherited from `Backtester`.
    rebalance_config : RebalanceConfig
        Rebalance rules used to generate training windows and rebalance dates.

    Notes
    -----
    Prices are expected to be indexed by date and to contain one column per
    configured ticker. If `prices` is not supplied to `run`, the engine downloads
    prices through `AssetsResearch` using the fields defined in `config`.

    The first allocation is not charged turnover. Subsequent rebalances charge
    `current_value * turnover * transaction_cost`, where turnover is the sum of
    absolute changes in portfolio weights.

    Examples
    --------
    >>> config = BacktestConfig(
    ...     tickers=["AAPL", "MSFT"],
    ...     initial_capital=100_000,
    ...     optimization_start="2020-01-01",
    ...     backtest_start="2021-01-01",
    ...     end="2022-01-01",
    ... )
    >>> rebalance = RebalanceConfig(lookback_months=12, rebalance_months=1)
    >>> engine = DynamicBacktester(config, rebalance)
    >>> result = engine.run(strategy, prices=prices)
    >>> result.evolution.tail()
    """

    def __init__(
        self,
        config: BacktestConfig,
        rebalance_config: RebalanceConfig,
    ) -> None:
        super().__init__(config=config)
        self.rebalance_config = rebalance_config

    @staticmethod
    def _first_index_on_or_after(index: pd.Index, date: pd.Timestamp) -> Optional[pd.Timestamp]:
        candidates = index[index >= date]
        if len(candidates) == 0:
            return None
        return pd.Timestamp(candidates[0])

    def _rebalance_dates(self, prices_backtest: pd.DataFrame) -> list[pd.Timestamp]:
        """
        Build trading dates where allocations should be recalculated.

        Parameters
        ----------
        prices_backtest : pandas.DataFrame
            Backtest price window used to infer tradable rebalance dates.

        Returns
        -------
        list of pandas.Timestamp
            Unique trading dates mapped from monthly rebalance anchors.
        """
        returns_index = prices_backtest.pct_change().dropna().index
        if len(returns_index) == 0:
            raise ValueError("backtest window must contain at least one return row.")

        trading_index = pd.DatetimeIndex(prices_backtest.index)
        first_date = pd.Timestamp(trading_index[0])
        last_return_date = pd.Timestamp(returns_index[-1])

        anchors: list[pd.Timestamp] = []
        anchor = first_date
        while anchor <= last_return_date:
            anchors.append(anchor)
            anchor = anchor + pd.DateOffset(months=self.rebalance_config.rebalance_months)

        mapped: list[pd.Timestamp] = []
        for anchor in anchors:
            trading_date = self._first_index_on_or_after(trading_index, anchor)
            if trading_date is not None and trading_date <= last_return_date:
                mapped.append(trading_date)

        unique_dates = list(dict.fromkeys(mapped))
        if not unique_dates:
            raise ValueError("no rebalance dates could be generated.")
        return unique_dates

    def _training_start(self, rebalance_date: pd.Timestamp) -> pd.Timestamp:
        """
        Return the inclusive start date for a rebalance training window.

        Parameters
        ----------
        rebalance_date : pandas.Timestamp
            Date at which a new allocation is estimated.

        Returns
        -------
        pandas.Timestamp
            Training window start. This is either `optimization_start` for
            expanding windows or the later of `optimization_start` and the
            rolling lookback start.
        """
        optimization_start = pd.Timestamp(self.config.optimization_start)
        if self.rebalance_config.expanding_window:
            return optimization_start

        lookback_start = rebalance_date - pd.DateOffset(
            months=self.rebalance_config.lookback_months,
        )
        return max(optimization_start, lookback_start)

    def _training_prices(
        self,
        prices: pd.DataFrame,
        rebalance_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Return the price window used to estimate weights at a rebalance date.

        Parameters
        ----------
        prices : pandas.DataFrame
            Full prepared price table available to the dynamic engine.
        rebalance_date : pandas.Timestamp
            Date at which the strategy is optimized.

        Returns
        -------
        pandas.DataFrame
            Training prices ending at `rebalance_date`.
        """
        training = slice_time_window(
            prices,
            start=self._training_start(rebalance_date),
            end=rebalance_date,
        )
        if len(training) < 2:
            raise ValueError(
                "training window must contain at least two price rows. "
                "Increase the available lookback data or move backtest_start later."
            )
        return training

    def _prepare_full_optimization_benchmark(
        self,
        optimization_benchmark_prices: Optional[pd.Series | pd.DataFrame],
    ) -> Optional[pd.Series]:
        """
        Normalize a benchmark series for dynamic strategy optimization.

        Parameters
        ----------
        optimization_benchmark_prices : pandas.Series or pandas.DataFrame, optional
            Full benchmark price history supplied to `run`.

        Returns
        -------
        pandas.Series or None
            Clean benchmark prices ready to be sliced per training window.
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

        return normalize_benchmark_prices(
            optimization_benchmark_prices,
            label=label,
        )

    def _training_benchmark(
        self,
        benchmark: Optional[pd.Series],
        rebalance_date: pd.Timestamp,
    ) -> Optional[pd.Series]:
        """
        Return the benchmark window aligned to one training window.

        Parameters
        ----------
        benchmark : pandas.Series, optional
            Full normalized benchmark price history.
        rebalance_date : pandas.Timestamp
            Date at which the strategy is optimized.

        Returns
        -------
        pandas.Series or None
            Benchmark prices aligned to the training window, or `None` when no
            benchmark was supplied.
        """
        if benchmark is None:
            return None

        sliced = slice_time_window(
            benchmark,
            start=self._training_start(rebalance_date),
            end=rebalance_date,
        )
        if len(sliced) < 2:
            raise ValueError(
                "optimization benchmark training window must contain at least two rows."
            )
        return sliced

    def _simulate_strategy(
        self,
        strategy: AllocationStrategy,
        *,
        full_prices: pd.DataFrame,
        prices_backtest: pd.DataFrame,
        rebalance_dates: Sequence[pd.Timestamp],
        optimization_benchmark: Optional[pd.Series],
    ) -> DynamicBacktestStrategyResult:
        """
        Simulate one strategy through all rebalance windows.

        Parameters
        ----------
        strategy : AllocationStrategy
            Strategy optimized at every rebalance date.
        full_prices : pandas.DataFrame
            Full prepared price table used for training windows.
        prices_backtest : pandas.DataFrame
            Backtest price window used for simulated returns.
        rebalance_dates : sequence of pandas.Timestamp
            Dates where the strategy allocation is recalculated.
        optimization_benchmark : pandas.Series, optional
            Benchmark prices sliced and passed to each strategy optimization.

        Returns
        -------
        DynamicBacktestStrategyResult
            Per-strategy dynamic output with allocations, returns, evolution,
            weights history, turnover, and transaction costs.
        """
        returns_backtest = prices_backtest.pct_change().dropna()
        current_value = float(self.config.initial_capital)
        previous_weights: Optional[np.ndarray] = None

        allocations: Dict[pd.Timestamp, StrategyAllocation] = {}
        weights_rows: list[pd.Series] = []
        turnover_rows: dict[pd.Timestamp, float] = {}
        cost_rows: dict[pd.Timestamp, float] = {}
        returns_rows: dict[pd.Timestamp, float] = {}
        evolution_rows: dict[pd.Timestamp, float] = {}
        pre_back_rows: list[pd.Series] = []

        for position, rebalance_date in enumerate(rebalance_dates):
            next_rebalance = (
                rebalance_dates[position + 1]
                if position + 1 < len(rebalance_dates)
                else None
            )

            training_prices = self._training_prices(full_prices, rebalance_date)
            training_benchmark = self._training_benchmark(
                optimization_benchmark,
                rebalance_date,
            )
            allocation = strategy.optimize(
                training_prices,
                optimization_benchmark_prices=training_benchmark,
            )
            weights = np.asarray(allocation.weights, dtype=float)
            allocations[rebalance_date] = allocation
            pre_back_metrics = self._compute_allocation_pre_back_metrics(
                allocation=allocation,
                prices=training_prices,
                benchmark_prices=training_benchmark,
            )
            pre_back_metrics.name = rebalance_date
            pre_back_rows.append(pre_back_metrics)

            if previous_weights is None:
                turnover = 0.0
            else:
                turnover = float(np.abs(weights - previous_weights).sum())

            transaction_cost = (
                current_value * turnover * self.rebalance_config.transaction_cost
            )
            current_value -= transaction_cost
            turnover_rows[rebalance_date] = turnover
            cost_rows[rebalance_date] = float(transaction_cost)

            weights_row = allocation.weights_by_ticker.copy()
            weights_row.name = rebalance_date
            weights_rows.append(weights_row)

            if next_rebalance is None:
                segment_returns = returns_backtest[
                    returns_backtest.index > rebalance_date
                ]
            else:
                segment_returns = returns_backtest[
                    (returns_backtest.index > rebalance_date)
                    & (returns_backtest.index <= next_rebalance)
                ]

            for date, asset_returns in segment_returns.iterrows():
                portfolio_return = float(
                    asset_returns.to_numpy(dtype=float) @ weights
                )
                current_value *= 1.0 + portfolio_return
                returns_rows[pd.Timestamp(date)] = portfolio_return
                evolution_rows[pd.Timestamp(date)] = current_value

            previous_weights = weights

        weights_history = pd.DataFrame(weights_rows)
        weights_history.index.name = "date"
        portfolio_returns = pd.Series(returns_rows, name=strategy.name).sort_index()
        evolution = pd.Series(evolution_rows, name=strategy.name).sort_index()
        turnover = pd.Series(turnover_rows, name=strategy.name).sort_index()
        transaction_costs = pd.Series(cost_rows, name=strategy.name).sort_index()
        pre_back_metrics = pd.DataFrame(pre_back_rows)
        pre_back_metrics.index.name = "date"

        return DynamicBacktestStrategyResult(
            name=strategy.name,
            allocations=allocations,
            portfolio_returns=portfolio_returns,
            evolution=evolution,
            weights_history=weights_history,
            turnover=turnover,
            transaction_costs=transaction_costs,
            pre_back_metrics=pre_back_metrics,
        )

    def run(
        self,
        strategies: AllocationStrategy | Sequence[AllocationStrategy],
        *,
        prices: Optional[pd.DataFrame] = None,
        benchmark_prices: Optional[pd.Series | pd.DataFrame] = None,
        optimization_benchmark_prices: Optional[pd.Series | pd.DataFrame] = None,
    ) -> DynamicBacktestResult:
        """
        Execute a dynamic backtest for one or more allocation strategies.

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
            Benchmark prices passed to strategies during each optimization
            window when the strategy requires a reference benchmark.

        Returns
        -------
        DynamicBacktestResult
            Full dynamic backtest result. It contains the prepared prices,
            backtest prices, per-strategy results, return series, wealth
            evolution, summary metrics, weights history, turnover, and
            transaction costs.

        Raises
        ------
        ValueError
            If the price data cannot produce valid optimization or backtest
            windows, no rebalance dates can be generated, a benchmark window is
            too short, or strategy names are duplicated.
        """
        resolved_strategies = self._resolve_strategies(strategies)
        full_prices = self._prepare_prices(prices)
        prices_backtest = slice_time_window(
            full_prices,
            start=self._backtest_window_start(),
            end=self.config.end,
        )
        if len(prices_backtest) < 2:
            raise ValueError("backtest window must contain at least two price rows.")

        rebalance_dates = self._rebalance_dates(prices_backtest)
        optimization_benchmark = self._prepare_full_optimization_benchmark(
            optimization_benchmark_prices,
        )

        strategy_results: Dict[str, DynamicBacktestStrategyResult] = {}
        returns_data: Dict[str, pd.Series] = {}
        evolution_data: Dict[str, pd.Series] = {}
        weights_frames: list[pd.DataFrame] = []
        turnover_data: Dict[str, pd.Series] = {}
        cost_data: Dict[str, pd.Series] = {}
        pre_back_data: Dict[str, pd.Series] = {}

        for strategy in resolved_strategies:
            result = self._simulate_strategy(
                strategy,
                full_prices=full_prices,
                prices_backtest=prices_backtest,
                rebalance_dates=rebalance_dates,
                optimization_benchmark=optimization_benchmark,
            )
            strategy_results[result.name] = result
            returns_data[result.name] = result.portfolio_returns
            evolution_data[result.name] = result.evolution
            turnover_data[result.name] = result.turnover
            cost_data[result.name] = result.transaction_costs
            pre_back_data[result.name] = result.pre_back_metrics.mean()

            weights_frame = result.weights_history.copy()
            weights_frame.insert(0, "strategy", result.name)
            weights_frames.append(weights_frame)

        benchmark_returns, benchmark_evolution = self._build_benchmark_result(
            benchmark_prices=benchmark_prices,
        )
        if benchmark_returns is not None and benchmark_evolution is not None:
            returns_data[benchmark_returns.name] = benchmark_returns
            evolution_data[benchmark_evolution.name] = benchmark_evolution

        returns = pd.DataFrame(returns_data)
        evolution = pd.DataFrame(evolution_data)

        weights_history = (
            pd.concat(weights_frames)
            if weights_frames
            else pd.DataFrame()
        )
        turnover = pd.DataFrame(turnover_data)
        transaction_costs = pd.DataFrame(cost_data)
        pre_back_metrics = pd.DataFrame(pre_back_data)
        gross_metrics = self._compute_performance_metrics(
            returns=returns,
            benchmark_returns=benchmark_returns,
            use_evolution_returns=False,
        )
        net_metrics = self._compute_performance_metrics(
            returns=returns,
            evolution=evolution,
            benchmark_returns=benchmark_returns,
            use_evolution_returns=True,
        )
        execution_metrics = self._compute_execution_metrics(
            columns=net_metrics.columns,
            turnover=turnover,
            transaction_costs=transaction_costs,
        )
        metrics = self._combine_summary_metrics(
            net_metrics=net_metrics,
            execution_metrics=execution_metrics,
        )

        return DynamicBacktestResult(
            config=self.config,
            rebalance_config=self.rebalance_config,
            prices=full_prices,
            prices_backtest=prices_backtest,
            strategy_results=strategy_results,
            returns=returns,
            evolution=evolution,
            metrics=metrics,
            pre_back_metrics=pre_back_metrics,
            gross_metrics=gross_metrics,
            net_metrics=net_metrics,
            execution_metrics=execution_metrics,
            weights_history=weights_history,
            turnover=turnover,
            transaction_costs=transaction_costs,
        )


DynamicBacktestEngine = DynamicBacktester


__all__ = [
    "DynamicBacktestEngine",
    "DynamicBacktester",
]
