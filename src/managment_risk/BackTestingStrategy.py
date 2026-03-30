from __future__ import annotations
"""
Backtesting helpers for portfolio allocation strategies.

This module provides a small orchestration layer on top of the optimization
engines already available in `asset_allocation`. The goal is to keep the
optimization logic inside the specialized portfolio classes while exposing a
simple API to:

- define a backtesting configuration,
- plug one or more allocation strategies,
- simulate the capital path on a holdout window,
- compare strategies against an optional benchmark, and
- summarize the main performance metrics.

The design is intentionally split in two layers:

- `AllocationStrategy` adapters decide how portfolio weights are optimized.
- `Backtester` coordinates data preparation, simulation, and reporting.

This keeps the codebase aligned with the rest of the project, where data
research, optimization, and evaluation are modeled as separate concerns.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    SRC_ROOT = Path(__file__).resolve().parents[1]

    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    from asset_allocation import (
        MaximumOmegaConfig,
        MinimumSemivarianceConfig,
        MinimumVarianceConfig,
        OptimizationConfig,
        PortfolioOptimization,
        PortfolioOptimizationPostModern,
        PostModernOptimizationConfig,
    )
    from security_selection.AssetsResearch import AssetsResearch
else:
    try:
        from ..asset_allocation import (
            MaximumOmegaConfig,
            MinimumSemivarianceConfig,
            MinimumVarianceConfig,
            OptimizationConfig,
            PortfolioOptimization,
            PortfolioOptimizationPostModern,
            PostModernOptimizationConfig,
        )
        from ..security_selection.AssetsResearch import AssetsResearch
    except ImportError:
        from asset_allocation import (
            MaximumOmegaConfig,
            MinimumSemivarianceConfig,
            MinimumVarianceConfig,
            OptimizationConfig,
            PortfolioOptimization,
            PortfolioOptimizationPostModern,
            PostModernOptimizationConfig,
        )
        from security_selection.AssetsResearch import AssetsResearch


@dataclass(frozen=True)
class BacktestConfig:
    """
    Global configuration shared by the backtesting engine.

    Parameters
    ----------
    tickers : Sequence[str]
        Ordered asset universe used by the strategies and the simulation.
    initial_capital : float
        Starting capital used to build the portfolio wealth path.
    optimization_start : str
        Start date of the historical window used to estimate optimal weights.
    backtest_start : str
        First date of the out-of-sample simulation window.
    end : str | None, default None
        Optional exclusive end date for both the optimization and backtest
        datasets.
    price_field : str, default "Close"
        Price field extracted from Yahoo Finance when prices are downloaded
        through `AssetsResearch`.
    benchmark_ticker : str | None, default None
        Optional ticker used as passive benchmark during the backtest.
    benchmark_label : str | None, default None
        Custom display name for the benchmark series. When omitted, defaults
        to `benchmark_ticker` or `"Benchmark"`.
    reuse_optimization_window : bool, default False
        When True, reuse the optimization window as the evaluation window.
        This enables an in-sample evaluation mode and requires
        `backtest_start` to match `optimization_start`.
    risk_free_rate : float, default 0.0
        Annualized risk-free rate used when computing Sharpe ratios in the
        summary table.
    trading_days : int, default 252
        Number of trading days used to annualize returns and volatility.

    Notes
    -----
    The configuration validates that:

    - at least one ticker is provided,
    - capital and trading days are strictly positive,
    - the backtest window starts after the optimization window, unless
      `reuse_optimization_window=True`,
    - `end`, when provided, is later than `backtest_start`.
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
    """
    Optimization output consumed by the backtester.

    Attributes
    ----------
    name : str
        User-facing strategy label used in result tables.
    weights : numpy.ndarray
        Optimal portfolio weights returned by the optimization routine.
    weights_by_ticker : pandas.Series
        Same weights indexed by ticker symbol for easier inspection.
    optimization_result : Any
        Native optimization result returned by the underlying portfolio class.
        It is intentionally kept generic because mean-variance and
        post-modern optimizers expose different result dataclasses.
    """

    name: str
    weights: np.ndarray
    weights_by_ticker: pd.Series
    optimization_result: Any


@dataclass
class BacktestStrategyResult:
    """
    Per-strategy result returned by the backtesting engine.

    Attributes
    ----------
    name : str
        Strategy label used in plots, tables, and portfolio series.
    weights : numpy.ndarray
        Optimal weights applied during the backtest window.
    weights_by_ticker : pandas.Series
        Weight vector indexed by ticker symbol.
    optimization_result : Any
        Native optimization result associated with the strategy.
    portfolio_returns : pandas.Series
        Out-of-sample daily portfolio returns generated during the backtest.
    evolution : pandas.Series
        Capital path obtained by compounding `portfolio_returns` from
        `initial_capital`.
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
    Full result returned by the backtester.

    Attributes
    ----------
    config : BacktestConfig
        Configuration used to run the backtest.
    prices_optimization : pandas.DataFrame
        Historical price window used to estimate strategy weights.
    prices_backtest : pandas.DataFrame
        Holdout price window used for the simulation.
    strategy_results : dict[str, BacktestStrategyResult]
        Detailed result per simulated strategy, keyed by strategy name.
    returns : pandas.DataFrame
        Daily return table containing all strategy series and the optional
        benchmark.
    evolution : pandas.DataFrame
        Capital path table containing all strategy series and the optional
        benchmark.
    metrics : pandas.DataFrame
        Summary metrics table aligned with the style used in the course
        notebooks.
    """

    config: BacktestConfig
    prices_optimization: pd.DataFrame
    prices_backtest: pd.DataFrame
    strategy_results: Dict[str, BacktestStrategyResult]
    returns: pd.DataFrame
    evolution: pd.DataFrame
    metrics: pd.DataFrame


class AllocationStrategy(ABC):
    """
    Common interface used by the backtesting engine.

    Concrete subclasses are responsible only for weight estimation. They do
    not perform the actual capital simulation; that work is delegated to
    `Backtester`.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self._name = name

    @property
    @abstractmethod
    def default_name(self) -> str:
        """Default human-readable label for the strategy."""

    @property
    def name(self) -> str:
        """Return the user-defined name or the strategy default label."""
        return self._name or self.default_name

    @abstractmethod
    def optimize(self, prices: pd.DataFrame) -> StrategyAllocation:
        """
        Optimize the portfolio using the provided price window.

        Parameters
        ----------
        prices : pandas.DataFrame
            Price table used as in-sample optimization window.

        Returns
        -------
        StrategyAllocation
            Allocation object consumed later by the backtesting engine.
        """


def _default_weights(n_assets: int) -> np.ndarray:
    """Return an equally weighted feasible portfolio."""
    return np.ones(n_assets, dtype=float) / float(n_assets)


def _normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a price table used by the backtester."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")

    cleaned = prices.sort_index().dropna().copy()
    if cleaned.empty:
        raise ValueError("prices must contain at least one complete observation.")
    if len(cleaned.columns) == 0:
        raise ValueError("prices must contain at least one asset column.")
    return cleaned


def _normalize_benchmark_prices(
    prices: pd.Series | pd.DataFrame,
    label: str,
) -> pd.Series:
    """Normalize benchmark prices into a clean named series."""
    if isinstance(prices, pd.Series):
        benchmark = prices.sort_index().dropna().copy()
    elif isinstance(prices, pd.DataFrame):
        cleaned = prices.sort_index().dropna().copy()
        if cleaned.shape[1] != 1:
            raise ValueError("benchmark_prices must contain exactly one column.")
        benchmark = cleaned.iloc[:, 0]
    else:
        raise TypeError("benchmark_prices must be a pandas Series or DataFrame.")

    if benchmark.empty:
        raise ValueError("benchmark_prices must contain at least one observation.")

    benchmark.name = label
    return benchmark


def _slice_time_window(
    prices: pd.DataFrame | pd.Series,
    *,
    start: str | pd.Timestamp,
    end: Optional[str] = None,
) -> pd.DataFrame | pd.Series:
    """Return a copy restricted to the configured [start, end) window."""
    start_timestamp = pd.Timestamp(start)
    mask = prices.index >= start_timestamp

    if end is not None:
        end_timestamp = pd.Timestamp(end)
        mask &= prices.index < end_timestamp

    return prices.loc[mask].copy()


def _window_bounds(prices: pd.DataFrame) -> tuple[str, str]:
    """Infer constructor dates for optimizer instances from a price table."""
    start = pd.Timestamp(prices.index.min()).date().isoformat()
    end = (pd.Timestamp(prices.index.max()) + pd.Timedelta(days=1)).date().isoformat()
    return start, end


def _build_mean_variance_optimizer(
    prices: pd.DataFrame,
    initial_weights: Optional[Iterable[float]] = None,
) -> PortfolioOptimization:
    """
    Build a `PortfolioOptimization` instance from an in-memory price table.

    The helper mirrors the constructor requirements of the portfolio class and
    then seeds the internal price/return caches to avoid unnecessary network
    calls during the backtest.
    """
    normalized_prices = _normalize_prices(prices)
    tickers = list(normalized_prices.columns)
    weight = _default_weights(len(tickers))
    if initial_weights is not None:
        weight = np.asarray(initial_weights, dtype=float)

    start, end = _window_bounds(normalized_prices)
    optimizer = PortfolioOptimization(
        tickers,
        start=start,
        end=end,
        price_field="Close",
        weight=weight,
    )
    optimizer._set_prices_cache(normalized_prices)
    optimizer._set_returns_cache(normalized_prices.pct_change().dropna())
    return optimizer


def _build_post_modern_optimizer(
    prices: pd.DataFrame,
    initial_weights: Optional[Iterable[float]] = None,
) -> PortfolioOptimizationPostModern:
    """
    Build a `PortfolioOptimizationPostModern` instance from a price table.

    As in the mean-variance helper, the optimizer receives the relevant prices
    and returns through the cache so the backtest can work with preloaded
    in-memory datasets.
    """
    normalized_prices = _normalize_prices(prices)
    tickers = list(normalized_prices.columns)
    weight = _default_weights(len(tickers))
    if initial_weights is not None:
        weight = np.asarray(initial_weights, dtype=float)

    start, end = _window_bounds(normalized_prices)
    optimizer = PortfolioOptimizationPostModern(
        tickers,
        start=start,
        end=end,
        price_field="Close",
        weight=weight,
    )
    optimizer._set_prices_cache(normalized_prices)
    optimizer._set_returns_cache(normalized_prices.pct_change().dropna())
    return optimizer


class MeanVarianceStrategy(AllocationStrategy):
    """
    Adapter around `PortfolioOptimization` for mean-variance objectives.

    Parameters
    ----------
    objective : {"minimum_variance", "maximum_sharpe"}
        Optimization objective delegated to `PortfolioOptimization`.
    config : OptimizationConfig | None, default None
        Optional configuration object forwarded to the optimizer. When omitted,
        a default config compatible with the selected objective is created.
    name : str | None, default None
        Optional custom label used in result tables.
    """

    _DEFAULT_NAMES = {
        "minimum_variance": "Min Var",
        "maximum_sharpe": "Max Sharpe",
    }

    def __init__(
        self,
        objective: str,
        config: Optional[OptimizationConfig] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        if objective not in self._DEFAULT_NAMES:
            raise ValueError(
                "objective must be 'minimum_variance' or 'maximum_sharpe'."
            )

        self.objective = objective
        self.config = config

    @property
    def default_name(self) -> str:
        """Default display name associated with the selected objective."""
        return self._DEFAULT_NAMES[self.objective]

    def optimize(self, prices: pd.DataFrame) -> StrategyAllocation:
        """
        Optimize portfolio weights on the provided in-sample window.

        Parameters
        ----------
        prices : pandas.DataFrame
            Historical price table used to estimate the optimal weights.

        Returns
        -------
        StrategyAllocation
            Allocation bundle ready to be simulated by `Backtester`.

        Raises
        ------
        RuntimeError
            If the underlying optimizer does not converge successfully.
        """
        config = self.config
        if config is None:
            if self.objective == "minimum_variance":
                config = MinimumVarianceConfig()
            else:
                config = OptimizationConfig()

        optimizer = _build_mean_variance_optimizer(
            prices,
            initial_weights=config.initial_weights,
        )

        if self.objective == "minimum_variance":
            result = optimizer.optimize_minimum_variance(config=config)
        else:
            result = optimizer.optimize_maximum_sharpe(config=config)

        if not result.success:
            raise RuntimeError(f"{self.name} optimization failed: {result.message}")

        return StrategyAllocation(
            name=self.name,
            weights=result.weights,
            weights_by_ticker=result.weights_by_ticker,
            optimization_result=result,
        )


class PostModernStrategy(AllocationStrategy):
    """
    Adapter around `PortfolioOptimizationPostModern` objectives.

    Parameters
    ----------
    objective : {"minimum_semivariance", "maximum_omega"}
        Post-modern optimization objective delegated to the corresponding
        optimizer.
    config : PostModernOptimizationConfig | None, default None
        Optional configuration forwarded to the optimizer. When omitted, a
        default config is instantiated based on the objective.
    name : str | None, default None
        Optional custom label used in outputs.
    """

    _DEFAULT_NAMES = {
        "minimum_semivariance": "Min Semivar",
        "maximum_omega": "Max Omega",
    }

    def __init__(
        self,
        objective: str,
        config: Optional[PostModernOptimizationConfig] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        if objective not in self._DEFAULT_NAMES:
            raise ValueError(
                "objective must be 'minimum_semivariance' or 'maximum_omega'."
            )

        self.objective = objective
        self.config = config

    @property
    def default_name(self) -> str:
        """Default display name associated with the selected objective."""
        return self._DEFAULT_NAMES[self.objective]

    def optimize(self, prices: pd.DataFrame) -> StrategyAllocation:
        """
        Optimize post-modern portfolio weights on the provided price window.

        Parameters
        ----------
        prices : pandas.DataFrame
            Historical in-sample price table.

        Returns
        -------
        StrategyAllocation
            Allocation bundle ready to be simulated by `Backtester`.

        Raises
        ------
        RuntimeError
            If the underlying optimizer does not converge successfully.
        """
        config = self.config
        if config is None:
            if self.objective == "minimum_semivariance":
                config = MinimumSemivarianceConfig()
            else:
                config = MaximumOmegaConfig()

        optimizer = _build_post_modern_optimizer(
            prices,
            initial_weights=config.initial_weights,
        )

        if self.objective == "minimum_semivariance":
            result = optimizer.optimize_minimum_semivariance(config=config)
        else:
            result = optimizer.optimize_maximum_omega(config=config)

        if not result.success:
            raise RuntimeError(f"{self.name} optimization failed: {result.message}")

        return StrategyAllocation(
            name=self.name,
            weights=result.weights,
            weights_by_ticker=result.weights_by_ticker,
            optimization_result=result,
        )


class Backtester:
    """
    Single-split backtesting engine for portfolio allocation strategies.

    `Backtester` implements the workflow used throughout the course notebooks:

    1. load or receive a price table,
    2. separate optimization and backtest windows,
    3. optimize one or more strategies on the in-sample window,
    4. simulate the out-of-sample portfolio path, and
    5. compare performance metrics against an optional benchmark.

    Parameters
    ----------
    config : BacktestConfig
        Global configuration controlling dates, capital, benchmark, and
        annualization conventions.
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
        """Return validated prices aligned with the configured ticker order."""
        if prices is None:
            return _normalize_prices(self._load_prices())

        normalized = _normalize_prices(prices)
        missing = [ticker for ticker in self.config.tickers if ticker not in normalized.columns]
        if missing:
            raise ValueError(f"prices is missing configured tickers: {missing}")

        ordered = normalized[self.config.tickers]
        prepared = _slice_time_window(
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
        """Split the full price table into optimization and backtest windows."""
        if self.config.reuse_optimization_window:
            prices_window = _slice_time_window(
                prices,
                start=self.config.optimization_start,
                end=self.config.end,
            )
            prices_optimization = prices_window.copy()
            prices_backtest = prices_window.copy()
        else:
            prices_optimization = _slice_time_window(
                prices,
                start=self.config.optimization_start,
                end=self.config.backtest_start,
            )
            prices_backtest = _slice_time_window(
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
        """Normalize the strategy input into a validated list."""
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
        """Simulate portfolio returns and wealth path for one allocation."""
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
        return _normalize_benchmark_prices(
            benchmark_prices,
            label=str(self.config.benchmark_label),
        )

    def _build_benchmark_result(
        self,
        benchmark_prices: Optional[pd.Series | pd.DataFrame] = None,
    ) -> tuple[Optional[pd.Series], Optional[pd.Series]]:
        """Build benchmark return and wealth series when a benchmark is available."""
        if benchmark_prices is None and self.config.benchmark_ticker is None:
            return None, None

        if benchmark_prices is None:
            benchmark = self._load_benchmark_prices()
        else:
            benchmark = _normalize_benchmark_prices(
                benchmark_prices,
                label=str(self.config.benchmark_label),
            )

        benchmark = _slice_time_window(
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

        The metrics intentionally follow the same convention already used in
        the course notebook:

        - annualized expected return,
        - effective total return,
        - annualized volatility,
        - Sharpe ratio,
        - downside risk,
        - upside risk, and
        - Omega ratio.
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
    ) -> BacktestResult:
        """
        Execute the backtest for one or more allocation strategies.

        Parameters
        ----------
        strategies : AllocationStrategy | Sequence[AllocationStrategy]
            Strategy instance or collection of strategy instances to be
            optimized and simulated.
        prices : pandas.DataFrame | None, default None
            Optional in-memory price table. When omitted, prices are downloaded
            using `AssetsResearch` and the dates defined in `config`.
        benchmark_prices : pandas.Series | pandas.DataFrame | None, default None
            Optional benchmark price series. When omitted and
            `config.benchmark_ticker` is configured, benchmark data is
            downloaded automatically.

        Returns
        -------
        BacktestResult
            Structured result containing the split datasets, per-strategy
            allocations, simulated return series, capital evolution, and
            summary metrics.

        Notes
        -----
        This implementation performs a single train/test split. It does not
        rebalance dynamically or use rolling windows; those features can be
        added later on top of the same strategy interface. When
        `reuse_optimization_window=True`, the evaluation is performed
        in-sample on the same window used for optimization.
        """
        resolved_strategies = self._resolve_strategies(strategies)
        full_prices = self._prepare_prices(prices)
        prices_optimization, prices_backtest = self._split_prices(full_prices)

        strategy_results: Dict[str, BacktestStrategyResult] = {}
        returns_data: Dict[str, pd.Series] = {}
        evolution_data: Dict[str, pd.Series] = {}

        for strategy in resolved_strategies:
            allocation = strategy.optimize(prices_optimization)
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


# Backward-compatible alias for the previous class name.
BackTestingStrategy = Backtester


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
