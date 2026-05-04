"""Shared performance metrics for portfolios and backtests.

The calculator works from realized return and wealth series. It keeps the
financial formulas in one place so portfolio analysis and backtesting report the
same metrics without forcing the simulation engines to build portfolio objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


METRIC_LABELS = [
    "Rendimiento esperado",
    "Rendimiento realizado",
    "Volatilidad",
    "Ratio de sharpe",
    "Downside risk",
    "Upside risk",
    "Omega",
    "Beta",
    "Alpha de Jensen",
    "Ratio de Treynor",
    "Ratio de Sortino",
]


@dataclass
class PerformanceMetricsCalculator:
    """
    Compute performance metrics from realized portfolio or strategy series.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Return series to analyze. Columns represent portfolios, strategies, or
        benchmark series.
    evolution : pandas.Series or pandas.DataFrame, optional
        Wealth paths for the same columns. When supplied, realized return is
        computed from this net wealth path.
    initial_value : float, default 1.0
        Starting value used to evaluate realized return from `evolution`.
    benchmark_returns : pandas.Series or pandas.DataFrame, optional
        Benchmark returns used for beta, Jensen alpha, and Treynor ratio.
    benchmark_name : str, optional
        Name of a column in `returns`/`evolution` that should be used as the
        benchmark when `benchmark_returns` is omitted.
    downside_benchmark_returns : pandas.Series or pandas.DataFrame, optional
        Benchmark hurdle used only for downside, upside, Omega, and Sortino.
    risk_free_rate : float, default 0.0
        Annual risk-free rate used by Sharpe, Jensen alpha, Treynor, and Sortino.
    threshold : float, default 0.0
        Minimum acceptable return used by downside metrics.
    trading_days : int, default 252
        Number of trading days used to annualize return and risk metrics.
    use_evolution_returns : bool, default False
        If true and `evolution` is supplied, all risk metrics are computed from
        returns implied by the wealth path. This lets dynamic backtests include
        transaction costs in the reported metrics.
    """

    returns: pd.Series | pd.DataFrame
    evolution: Optional[pd.Series | pd.DataFrame] = None
    initial_value: float = 1.0
    benchmark_returns: Optional[pd.Series | pd.DataFrame] = None
    benchmark_name: Optional[str] = None
    downside_benchmark_returns: Optional[pd.Series | pd.DataFrame] = None
    risk_free_rate: float = 0.0
    threshold: float = 0.0
    trading_days: int = 252
    use_evolution_returns: bool = False

    def __post_init__(self) -> None:
        if self.initial_value <= 0.0:
            raise ValueError("initial_value must be greater than zero.")
        if self.trading_days <= 0:
            raise ValueError("trading_days must be greater than zero.")

        self._returns = self._as_frame(self.returns, default_name="portfolio")
        self._evolution = (
            self._as_frame(self.evolution, default_name="portfolio")
            if self.evolution is not None
            else None
        )
        self._benchmark_returns = self._as_series(
            self.benchmark_returns,
            label=self.benchmark_name,
        )
        self._downside_benchmark_returns = self._as_series(
            self.downside_benchmark_returns,
        )

    @staticmethod
    def _as_frame(
        data: pd.Series | pd.DataFrame,
        *,
        default_name: str,
    ) -> pd.DataFrame:
        if isinstance(data, pd.Series):
            name = data.name or default_name
            frame = data.to_frame(name=name)
        elif isinstance(data, pd.DataFrame):
            frame = data.copy()
        else:
            raise TypeError("data must be a pandas Series or DataFrame.")

        frame = frame.sort_index().apply(pd.to_numeric, errors="coerce")
        if frame.empty or len(frame.columns) == 0:
            raise ValueError("data must contain at least one column.")
        return frame

    @staticmethod
    def _as_series(
        data: Optional[pd.Series | pd.DataFrame],
        *,
        label: Optional[str] = None,
    ) -> Optional[pd.Series]:
        if data is None:
            return None
        if isinstance(data, pd.Series):
            series = data.sort_index().copy()
        elif isinstance(data, pd.DataFrame):
            frame = data.sort_index().copy()
            if frame.shape[1] != 1:
                raise ValueError("benchmark data must contain exactly one column.")
            series = frame.iloc[:, 0]
        else:
            raise TypeError("benchmark data must be a pandas Series or DataFrame.")

        series = pd.to_numeric(series, errors="coerce").dropna()
        if series.empty:
            raise ValueError("benchmark data must contain at least one observation.")
        if label is not None:
            series.name = label
        elif series.name is None:
            series.name = "Benchmark"
        return series

    def _returns_from_evolution(self) -> pd.DataFrame:
        if self._evolution is None:
            return self._returns

        data: dict[str, pd.Series] = {}
        for column in self._evolution.columns:
            wealth = self._evolution[column].dropna()
            if wealth.empty:
                data[column] = pd.Series(dtype=float, name=column)
                continue
            previous = wealth.shift(1)
            previous.iloc[0] = self.initial_value
            data[column] = (wealth / previous - 1.0).rename(column)
        return pd.DataFrame(data)

    def _metric_returns(self) -> pd.DataFrame:
        if self.use_evolution_returns and self._evolution is not None:
            return self._returns_from_evolution()
        return self._returns

    def _realized_return(self) -> pd.Series:
        if self._evolution is None:
            return (1.0 + self._returns).prod() - 1.0

        values: dict[str, float] = {}
        for column in self._evolution.columns:
            wealth = self._evolution[column].dropna()
            values[column] = (
                float(wealth.iloc[-1] / self.initial_value - 1.0)
                if not wealth.empty
                else float("nan")
            )
        return pd.Series(values, dtype=float)

    def _benchmark_for_relative_metrics(
        self,
        metric_returns: pd.DataFrame,
    ) -> Optional[pd.Series]:
        if self._benchmark_returns is not None:
            return self._benchmark_returns
        if self.benchmark_name is not None and self.benchmark_name in metric_returns.columns:
            benchmark = metric_returns[self.benchmark_name].dropna()
            benchmark.name = self.benchmark_name
            return benchmark
        return None

    def _adjusted_returns(self, series: pd.Series) -> pd.Series:
        threshold = float(self.threshold)
        if self._downside_benchmark_returns is None:
            return series.dropna() - threshold

        aligned = pd.concat(
            [
                series.rename("portfolio"),
                self._downside_benchmark_returns.rename("benchmark"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        if len(aligned) < 2:
            raise ValueError(
                "downside benchmark returns must overlap analyzed returns "
                "with at least two observations."
            )
        return aligned["portfolio"] - (aligned["benchmark"] + threshold)

    def _downside_metrics(
        self,
        metric_returns: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        downside: dict[str, float] = {}
        upside: dict[str, float] = {}
        omega: dict[str, float] = {}

        for column in metric_returns.columns:
            adjusted = self._adjusted_returns(metric_returns[column])
            down_value = float(
                adjusted.where(adjusted < 0.0, 0.0).std() * np.sqrt(self.trading_days)
            )
            up_value = float(
                adjusted.where(adjusted > 0.0, 0.0).std() * np.sqrt(self.trading_days)
            )
            downside[column] = down_value
            upside[column] = up_value
            omega[column] = (
                float("nan")
                if np.isclose(down_value, 0.0)
                else float(up_value / down_value)
            )

        return (
            pd.Series(downside, dtype=float),
            pd.Series(upside, dtype=float),
            pd.Series(omega, dtype=float),
        )

    def _benchmark_metrics(
        self,
        metric_returns: pd.DataFrame,
        expected_return: pd.Series,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        benchmark = self._benchmark_for_relative_metrics(metric_returns)
        beta = pd.Series(np.nan, index=metric_returns.columns, dtype=float)
        alpha = pd.Series(np.nan, index=metric_returns.columns, dtype=float)
        treynor = pd.Series(np.nan, index=metric_returns.columns, dtype=float)

        if benchmark is None:
            return beta, alpha, treynor

        market_return = float(benchmark.mean() * self.trading_days)
        for column in metric_returns.columns:
            aligned = pd.concat(
                [
                    metric_returns[column].rename("portfolio"),
                    benchmark.rename("benchmark"),
                ],
                axis=1,
                join="inner",
            ).dropna()
            if len(aligned) < 2:
                continue

            benchmark_variance = float(aligned["benchmark"].var())
            if np.isclose(benchmark_variance, 0.0):
                continue

            beta_value = float(
                aligned["portfolio"].cov(aligned["benchmark"]) / benchmark_variance
            )
            beta[column] = beta_value

            capm_return = self.risk_free_rate + beta_value * (
                market_return - self.risk_free_rate
            )
            alpha[column] = float(expected_return[column] - capm_return)
            if not np.isclose(beta_value, 0.0):
                treynor[column] = float(
                    (expected_return[column] - self.risk_free_rate) / beta_value
                )

        return beta, alpha, treynor

    def metrics_table(self) -> pd.DataFrame:
        """
        Return a metrics table with one column per analyzed portfolio/strategy.

        Returns
        -------
        pandas.DataFrame
            Rows are metric names and columns are return/evolution series names.
        """
        metric_returns = self._metric_returns()
        expected_return = metric_returns.mean() * self.trading_days
        realized_return = self._realized_return()
        volatility = metric_returns.std() * np.sqrt(self.trading_days)
        sharpe = (
            (expected_return - self.risk_free_rate)
            / volatility.replace(0.0, np.nan)
        )
        downside, upside, omega = self._downside_metrics(metric_returns)
        sortino = (
            (expected_return - self.risk_free_rate)
            / downside.replace(0.0, np.nan)
        )
        beta, alpha, treynor = self._benchmark_metrics(
            metric_returns=metric_returns,
            expected_return=expected_return,
        )

        table = pd.DataFrame(
            {
                "Rendimiento esperado": expected_return,
                "Rendimiento realizado": realized_return,
                "Volatilidad": volatility,
                "Ratio de sharpe": sharpe,
                "Downside risk": downside,
                "Upside risk": upside,
                "Omega": omega,
                "Beta": beta,
                "Alpha de Jensen": alpha,
                "Ratio de Treynor": treynor,
                "Ratio de Sortino": sortino,
            }
        ).T
        return table.reindex(METRIC_LABELS)


__all__ = [
    "METRIC_LABELS",
    "PerformanceMetricsCalculator",
]
