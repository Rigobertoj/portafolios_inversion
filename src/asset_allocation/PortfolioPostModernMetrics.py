from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

try:
    from .AssetsResearch import AssetsResearch
except ImportError:
    from AssetsResearch import AssetsResearch


MetricValue = pd.DataFrame | pd.Series


@dataclass(frozen=True)
class _PostModernMetricContext:
    """Shared inputs used by the post-modern metrics components."""

    returns: pd.DataFrame
    correlation: pd.DataFrame
    threshold: float


class _PostModernMetricComponent(ABC):
    """Component contract used to model metric calculations as a tree."""

    name: str

    @abstractmethod
    def compute(self, context: _PostModernMetricContext) -> MetricValue:
        """Compute the metric value for the provided context."""


@dataclass(frozen=True)
class _ReturnsBelowMetric(_PostModernMetricComponent):
    """Leaf component that keeps only returns below the configured threshold."""

    name: str = "returns_below"

    def compute(self, context: _PostModernMetricContext) -> pd.DataFrame:
        return context.returns.where(context.returns < context.threshold, 0.0)


@dataclass(frozen=True)
class _CorrelationMetric(_PostModernMetricComponent):
    """Leaf component that returns the assets correlation matrix."""

    name: str = "correlation"

    def compute(self, context: _PostModernMetricContext) -> pd.DataFrame:
        return context.correlation.copy()


@dataclass(frozen=True)
class _DownsideRiskMetric(_PostModernMetricComponent):
    """Leaf component that calculates downside risk from filtered returns."""

    returns_metric: _ReturnsBelowMetric = field(default_factory=_ReturnsBelowMetric)
    name: str = "downside_risk"

    def compute(self, context: _PostModernMetricContext) -> pd.Series:
        filtered_returns = self.returns_metric.compute(context)
        return filtered_returns.std()


@dataclass(frozen=True)
class _MetricComposite(_PostModernMetricComponent):
    """Composite component that combines child metric results into a new metric."""

    name: str
    children: tuple[_PostModernMetricComponent, ...]
    reducer: Callable[[Mapping[str, MetricValue], _PostModernMetricContext], MetricValue]

    def compute(self, context: _PostModernMetricContext) -> MetricValue:
        values = {child.name: child.compute(context) for child in self.children}
        return self.reducer(values, context)


def _reduce_semivariance_matrix(
    values: Mapping[str, MetricValue],
    _context: _PostModernMetricContext,
) -> pd.DataFrame:
    downside_risk = values["downside_risk"]
    correlation = values["correlation"]

    if not isinstance(downside_risk, pd.Series):
        raise TypeError("downside_risk reducer input must be a pandas Series.")
    if not isinstance(correlation, pd.DataFrame):
        raise TypeError("correlation reducer input must be a pandas DataFrame.")

    vector = downside_risk.to_numpy(dtype=float)
    risk_matrix = np.outer(vector, vector)
    downside_risk_matrix = pd.DataFrame(
        risk_matrix,
        index=downside_risk.index,
        columns=downside_risk.index,
    )

    aligned_correlation = correlation.loc[downside_risk.index, downside_risk.index]
    return downside_risk_matrix * aligned_correlation


@dataclass
class PortfolioPostModernMetrics(AssetsResearch):
    """
    Compute post-modern portfolio metrics focused on downside behavior.

    The class keeps the original public API, but internally models the
    calculation flow as small reusable components. The semivariance matrix is
    built through a Composite that combines downside risk and correlation.
    """

    returns_down: Optional[pd.DataFrame] = field(init=False, repr=False, default=None)
    shortfall_risk: Optional[np.ndarray] = field(init=False, repr=False, default=None)

    __active_threshold: float = field(init=False, repr=False, default=0.0)
    __returns_below_cache: dict[float, pd.DataFrame] = field(
        init=False, repr=False, default_factory=dict
    )
    __downside_risk_cache: dict[float, pd.Series] = field(
        init=False, repr=False, default_factory=dict
    )
    __semivariance_cache: dict[float, pd.DataFrame] = field(
        init=False, repr=False, default_factory=dict
    )
    __returns_metric: _ReturnsBelowMetric = field(init=False, repr=False)
    __downside_risk_metric: _DownsideRiskMetric = field(init=False, repr=False)
    __semivariance_metric: _MetricComposite = field(init=False, repr=False)

    def __post_init__(
        self,
        tickers: Iterable[str],
        start: str,
        end: Optional[str],
        price_field: str,
    ) -> None:
        super().__post_init__(
            tickers=tickers,
            start=start,
            end=end,
            price_field=price_field,
        )

        self.__returns_metric = _ReturnsBelowMetric()
        self.__downside_risk_metric = _DownsideRiskMetric(
            returns_metric=self.__returns_metric
        )
        self.__semivariance_metric = _MetricComposite(
            name="semivariance_matrix",
            children=(self.__downside_risk_metric, _CorrelationMetric()),
            reducer=_reduce_semivariance_matrix,
        )
        self._reset_post_modern_cache()

    def _reset_cache(self) -> None:
        """Reset research caches and derived downside metrics."""
        super()._reset_cache()
        if hasattr(self, "_PortfolioPostModernMetrics__returns_below_cache"):
            self._reset_post_modern_cache()

    def _reset_post_modern_cache(self) -> None:
        """Reset cached post-modern metrics when source data changes."""
        self.returns_down = None
        self.shortfall_risk = None
        self.__active_threshold = 0.0
        self.__returns_below_cache = {}
        self.__downside_risk_cache = {}
        self.__semivariance_cache = {}

    @staticmethod
    def _normalize_threshold(value: Optional[float]) -> float:
        if value is None:
            raise ValueError("threshold cannot be None in this context.")

        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("below must be a numeric value.") from exc

    def _resolve_threshold(self, below: Optional[float]) -> float:
        if below is None:
            return self.__active_threshold
        return self._normalize_threshold(below)

    def _build_context(self, below: float) -> _PostModernMetricContext:
        returns = self.get_returns()
        return _PostModernMetricContext(
            returns=returns,
            correlation=returns.corr(),
            threshold=below,
        )

    def returns_below(self, below: float = 0.0) -> pd.DataFrame:
        """
        Return daily returns below a downside threshold and zero-out the rest.

        Parameters
        ----------
        below : float, default 0.0
            Threshold used to keep only downside observations.

        Returns
        -------
        pandas.DataFrame
            Daily returns where values above the threshold are replaced by zero.
        """
        threshold = self._normalize_threshold(below)
        filtered_returns = self.__returns_below_cache.get(threshold)
        if filtered_returns is None:
            filtered_returns = self.__returns_metric.compute(self._build_context(threshold))
            self.__returns_below_cache[threshold] = filtered_returns

        self.__active_threshold = threshold
        self.returns_down = filtered_returns.copy()

        cached_risk = self.__downside_risk_cache.get(threshold)
        self.shortfall_risk = (
            None if cached_risk is None else cached_risk.to_numpy(dtype=float)
        )

        return filtered_returns.copy()

    def downside_risk(self, below: Optional[float] = None) -> np.ndarray:
        """
        Compute the downside standard deviation for each asset.

        Parameters
        ----------
        below : float | None, default None
            Threshold used to identify downside returns. When omitted, the last
            threshold used by `returns_below()` is reused.

        Returns
        -------
        numpy.ndarray
            Downside risk vector aligned with `self.tickers`.
        """
        threshold = self._resolve_threshold(below)
        downside_risk = self.__downside_risk_cache.get(threshold)
        if downside_risk is None:
            downside_risk = self.__downside_risk_metric.compute(
                self._build_context(threshold)
            )
            self.__downside_risk_cache[threshold] = downside_risk

        self.__active_threshold = threshold
        self.returns_down = self.returns_below(threshold)
        self.shortfall_risk = downside_risk.to_numpy(dtype=float)
        return self.shortfall_risk.copy()

    def semivariance_matrix(self, below: Optional[float] = None) -> pd.DataFrame:
        """
        Build the downside semivariance matrix for the asset universe.

        Parameters
        ----------
        below : float | None, default None
            Threshold used to identify downside returns. When omitted, the last
            threshold used by `returns_below()` is reused.

        Returns
        -------
        pandas.DataFrame
            Semivariance matrix indexed by ticker.
        """
        threshold = self._resolve_threshold(below)
        semivariance = self.__semivariance_cache.get(threshold)
        if semivariance is None:
            semivariance = self.__semivariance_metric.compute(
                self._build_context(threshold)
            )
            if not isinstance(semivariance, pd.DataFrame):
                raise TypeError("semivariance matrix must be a pandas DataFrame.")
            self.__semivariance_cache[threshold] = semivariance

        self.__active_threshold = threshold
        self.returns_down = self.returns_below(threshold)
        self.shortfall_risk = self.downside_risk(threshold)
        return semivariance.copy()

    def semivarianza_down(self, below: Optional[float] = None) -> pd.DataFrame:
        """Backward-compatible alias for `semivariance_matrix()`."""
        return self.semivariance_matrix(below=below)


if __name__ == "__main__":
    pass
