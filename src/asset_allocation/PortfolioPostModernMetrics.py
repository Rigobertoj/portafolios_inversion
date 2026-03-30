from __future__ import annotations

from dataclasses import dataclass, field
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from PortfolioElementaryMetrics import PortfolioElementaryMetrics
else:
    from .PortfolioElementaryMetrics import PortfolioElementaryMetrics


@dataclass
class PortfolioPostModernMetrics(PortfolioElementaryMetrics):
    """
    Compute post-modern portfolio metrics focused on downside behavior.

    The class extends `PortfolioElementaryMetrics` by adding downside and
    upside calculations built from reference-adjusted returns. The downside
    reference can be a scalar threshold/MAR and, when needed, an aligned
    benchmark return series. It exposes semivariance-based metrics and the
    Omega ratio while keeping the portfolio weight handling inherited from the
    elementary metrics layer.

    Notes
    -----
    The public methods in this class follow a consistent reference rule:

    - when only `threshold` is provided, the downside reference is a scalar
      hurdle or minimum acceptable return (MAR),
    - when `benchmark_returns` is also provided, the effective reference
      becomes `benchmark_returns + threshold`,
    - therefore, `threshold=0` with a benchmark produces a pure target
      semivariance setup, while a non-zero threshold represents a benchmark
      plus spread.
    """

    returns_down: Optional[pd.DataFrame] = field(init=False, repr=False, default=None)
    returns_up: Optional[pd.DataFrame] = field(init=False, repr=False, default=None)
    shortfall_risk: Optional[np.ndarray] = field(init=False, repr=False, default=None)
    upside_potential: Optional[np.ndarray] = field(init=False, repr=False, default=None)

    __active_threshold: float = field(init=False, repr=False, default=0.0)
    __returns_below_cache: dict[float, pd.DataFrame] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )
    __returns_above_cache: dict[float, pd.DataFrame] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )
    __downside_risk_cache: dict[float, pd.Series] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )
    __upside_risk_cache: dict[float, pd.Series] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )
    __semivariance_cache: dict[float, pd.DataFrame] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )
    __omega_cache: dict[float, pd.Series] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )

    def __post_init__(
        self,
        tickers: Iterable[str],
        start: str,
        end: Optional[str],
        price_field: str,
        weight: Iterable[float],
    ) -> None:
        super().__post_init__(
            tickers=tickers,
            start=start,
            end=end,
            price_field=price_field,
            weight=weight,
        )
        self._reset_post_modern_cache()

    def _reset_cache(self) -> None:
        """Reset research caches and all post-modern derived metrics."""
        super()._reset_cache()
        if hasattr(self, "_PortfolioPostModernMetrics__returns_below_cache"):
            self._reset_post_modern_cache()

    def _reset_post_modern_cache(self) -> None:
        """Reset post-modern cached metrics when source data changes."""
        self.returns_down = None
        self.returns_up = None
        self.shortfall_risk = None
        self.upside_potential = None
        self.__active_threshold = 0.0
        self.__returns_below_cache = {}
        self.__returns_above_cache = {}
        self.__downside_risk_cache = {}
        self.__upside_risk_cache = {}
        self.__semivariance_cache = {}
        self.__omega_cache = {}

    @staticmethod
    def _normalize_threshold(value: Optional[float]) -> float:
        if value is None:
            raise ValueError("threshold cannot be None in this context.")

        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("threshold must be a numeric value.") from exc

    def _resolve_threshold(self, threshold: Optional[float]) -> float:
        if threshold is None:
            return self.__active_threshold
        return self._normalize_threshold(threshold)

    @staticmethod
    def _normalize_benchmark_returns(
        benchmark_returns: pd.Series | pd.DataFrame,
    ) -> pd.Series:
        """Normalize benchmark returns into a clean aligned series."""
        if isinstance(benchmark_returns, pd.Series):
            benchmark = benchmark_returns.sort_index().dropna().copy()
        elif isinstance(benchmark_returns, pd.DataFrame):
            cleaned = benchmark_returns.sort_index().dropna().copy()
            if cleaned.shape[1] != 1:
                raise ValueError("benchmark_returns must contain exactly one column.")
            benchmark = cleaned.iloc[:, 0]
        else:
            raise TypeError("benchmark_returns must be a pandas Series or DataFrame.")

        benchmark = pd.to_numeric(benchmark, errors="coerce").dropna()
        if benchmark.empty:
            raise ValueError("benchmark_returns must contain at least one observation.")
        return benchmark

    def _reference_adjusted_returns(
        self,
        threshold: float,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Return asset returns expressed as deviations from the downside reference.

        When `benchmark_returns` is provided, the effective downside reference
        becomes `benchmark_returns + threshold`, which naturally supports pure
        benchmark target semivariance (`threshold=0`) and benchmark-plus-spread
        scenarios.
        """
        returns = self.get_returns(self._assets_order())
        if benchmark_returns is None:
            return returns - threshold

        benchmark = self._normalize_benchmark_returns(benchmark_returns)
        aligned_returns, aligned_benchmark = returns.align(benchmark, join="inner", axis=0)
        if len(aligned_returns) < 2:
            raise ValueError(
                "benchmark_returns must overlap asset returns with at least two observations."
            )

        reference = aligned_benchmark + threshold
        return aligned_returns.sub(reference, axis=0)

    def returns_below(
        self,
        threshold: float = 0.0,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Return downside observations relative to the configured reference.

        Values above the downside reference become zero. Values below the
        reference are stored as negative deviations from that reference.

        Parameters
        ----------
        threshold : float, default 0.0
            Scalar hurdle expressed in the same periodicity as the asset
            returns. This acts as a traditional MAR when no benchmark is
            provided.
        benchmark_returns : pandas.Series | pandas.DataFrame | None
            Optional benchmark return series aligned by date. When supplied,
            downside observations are computed against
            `benchmark_returns + threshold`.
        """
        normalized_threshold = self._normalize_threshold(threshold)
        filtered_returns = None
        if benchmark_returns is None:
            filtered_returns = self.__returns_below_cache.get(normalized_threshold)

        if filtered_returns is None:
            deviations = self._reference_adjusted_returns(
                normalized_threshold,
                benchmark_returns=benchmark_returns,
            )
            filtered_returns = deviations.where(deviations < 0.0, 0.0)
            if benchmark_returns is None:
                self.__returns_below_cache[normalized_threshold] = filtered_returns

        self.__active_threshold = normalized_threshold
        self.returns_down = filtered_returns.copy()

        cached_risk = (
            None
            if benchmark_returns is not None
            else self.__downside_risk_cache.get(normalized_threshold)
        )
        self.shortfall_risk = (
            None if cached_risk is None else cached_risk.to_numpy(dtype=float)
        )

        return filtered_returns.copy()

    def returns_above(
        self,
        threshold: float = 0.0,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Return upside observations relative to the configured reference.

        Values below the downside reference become zero. Values above the
        reference are stored as positive deviations from that reference.

        The downside reference follows the same rule as `returns_below()`:
        scalar `threshold` alone behaves as a MAR, while
        `benchmark_returns + threshold` behaves as a benchmark-relative
        hurdle.
        """
        normalized_threshold = self._normalize_threshold(threshold)
        filtered_returns = None
        if benchmark_returns is None:
            filtered_returns = self.__returns_above_cache.get(normalized_threshold)

        if filtered_returns is None:
            deviations = self._reference_adjusted_returns(
                normalized_threshold,
                benchmark_returns=benchmark_returns,
            )
            filtered_returns = deviations.where(deviations > 0.0, 0.0)
            if benchmark_returns is None:
                self.__returns_above_cache[normalized_threshold] = filtered_returns

        self.__active_threshold = normalized_threshold
        self.returns_up = filtered_returns.copy()

        cached_risk = (
            None
            if benchmark_returns is not None
            else self.__upside_risk_cache.get(normalized_threshold)
        )
        self.upside_potential = (
            None if cached_risk is None else cached_risk.to_numpy(dtype=float)
        )

        return filtered_returns.copy()

    def _downside_risk_series(
        self,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> pd.Series:
        """Return annualized downside risk per asset as a pandas Series."""
        normalized_threshold = self._resolve_threshold(threshold)
        downside_risk = None
        if benchmark_returns is None:
            downside_risk = self.__downside_risk_cache.get(normalized_threshold)

        if downside_risk is None:
            filtered_returns = self.returns_below(
                normalized_threshold,
                benchmark_returns=benchmark_returns,
            )
            downside_risk = filtered_returns.std() * np.sqrt(252.0)
            downside_risk = downside_risk.loc[self._assets_order()]
            if benchmark_returns is None:
                self.__downside_risk_cache[normalized_threshold] = downside_risk

        self.__active_threshold = normalized_threshold
        self.shortfall_risk = downside_risk.to_numpy(dtype=float)
        return downside_risk.copy()

    def downside_risk(
        self,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Return annualized downside risk aligned with `self.tickers`.

        The risk is measured relative to the same downside reference used by
        `returns_below()`: a scalar MAR when only `threshold` is provided, or
        a benchmark-relative hurdle when `benchmark_returns` is supplied.
        """
        return self._downside_risk_series(
            threshold,
            benchmark_returns=benchmark_returns,
        ).to_numpy(dtype=float)

    def _upside_risk_series(
        self,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> pd.Series:
        """Return annualized upside risk per asset as a pandas Series."""
        normalized_threshold = self._resolve_threshold(threshold)
        upside_risk = None
        if benchmark_returns is None:
            upside_risk = self.__upside_risk_cache.get(normalized_threshold)

        if upside_risk is None:
            filtered_returns = self.returns_above(
                normalized_threshold,
                benchmark_returns=benchmark_returns,
            )
            upside_risk = filtered_returns.std() * np.sqrt(252.0)
            upside_risk = upside_risk.loc[self._assets_order()]
            if benchmark_returns is None:
                self.__upside_risk_cache[normalized_threshold] = upside_risk

        self.__active_threshold = normalized_threshold
        self.upside_potential = upside_risk.to_numpy(dtype=float)
        return upside_risk.copy()

    def upside_risk(
        self,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Return annualized upside risk aligned with `self.tickers`.

        The upside is measured against the same reference used for downside
        calculations, which keeps Omega-style interpretations consistent across
        MAR-based and benchmark-relative use cases.
        """
        return self._upside_risk_series(
            threshold,
            benchmark_returns=benchmark_returns,
        ).to_numpy(dtype=float)

    def semivariance_matrix(
        self,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build the annualized semivariance matrix for the asset universe.

        The matrix is formed with the annualized downside-risk vector and the
        correlation matrix of the original asset returns.

        Parameters
        ----------
        threshold : float | None, default None
            Scalar MAR or spread over the benchmark, expressed in the same
            periodicity as the return series. When omitted, the last active
            scalar threshold is reused.
        benchmark_returns : pandas.Series | pandas.DataFrame | None
            Optional benchmark return series used to build target
            semivariance. When provided, the downside reference becomes
            `benchmark_returns + threshold`.
        """
        normalized_threshold = self._resolve_threshold(threshold)
        semivariance = None
        if benchmark_returns is None:
            semivariance = self.__semivariance_cache.get(normalized_threshold)

        if semivariance is None:
            downside_risk = self._downside_risk_series(
                normalized_threshold,
                benchmark_returns=benchmark_returns,
            )
            vector = downside_risk.to_numpy(dtype=float)
            semivariance = pd.DataFrame(
                np.outer(vector, vector),
                index=downside_risk.index,
                columns=downside_risk.index,
            )
            correlation_returns = self.get_returns(self._assets_order())
            if benchmark_returns is not None:
                deviations = self._reference_adjusted_returns(
                    normalized_threshold,
                    benchmark_returns=benchmark_returns,
                )
                correlation_returns = correlation_returns.loc[deviations.index]

            correlation = correlation_returns.corr().loc[downside_risk.index, downside_risk.index]
            semivariance = semivariance * correlation
            if benchmark_returns is None:
                self.__semivariance_cache[normalized_threshold] = semivariance

        self.__active_threshold = normalized_threshold
        self.returns_down = self.returns_below(
            normalized_threshold,
            benchmark_returns=benchmark_returns,
        )
        self.shortfall_risk = self.downside_risk(
            normalized_threshold,
            benchmark_returns=benchmark_returns,
        )
        return semivariance.copy()

    def portfolio_semivariance(
        self,
        weight: Optional[Iterable[float]] = None,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> float:
        """
        Compute the annualized portfolio semivariance.

        This is the portfolio-level counterpart of `semivariance_matrix()` and
        therefore supports both a scalar MAR and a benchmark-relative target
        reference.
        """
        vector = self._resolve_weight(weight)
        semivariance = self.semivariance_matrix(
            threshold,
            benchmark_returns=benchmark_returns,
        ).to_numpy(dtype=float)
        value = self._portfolio_variance_from_inputs(vector, semivariance)
        return max(value, 0.0)

    def portfolio_downside_risk(
        self,
        weight: Optional[Iterable[float]] = None,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> float:
        """
        Compute the annualized portfolio downside risk.

        This is simply the square root of portfolio semivariance under the
        selected downside reference.
        """
        semivariance = self.portfolio_semivariance(
            weight=weight,
            threshold=threshold,
            benchmark_returns=benchmark_returns,
        )
        return float(np.sqrt(semivariance))

    def asset_omega_ratio(
        self,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Compute the Omega ratio of each asset as upside risk over downside risk.

        The upside and downside components are always measured against the same
        reference, so this method supports both a scalar MAR and a
        benchmark-relative hurdle.
        """
        normalized_threshold = self._resolve_threshold(threshold)
        omega = None
        if benchmark_returns is None:
            omega = self.__omega_cache.get(normalized_threshold)

        if omega is None:
            upside = self._upside_risk_series(
                normalized_threshold,
                benchmark_returns=benchmark_returns,
            )
            downside = self._downside_risk_series(
                normalized_threshold,
                benchmark_returns=benchmark_returns,
            )
            omega = upside / downside.replace(0.0, np.nan)
            omega = omega.loc[self._assets_order()]
            if benchmark_returns is None:
                self.__omega_cache[normalized_threshold] = omega

        self.__active_threshold = normalized_threshold
        return omega.copy()

    def portfolio_omega_ratio(
        self,
        weight: Optional[Iterable[float]] = None,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> float:
        """
        Compute the portfolio Omega ratio as the weighted sum of asset Omegas.

        As with the rest of the post-modern methods, `benchmark_returns`
        changes the reference from a scalar MAR to a benchmark-relative
        hurdle.
        """
        vector = self._resolve_weight(weight)
        omega = self.asset_omega_ratio(
            threshold,
            benchmark_returns=benchmark_returns,
        ).to_numpy(dtype=float)
        if not np.isfinite(omega).all():
            raise ValueError("omega ratio is undefined for assets with zero downside risk.")
        return float(vector @ omega)

    def semivarianza_down(
        self,
        threshold: Optional[float] = None,
        benchmark_returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Backward-compatible alias for `semivariance_matrix()`."""
        return self.semivariance_matrix(
            threshold=threshold,
            benchmark_returns=benchmark_returns,
        )


if __name__ == "__main__":
    pass
