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
    upside calculations built from threshold-adjusted returns. It exposes
    semivariance-based metrics and the Omega ratio while keeping the portfolio
    weight handling inherited from the elementary metrics layer.
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

    def _threshold_adjusted_returns(self, threshold: float) -> pd.DataFrame:
        """Return asset returns expressed as deviations from the threshold."""
        return self.get_returns(self._assets_order()) - threshold

    def returns_below(self, threshold: float = 0.0) -> pd.DataFrame:
        """
        Return threshold-adjusted downside observations and zero-out the rest.

        Values above the threshold become zero. Values below the threshold are
        stored as negative deviations from that threshold.
        """
        normalized_threshold = self._normalize_threshold(threshold)
        filtered_returns = self.__returns_below_cache.get(normalized_threshold)
        if filtered_returns is None:
            deviations = self._threshold_adjusted_returns(normalized_threshold)
            filtered_returns = deviations.where(deviations < 0.0, 0.0)
            self.__returns_below_cache[normalized_threshold] = filtered_returns

        self.__active_threshold = normalized_threshold
        self.returns_down = filtered_returns.copy()

        cached_risk = self.__downside_risk_cache.get(normalized_threshold)
        self.shortfall_risk = (
            None if cached_risk is None else cached_risk.to_numpy(dtype=float)
        )

        return filtered_returns.copy()

    def returns_above(self, threshold: float = 0.0) -> pd.DataFrame:
        """
        Return threshold-adjusted upside observations and zero-out the rest.

        Values below the threshold become zero. Values above the threshold are
        stored as positive deviations from that threshold.
        """
        normalized_threshold = self._normalize_threshold(threshold)
        filtered_returns = self.__returns_above_cache.get(normalized_threshold)
        if filtered_returns is None:
            deviations = self._threshold_adjusted_returns(normalized_threshold)
            filtered_returns = deviations.where(deviations > 0.0, 0.0)
            self.__returns_above_cache[normalized_threshold] = filtered_returns

        self.__active_threshold = normalized_threshold
        self.returns_up = filtered_returns.copy()

        cached_risk = self.__upside_risk_cache.get(normalized_threshold)
        self.upside_potential = (
            None if cached_risk is None else cached_risk.to_numpy(dtype=float)
        )

        return filtered_returns.copy()

    def _downside_risk_series(self, threshold: Optional[float] = None) -> pd.Series:
        """Return annualized downside risk per asset as a pandas Series."""
        normalized_threshold = self._resolve_threshold(threshold)
        downside_risk = self.__downside_risk_cache.get(normalized_threshold)
        if downside_risk is None:
            filtered_returns = self.returns_below(normalized_threshold)
            downside_risk = filtered_returns.std() * np.sqrt(252.0)
            downside_risk = downside_risk.loc[self._assets_order()]
            self.__downside_risk_cache[normalized_threshold] = downside_risk

        self.__active_threshold = normalized_threshold
        self.shortfall_risk = downside_risk.to_numpy(dtype=float)
        return downside_risk.copy()

    def downside_risk(self, threshold: Optional[float] = None) -> np.ndarray:
        """Return annualized downside risk aligned with `self.tickers`."""
        return self._downside_risk_series(threshold).to_numpy(dtype=float)

    def _upside_risk_series(self, threshold: Optional[float] = None) -> pd.Series:
        """Return annualized upside risk per asset as a pandas Series."""
        normalized_threshold = self._resolve_threshold(threshold)
        upside_risk = self.__upside_risk_cache.get(normalized_threshold)
        if upside_risk is None:
            filtered_returns = self.returns_above(normalized_threshold)
            upside_risk = filtered_returns.std() * np.sqrt(252.0)
            upside_risk = upside_risk.loc[self._assets_order()]
            self.__upside_risk_cache[normalized_threshold] = upside_risk

        self.__active_threshold = normalized_threshold
        self.upside_potential = upside_risk.to_numpy(dtype=float)
        return upside_risk.copy()

    def upside_risk(self, threshold: Optional[float] = None) -> np.ndarray:
        """Return annualized upside risk aligned with `self.tickers`."""
        return self._upside_risk_series(threshold).to_numpy(dtype=float)

    def semivariance_matrix(self, threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Build the annualized semivariance matrix for the asset universe.

        The matrix is formed with the annualized downside-risk vector and the
        correlation matrix of the original asset returns.
        """
        normalized_threshold = self._resolve_threshold(threshold)
        semivariance = self.__semivariance_cache.get(normalized_threshold)
        if semivariance is None:
            downside_risk = self._downside_risk_series(normalized_threshold)
            vector = downside_risk.to_numpy(dtype=float)
            semivariance = pd.DataFrame(
                np.outer(vector, vector),
                index=downside_risk.index,
                columns=downside_risk.index,
            )
            correlation = self.correlation(self._assets_order()).loc[
                downside_risk.index,
                downside_risk.index,
            ]
            semivariance = semivariance * correlation
            self.__semivariance_cache[normalized_threshold] = semivariance

        self.__active_threshold = normalized_threshold
        self.returns_down = self.returns_below(normalized_threshold)
        self.shortfall_risk = self.downside_risk(normalized_threshold)
        return semivariance.copy()

    def portfolio_semivariance(
        self,
        weight: Optional[Iterable[float]] = None,
        threshold: Optional[float] = None,
    ) -> float:
        """Compute the annualized portfolio semivariance."""
        vector = self._resolve_weight(weight)
        semivariance = self.semivariance_matrix(threshold).to_numpy(dtype=float)
        value = self._portfolio_variance_from_inputs(vector, semivariance)
        return max(value, 0.0)

    def portfolio_downside_risk(
        self,
        weight: Optional[Iterable[float]] = None,
        threshold: Optional[float] = None,
    ) -> float:
        """Compute the annualized portfolio downside risk."""
        semivariance = self.portfolio_semivariance(weight=weight, threshold=threshold)
        return float(np.sqrt(semivariance))

    def asset_omega_ratio(self, threshold: Optional[float] = None) -> pd.Series:
        """
        Compute the Omega ratio of each asset as upside risk over downside risk.
        """
        normalized_threshold = self._resolve_threshold(threshold)
        omega = self.__omega_cache.get(normalized_threshold)
        if omega is None:
            upside = self._upside_risk_series(normalized_threshold)
            downside = self._downside_risk_series(normalized_threshold)
            omega = upside / downside.replace(0.0, np.nan)
            omega = omega.loc[self._assets_order()]
            self.__omega_cache[normalized_threshold] = omega

        self.__active_threshold = normalized_threshold
        return omega.copy()

    def portfolio_omega_ratio(
        self,
        weight: Optional[Iterable[float]] = None,
        threshold: Optional[float] = None,
    ) -> float:
        """
        Compute the portfolio Omega ratio as the weighted sum of asset Omegas.
        """
        vector = self._resolve_weight(weight)
        omega = self.asset_omega_ratio(threshold).to_numpy(dtype=float)
        if not np.isfinite(omega).all():
            raise ValueError("omega ratio is undefined for assets with zero downside risk.")
        return float(vector @ omega)

    def semivarianza_down(self, threshold: Optional[float] = None) -> pd.DataFrame:
        """Backward-compatible alias for `semivariance_matrix()`."""
        return self.semivariance_matrix(threshold=threshold)


if __name__ == "__main__":
    pass
