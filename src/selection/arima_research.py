"""ARIMA modeling helpers aligned with the public selection API.

This module wraps statsmodels ARIMA in a small reporting interface that returns
pandas tables for parameters, diagnostics, summaries, and forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA as _StatsmodelsARIMA
except ModuleNotFoundError:
    _StatsmodelsARIMA = None


def _validate_order(order: tuple[int, ...], *, name: str, size: int) -> None:
    """Validate the integer tuple expected by statsmodels ARIMA."""
    if len(order) != size:
        raise ValueError(f"{name} must contain exactly {size} integers.")
    if any(int(value) < 0 for value in order):
        raise ValueError(f"{name} values must be non-negative integers.")


@dataclass
class ARIMAResearch:
    """
    Wrap statsmodels ARIMA with a project-friendly reporting API.

    The helper keeps a compact `summary()` table in pandas while still exposing
    the native statsmodels text output through `statsmodels_summary()`.

    Parameters
    ----------
    series : pandas.Series, pandas.DataFrame, sequence of float, or numpy.ndarray
        Target time series. DataFrames must contain exactly one column.
    order : tuple of int, default (1, 0, 0)
        Non-seasonal ARIMA `(p, d, q)` order.
    seasonal_order : tuple of int, default (0, 0, 0, 0)
        Seasonal ARIMA `(P, D, Q, s)` order.
    trend : str, optional
        Trend parameter forwarded to statsmodels.
    enforce_stationarity : bool, default True
        Whether statsmodels enforces stationarity.
    enforce_invertibility : bool, default True
        Whether statsmodels enforces invertibility.

    Raises
    ------
    ValueError
        If orders have the wrong shape, contain negative values, or the target
        series is empty.
    """

    series: pd.Series | pd.DataFrame | Sequence[float] | np.ndarray
    order: tuple[int, int, int] = (1, 0, 0)
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)
    trend: Optional[str] = None
    enforce_stationarity: bool = True
    enforce_invertibility: bool = True
    _series: pd.Series = field(init=False, repr=False)
    _result: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_order(self.order, name="order", size=3)
        _validate_order(self.seasonal_order, name="seasonal_order", size=4)
        self._series = self._normalize_series(self.series)

    @staticmethod
    def _normalize_series(
        series: pd.Series | pd.DataFrame | Sequence[float] | np.ndarray,
    ) -> pd.Series:
        """Normalize the input into a single non-empty float series."""
        if isinstance(series, pd.DataFrame):
            if series.shape[1] != 1:
                raise ValueError(
                    "series DataFrame must contain exactly one column."
                )
            normalized = series.iloc[:, 0]
        elif isinstance(series, pd.Series):
            normalized = series.copy()
        else:
            normalized = pd.Series(series, dtype=float)

        normalized = normalized.dropna().astype(float)
        if normalized.empty:
            raise ValueError("series must contain at least one non-null value.")
        return normalized

    @property
    def target(self) -> pd.Series:
        """Return the normalized target series used by the ARIMA model."""
        return self._series.copy()

    def _build_model(self):
        """Instantiate the underlying statsmodels ARIMA model."""
        statsmodels_arima = _StatsmodelsARIMA
        if statsmodels_arima is None:
            try:
                from statsmodels.tsa.arima.model import ARIMA as statsmodels_arima
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "statsmodels is required to use ARIMAResearch. "
                    "Install it to fit ARIMA models."
                ) from exc

        return statsmodels_arima(
            self._series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )

    def fit(self, **fit_kwargs) -> "ARIMAResearch":
        """
        Fit the ARIMA model and cache the statsmodels result object.

        Parameters
        ----------
        **fit_kwargs
            Keyword arguments forwarded to `statsmodels` model fitting.

        Returns
        -------
        ARIMAResearch
            The fitted instance, returned for chaining.
        """
        self._result = self._build_model().fit(**fit_kwargs)
        return self

    def fitted_result(self):
        """Return the cached fitted result, fitting lazily if needed."""
        if self._result is None:
            self.fit()
        return self._result

    def residuals(self) -> pd.Series:
        """Return the fitted model residuals as a pandas series."""
        result = self.fitted_result()
        residuals = result.resid
        if isinstance(residuals, pd.Series):
            return residuals.dropna()
        return pd.Series(residuals, index=self._series.index).dropna()

    def parameters(self) -> pd.DataFrame:
        """
        Return parameter estimates and confidence intervals.

        Returns
        -------
        pandas.DataFrame
            Table with coefficients, standard errors, test statistics, p-values,
            and confidence interval bounds.
        """
        result = self.fitted_result()
        index = pd.Index(result.param_names, name="parameter")
        conf_int = result.conf_int()

        if isinstance(conf_int, pd.DataFrame):
            ci_low = conf_int.iloc[:, 0].astype(float)
            ci_high = conf_int.iloc[:, 1].astype(float)
            ci_low.index = index
            ci_high.index = index
        else:
            ci_low = pd.Series(conf_int[:, 0], index=index, dtype=float)
            ci_high = pd.Series(conf_int[:, 1], index=index, dtype=float)

        return pd.DataFrame(
            {
                "coef": pd.Series(result.params, index=index, dtype=float),
                "std_err": pd.Series(result.bse, index=index, dtype=float),
                "z": pd.Series(result.tvalues, index=index, dtype=float),
                "p_value": pd.Series(result.pvalues, index=index, dtype=float),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    def summary(self) -> pd.DataFrame:
        """
        Return a compact numeric summary of the fitted ARIMA model.

        Returns
        -------
        pandas.DataFrame
            One-column table with observations, information criteria,
            log-likelihood, residual moments, and variance estimate.
        """
        result = self.fitted_result()
        params = self.parameters()
        residuals = self.residuals()

        sigma2 = float("nan")
        if "sigma2" in params.index:
            sigma2 = float(params.loc["sigma2", "coef"])

        metrics = {
            "observations": float(result.nobs),
            "aic": float(result.aic),
            "bic": float(result.bic),
            "hqic": float(result.hqic),
            "log_likelihood": float(result.llf),
            "sigma2": sigma2,
            "residual_mean": float(residuals.mean()),
            "residual_std": float(residuals.std()),
        }
        return pd.DataFrame({"value": pd.Series(metrics, dtype=float)})

    def report(self) -> pd.DataFrame:
        """Alias around `summary()` for a report-oriented workflow."""
        return self.summary()

    def forecast(self, steps: int = 5, alpha: float = 0.05) -> pd.DataFrame:
        """
        Return point forecasts and confidence intervals.

        Parameters
        ----------
        steps : int, default 5
            Number of future periods to forecast.
        alpha : float, default 0.05
            Significance level used for confidence intervals.

        Returns
        -------
        pandas.DataFrame
            Statsmodels forecast summary frame.
        """
        if steps <= 0:
            raise ValueError("steps must be greater than zero.")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1.")

        return self.fitted_result().get_forecast(steps=steps).summary_frame(
            alpha=alpha
        )

    def statsmodels_summary(self) -> str:
        """Return the native text summary generated by statsmodels."""
        return self.fitted_result().summary().as_text()


__all__ = [
    "ARIMAResearch",
]
