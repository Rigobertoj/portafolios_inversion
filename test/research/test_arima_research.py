import numpy as np
import pandas as pd
import pytest

from src.research import ARIMAResearch

pytest.importorskip(
    "statsmodels.tsa.arima.model",
    reason="statsmodels is required for ARIMAResearch fit tests",
)


def _sample_series() -> pd.Series:
    rng = np.random.default_rng(42)
    values = np.zeros(120)
    noise = rng.normal(scale=0.3, size=120)

    for idx in range(1, len(values)):
        values[idx] = 0.6 * values[idx - 1] + noise[idx]

    return pd.Series(
        values,
        index=pd.date_range("2024-01-01", periods=len(values), freq="D"),
        name="returns",
    )


def test_summary_and_report_return_compact_numeric_table():
    research = ARIMAResearch(series=_sample_series(), order=(1, 0, 0))

    summary = research.summary()
    report = research.report()

    pd.testing.assert_frame_equal(summary, report)
    assert set(summary.index) == {
        "observations",
        "aic",
        "bic",
        "hqic",
        "log_likelihood",
        "sigma2",
        "residual_mean",
        "residual_std",
    }
    assert set(summary.columns) == {"value"}


def test_parameters_table_includes_inference_columns():
    research = ARIMAResearch(series=_sample_series(), order=(1, 0, 0))

    params = research.parameters()

    assert {"coef", "std_err", "z", "p_value", "ci_low", "ci_high"} == set(
        params.columns
    )
    assert "ar.L1" in params.index
    assert "sigma2" in params.index


def test_forecast_returns_requested_number_of_rows():
    research = ARIMAResearch(series=_sample_series(), order=(1, 0, 0))

    forecast = research.forecast(steps=3, alpha=0.1)

    assert len(forecast) == 3
    assert {"mean", "mean_se", "mean_ci_lower", "mean_ci_upper"} == set(
        forecast.columns
    )


def test_statsmodels_summary_exposes_native_text_output():
    research = ARIMAResearch(series=_sample_series(), order=(1, 0, 0))

    text_summary = research.statsmodels_summary()

    assert "ARIMA(1, 0, 0)" in text_summary


def test_invalid_forecast_steps_raise_value_error():
    research = ARIMAResearch(series=_sample_series(), order=(1, 0, 0))

    with pytest.raises(ValueError, match="steps must be greater than zero"):
        research.forecast(steps=0)
