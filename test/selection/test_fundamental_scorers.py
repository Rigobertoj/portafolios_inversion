import pandas as pd

from src.selection.fundamental_scorers import (
    FundamentalScoreConfig,
    FUNDAMENTAL_METRIC_SIGNAL_SPECS,
    GrowthScoreConfig,
    MetricSignalSpec,
    ValueScoreConfig,
    score_fundamentals,
)


def test_value_score_rewards_lower_multiples_and_quality():
    metrics = pd.DataFrame(
        {
            "ticker": ["CHEAP", "EXPENSIVE"],
            "trailing_pe": [10.0, 35.0],
            "price_to_book": [1.2, 9.0],
            "debt_to_equity": [0.3, 2.5],
            "roe": [0.22, 0.06],
            "profit_margin": [0.18, 0.04],
            "free_cash_flow_margin": [0.12, -0.02],
            "current_ratio": [2.0, 0.8],
            "market_cap": [1_000.0, 2_000.0],
        }
    )

    ranking = score_fundamentals(metrics, ValueScoreConfig())

    assert ranking.loc[0, "ticker"] == "CHEAP"
    assert ranking.loc[0, "fundamental_score"] > ranking.loc[1, "fundamental_score"]


def test_growth_score_rewards_growth_and_penalizes_expensive_growth():
    metrics = pd.DataFrame(
        {
            "ticker": ["GROWER", "SLOW"],
            "revenue_growth": [0.35, 0.02],
            "eps_growth": [0.42, -0.05],
            "earnings_growth": [0.30, 0.01],
            "roe": [0.28, 0.08],
            "profit_margin": [0.22, 0.06],
            "operating_margin": [0.30, 0.07],
            "peg_ratio": [1.1, 4.0],
            "debt_to_equity": [0.2, 1.8],
        }
    )

    ranking = score_fundamentals(metrics, GrowthScoreConfig())

    assert ranking.loc[0, "ticker"] == "GROWER"
    assert ranking.loc[0, "fundamental_score"] > ranking.loc[1, "fundamental_score"]


def test_signal_score_uses_period_yoy_and_historical_change():
    periods = pd.to_datetime(
        ["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31"]
    )
    metric_history = pd.DataFrame(
        [
            {"ticker": ticker, "period": period, "revenue": revenue, "gross_margin": margin}
            for ticker, revenues, margins in [
                ("AAA", [100.0, 110.0, 121.0, 133.1, 150.0], [0.30, 0.31, 0.32, 0.33, 0.35]),
                ("BBB", [100.0, 98.0, 95.0, 93.0, 90.0], [0.30, 0.29, 0.28, 0.26, 0.25]),
            ]
            for period, revenue, margin in zip(periods, revenues, margins)
        ]
    )
    metrics = metric_history[metric_history["period"] == periods[-1]].copy()
    config = FundamentalScoreConfig(
        name="test_signals",
        signal_specs=(
            MetricSignalSpec("revenue", "period_change", 0.4, True, change_method="relative"),
            MetricSignalSpec("revenue", "yoy_change", 0.4, True, change_method="relative"),
            MetricSignalSpec("gross_margin", "historical_avg_change", 0.2, True, change_method="difference"),
        ),
    )

    ranking = score_fundamentals(
        metrics,
        config,
        metric_history=metric_history,
        frequency="quarterly",
    )

    assert ranking.loc[0, "ticker"] == "AAA"
    assert "revenue__period_change" in ranking.columns
    assert "revenue__yoy_change_score" in ranking.columns
    assert "gross_margin__historical_avg_change_score" in ranking.columns
    assert ranking["score_coverage"].min() == 1.0


def test_fundamental_catalog_covers_reference_metric_rows():
    assert len(FUNDAMENTAL_METRIC_SIGNAL_SPECS) == 91
    assert {spec.signal for spec in FUNDAMENTAL_METRIC_SIGNAL_SPECS} == {"level"}
    assert "net_operating_cycle" in {spec.metric for spec in FUNDAMENTAL_METRIC_SIGNAL_SPECS}
    assert "enterprise_value_to_ebitda" in {spec.metric for spec in FUNDAMENTAL_METRIC_SIGNAL_SPECS}
