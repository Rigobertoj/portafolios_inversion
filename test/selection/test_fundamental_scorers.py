import pandas as pd

from src.selection.fundamental_scorers import GrowthScoreConfig, ValueScoreConfig, score_fundamentals


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
