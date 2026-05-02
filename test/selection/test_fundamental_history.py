import numpy as np
import pandas as pd

from src.selection import FundamentalSelector
from src.selection.fundamental_metrics import build_fundamental_metric_history
from src.selection.fundamentals import FundamentalData


def _history_record(ticker="AAA"):
    periods = pd.to_datetime(
        ["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"]
    )
    income = pd.DataFrame(
        {
            period: {
                "Total Revenue": revenue,
                "Net Income": net_income,
                "Operating Income": operating_income,
                "Diluted EPS": eps,
            }
            for period, revenue, net_income, operating_income, eps in zip(
                periods,
                [100.0, 110.0, 121.0, 133.1],
                [10.0, 12.0, 14.0, 16.0],
                [15.0, 17.0, 19.0, 22.0],
                [1.0, 1.2, 1.4, 1.6],
            )
        }
    )
    balance = pd.DataFrame(
        {
            period: {
                "Stockholders Equity": equity,
                "Total Debt": 40.0,
                "Current Assets": 80.0,
                "Current Liabilities": 40.0,
                "Ordinary Shares Number": 10.0,
            }
            for period, equity in zip(periods, [90.0, 95.0, 100.0, 105.0])
        }
    )
    cash = pd.DataFrame(
        {
            period: {"Operating Cash Flow": ocf, "Capital Expenditure": -5.0}
            for period, ocf in zip(periods, [20.0, 22.0, 24.0, 27.0])
        }
    )
    prices = pd.Series([20.0, 22.0, 24.0, 26.0], index=periods, name="Close")

    return FundamentalData(
        ticker=ticker,
        info={"sharesOutstanding": 10.0},
        income_statement=income,
        balance_sheet=balance,
        cash_flow=cash,
        quarterly_income_statement=income,
        quarterly_balance_sheet=balance,
        quarterly_cash_flow=cash,
        prices=prices,
    )


class StaticProvider:
    def __init__(self, records):
        self.records = records
        self.calls = 0

    def fetch_many(self, tickers):
        self.calls += 1
        return {ticker: self.records[ticker] for ticker in tickers}


def test_build_fundamental_metric_history_tracks_period_growth():
    history = build_fundamental_metric_history(
        _history_record(),
        frequency="quarterly",
        trailing_periods=4,
    )

    assert history.shape[0] == 4
    assert history.loc[0, "frequency"] == "quarterly"
    assert history.loc[3, "revenue"] == 133.1
    np.testing.assert_allclose(history.loc[1, "revenue_period_growth"], 0.10)
    assert "eps_yoy_growth" in history.columns


def test_selector_rank_over_time_and_selection_report():
    provider = StaticProvider(
        {
            "AAA": _history_record("AAA"),
            "BBB": _history_record("BBB"),
        }
    )
    selector = FundamentalSelector(strategy="growth", provider=provider)

    ranking_history = selector.rank_over_time(
        ["AAA", "BBB"],
        frequency="quarterly",
        trailing_periods=4,
    )
    latest = selector.rank(["AAA", "BBB"])
    report = selector.selection_report(latest, top_k=1)

    assert set(ranking_history["frequency"]) == {"quarterly"}
    assert "fundamental_score" in ranking_history.columns
    assert set(report) == {
        "selected",
        "metric_snapshot",
        "score_weights",
        "score_components",
    }
    assert report["score_weights"]["metric"].tolist()


def test_metric_evolution_pivots_one_metric_by_ticker_and_period():
    provider = StaticProvider(
        {
            "AAA": _history_record("AAA"),
            "BBB": _history_record("BBB"),
        }
    )
    selector = FundamentalSelector(strategy="growth", provider=provider)
    selector.rank_over_time(
        ["AAA", "BBB"],
        frequency="quarterly",
        trailing_periods=4,
    )

    score_evolution = selector.metric_evolution("fundamental_score", top_k=2)
    eps_evolution = selector.metric_evolution("eps", tickers=["AAA"])

    assert score_evolution.shape == (2, 4)
    assert eps_evolution.index.tolist() == ["AAA"]
    assert eps_evolution.shape == (1, 4)
    np.testing.assert_allclose(eps_evolution.iloc[0, -1], 1.6)
