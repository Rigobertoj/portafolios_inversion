import pandas as pd

from src.selection import FundamentalSelector
from src.selection.fundamentals import FundamentalData


class StaticFundamentalsProvider:
    def __init__(self, records):
        self.records = records

    def fetch_many(self, tickers):
        return {ticker: self.records[ticker] for ticker in tickers}


def _record(ticker, pe, pb, roe, margin, debt):
    periods = pd.to_datetime(["2023-12-31"])
    income = pd.DataFrame(
        {
            periods[0]: {
                "Total Revenue": 1000.0,
                "Net Income": roe * 500.0,
                "Operating Income": margin * 1000.0,
            }
        }
    )
    balance = pd.DataFrame(
        {
            periods[0]: {
                "Stockholders Equity": 500.0,
                "Total Debt": debt * 500.0,
                "Current Assets": 300.0,
                "Current Liabilities": 150.0,
                "Ordinary Shares Number": 100.0,
            }
        }
    )
    cash = pd.DataFrame(
        {periods[0]: {"Operating Cash Flow": 150.0, "Capital Expenditure": -30.0}}
    )
    price = pe * (roe * 500.0 / 100.0)
    return FundamentalData(
        ticker=ticker,
        info={
            "currentPrice": price,
            "trailingPE": pe,
            "priceToBook": pb,
            "sharesOutstanding": 100.0,
            "profitMargins": margin,
        },
        income_statement=income,
        balance_sheet=balance,
        cash_flow=cash,
        quarterly_income_statement=income,
        quarterly_balance_sheet=balance,
        quarterly_cash_flow=cash,
        prices=pd.Series([price], index=pd.to_datetime(["2024-01-03"]), name="Close"),
    )


def test_fundamental_selector_rank_and_select_top():
    provider = StaticFundamentalsProvider(
        {
            "AAA": _record("AAA", pe=9.0, pb=1.1, roe=0.22, margin=0.20, debt=0.2),
            "BBB": _record("BBB", pe=30.0, pb=7.0, roe=0.06, margin=0.03, debt=2.0),
            "CCC": _record("CCC", pe=14.0, pb=2.0, roe=0.18, margin=0.14, debt=0.5),
        }
    )
    selector = FundamentalSelector(strategy="value", provider=provider)

    ranking = selector.rank(["bbb", "aaa", "ccc"])
    selected = selector.select_top(ranking, top_k=2)

    assert ranking.loc[0, "ticker"] == "AAA"
    assert selected["ticker"].tolist() == ["AAA", "CCC"]
    assert selector.raw_data.keys() == {"BBB", "AAA", "CCC"}
    assert "fundamental_score" in ranking.columns


def test_run_pipeline_returns_selected_tickers():
    provider = StaticFundamentalsProvider(
        {
            "AAA": _record("AAA", pe=9.0, pb=1.1, roe=0.22, margin=0.20, debt=0.2),
            "BBB": _record("BBB", pe=30.0, pb=7.0, roe=0.06, margin=0.03, debt=2.0),
        }
    )
    selector = FundamentalSelector(strategy="value", provider=provider)

    result = selector.run_pipeline(["AAA", "BBB"], top_k=1)

    assert result["selected_tickers"] == ["AAA"]
    assert result["selected"].shape[0] == 1


def test_set_strategy_updates_score_config_and_clears_rankings():
    provider = StaticFundamentalsProvider(
        {
            "AAA": _record("AAA", pe=9.0, pb=1.1, roe=0.22, margin=0.20, debt=0.2),
            "BBB": _record("BBB", pe=30.0, pb=7.0, roe=0.06, margin=0.03, debt=2.0),
        }
    )
    selector = FundamentalSelector(strategy="value", provider=provider)
    selector.rank(["AAA", "BBB"])

    assert selector.score_config.name == "value"
    assert not selector.metrics_.empty
    assert not selector.ranking_.empty

    selector.set_strategy("growth")

    assert selector.strategy == "growth"
    assert selector.score_config.name == "growth"
    assert "revenue_growth" in selector.score_config.metric_weights
    assert not selector.metrics_.empty
    assert selector.ranking_.empty
    assert selector.ranking_history_.empty
