import pandas as pd

from src.selection.fundamentals import YahooFundamentalsProvider


class FakeTicker:
    instances = 0

    def __init__(self, ticker):
        FakeTicker.instances += 1
        self.ticker = ticker
        self.info = {"currentPrice": 10.0}
        self.income_stmt = pd.DataFrame()
        self.balance_sheet = pd.DataFrame()
        self.cash_flow = pd.DataFrame()
        self.quarterly_income_stmt = pd.DataFrame()
        self.quarterly_balance_sheet = pd.DataFrame()
        self.quarterly_cash_flow = pd.DataFrame()

    def history(self, start=None, end=None):
        return pd.DataFrame(
            {"Close": [10.0]},
            index=pd.to_datetime(["2024-01-01"]),
        )


def test_yahoo_fundamentals_provider_caches_records_by_ticker():
    FakeTicker.instances = 0
    provider = YahooFundamentalsProvider(ticker_factory=FakeTicker)

    first = provider.fetch("aaa")
    second = provider.fetch("AAA")

    assert first is second
    assert FakeTicker.instances == 1

    provider.clear_cache()
    provider.fetch("AAA")
    assert FakeTicker.instances == 2
