import numpy as np
import pandas as pd

from src.selection.fundamental_metrics import build_fundamental_metrics
from src.selection.fundamentals import FundamentalData


def _fundamental_data() -> FundamentalData:
    periods = pd.to_datetime(["2022-12-31", "2023-12-31"])
    income = pd.DataFrame(
        {
            periods[1]: {
                "Total Revenue": 1200.0,
                "Net Income": 180.0,
                "Operating Income": 240.0,
                "Diluted EPS": 3.0,
            },
            periods[0]: {
                "Total Revenue": 1000.0,
                "Net Income": 150.0,
                "Operating Income": 200.0,
                "Diluted EPS": 2.5,
            },
        }
    )
    balance = pd.DataFrame(
        {
            periods[1]: {
                "Stockholders Equity": 900.0,
                "Total Debt": 300.0,
                "Current Assets": 600.0,
                "Current Liabilities": 300.0,
                "Ordinary Shares Number": 60.0,
            },
            periods[0]: {
                "Stockholders Equity": 800.0,
                "Total Debt": 320.0,
                "Current Assets": 550.0,
                "Current Liabilities": 275.0,
                "Ordinary Shares Number": 60.0,
            },
        }
    )
    cash = pd.DataFrame(
        {
            periods[1]: {"Operating Cash Flow": 220.0, "Capital Expenditure": -40.0},
            periods[0]: {"Operating Cash Flow": 190.0, "Capital Expenditure": -35.0},
        }
    )
    prices = pd.Series(
        [28.0, 30.0],
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        name="Close",
    )

    return FundamentalData(
        ticker="AAA",
        info={},
        income_statement=income,
        balance_sheet=balance,
        cash_flow=cash,
        quarterly_income_statement=income,
        quarterly_balance_sheet=balance,
        quarterly_cash_flow=cash,
        prices=prices,
    )


def test_build_fundamental_metrics_computes_core_ratios():
    metrics = build_fundamental_metrics(_fundamental_data())

    assert metrics["ticker"] == "AAA"
    assert metrics["current_price"] == 30.0
    assert metrics["eps"] == 3.0
    assert metrics["trailing_pe"] == 10.0
    assert metrics["price_to_book"] == 2.0
    assert metrics["free_cash_flow"] == 180.0
    assert metrics["current_ratio"] == 2.0

    np.testing.assert_allclose(metrics["roe"], 180.0 / 900.0)
    np.testing.assert_allclose(metrics["profit_margin"], 180.0 / 1200.0)
    np.testing.assert_allclose(metrics["revenue_growth"], 0.20)
    np.testing.assert_allclose(metrics["eps_growth"], 0.20)
