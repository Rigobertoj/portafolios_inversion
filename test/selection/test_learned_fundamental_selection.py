import numpy as np
import pandas as pd

from src.selection import (
    FundamentalScoreConfig,
    LearnedFundamentalSelector,
    LearnedScoreConfigFactory,
    MetricSignalSpec,
    XGBoostFundamentalModel,
    build_forward_return_targets,
    build_fundamental_learning_panel,
)
from src.selection.fundamentals import FundamentalData


class StaticProvider:
    def __init__(self, records):
        self.records = records

    def fetch_many(self, tickers):
        return {ticker: self.records[ticker] for ticker in tickers}


class ImportanceRegressor:
    def fit(self, x, y):
        self.feature_importances_ = np.array([0.80, 0.20])[: x.shape[1]]
        return self

    def predict(self, x):
        return x.iloc[:, 0].to_numpy(dtype=float)


def _record(ticker, sector="Technology", revenue_scale=1.0, roe=0.20, debt=0.30):
    periods = pd.to_datetime(["2023-01-31", "2023-04-30", "2023-07-31", "2023-10-31"])
    income = pd.DataFrame(
        {
            period: {
                "Total Revenue": revenue_scale * revenue,
                "Net Income": roe * 500.0,
                "Operating Income": roe * 650.0,
                "Diluted EPS": eps,
            }
            for period, revenue, eps in zip(periods, [100.0, 110.0, 125.0, 140.0], [1.0, 1.1, 1.25, 1.4])
        }
    )
    balance = pd.DataFrame(
        {
            period: {
                "Stockholders Equity": 500.0,
                "Total Debt": debt * 500.0,
                "Current Assets": 300.0,
                "Current Liabilities": 150.0,
                "Ordinary Shares Number": 100.0,
            }
            for period in periods
        }
    )
    cash = pd.DataFrame(
        {
            period: {
                "Operating Cash Flow": 150.0,
                "Capital Expenditure": -30.0,
                "Cash Dividends Paid": -10.0,
            }
            for period in periods
        }
    )
    price_dates = pd.to_datetime(
        [
            "2023-01-31",
            "2023-04-30",
            "2023-07-31",
            "2023-10-31",
            "2024-01-31",
        ]
    )
    prices = pd.Series([100.0, 110.0, 121.0, 130.0, 143.0], index=price_dates, name="Close")
    dividends = pd.Series([2.0, 2.0], index=pd.to_datetime(["2023-03-15", "2023-09-15"]), name="Dividends")
    return FundamentalData(
        ticker=ticker,
        info={"sector": sector, "industry": f"{sector} Services", "sharesOutstanding": 100.0},
        income_statement=income,
        balance_sheet=balance,
        cash_flow=cash,
        quarterly_income_statement=income,
        quarterly_balance_sheet=balance,
        quarterly_cash_flow=cash,
        prices=prices,
        dividends=dividends,
    )


def test_build_forward_return_targets_uses_forward_price_and_dividends():
    metric_history = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "period": pd.to_datetime(["2023-01-31"]),
            "roe": [0.20],
        }
    )
    record = _record("AAA")

    targets = build_forward_return_targets(
        metric_history,
        {"AAA": record},
        horizon_months=3,
        reporting_lag_days=0,
    )

    np.testing.assert_allclose(targets.loc[0, "forward_price_return_3m"], 0.10)
    np.testing.assert_allclose(targets.loc[0, "forward_dividend_return_3m"], 0.02)
    np.testing.assert_allclose(targets.loc[0, "forward_total_return_3m"], 0.12)


def test_build_fundamental_learning_panel_adds_signal_features_and_targets():
    records = {
        "AAA": _record("AAA", sector="Technology"),
        "BBB": _record("BBB", sector="Technology", revenue_scale=0.8),
    }

    panel = build_fundamental_learning_panel(
        records,
        frequency="quarterly",
        trailing_periods=4,
        horizon_months=3,
        reporting_lag_days=0,
        signal_specs=[
            MetricSignalSpec("revenue", "level"),
            MetricSignalSpec("revenue", "period_change", change_method="relative"),
        ],
    )

    assert "sector" in panel.columns
    assert "revenue__level" in panel.columns
    assert "revenue__period_change" in panel.columns
    assert "forward_total_return_3m" in panel.columns
    assert panel["revenue__period_change"].notna().sum() > 0


def test_learned_score_config_factory_converts_impacts_to_weights():
    impact = pd.DataFrame(
        {
            "group": ["Technology", "Technology"],
            "feature": ["roe__level", "debt_to_equity__level"],
            "metric": ["roe", "debt_to_equity"],
            "signal": ["level", "level"],
            "importance": [0.70, 0.30],
            "higher_is_better": [True, False],
            "adjusted_impact": [0.60, 0.40],
        }
    )

    configs = LearnedScoreConfigFactory(max_features=5).from_impact_report(impact)
    config = configs["Technology"]

    specs = config.resolved_signal_specs()
    np.testing.assert_allclose(sum(spec.weight for spec in specs), 1.0)
    assert specs[0].metric == "roe"
    assert {spec.metric: spec.higher_is_better for spec in specs}["debt_to_equity"] is False


def test_learned_fundamental_selector_applies_group_specific_config():
    provider = StaticProvider(
        {
            "AAA": _record("AAA", sector="Technology", roe=0.30, debt=0.20),
            "BBB": _record("BBB", sector="Technology", roe=0.05, debt=1.00),
        }
    )
    config = FundamentalScoreConfig(
        name="learned_technology",
        signal_specs=[
            MetricSignalSpec("roe", "level", weight=0.70, higher_is_better=True),
            MetricSignalSpec("debt_to_equity", "level", weight=0.30, higher_is_better=False),
        ],
    )
    selector = LearnedFundamentalSelector(
        {"Technology": config},
        provider=provider,
        group_by="sector",
    )

    ranking = selector.rank(["AAA", "BBB"])
    report = selector.selection_report(ranking, top_k=1)

    assert ranking.loc[0, "ticker"] == "AAA"
    assert ranking.loc[0, "score_group"] == "Technology"
    assert report["selected"].loc[0, "ticker"] == "AAA"
    assert not report["score_weights"].empty


def test_xgboost_fundamental_model_uses_injected_estimator_for_impact_report():
    panel = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E", "F"],
            "sector": ["Technology"] * 6,
            "period": pd.date_range("2022-01-31", periods=6, freq="QE"),
            "available_at": pd.date_range("2022-03-31", periods=6, freq="QE"),
            "roe__level": [0.10, 0.12, 0.14, 0.20, 0.23, 0.25],
            "debt_to_equity__level": [0.80, 0.70, 0.65, 0.60, 0.50, 0.40],
            "forward_total_return_12m": [0.01, 0.02, 0.03, 0.08, 0.10, 0.11],
        }
    )
    model = XGBoostFundamentalModel(
        group_by="sector",
        min_group_size=3,
        min_feature_coverage=0.5,
        n_splits=2,
        estimator_factory=ImportanceRegressor,
    )

    model.fit(panel)

    assert not model.impact_report_.empty
    assert {"Technology", "__global__"}.issubset(set(model.impact_report_["group"]))
    assert "adjusted_impact" in model.impact_report_.columns
