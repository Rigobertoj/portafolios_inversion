import pandas as pd
import pytest

from src.client_policy import mexican_moderate_aggressive_growth_policy
from src.research.macro_liquidity import (
    BeaApiProvider,
    BlsApiProvider,
    EconomicSeriesSpec,
    FeatureBuilder,
    FedDdpProvider,
    ForecastScenario,
    FredApiProvider,
    MacroLiquidityPolicy,
    MacroLiquidityResearch,
    MacroLiquidityWorkflow,
    MissingCredentialError,
    ProviderConfig,
    ProviderCredentials,
    ProviderRegistry,
    RegimeForecast,
    TreasuryFiscalDataProvider,
    build_asset_class_view,
)
from src.strategy import build_selection_context


def _sample_series_frame():
    dates = pd.date_range("2024-01-31", periods=8, freq=pd.offsets.MonthEnd())
    values = {
        "real_gdp": [100, 101, 102, 104, 106, 108, 111, 114],
        "cpi": [110, 111, 112, 112, 111, 110, 109, 108],
        "unemployment_rate": [4.3, 4.2, 4.1, 4.0, 3.9, 3.8, 3.8, 3.7],
        "reserve_balances": [3.0, 3.1, 3.3, 3.5, 3.8, 4.1, 4.3, 4.6],
        "tga": [900, 860, 820, 760, 690, 630, 590, 540],
        "hy_oas": [4.8, 4.6, 4.3, 4.0, 3.7, 3.4, 3.2, 3.0],
    }
    return pd.DataFrame(
        [
            {"date": date, "series": series, "value": value}
            for series, series_values in values.items()
            for date, value in zip(dates, series_values)
        ]
    )


def _sample_specs():
    return (
        EconomicSeriesSpec("real_gdp", "macro", "growth", True),
        EconomicSeriesSpec("cpi", "macro", "inflation", False),
        EconomicSeriesSpec("unemployment_rate", "macro", "labor", False),
        EconomicSeriesSpec("reserve_balances", "liquidity", "fed_liquidity", True),
        EconomicSeriesSpec("tga", "liquidity", "treasury_liquidity", False),
        EconomicSeriesSpec("hy_oas", "liquidity", "financial_conditions", False),
    )


def _macro_model_specs():
    return (
        EconomicSeriesSpec(
            "real_gdp",
            "macro",
            "growth",
            True,
            frequency="Q",
            transform="yoy_change",
            periods=4,
            weight=1.2,
        ),
        EconomicSeriesSpec(
            "cpi",
            "macro",
            "inflation",
            False,
            frequency="M",
            transform="yoy_change",
            periods=12,
            weight=1.2,
        ),
        EconomicSeriesSpec(
            "unemployment_rate",
            "macro",
            "labor",
            False,
            frequency="M",
            transform="level",
            periods=1,
        ),
        EconomicSeriesSpec(
            "reserve_balances",
            "liquidity",
            "fed_liquidity",
            True,
            frequency="M",
            transform="yoy_change",
            periods=12,
        ),
        EconomicSeriesSpec(
            "hy_oas",
            "liquidity",
            "financial_conditions",
            False,
            frequency="M",
            transform="level",
            periods=1,
        ),
    )


def _macro_model_frame():
    monthly_dates = pd.date_range("2020-01-31", periods=48, freq=pd.offsets.MonthEnd())
    quarterly_dates = pd.date_range("2020-03-31", periods=16, freq=pd.offsets.QuarterEnd())
    rows = []
    for idx, date in enumerate(quarterly_dates):
        rows.append({"date": date, "series": "real_gdp", "value": 100 + idx * 1.1})
    for idx, date in enumerate(monthly_dates):
        rows.extend(
            [
                {"date": date, "series": "cpi", "value": 100 + idx * 0.25},
                {
                    "date": date,
                    "series": "unemployment_rate",
                    "value": 4.8 - min(idx, 30) * 0.03 + max(idx - 36, 0) * 0.04,
                },
                {"date": date, "series": "reserve_balances", "value": 2.5 + idx * 0.04},
                {"date": date, "series": "hy_oas", "value": 5.0 - idx * 0.03},
            ]
        )
    return pd.DataFrame(rows)


def test_current_expected_and_client_overlay_build_selection_context():
    result = MacroLiquidityResearch(_sample_specs(), min_periods=3).analyze_current(
        _sample_series_frame()
    )
    forecast = RegimeForecast(
        scenarios=(
            ForecastScenario(
                name="base",
                macro_regime="desaceleracion_ordenada",
                liquidity_regime="restrictiva",
                probability=0.65,
                confidence=0.70,
            ),
            ForecastScenario(
                name="upside",
                macro_regime="expansion_desinflacionaria",
                liquidity_regime="neutral",
                probability=0.35,
                confidence=0.50,
            ),
        )
    )
    asset_view = build_asset_class_view(result.current, forecast)
    context = build_selection_context(
        current=result.current,
        forecast=forecast,
        asset_view=asset_view,
        policy=mexican_moderate_aggressive_growth_policy(),
    )

    assert result.current.macro_regime == "expansion_desinflacionaria"
    assert result.current.liquidity_regime == "expansiva"
    assert context.expected_liquidity_regime == "restrictiva"
    assert context.base_currency == "MXN"
    assert "fossil_fuels" in context.excluded_sectors
    assert "quality" in context.preferred_styles
    assert context.score_tilts["profitability"] > 0


def test_provider_config_reports_detached_missing_credentials():
    config = ProviderConfig(env={})

    report = config.availability_report()

    assert "FRED" in set(report["provider"])
    assert not bool(report.loc[report["provider"] == "FRED", "available"].iloc[0])
    assert not bool(report.loc[report["provider"] == "FRED", "ready_for_fetch"].iloc[0])
    with pytest.raises(MissingCredentialError):
        config.require_token("FRED")


def test_provider_config_separates_credentials_from_provider_readiness():
    config = ProviderConfig(
        env={
            "FRED_API_KEY": "fred-token",
            "BEA_API_KEY": "bea-token",
            "EIA_API_KEY": "eia-token",
        }
    )

    report = config.availability_report()
    fred = report[report["provider"] == "FRED"].iloc[0]
    eia = report[report["provider"] == "EIA"].iloc[0]

    assert bool(fred["credential_available"])
    assert bool(fred["provider_implemented"])
    assert bool(fred["ready_for_fetch"])
    assert bool(eia["credential_available"])
    assert not bool(eia["provider_implemented"])
    assert not bool(eia["ready_for_fetch"])


def test_fred_provider_consumes_credentials_without_owning_env(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"observations": [{"date": "2024-01-01", "value": "5.25"}]}

    def fake_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("src.research.macro_liquidity.providers.requests.get", fake_get)
    credentials = ProviderCredentials(env={}, tokens={"FRED": "not-a-real-token"})
    provider = FredApiProvider(credentials=credentials, timeout=7)
    spec = EconomicSeriesSpec(
        "fed_funds",
        "macro",
        "policy",
        False,
        provider="FRED",
        provider_code="FEDFUNDS",
    )

    frame = provider.fetch((spec,), start="2024-01-01", end="2024-01-31")

    assert captured["params"]["api_key"] == "not-a-real-token"
    assert captured["params"]["series_id"] == "FEDFUNDS"
    assert captured["timeout"] == 7
    assert list(frame.columns) == ["date", "series", "value"]
    assert frame.loc[0, "series"] == "fed_funds"


def test_bea_provider_fetches_json_with_provider_params(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "BEAAPI": {
                    "Results": {
                        "Data": [
                            {"TimePeriod": "2024Q1", "DataValue": "23,456.7"},
                            {"TimePeriod": "2024Q2", "DataValue": "23,900.1"},
                        ]
                    }
                }
            }

    def fake_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("src.research.macro_liquidity.providers.requests.get", fake_get)
    provider = BeaApiProvider(credentials=ProviderCredentials(env={}, tokens={"BEA": "bea-token"}))
    spec = EconomicSeriesSpec(
        "bea_real_gdp",
        "macro",
        "growth",
        True,
        provider="BEA",
        provider_params={
            "DatasetName": "NIPA",
            "TableName": "T10106",
            "LineNumber": "1",
            "Frequency": "Q",
        },
    )

    frame = provider.fetch((spec,), start="2024-01-01", end="2024-12-31")

    assert captured["params"]["UserID"] == "bea-token"
    assert captured["params"]["DatasetName"] == "NIPA"
    assert captured["params"]["Year"] == "2024"
    assert frame["series"].unique().tolist() == ["bea_real_gdp"]
    assert frame["value"].tolist() == [23456.7, 23900.1]


def test_bls_provider_posts_series_payload_and_parses_observations(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "status": "REQUEST_SUCCEEDED",
                "Results": {
                    "series": [
                        {
                            "seriesID": "LNS14000000",
                            "data": [
                                {"year": "2024", "period": "M02", "value": "3.9"},
                                {"year": "2024", "period": "M01", "value": "3.7"},
                            ],
                        }
                    ]
                },
            }

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("src.research.macro_liquidity.providers.requests.post", fake_post)
    provider = BlsApiProvider(credentials=ProviderCredentials(env={}, tokens={"BLS": "bls-token"}))
    spec = EconomicSeriesSpec(
        "bls_unemployment_rate",
        "macro",
        "labor",
        False,
        provider="BLS",
        provider_code="LNS14000000",
    )

    frame = provider.fetch((spec,), start="2024-01-01", end="2024-12-31")

    assert captured["json"]["seriesid"] == ["LNS14000000"]
    assert captured["json"]["registrationkey"] == "bls-token"
    assert captured["json"]["startyear"] == "2024"
    assert frame["series"].unique().tolist() == ["bls_unemployment_rate"]
    assert frame["value"].tolist() == [3.7, 3.9]


def test_treasury_provider_fetches_fiscal_data_json(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {"record_date": "2024-01-31", "close_today_bal": "850.5"},
                    {"record_date": "2024-02-29", "close_today_bal": "825.0"},
                ]
            }

    def fake_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("src.research.macro_liquidity.providers.requests.get", fake_get)
    provider = TreasuryFiscalDataProvider()
    spec = EconomicSeriesSpec(
        "treasury_cash_balance",
        "liquidity",
        "treasury_liquidity",
        False,
        provider="TREASURY",
        provider_code="v1/accounting/dts/dts_table_1",
        provider_params={
            "_date_field": "record_date",
            "_value_field": "close_today_bal",
            "_fields": "record_date,close_today_bal",
        },
    )

    frame = provider.fetch((spec,), start="2024-01-01", end="2024-03-01")

    assert captured["url"].endswith("/v1/accounting/dts/dts_table_1")
    assert "record_date:gte:2024-01-01" in captured["params"]["filter"]
    assert frame["series"].unique().tolist() == ["treasury_cash_balance"]
    assert frame["value"].tolist() == [850.5, 825.0]


def test_fed_ddp_provider_fetches_csv_package(monkeypatch):
    captured = {}

    class FakeResponse:
        text = (
            "Metadata row,ignored\n"
            "Time Period,H15/H15/RIFLGFCY10_N.B\n"
            "2024-01-01,3.88\n"
            "2024-01-02,3.92\n"
        )

        def raise_for_status(self):
            return None

    def fake_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("src.research.macro_liquidity.providers.requests.get", fake_get)
    provider = FedDdpProvider()
    spec = EconomicSeriesSpec(
        "h15_ten_year_yield",
        "macro",
        "policy",
        False,
        provider="FED_DDP",
        provider_code="abc123",
        provider_params={"_value_column": "H15/H15/RIFLGFCY10_N.B", "_release": "H15"},
    )

    frame = provider.fetch((spec,), start="2024-01-01", end="2024-01-31")

    assert captured["url"].endswith("/Download.aspx")
    assert captured["params"]["rel"] == "H15"
    assert captured["params"]["series"] == "abc123"
    assert frame["series"].unique().tolist() == ["h15_ten_year_yield"]
    assert frame["value"].tolist() == [3.88, 3.92]


def test_provider_registry_routes_mixed_specs():
    class StaticProvider:
        def __init__(self, rows):
            self.rows = rows

        def fetch(self, specs, start=None, end=None):
            return pd.DataFrame(self.rows)

    registry = ProviderRegistry(
        {
            "BEA": StaticProvider(
                [{"date": "2024-03-31", "series": "bea_real_gdp", "value": 2.1}]
            ),
            "TREASURY": StaticProvider(
                [{"date": "2024-03-31", "series": "treasury_cash_balance", "value": 825.0}]
            ),
        }
    )
    specs = (
        EconomicSeriesSpec("bea_real_gdp", "macro", "growth", True, provider="BEA"),
        EconomicSeriesSpec(
            "treasury_cash_balance",
            "liquidity",
            "treasury_liquidity",
            False,
            provider="TREASURY",
        ),
    )

    frame = registry.fetch(specs)

    assert set(frame["series"]) == {"bea_real_gdp", "treasury_cash_balance"}
    assert list(frame.columns) == ["date", "series", "value"]


def test_provider_registry_can_continue_when_one_provider_fails():
    class FailingProvider:
        def fetch(self, specs, start=None, end=None):
            raise ValueError("BEA API error for bea_real_gdp: UserId is not active")

    class StaticProvider:
        def fetch(self, specs, start=None, end=None):
            return pd.DataFrame(
                [{"date": "2024-01-31", "series": "bls_unemployment_rate", "value": 3.7}]
            )

    registry = ProviderRegistry(
        {"BEA": FailingProvider(), "BLS": StaticProvider()},
        continue_on_error=True,
    )
    specs = (
        EconomicSeriesSpec("bea_real_gdp", "macro", "growth", True, provider="BEA"),
        EconomicSeriesSpec(
            "bls_unemployment_rate",
            "macro",
            "labor",
            False,
            provider="BLS",
        ),
    )

    frame = registry.fetch(specs)
    errors = registry.error_report()

    assert frame["series"].tolist() == ["bls_unemployment_rate"]
    assert errors.loc[0, "provider"] == "BEA"
    assert errors.loc[0, "series"] == "bea_real_gdp"
    assert errors.loc[0, "error_type"] == "ValueError"
    assert "UserId is not active" in errors.loc[0, "message"]


def test_feature_builder_uses_revised_monthly_policy_and_interpolates_quarterly_gdp():
    policy = MacroLiquidityPolicy()
    result = FeatureBuilder(_macro_model_specs(), policy=policy, min_periods=6).build(
        _macro_model_frame(),
        as_of="2024-04-30",
    )

    gdp_features = result.features[result.features["series"] == "real_gdp"]
    assert result.methodology["data_mode"] == "revised"
    assert result.methodology["real_time_safe"] is False
    assert result.methodology["canonical_frequency"] == "M"
    assert result.methodology["rebalance_frequency"] == "Q"
    assert gdp_features["interpolated"].any()
    assert set(result.validation["status"]).issubset({"active", "candidate"})


def test_macro_liquidity_workflow_builds_auditable_model_to_selection_flow(tmp_path):
    policy = MacroLiquidityPolicy()
    policy = MacroLiquidityPolicy(
        data=policy.data,
        validation=policy.validation,
        confidence=policy.confidence,
        forecast=policy.forecast,
        scenario=policy.scenario,
        audit=type(policy.audit)(run_root=str(tmp_path), write_artifacts=True),
    )

    workflow = MacroLiquidityWorkflow(_macro_model_specs(), policy=policy, min_periods=6)
    result = workflow.run(
        _macro_model_frame(),
        as_of="2024-04-30",
        client_policy=mexican_moderate_aggressive_growth_policy(),
        write_audit=True,
    )

    tables = result.summary_tables()

    assert result.regime_view.expected.macro_regime != "unknown"
    assert result.scenarios.forecast.base_case.name == "base"
    assert result.selection_context is not None
    assert "forecast_diagnostics" in tables
    assert "selection_context" in tables
    assert result.audit_dir is not None
    assert (result.audit_dir / "run_summary.json").exists()
    assert (result.audit_dir / "forecast_table.csv").exists()
    assert (result.audit_dir / "model_diagnostics" / "model_selection.csv").exists()
    assert (result.audit_dir / "model_diagnostics" / "rolling_backtests.csv").exists()
    assert "model_selection" in tables
    assert "rolling_backtests" in tables
    assert set(result.forecast.model_reports["model"]).issuperset(
        {"mean", "naive", "drift", "ar1", "mean_reversion", "bootstrap_ar1"}
    )
    assert set(result.forecast.diagnostics["benchmark_model"]) == {"ar1"}
    assert result.forecast.diagnostics["model"].notna().all()
