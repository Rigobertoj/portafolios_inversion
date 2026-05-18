"""End-to-end macro/liquidity research workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .asset_view import AssetClassView, build_asset_class_view
from .audit import ResearchRunAudit
from .catalog import EconomicSeriesSpec
from .classification import RegimeClassifier, RegimeView
from .features import FeatureBuildResult, FeatureBuilder
from .forecasting import ForecastResult, MacroLiquidityForecaster
from .nowcast import MacroLiquidityNowcaster, NowcastResult
from .policy import MacroLiquidityPolicy
from .scenarios import ScenarioBuildResult, ScenarioEngine, ScenarioOverride


@dataclass(frozen=True)
class MacroLiquidityWorkflowResult:
    """Complete research output from data to selection-ready posture."""

    features: FeatureBuildResult
    nowcast: NowcastResult
    forecast: ForecastResult
    regime_view: RegimeView
    scenarios: ScenarioBuildResult
    asset_view: AssetClassView
    selection_context: object | None
    audit_dir: Path | None

    def summary_tables(self) -> dict[str, pd.DataFrame]:
        """Return key notebook tables."""

        tables = {
            "feature_validation": self.features.validation,
            "nowcast_confidence": self.nowcast.confidence_components,
            "nowcast_contributors": self.nowcast.contributors,
            "forecast_diagnostics": self.forecast.diagnostics,
            "model_selection": self.forecast.model_reports,
            "rolling_backtests": self.forecast.rolling_backtests,
            "forecast_table": self.forecast.forecasts,
            "regime_probabilities": self.regime_view.probabilities,
            "scenario_table": self.scenarios.scenario_table,
            "asset_view": self.asset_view.to_frame(),
        }
        if self.selection_context is not None:
            tables["selection_context"] = self.selection_context.to_frame()
        return tables


class MacroLiquidityWorkflow:
    """Run the implemented research pipeline after data ingestion."""

    def __init__(
        self,
        specs: Iterable[EconomicSeriesSpec],
        policy: Optional[MacroLiquidityPolicy] = None,
        min_periods: int = 6,
    ) -> None:
        self.specs = tuple(specs)
        self.policy = policy or MacroLiquidityPolicy()
        self.min_periods = max(int(min_periods), 2)

    def run(
        self,
        series_data: pd.DataFrame,
        as_of: Optional[str | pd.Timestamp] = None,
        client_policy: object | None = None,
        scenario_overrides: Optional[Iterable[ScenarioOverride]] = None,
        provider_errors: Optional[pd.DataFrame] = None,
        write_audit: bool = True,
    ) -> MacroLiquidityWorkflowResult:
        """Run feature, nowcast, forecast, regime, scenario, and asset-view stages."""

        features = FeatureBuilder(self.specs, self.policy, self.min_periods).build(
            series_data,
            as_of=as_of,
        )
        nowcast = MacroLiquidityNowcaster(self.policy).fit_transform(features.features)
        forecast = MacroLiquidityForecaster(self.policy).fit_predict(nowcast.scores)
        regime_view = RegimeClassifier(self.policy).classify(nowcast, forecast)
        scenarios = ScenarioEngine(self.policy).build(regime_view, overrides=scenario_overrides)
        asset_view = build_asset_class_view(regime_view.current, scenarios.forecast)

        selection_context = None
        if client_policy is not None:
            from src.strategy import build_selection_context

            selection_context = build_selection_context(
                current=regime_view.current,
                forecast=scenarios.forecast,
                asset_view=asset_view,
                policy=client_policy,
            )

        audit_dir = None
        if write_audit:
            audit = ResearchRunAudit(self.policy)
            audit_dir = audit.prepare()
            audit.write_json("input_manifest.json", features.methodology)
            audit.write_frame("aligned_series.csv", features.aligned_series)
            audit.write_frame("feature_panel.csv", features.features)
            audit.write_frame("feature_validation.csv", features.validation)
            audit.write_frame("nowcast_scores.csv", nowcast.scores)
            audit.write_frame("nowcast_confidence.csv", nowcast.confidence_components)
            audit.write_frame("nowcast_contributors.csv", nowcast.contributors)
            audit.write_frame("forecast_table.csv", forecast.forecasts)
            audit.write_frame("model_diagnostics/forecast_diagnostics.csv", forecast.diagnostics)
            audit.write_frame("model_diagnostics/model_selection.csv", forecast.model_reports)
            audit.write_frame("model_diagnostics/rolling_backtests.csv", forecast.rolling_backtests)
            audit.write_frame("regime_probabilities.csv", regime_view.probabilities)
            audit.write_frame("scenario_table.csv", scenarios.scenario_table)
            audit.write_frame("scenario_overrides.csv", scenarios.override_log)
            audit.write_frame("asset_view.csv", asset_view.to_frame())
            if selection_context is not None:
                audit.write_frame("selection_context.csv", selection_context.to_frame())
            if provider_errors is not None:
                audit.write_frame("provider_errors.csv", provider_errors)
            audit.write_run_summary(
                features.methodology,
                current_label=regime_view.current.label(),
                expected_label=f"{regime_view.expected.macro_regime} + {regime_view.expected.liquidity_regime}",
                errors=provider_errors,
            )
            audit.write_markdown(
                "research_report.md",
                self._markdown_report(regime_view, scenarios, asset_view),
            )

        return MacroLiquidityWorkflowResult(
            features=features,
            nowcast=nowcast,
            forecast=forecast,
            regime_view=regime_view,
            scenarios=scenarios,
            asset_view=asset_view,
            selection_context=selection_context,
            audit_dir=audit_dir,
        )

    @staticmethod
    def _markdown_report(
        regime_view: RegimeView,
        scenarios: ScenarioBuildResult,
        asset_view: AssetClassView,
    ) -> str:
        return "\n".join(
            [
                "# Macro Liquidity Research Report",
                "",
                f"Current regime: {regime_view.current.label()}",
                f"Expected regime: {regime_view.expected.macro_regime} + {regime_view.expected.liquidity_regime}",
                f"Transition: {regime_view.transition}",
                f"Forecast confidence: {regime_view.confidence:.2f}",
                "",
                "## Scenarios",
                "```text",
                scenarios.scenario_table.to_string(index=False),
                "```",
                "",
                "## Asset View",
                "```text",
                asset_view.to_frame().to_string(index=False),
                "```",
                "",
            ]
        )
