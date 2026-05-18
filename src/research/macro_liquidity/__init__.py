"""Macro and liquidity research primitives.

This package keeps economic research independent from client-specific policy.
It diagnoses the current macro/liquidity regime, accepts in-house regime
forecasts, and translates both into an asset-class view that downstream strategy
code can overlay with a client mandate.
"""

from .asset_view import AssetClassView, build_asset_class_view
from .audit import ResearchRunAudit
from .catalog import EconomicSeriesSpec, default_us_macro_liquidity_catalog
from .classification import RegimeClassifier, RegimeView
from .features import FeatureBuilder, FeatureBuildResult
from .forecast import ForecastScenario, RegimeForecast
from .forecasting import ForecastResult, MacroLiquidityForecaster
from .model_selection import (
    TimeSeriesFit,
    TimeSeriesModelSelectionResult,
    TimeSeriesModelSelector,
)
from .nowcast import MacroLiquidityNowcaster, NowcastResult
from .pipeline import MacroLiquidityWorkflow, MacroLiquidityWorkflowResult
from .policy import (
    AuditPolicy,
    ConfidencePolicy,
    DataPolicy,
    ForecastPolicy,
    MacroLiquidityPolicy,
    ScenarioPolicy,
    ValidationPolicy,
)
from .providers import (
    ApiConnectionSpec,
    BeaApiProvider,
    BlsApiProvider,
    FedDdpProvider,
    FredApiProvider,
    LocalSeriesProvider,
    MissingCredentialError,
    ProviderConfig,
    ProviderCredentials,
    ProviderFetchError,
    ProviderRegistry,
    TreasuryFiscalDataProvider,
    default_api_connection_specs,
)
from .scenarios import ScenarioBuildResult, ScenarioEngine, ScenarioOverride
from .regimes import (
    CurrentRegimeSnapshot,
    MacroLiquidityResearch,
    MacroLiquidityResult,
    classify_liquidity_regime,
    classify_macro_regime,
)

__all__ = [
    "ApiConnectionSpec",
    "AuditPolicy",
    "AssetClassView",
    "BeaApiProvider",
    "BlsApiProvider",
    "ConfidencePolicy",
    "CurrentRegimeSnapshot",
    "DataPolicy",
    "EconomicSeriesSpec",
    "FedDdpProvider",
    "FeatureBuildResult",
    "FeatureBuilder",
    "ForecastScenario",
    "ForecastPolicy",
    "ForecastResult",
    "FredApiProvider",
    "LocalSeriesProvider",
    "MacroLiquidityForecaster",
    "MacroLiquidityPolicy",
    "MacroLiquidityResearch",
    "MacroLiquidityResult",
    "MacroLiquidityWorkflow",
    "MacroLiquidityWorkflowResult",
    "MissingCredentialError",
    "NowcastResult",
    "ProviderConfig",
    "ProviderCredentials",
    "ProviderFetchError",
    "ProviderRegistry",
    "RegimeClassifier",
    "RegimeForecast",
    "RegimeView",
    "ResearchRunAudit",
    "ScenarioBuildResult",
    "ScenarioEngine",
    "ScenarioOverride",
    "ScenarioPolicy",
    "TreasuryFiscalDataProvider",
    "TimeSeriesFit",
    "TimeSeriesModelSelectionResult",
    "TimeSeriesModelSelector",
    "ValidationPolicy",
    "build_asset_class_view",
    "classify_liquidity_regime",
    "classify_macro_regime",
    "default_api_connection_specs",
    "default_us_macro_liquidity_catalog",
]
