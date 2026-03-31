from src.asset_allocation import (
    MaximumOmegaConfig,
    MinimumSemivarianceConfig,
    MinimumVarianceConfig,
    OptimizationConfig,
    OptimizationResult,
    PortfolioElementaryAnalysis,
    PortfolioElementaryMetrics,
    PortfolioOptimization,
    PortfolioOptimizationPostModern,
    PortfolioPostModernMetrics,
    PostModernOptimizationConfig,
    PostModernOptimizationResult,
)
from src.backtesting import (
    AllocationStrategy,
    BacktestConfig,
    BacktestResult,
    BacktestStrategyResult,
    Backtester,
    MeanVarianceStrategy,
    PostModernStrategy,
    StaticBacktestEngine,
    StrategyAllocation,
)
from src.managment_risk import (
    AllocationStrategy as LegacyAllocationStrategy,
)
from src.managment_risk import (
    BacktestConfig as LegacyBacktestConfig,
)
from src.managment_risk import (
    BacktestResult as LegacyBacktestResult,
)
from src.managment_risk import (
    BacktestStrategyResult as LegacyBacktestStrategyResult,
)
from src.managment_risk import Backtester as LegacyBacktester
from src.managment_risk import (
    MeanVarianceStrategy as LegacyMeanVarianceStrategy,
)
from src.managment_risk import (
    PostModernStrategy as LegacyPostModernStrategy,
)
from src.managment_risk import StrategyAllocation as LegacyStrategyAllocation
from src.optimization import (
    MaximumOmegaConfig as NewMaximumOmegaConfig,
    MeanVarianceOptimizer,
    MinimumSemivarianceConfig as NewMinimumSemivarianceConfig,
    MinimumVarianceConfig as NewMinimumVarianceConfig,
    OptimizationConfig as NewOptimizationConfig,
    OptimizationResult as NewOptimizationResult,
    PostModernOptimizer,
    PostModernOptimizationConfig as NewPostModernOptimizationConfig,
    PostModernOptimizationResult as NewPostModernOptimizationResult,
)
from src.portfolio import (
    Portfolio,
    PortfolioBasicMetrics,
    PortfolioBenchmarkAnalysis,
    PortfolioDownsideMetrics,
    PortfolioPerformanceAnalysis,
)
from src.research import AssetsResearch
from src.security_selection import (
    AssetsResearch as LegacyAssetsResearch,
    CorrelationPortfolioSelector as LegacyCorrelationPortfolioSelector,
)
from src.selection import CorrelationPortfolioSelector, CorrelationSelector


def test_research_and_selection_exports_map_to_existing_classes():
    assert AssetsResearch is LegacyAssetsResearch
    assert CorrelationPortfolioSelector is LegacyCorrelationPortfolioSelector
    assert CorrelationSelector is LegacyCorrelationPortfolioSelector


def test_portfolio_exports_map_to_existing_classes():
    assert PortfolioBasicMetrics is not PortfolioElementaryMetrics
    assert PortfolioDownsideMetrics is PortfolioPostModernMetrics
    assert PortfolioBenchmarkAnalysis is PortfolioElementaryAnalysis
    assert Portfolio is not None
    assert PortfolioPerformanceAnalysis is not None


def test_optimization_exports_map_to_existing_classes():
    assert MeanVarianceOptimizer is PortfolioOptimization
    assert PostModernOptimizer is PortfolioOptimizationPostModern
    assert NewOptimizationConfig is OptimizationConfig
    assert NewMinimumVarianceConfig is MinimumVarianceConfig
    assert NewOptimizationResult is OptimizationResult
    assert NewPostModernOptimizationConfig is PostModernOptimizationConfig
    assert NewMinimumSemivarianceConfig is MinimumSemivarianceConfig
    assert NewMaximumOmegaConfig is MaximumOmegaConfig
    assert NewPostModernOptimizationResult is PostModernOptimizationResult


def test_backtesting_exports_map_to_existing_classes():
    assert StaticBacktestEngine is LegacyBacktester
    assert Backtester is LegacyBacktester
    assert AllocationStrategy is LegacyAllocationStrategy
    assert MeanVarianceStrategy is LegacyMeanVarianceStrategy
    assert PostModernStrategy is LegacyPostModernStrategy
    assert BacktestConfig is LegacyBacktestConfig
    assert BacktestResult is LegacyBacktestResult
    assert BacktestStrategyResult is LegacyBacktestStrategyResult
    assert StrategyAllocation is LegacyStrategyAllocation
