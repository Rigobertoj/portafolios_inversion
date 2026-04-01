import src.backtesting as backtesting_package
import src.optimization as optimization_package
import src.portfolio as portfolio_package
import src.research as research_package
import src.risk as risk_package
import src.selection as selection_package
from src.backtesting import (
    AllocationStrategy,
    BacktestConfig,
    Backtester,
    BacktestResult,
    BacktestStrategyResult,
    MeanVarianceStrategy,
    PostModernStrategy,
    StaticBacktestEngine,
    StrategyAllocation,
)
from src.optimization import (
    MeanVarianceOptimizer,
    MinimumVarianceConfig,
    OptimizationConfig,
    OptimizationResult,
    PostModernOptimizer,
    PostModernOptimizationResult,
)
from src.portfolio import (
    Portfolio,
    PortfolioBasicMetrics,
    PortfolioBenchmarkAnalysis,
    PortfolioDownsideMetrics,
    PortfolioPerformanceAnalysis,
)
from src.research import AssetsResearch
from src.risk import (
    PortfolioDrawdownAnalysis,
    PortfolioRelativeRisk,
    PortfolioTailRisk,
    RiskAnalyzer,
)
from src.selection import CorrelationPortfolioSelector, CorrelationSelector


def test_public_packages_expose_expected_symbols():
    assert set(research_package.__all__) == {"AssetsResearch"}
    assert set(selection_package.__all__) == {
        "CorrelationPortfolioSelector",
        "CorrelationSelector",
    }
    assert {
        "Portfolio",
        "PortfolioBasicMetrics",
        "PortfolioBenchmarkAnalysis",
        "PortfolioDownsideMetrics",
        "PortfolioPerformanceAnalysis",
    }.issubset(set(portfolio_package.__all__))
    assert {
        "MeanVarianceOptimizer",
        "OptimizationConfig",
        "OptimizationResult",
        "PostModernOptimizer",
        "PostModernOptimizationResult",
    }.issubset(set(optimization_package.__all__))
    assert {
        "AllocationStrategy",
        "BacktestConfig",
        "BacktestResult",
        "BacktestStrategyResult",
        "Backtester",
        "MeanVarianceStrategy",
        "PostModernStrategy",
        "StaticBacktestEngine",
        "StrategyAllocation",
    }.issubset(set(backtesting_package.__all__))
    assert {
        "PortfolioDrawdownAnalysis",
        "PortfolioRelativeRisk",
        "PortfolioTailRisk",
        "RiskAnalyzer",
    }.issubset(set(risk_package.__all__))


def test_research_and_selection_public_api_is_available():
    assert AssetsResearch is not None
    assert CorrelationPortfolioSelector is not None
    assert CorrelationSelector is not None


def test_portfolio_and_optimization_exports_use_new_modules():
    assert Portfolio.__module__ == "src.portfolio.portfolio"
    assert PortfolioBasicMetrics.__module__ == "src.portfolio.metrics_basic"
    assert PortfolioDownsideMetrics.__module__ == "src.portfolio.metrics_downside"
    assert PortfolioBenchmarkAnalysis.__module__ == "src.portfolio.benchmark_analysis"
    assert PortfolioPerformanceAnalysis.__module__ == "src.portfolio.performance_analysis"
    assert MeanVarianceOptimizer.__module__ == "src.optimization.mean_variance"
    assert PostModernOptimizer.__module__ == "src.optimization.postmodern"
    assert OptimizationConfig.__module__ == "src.optimization.configs"
    assert MinimumVarianceConfig.__module__ == "src.optimization.configs"
    assert OptimizationResult.__module__ == "src.optimization.results"
    assert PostModernOptimizationResult.__module__ == "src.optimization.results"


def test_backtesting_and_risk_exports_use_new_modules():
    assert Backtester.__module__ == "src.backtesting.engine_static"
    assert StaticBacktestEngine.__module__ == "src.backtesting.engine_static"
    assert AllocationStrategy.__module__ == "src.backtesting.strategies"
    assert MeanVarianceStrategy.__module__ == "src.backtesting.strategies"
    assert PostModernStrategy.__module__ == "src.backtesting.strategies"
    assert BacktestConfig.__module__ == "src.backtesting.results"
    assert BacktestResult.__module__ == "src.backtesting.results"
    assert BacktestStrategyResult.__module__ == "src.backtesting.results"
    assert StrategyAllocation.__module__ == "src.backtesting.results"
    assert PortfolioDrawdownAnalysis.__module__ == "src.risk.drawdown"
    assert PortfolioTailRisk.__module__ == "src.risk.var_cvar"
    assert PortfolioRelativeRisk.__module__ == "src.risk.tracking"
    assert RiskAnalyzer.__module__ == "src.risk.report"
