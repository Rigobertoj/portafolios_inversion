from src.managment_risk import Backtester as LegacyBacktester
from src.managment_risk.backtesting_strategy import Backtester as SnakeBacktester
from src.optimization import (
    PortfolioOptimizationPostMordern,
    PortfolioOptimizationPostModern,
)
from src.portfolio_utils import convert_pdf_to_images
from src.portfolio_utils.pdf_image_converter import (
    convert_pdf_to_images as SnakeConvertPdfToImages,
)
from src.research import AssetsResearch as ResearchAssetsResearch
from src.security_selection import (
    AssetsResearch as LegacyAssetsResearch,
    CorrelationPortfolioSelector as LegacyCorrelationSelector,
)
from src.security_selection.assets_research import AssetsResearch as SnakeAssetsResearch
from src.security_selection.correlation_portfolio_selector import (
    CorrelationPortfolioSelector as SnakeCorrelationSelector,
)
from src.selection import CorrelationPortfolioSelector as SelectionCorrelationSelector


def test_snake_case_security_selection_modules_preserve_legacy_exports():
    assert SnakeAssetsResearch is LegacyAssetsResearch
    assert ResearchAssetsResearch is LegacyAssetsResearch
    assert SnakeCorrelationSelector is LegacyCorrelationSelector
    assert SelectionCorrelationSelector is LegacyCorrelationSelector


def test_snake_case_portfolio_utils_module_preserves_public_functions():
    assert SnakeConvertPdfToImages is convert_pdf_to_images


def test_snake_case_managment_risk_module_preserves_backtesting_bridge():
    assert SnakeBacktester is LegacyBacktester


def test_optimization_keeps_the_legacy_postmodern_typo_alias():
    assert PortfolioOptimizationPostMordern is PortfolioOptimizationPostModern
