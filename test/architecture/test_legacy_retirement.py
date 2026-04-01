from pathlib import Path

from src.optimization import PortfolioOptimization, PortfolioOptimizationPostModern
from src.portfolio import (
    PortfolioElementaryAnalysis,
    PortfolioElementaryMetrics,
    PortfolioPostModernMetrics,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_legacy_packages_were_retired_from_src_tree():
    assert not (REPO_ROOT / "src" / "asset_allocation").exists()
    assert not (REPO_ROOT / "src" / "security_selection").exists()
    assert not (REPO_ROOT / "src" / "managment_risk").exists()


def test_legacy_class_names_remain_available_from_new_packages():
    assert PortfolioElementaryMetrics.__module__ == "src.portfolio.legacy_adapters"
    assert PortfolioPostModernMetrics.__module__ == "src.portfolio.legacy_adapters"
    assert PortfolioElementaryAnalysis.__module__ == "src.portfolio.legacy_adapters"
    assert PortfolioOptimization.__module__ == "src.optimization.mean_variance"
    assert PortfolioOptimizationPostModern.__module__ == "src.optimization.postmodern"
