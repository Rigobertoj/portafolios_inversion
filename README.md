# Investment Portfolios — Portfolio Construction & Quant Allocation

This repository contains my course work for **Investment Portfolios**, organized as a quant-style project: define the investment problem, model risk/return, build and optimize portfolios under realistic constraints, and evaluate performance with transparent assumptions.

The goal is not just to “compute weights”, but to build decision-ready portfolios with clear logic, reproducible code, and defensible interpretations.

---

## General Objective

To develop the ability to **design, build, and evaluate investment portfolios** by integrating:

- risk–return measurement and interpretation  
- diversification and dependency structure (covariances/correlations)  
- portfolio optimization under practical constraints  
- benchmark-aware portfolio construction  
- performance evaluation, attribution, and robustness checks

---

## Project Approach (Quant Workflow)

This repository follows a consistent pipeline:

1. **Data**: ingestion, cleaning, alignment, missing-data handling  
2. **Returns**: construction of simple/log returns, frequency consistency, annualization  
3. **Risk model**: covariance estimation, stability diagnostics, diversification analysis  
4. **Expected returns**: assumptions and estimation methods (historical / equilibrium-style / scenario-based)  
5. **Portfolio construction**: objective functions + constraints + transaction cost logic  
6. **Backtesting**: rebalancing rules, benchmarking, drawdowns, sensitivity tests  
7. **Interpretation**: decision narrative — what drives results and when it fails

---

## Core Topics Covered

### 1) Returns and Compounding
- simple vs log returns  
- aggregation across time and annualization  
- return decomposition and practical pitfalls

### 2) Risk Measurement
- variance/volatility, covariance, correlation  
- diversification and concentration risk  
- drawdowns and downside-focused views of risk  
- risk contribution by asset (who is actually driving portfolio risk)

### 3) Portfolio Theory and Efficient Allocation
- mean–variance logic and the efficient set  
- minimum-variance portfolios and risk budgeting intuition  
- trade-off between estimation error and optimality

### 4) Optimization in Practice
- long-only vs long/short setups  
- constraints: max weight, sector limits, leverage, liquidity filters  
- turnover control and transaction-cost awareness  
- robustness: sensitivity of weights to small changes in inputs

### 5) Benchmarking and Performance
- benchmark-relative performance and tracking error  
- Sharpe/Sortino-style metrics and interpretation  
- attribution basics: allocation vs selection effects  
- stability and regime awareness (why backtests can lie)

---

## Repository Structure

```
├── __init__.py
├── LICENSE
├── README.md
├── course_notes/
│   ├── 01/
│   │   ├── data/
│   │   ├── img/
│   │   ├── notebooks/
│   │   ├── notes/
│   │   └── outputs/
│   ├── 02/
│   │   ├── data/
│   │   ├── img/
│   │   ├── notebooks/
│   │   ├── notes/
│   │   └── outputs/
│   ├── 03/
│   │   ├── notebooks/
│   │   ├── notes/
│   │   └── outputs/
│   └── project_context.md
├── data/
│   ├── features/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── docs/
│   ├── backtesting/
│   ├── optimization/
│   ├── portfolio_optimization_architecture.md
│   └── selection/
├── research/
│   ├── 01.20250212.min_corelacion_research.ipynb
│   ├── 02.20250312.metrics_research.ipynb
│   ├── 03.20250316.min_semivar.ipynb
│   ├── 04.20250321.backtesting.ipynb
│   ├── 05.20250322.repaso.ipynb
│   ├── 06.20250401.fit_variance_model.ipynb
│   ├── 07.20260401.evolution_prices_MELI.ipynb
│   └── 08.20260501.energy_industry.ipynb
├── reports/
├── setup.py
├── src/
│   ├── __init__.py
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── _helpers.py
│   │   ├── engine_static.py
│   │   ├── results.py
│   │   └── strategies.py
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── configs.py
│   │   ├── mean_variance.py
│   │   ├── postmodern.py
│   │   └── results.py
│   ├── portfolio/
│   │   ├── __init__.py
│   │   ├── benchmark_analysis.py
│   │   ├── legacy_adapters.py
│   │   ├── metrics_basic.py
│   │   ├── metrics_downside.py
│   │   ├── performance_analysis.py
│   │   └── portfolio.py
│   ├── portfolio_utils/
│   │   ├── __init__.py
│   │   ├── PdfImageConverter.py
│   │   ├── pdf_image_converter.py
│   │   └── readme.md
│   ├── research/
│   │   ├── __init__.py
│   │   ├── arima_research.py
│   │   └── assets_research.py
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── drawdown.py
│   │   ├── report.py
│   │   ├── tracking.py
│   │   ├── volatility.py
│   │   └── var_cvar.py
│   └── selection/
│       ├── __init__.py
│       ├── correlation_selector.py
│       ├── fundamental_metrics.py
│       ├── fundamental_scorers.py
│       ├── fundamental_selector.py
│       └── fundamentals.py
├── test/
│   ├── conftest.py
│   ├── architecture/
│   ├── backtesting/
│   ├── managment_risk/
│   ├── optimization/
│   ├── portfolio_utils/
│   ├── research/
│   └── selection/
```

The current codebase is organized around domain-oriented packages in `src/`:

- `portfolio/`: portfolio objects, performance metrics, benchmark-aware analysis, and compatibility adapters
- `risk/`: drawdown, tracking, VaR/CVaR, volatility, and reporting helpers
- `selection/`: correlation and fundamental asset-selection logic
- `optimization/`: mean-variance and post-modern optimization workflows
- `backtesting/`: strategy execution, helpers, and result containers
- `research/`: reusable research-side utilities
- `portfolio_utils/`: supporting utilities and legacy helpers
