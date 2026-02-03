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
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── features/
│
├── docs/
│
├── env/
│
├── reports/
│
├── src/
│   ├── 01/
│   │   ├── data/
│   │   ├── notebooks/
│   │   └── outputs/
│   │
│   ├── 02/
│   │   ├── data/
│   │   ├── notebooks/
│   │   └── outputs/
│   │
│   ├── 03/
│   │   ├── data/
│   │   ├── notebooks/
│   │   └── outputs/
│   │
│   └── utils/
│
├── LICENSE
└── README.md
```

