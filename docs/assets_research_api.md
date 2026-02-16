# AssetsResearch API Reference

Quick reference for `AssetsResearch` without opening
`src/portfolio_utils/AssetsResearch.py`.

## Import

```python
from portfolio_utils import AssetsResearch
```

## Constructor

```python
research = AssetsResearch(
    tickers=["JPM", "V", "GS", "PGR"],
    start="2020-01-01",
    end="2025-01-01",    # optional
    price_field="Close", # optional (default: "Close")
)
```

Parameters:

- `tickers`: `str` or iterable of tickers.
- `start`: start date (`YYYY-MM-DD`).
- `end`: end date (`YYYY-MM-DD`), optional.
- `price_field`: column requested from Yahoo data (for example, `"Close"`).

Cached attributes:

- `prices`: cached prices (`pd.DataFrame`), filled by `download_prices()`.
- `returns`: cached returns (`pd.DataFrame`), filled by `compute_returns()`.

## Public Methods

| Method | Description | Return |
| --- | --- | --- |
| `download_prices()` | Downloads prices from Yahoo Finance and caches them in `self.prices`. | `pd.DataFrame` |
| `compute_returns()` | Computes daily returns from `self.prices` (`pct_change().dropna()`). Auto-downloads prices if needed. | `pd.DataFrame` |
| `get_prices(tickers=None)` | Returns cached prices (all or selected tickers). Auto-downloads prices if needed. | `pd.DataFrame` |
| `get_returns(tickers=None)` | Returns cached returns (all or selected tickers). Auto-computes returns if needed. | `pd.DataFrame` |
| `annual_return(tickers=None)` | Annualized expected return (`mean * 252`). | `pd.Series` |
| `annual_volatility(tickers=None)` | Annualized volatility (`std * sqrt(252)`). | `pd.Series` |
| `skew(tickers=None)` | Skewness of daily returns. | `pd.Series` |
| `vol_over_mean(tickers=None)` | Volatility divided by annualized return (`0` return is mapped to `NaN` before division). | `pd.Series` |
| `return_interval_pct(tickers=None, z=2.65)` | Return interval in percent (`low`, `high`) using `annual_return_pct +/- z * annual_volatility_pct`. | `pd.DataFrame` |
| `metrics(tickers=None)` | Consolidated metrics table with return, volatility, skew, ratio and interval columns. | `pd.DataFrame` |
| `describe_returns(tickers=None)` | Descriptive stats (`DataFrame.describe()`) on daily returns. | `pd.DataFrame` |
| `covariance(tickers=None)` | Covariance matrix on daily returns. | `pd.DataFrame` |

## `metrics()` Output Columns

`metrics()` returns one row per ticker with:

- `annual_return`
- `annual_volatility`
- `annual_return_pct`
- `annual_volatility_pct`
- `skew`
- `vol_over_mean`
- `return_interval_low_pct`
- `return_interval_high_pct`

## Validation and Error Behavior

- Requested subset tickers are validated in `get_prices()` and `get_returns()`.  
  If any ticker is missing, it raises `ValueError("Tickers not found in data: ...")`.
- If `price_field` is not present in downloaded Yahoo data, `download_prices()` raises `ValueError`.
- `metrics()` uses a fixed interval multiplier of `2.65`.
- `return_interval_pct()` allows custom `z` (default `2.65`).

## Typical Workflow

```python
research = AssetsResearch(["JPM", "V", "GS", "PGR"], start="2020-01-01")

prices = research.download_prices()
returns = research.compute_returns()
summary = research.metrics()
subset_cov = research.covariance(["JPM", "V"])
```

## Quick Introspection

```python
[name for name in dir(AssetsResearch) if not name.startswith("_")]
help(AssetsResearch)
```
