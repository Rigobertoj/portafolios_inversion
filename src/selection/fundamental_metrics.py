"""Pure fundamental metric calculations used by security selectors."""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd

from .fundamentals import FundamentalData

StatementFrequency = Literal["annual", "quarterly"]


def _clean_label(value: object) -> str:
    return " ".join(str(value).lower().replace("_", " ").split())


def _safe_divide(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or np.isclose(float(denominator), 0.0):
        return np.nan
    return float(numerator) / float(denominator)


def _info_number(info: dict[str, object], *keys: str) -> float:
    for key in keys:
        value = info.get(key)
        if value is None:
            continue
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric):
            return float(numeric)
    return np.nan


def _statement_row(statement: pd.DataFrame, aliases: Iterable[str]) -> pd.Series:
    if not isinstance(statement, pd.DataFrame) or statement.empty:
        return pd.Series(dtype=float)

    alias_map = {_clean_label(alias) for alias in aliases}
    for idx in statement.index:
        if _clean_label(idx) in alias_map:
            return pd.to_numeric(statement.loc[idx], errors="coerce").dropna()
    return pd.Series(dtype=float)


def _latest_from_statement(statement: pd.DataFrame, aliases: Iterable[str]) -> float:
    row = _statement_row(statement, aliases)
    if row.empty:
        return np.nan

    try:
        row = row.sort_index()
    except TypeError:
        pass
    value = row.dropna().iloc[-1] if not row.dropna().empty else np.nan
    return float(value) if pd.notna(value) else np.nan


def _growth_from_statement(statement: pd.DataFrame, aliases: Iterable[str]) -> float:
    row = _statement_row(statement, aliases)
    row = row.dropna()
    if row.shape[0] < 2:
        return np.nan

    try:
        row = row.sort_index()
    except TypeError:
        pass

    start = float(row.iloc[0])
    end = float(row.iloc[-1])
    periods = row.shape[0] - 1

    if start > 0 and end > 0:
        return (end / start) ** (1 / periods) - 1
    return _safe_divide(end - start, abs(start))


def _statement_for_frequency(
    data: FundamentalData,
    statement_name: str,
    frequency: StatementFrequency,
) -> pd.DataFrame:
    if frequency == "annual":
        return getattr(data, statement_name)
    if frequency == "quarterly":
        return getattr(data, f"quarterly_{statement_name}")
    raise ValueError("frequency must be either 'annual' or 'quarterly'.")


def _period_columns(*statements: pd.DataFrame) -> list[object]:
    columns: set[object] = set()
    for statement in statements:
        if isinstance(statement, pd.DataFrame) and not statement.empty:
            columns.update(statement.columns.tolist())
    return sorted(columns, key=lambda value: pd.Timestamp(value))


def _value_at_period(
    statement: pd.DataFrame,
    aliases: Iterable[str],
    period: object,
) -> float:
    row = _statement_row(statement, aliases)
    if row.empty or period not in row.index:
        return np.nan
    value = row.loc[period]
    return float(value) if pd.notna(value) else np.nan


def _price_at_period(prices: pd.Series, period: object) -> float:
    prices_clean = pd.to_numeric(prices, errors="coerce").dropna()
    if prices_clean.empty:
        return np.nan

    period_ts = pd.Timestamp(period)
    try:
        eligible = prices_clean[prices_clean.index <= period_ts]
    except TypeError:
        eligible = prices_clean
    if eligible.empty:
        eligible = prices_clean
    return float(eligible.iloc[-1])


def _period_growth(values: pd.Series, periods: int = 1) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.pct_change(periods=periods, fill_method=None)


def build_fundamental_metrics(data: FundamentalData) -> pd.Series:
    """Build a normalized metric row from one company's raw fundamental data."""
    info = data.info
    income = data.income_statement
    balance = data.balance_sheet
    cash = data.cash_flow

    latest_price = float(data.prices.dropna().iloc[-1]) if not data.prices.dropna().empty else np.nan
    current_price = _info_number(info, "currentPrice", "regularMarketPrice")
    if pd.isna(current_price):
        current_price = latest_price

    shares = _info_number(info, "sharesOutstanding", "impliedSharesOutstanding")
    if pd.isna(shares):
        shares = _latest_from_statement(
            balance,
            ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"],
        )

    net_income = _latest_from_statement(
        income,
        ["Net Income", "Net Income Common Stockholders", "Net Income From Continuing Operation Net Minority Interest"],
    )
    revenue = _latest_from_statement(income, ["Total Revenue", "Operating Revenue"])
    operating_income = _latest_from_statement(income, ["Operating Income", "EBIT"])
    equity = _latest_from_statement(
        balance,
        ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"],
    )
    total_debt = _latest_from_statement(balance, ["Total Debt", "Long Term Debt"])
    current_assets = _latest_from_statement(balance, ["Current Assets", "Total Current Assets"])
    current_liabilities = _latest_from_statement(
        balance,
        ["Current Liabilities", "Total Current Liabilities"],
    )
    operating_cash_flow = _latest_from_statement(cash, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    capital_expenditure = _latest_from_statement(cash, ["Capital Expenditure", "Capital Expenditures"])
    interest_expense = abs(_latest_from_statement(income, ["Interest Expense", "Interest Expense Non Operating"]))

    eps = _info_number(info, "trailingEps", "forwardEps")
    if pd.isna(eps):
        eps = _safe_divide(net_income, shares)

    market_cap = _info_number(info, "marketCap")
    if pd.isna(market_cap):
        market_cap = current_price * shares if pd.notna(current_price) and pd.notna(shares) else np.nan

    book_value_per_share = _safe_divide(equity, shares)
    free_cash_flow = np.nan
    if pd.notna(operating_cash_flow):
        capex = 0.0 if pd.isna(capital_expenditure) else float(capital_expenditure)
        free_cash_flow = float(operating_cash_flow) + capex

    trailing_pe = _info_number(info, "trailingPE")
    if pd.isna(trailing_pe):
        trailing_pe = _safe_divide(current_price, eps)

    price_to_book = _info_number(info, "priceToBook")
    if pd.isna(price_to_book):
        price_to_book = _safe_divide(current_price, book_value_per_share)

    revenue_growth = _info_number(info, "revenueGrowth")
    if pd.isna(revenue_growth):
        revenue_growth = _growth_from_statement(income, ["Total Revenue", "Operating Revenue"])

    earnings_growth = _info_number(info, "earningsGrowth")
    if pd.isna(earnings_growth):
        earnings_growth = _growth_from_statement(
            income,
            ["Net Income", "Net Income Common Stockholders", "Net Income From Continuing Operation Net Minority Interest"],
        )

    eps_growth = _growth_from_statement(income, ["Diluted EPS", "Basic EPS"])
    if pd.isna(eps_growth):
        eps_growth = earnings_growth

    peg_ratio = _info_number(info, "pegRatio")
    if pd.isna(peg_ratio) and pd.notna(trailing_pe) and pd.notna(eps_growth) and eps_growth > 0:
        peg_ratio = trailing_pe / (eps_growth * 100)

    metrics = {
        "ticker": data.ticker,
        "current_price": current_price,
        "market_cap": market_cap,
        "shares_outstanding": shares,
        "eps": eps,
        "revenue": revenue,
        "net_income": net_income,
        "total_equity": equity,
        "free_cash_flow": free_cash_flow,
        "trailing_pe": trailing_pe,
        "forward_pe": _info_number(info, "forwardPE"),
        "price_to_book": price_to_book,
        "peg_ratio": peg_ratio,
        "roe": _safe_divide(net_income, equity),
        "profit_margin": _info_number(info, "profitMargins"),
        "operating_margin": _safe_divide(operating_income, revenue),
        "free_cash_flow_margin": _safe_divide(free_cash_flow, revenue),
        "debt_to_equity": _safe_divide(total_debt, equity),
        "current_ratio": _safe_divide(current_assets, current_liabilities),
        "interest_coverage": _safe_divide(operating_income, interest_expense),
        "revenue_growth": revenue_growth,
        "earnings_growth": earnings_growth,
        "eps_growth": eps_growth,
        "dividend_yield": _info_number(info, "dividendYield"),
        "beta": _info_number(info, "beta"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
    }

    if pd.isna(metrics["profit_margin"]):
        metrics["profit_margin"] = _safe_divide(net_income, revenue)

    return pd.Series(metrics)


def build_fundamental_metric_history(
    data: FundamentalData,
    frequency: StatementFrequency = "quarterly",
    trailing_periods: int = 4,
) -> pd.DataFrame:
    """Build period-by-period fundamental metrics for one company.

    Parameters
    ----------
    data : FundamentalData
        Raw fundamental record for one ticker.
    frequency : {"annual", "quarterly"}, default "quarterly"
        Financial statement frequency to use.
    trailing_periods : int, default 4
        Maximum number of periods returned, counted from the most recent data.

    Returns
    -------
    pandas.DataFrame
        One row per reporting period with point-in-time ratios and growth fields.
    """
    if trailing_periods < 1:
        raise ValueError("trailing_periods must be >= 1.")

    income = _statement_for_frequency(data, "income_statement", frequency)
    balance = _statement_for_frequency(data, "balance_sheet", frequency)
    cash = _statement_for_frequency(data, "cash_flow", frequency)
    periods = _period_columns(income, balance, cash)[-trailing_periods:]
    if not periods:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    fallback_shares = _info_number(data.info, "sharesOutstanding", "impliedSharesOutstanding")

    for period in periods:
        revenue = _value_at_period(income, ["Total Revenue", "Operating Revenue"], period)
        net_income = _value_at_period(
            income,
            [
                "Net Income",
                "Net Income Common Stockholders",
                "Net Income From Continuing Operation Net Minority Interest",
            ],
            period,
        )
        operating_income = _value_at_period(income, ["Operating Income", "EBIT"], period)
        eps = _value_at_period(income, ["Diluted EPS", "Basic EPS"], period)
        equity = _value_at_period(
            balance,
            ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"],
            period,
        )
        total_debt = _value_at_period(balance, ["Total Debt", "Long Term Debt"], period)
        current_assets = _value_at_period(balance, ["Current Assets", "Total Current Assets"], period)
        current_liabilities = _value_at_period(
            balance,
            ["Current Liabilities", "Total Current Liabilities"],
            period,
        )
        shares = _value_at_period(
            balance,
            [
                "Ordinary Shares Number",
                "Share Issued",
                "Common Stock Shares Outstanding",
            ],
            period,
        )
        if pd.isna(shares):
            shares = fallback_shares
        if pd.isna(eps):
            eps = _safe_divide(net_income, shares)

        operating_cash_flow = _value_at_period(
            cash,
            ["Operating Cash Flow", "Total Cash From Operating Activities"],
            period,
        )
        capital_expenditure = _value_at_period(
            cash,
            ["Capital Expenditure", "Capital Expenditures"],
            period,
        )
        capex = 0.0 if pd.isna(capital_expenditure) else float(capital_expenditure)
        free_cash_flow = (
            float(operating_cash_flow) + capex
            if pd.notna(operating_cash_flow)
            else np.nan
        )
        price = _price_at_period(data.prices, period)
        book_value_per_share = _safe_divide(equity, shares)

        rows.append(
            {
                "ticker": data.ticker,
                "frequency": frequency,
                "period": pd.Timestamp(period),
                "current_price": price,
                "shares_outstanding": shares,
                "eps": eps,
                "revenue": revenue,
                "net_income": net_income,
                "operating_income": operating_income,
                "total_equity": equity,
                "free_cash_flow": free_cash_flow,
                "trailing_pe": _safe_divide(price, eps),
                "price_to_book": _safe_divide(price, book_value_per_share),
                "roe": _safe_divide(net_income, equity),
                "profit_margin": _safe_divide(net_income, revenue),
                "operating_margin": _safe_divide(operating_income, revenue),
                "free_cash_flow_margin": _safe_divide(free_cash_flow, revenue),
                "debt_to_equity": _safe_divide(total_debt, equity),
                "current_ratio": _safe_divide(current_assets, current_liabilities),
            }
        )

    history = pd.DataFrame(rows).sort_values(["ticker", "period"]).reset_index(drop=True)
    for base_metric in ["revenue", "net_income", "eps", "free_cash_flow"]:
        history[f"{base_metric}_period_growth"] = history.groupby("ticker")[base_metric].transform(
            lambda values: _period_growth(values, periods=1)
        )

        yoy_periods = 4 if frequency == "quarterly" else 1
        history[f"{base_metric}_yoy_growth"] = history.groupby("ticker")[base_metric].transform(
            lambda values: _period_growth(values, periods=yoy_periods)
        )

    history["earnings_growth"] = history["net_income_period_growth"]
    history["revenue_growth"] = history["revenue_period_growth"]
    history["eps_growth"] = history["eps_period_growth"]
    return history


def build_metrics_frame(records: Iterable[FundamentalData]) -> pd.DataFrame:
    """Build a metrics table for a group of companies."""
    rows = [build_fundamental_metrics(record) for record in records]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("ticker", drop=False)


def build_metric_history_frame(
    records: Iterable[FundamentalData],
    frequency: StatementFrequency = "quarterly",
    trailing_periods: int = 4,
) -> pd.DataFrame:
    """Build a period-by-period metrics table for a group of companies."""
    frames = [
        build_fundamental_metric_history(
            record,
            frequency=frequency,
            trailing_periods=trailing_periods,
        )
        for record in records
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


__all__ = [
    "StatementFrequency",
    "build_fundamental_metric_history",
    "build_fundamental_metrics",
    "build_metric_history_frame",
    "build_metrics_frame",
]
