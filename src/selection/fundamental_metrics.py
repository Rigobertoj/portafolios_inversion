"""Pure fundamental metric calculations used by security selectors.

The functions in this module transform raw Yahoo Finance statements and profile
fields into normalized value, quality, growth, leverage, and liquidity metrics.
"""

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


def _abs_or_nan(value: float) -> float:
    return abs(float(value)) if pd.notna(value) else np.nan


def build_fundamental_metrics(data: FundamentalData) -> pd.Series:
    """
    Build a normalized metric row from one company's raw fundamental data.

    Parameters
    ----------
    data : FundamentalData
        Raw Yahoo Finance record for one ticker.

    Returns
    -------
    pandas.Series
        Metric row containing valuation, profitability, growth, leverage,
        liquidity, and descriptive fields.
    """
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
    cost_of_revenue = abs(_latest_from_statement(income, ["Cost Of Revenue", "Cost Of Goods Sold", "Cost of Revenue"]))
    gross_profit = _latest_from_statement(income, ["Gross Profit"])
    sga_expense = _latest_from_statement(
        income,
        [
            "Selling General And Administration",
            "Selling General Administrative",
            "Selling General And Administrative",
        ],
    )
    operating_income = _latest_from_statement(income, ["Operating Income", "EBIT"])
    pretax_income = _latest_from_statement(income, ["Pretax Income", "Income Before Tax"])
    tax_expense = abs(_latest_from_statement(income, ["Tax Provision", "Income Tax Expense"]))
    ebitda = _latest_from_statement(income, ["EBITDA", "Normalized EBITDA"])
    equity = _latest_from_statement(
        balance,
        ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"],
    )
    total_debt = _latest_from_statement(balance, ["Total Debt", "Long Term Debt"])
    long_term_debt = _latest_from_statement(balance, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"])
    total_assets = _latest_from_statement(balance, ["Total Assets"])
    fixed_assets = _latest_from_statement(
        balance,
        ["Net PPE", "Property Plant Equipment", "Gross PPE", "Net Property Plant And Equipment"],
    )
    tangible_equity = _latest_from_statement(
        balance,
        ["Tangible Book Value", "Net Tangible Assets", "Tangible Common Equity"],
    )
    cash_and_short_term = _latest_from_statement(
        balance,
        [
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Cash Equivalents",
            "Cash And Short Term Investments",
        ],
    )
    receivables = _latest_from_statement(
        balance,
        ["Accounts Receivable", "Net Receivables", "Receivables"],
    )
    inventory = _latest_from_statement(balance, ["Inventory"])
    payables = _latest_from_statement(
        balance,
        ["Accounts Payable", "Payables", "Payables And Accrued Expenses"],
    )
    current_assets = _latest_from_statement(balance, ["Current Assets", "Total Current Assets"])
    current_liabilities = _latest_from_statement(
        balance,
        ["Current Liabilities", "Total Current Liabilities"],
    )
    operating_cash_flow = _latest_from_statement(cash, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    capital_expenditure = _latest_from_statement(cash, ["Capital Expenditure", "Capital Expenditures"])
    dividends_paid = abs(
        _latest_from_statement(
            cash,
            ["Cash Dividends Paid", "Common Stock Dividend Paid", "Cash Dividends Paid Direct"],
        )
    )
    interest_expense = abs(_latest_from_statement(income, ["Interest Expense", "Interest Expense Non Operating"]))

    eps = _info_number(info, "trailingEps", "forwardEps")
    if pd.isna(eps):
        eps = _safe_divide(net_income, shares)

    market_cap = _info_number(info, "marketCap")
    if pd.isna(market_cap):
        market_cap = current_price * shares if pd.notna(current_price) and pd.notna(shares) else np.nan

    book_value_per_share = _safe_divide(equity, shares)
    tangible_book_value_per_share = _safe_divide(tangible_equity, shares)
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

    price_to_tangible_book = _safe_divide(current_price, tangible_book_value_per_share)

    price_to_sales = _info_number(info, "priceToSalesTrailing12Months")
    if pd.isna(price_to_sales):
        price_to_sales = _safe_divide(market_cap, revenue)

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

    enterprise_value = _info_number(info, "enterpriseValue")
    if pd.isna(enterprise_value) and pd.notna(market_cap):
        net_debt = total_debt - cash_and_short_term if pd.notna(cash_and_short_term) else total_debt
        enterprise_value = market_cap + net_debt if pd.notna(net_debt) else np.nan

    invested_capital = np.nan
    if pd.notna(total_debt) or pd.notna(equity):
        debt_value = 0.0 if pd.isna(total_debt) else float(total_debt)
        equity_value = 0.0 if pd.isna(equity) else float(equity)
        cash_value = 0.0 if pd.isna(cash_and_short_term) else float(cash_and_short_term)
        invested_capital = debt_value + equity_value - cash_value
    total_capital = (
        (0.0 if pd.isna(total_debt) else float(total_debt))
        + (0.0 if pd.isna(equity) else float(equity))
        if pd.notna(total_debt) or pd.notna(equity)
        else np.nan
    )
    net_debt = total_debt - cash_and_short_term if pd.notna(cash_and_short_term) else np.nan
    tax_rate = _safe_divide(tax_expense, pretax_income)
    tax_complement = 1.0 - tax_rate if pd.notna(tax_rate) else np.nan
    nopat = operating_income * tax_complement if pd.notna(operating_income) and pd.notna(tax_complement) else np.nan
    pretax_return_on_assets = _safe_divide(pretax_income, total_assets)
    ebit_return_on_assets = _safe_divide(operating_income, total_assets)
    equity_multiplier = _safe_divide(total_assets, equity)
    earnings_retention = 1.0 - _info_number(info, "payoutRatio")
    if pd.isna(earnings_retention) and pd.notna(dividends_paid) and pd.notna(net_income):
        earnings_retention = 1.0 - _safe_divide(dividends_paid, net_income)
    reinvestment_rate = (
        _safe_divide(net_income - dividends_paid, equity)
        if pd.notna(net_income) and pd.notna(dividends_paid)
        else np.nan
    )
    cash_turnover = _safe_divide(revenue, cash_and_short_term)
    receivables_turnover = _safe_divide(revenue, receivables)
    inventory_turnover = _safe_divide(cost_of_revenue, inventory)
    if pd.isna(inventory_turnover):
        inventory_turnover = _safe_divide(revenue, inventory)
    payables_turnover = _safe_divide(cost_of_revenue, payables)
    if pd.isna(payables_turnover):
        payables_turnover = _safe_divide(revenue, payables)
    current_assets_turnover = _safe_divide(revenue, current_assets)
    fixed_assets_turnover = _safe_divide(revenue, fixed_assets)
    total_assets_turnover = _safe_divide(revenue, total_assets)
    working_capital_turnover = _safe_divide(revenue, current_assets - current_liabilities)
    days_inventory_on_hand = _safe_divide(365.0, inventory_turnover)
    days_sales_outstanding = _safe_divide(365.0, receivables_turnover)
    operating_cycle = (
        days_inventory_on_hand + days_sales_outstanding
        if pd.notna(days_inventory_on_hand) and pd.notna(days_sales_outstanding)
        else np.nan
    )
    days_payables_outstanding = _safe_divide(365.0, payables_turnover)
    net_operating_cycle = (
        operating_cycle - days_payables_outstanding
        if pd.notna(operating_cycle) and pd.notna(days_payables_outstanding)
        else np.nan
    )

    metrics = {
        "ticker": data.ticker,
        "current_price": current_price,
        "market_cap": market_cap,
        "enterprise_value": enterprise_value,
        "shares_outstanding": shares,
        "eps": eps,
        "eps_recurring": eps,
        "eps_basic": _latest_from_statement(income, ["Basic EPS", "Basic Average Shares"]),
        "eps_diluted": _latest_from_statement(income, ["Diluted EPS"]),
        "revenue": revenue,
        "cost_of_revenue": cost_of_revenue,
        "gross_profit": gross_profit,
        "net_income": net_income,
        "operating_income": operating_income,
        "pretax_income": pretax_income,
        "ebitda": ebitda,
        "total_equity": equity,
        "total_assets": total_assets,
        "total_debt": total_debt,
        "long_term_debt": long_term_debt,
        "fixed_assets": fixed_assets,
        "cash_and_short_term": cash_and_short_term,
        "receivables": receivables,
        "inventory": inventory,
        "current_assets": current_assets,
        "payables": payables,
        "current_liabilities": current_liabilities,
        "free_cash_flow": free_cash_flow,
        "trailing_pe": trailing_pe,
        "forward_pe": _info_number(info, "forwardPE"),
        "price_to_sales": price_to_sales,
        "price_to_book": price_to_book,
        "price_to_tangible_book": price_to_tangible_book,
        "price_to_cash_flow": _safe_divide(market_cap, operating_cash_flow),
        "price_to_free_cash_flow": _safe_divide(market_cap, free_cash_flow),
        "peg_ratio": peg_ratio,
        "enterprise_value_to_sales": _info_number(info, "enterpriseToRevenue"),
        "enterprise_value_to_ebitda": _info_number(info, "enterpriseToEbitda"),
        "enterprise_value_to_ebit": _safe_divide(enterprise_value, operating_income),
        "gross_margin": _safe_divide(gross_profit, revenue),
        "sga_to_sales": _safe_divide(_abs_or_nan(sga_expense), revenue),
        "pretax_margin": _safe_divide(pretax_income, revenue),
        "roe": _safe_divide(net_income, equity),
        "return_on_assets": _safe_divide(net_income, total_assets),
        "return_on_common_equity": _safe_divide(net_income, equity),
        "return_on_total_capital": _safe_divide(operating_income, total_capital),
        "return_on_invested_capital": _safe_divide(nopat, invested_capital),
        "cash_flow_return_on_invested_capital": _safe_divide(operating_cash_flow, invested_capital),
        "profit_margin": _info_number(info, "profitMargins"),
        "operating_margin": _safe_divide(operating_income, revenue),
        "free_cash_flow_margin": _safe_divide(free_cash_flow, revenue),
        "free_cash_flow_conversion_ratio": _safe_divide(free_cash_flow, net_income),
        "capex_to_sales": _safe_divide(_abs_or_nan(capital_expenditure), revenue),
        "sales_per_share": _safe_divide(revenue, shares),
        "operating_income_per_share": _safe_divide(operating_income, shares),
        "dividends_per_share": _info_number(info, "dividendRate"),
        "cash_flow_per_share": _safe_divide(operating_cash_flow, shares),
        "free_cash_flow_per_share": _safe_divide(free_cash_flow, shares),
        "book_value_per_share": book_value_per_share,
        "tangible_book_value_per_share": tangible_book_value_per_share,
        "diluted_shares_outstanding": shares,
        "basic_shares_outstanding": shares,
        "total_shares_outstanding": shares,
        "cash_and_short_term_turnover": cash_turnover,
        "receivables_turnover": receivables_turnover,
        "current_assets_turnover": current_assets_turnover,
        "fixed_assets_turnover": fixed_assets_turnover,
        "total_assets_turnover": total_assets_turnover,
        "asset_turnover_dupont": total_assets_turnover,
        "pretax_return_on_assets": pretax_return_on_assets,
        "tax_rate_complement": tax_complement,
        "return_on_assets_dupont": _safe_divide(net_income, total_assets),
        "equity_multiplier": equity_multiplier,
        "return_on_equity_dupont": _safe_divide(net_income, equity),
        "earnings_retention": earnings_retention,
        "reinvestment_rate": reinvestment_rate,
        "ebit_return_on_assets": ebit_return_on_assets,
        "interest_as_percent_assets": _safe_divide(interest_expense, total_assets),
        "debt_to_equity": _safe_divide(total_debt, equity),
        "long_term_debt_to_total_equity": _safe_divide(long_term_debt, equity),
        "long_term_debt_to_total_capital": _safe_divide(long_term_debt, total_capital),
        "long_term_debt_to_total_assets": _safe_divide(long_term_debt, total_assets),
        "total_debt_to_total_assets": _safe_divide(total_debt, total_assets),
        "total_debt_to_total_capital": _safe_divide(total_debt, total_capital),
        "net_debt_to_total_equity": _safe_divide(net_debt, equity),
        "net_debt_to_total_capital": _safe_divide(
            net_debt,
            total_capital,
        ),
        "current_ratio": _safe_divide(current_assets, current_liabilities),
        "quick_ratio": _safe_divide(
            (0.0 if pd.isna(cash_and_short_term) else float(cash_and_short_term))
            + (0.0 if pd.isna(receivables) else float(receivables)),
            current_liabilities,
        ),
        "cash_ratio": _safe_divide(cash_and_short_term, current_liabilities),
        "cash_and_short_term_to_current_assets": _safe_divide(cash_and_short_term, current_assets),
        "cfo_to_current_liabilities": _safe_divide(operating_cash_flow, current_liabilities),
        "inventory_turnover": inventory_turnover,
        "payables_turnover": payables_turnover,
        "asset_turnover": total_assets_turnover,
        "working_capital_turnover": working_capital_turnover,
        "days_inventory_on_hand": days_inventory_on_hand,
        "days_sales_outstanding": days_sales_outstanding,
        "operating_cycle": operating_cycle,
        "days_payables_outstanding": days_payables_outstanding,
        "net_operating_cycle": net_operating_cycle,
        "net_debt_to_ebitda": _safe_divide(
            net_debt,
            ebitda,
        ),
        "net_debt_to_ebitda_minus_capex": _safe_divide(net_debt, ebitda + capital_expenditure),
        "total_debt_to_ebitda": _safe_divide(total_debt, ebitda),
        "interest_coverage": _safe_divide(operating_income, interest_expense),
        "ebitda_to_interest_expense": _safe_divide(ebitda, interest_expense),
        "fixed_charge_coverage_ratio": _safe_divide(operating_income, interest_expense),
        "cfo_to_interest_expense": _safe_divide(operating_cash_flow, interest_expense),
        "cash_dividend_coverage_ratio": _safe_divide(operating_cash_flow, dividends_paid),
        "long_term_debt_to_ebitda": _safe_divide(long_term_debt, ebitda),
        "net_debt_to_ffo": _safe_divide(net_debt, operating_cash_flow),
        "long_term_debt_to_ffo": _safe_divide(long_term_debt, operating_cash_flow),
        "cfo_to_total_debt": _safe_divide(operating_cash_flow, total_debt),
        "ebitda_minus_capex_to_interest_expense": _safe_divide(ebitda + capital_expenditure, interest_expense),
        "revenue_growth": revenue_growth,
        "earnings_growth": earnings_growth,
        "eps_growth": eps_growth,
        "dividend_yield": _info_number(info, "dividendYield"),
        "dividend_payout_ratio": _info_number(info, "payoutRatio"),
        "beta": _info_number(info, "beta"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
    }

    if pd.isna(metrics["enterprise_value_to_sales"]):
        metrics["enterprise_value_to_sales"] = _safe_divide(enterprise_value, revenue)
    if pd.isna(metrics["enterprise_value_to_ebitda"]):
        metrics["enterprise_value_to_ebitda"] = _safe_divide(enterprise_value, ebitda)
    if pd.isna(metrics["profit_margin"]):
        metrics["profit_margin"] = _safe_divide(net_income, revenue)
    if pd.isna(metrics["dividend_payout_ratio"]):
        metrics["dividend_payout_ratio"] = _safe_divide(dividends_paid, net_income)

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
        cost_of_revenue = abs(
            _value_at_period(income, ["Cost Of Revenue", "Cost Of Goods Sold", "Cost of Revenue"], period)
        )
        gross_profit = _value_at_period(income, ["Gross Profit"], period)
        sga_expense = _value_at_period(
            income,
            [
                "Selling General And Administration",
                "Selling General Administrative",
                "Selling General And Administrative",
            ],
            period,
        )
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
        pretax_income = _value_at_period(income, ["Pretax Income", "Income Before Tax"], period)
        tax_expense = abs(_value_at_period(income, ["Tax Provision", "Income Tax Expense"], period))
        ebitda = _value_at_period(income, ["EBITDA", "Normalized EBITDA"], period)
        eps = _value_at_period(income, ["Diluted EPS", "Basic EPS"], period)
        equity = _value_at_period(
            balance,
            ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"],
            period,
        )
        total_debt = _value_at_period(balance, ["Total Debt", "Long Term Debt"], period)
        long_term_debt = _value_at_period(
            balance,
            ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
            period,
        )
        total_assets = _value_at_period(balance, ["Total Assets"], period)
        fixed_assets = _value_at_period(
            balance,
            ["Net PPE", "Property Plant Equipment", "Gross PPE", "Net Property Plant And Equipment"],
            period,
        )
        tangible_equity = _value_at_period(
            balance,
            ["Tangible Book Value", "Net Tangible Assets", "Tangible Common Equity"],
            period,
        )
        cash_and_short_term = _value_at_period(
            balance,
            [
                "Cash Cash Equivalents And Short Term Investments",
                "Cash And Cash Equivalents",
                "Cash And Short Term Investments",
            ],
            period,
        )
        receivables = _value_at_period(
            balance,
            ["Accounts Receivable", "Net Receivables", "Receivables"],
            period,
        )
        inventory = _value_at_period(balance, ["Inventory"], period)
        payables = _value_at_period(
            balance,
            ["Accounts Payable", "Payables", "Payables And Accrued Expenses"],
            period,
        )
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
        dividends_paid = abs(
            _value_at_period(
                cash,
                ["Cash Dividends Paid", "Common Stock Dividend Paid", "Cash Dividends Paid Direct"],
                period,
            )
        )
        capex = 0.0 if pd.isna(capital_expenditure) else float(capital_expenditure)
        free_cash_flow = (
            float(operating_cash_flow) + capex
            if pd.notna(operating_cash_flow)
            else np.nan
        )
        price = _price_at_period(data.prices, period)
        book_value_per_share = _safe_divide(equity, shares)
        tangible_book_value_per_share = _safe_divide(tangible_equity, shares)
        interest_expense = abs(
            _value_at_period(
                income,
                ["Interest Expense", "Interest Expense Non Operating"],
                period,
            )
        )
        market_cap = price * shares if pd.notna(price) and pd.notna(shares) else np.nan
        net_debt = total_debt - cash_and_short_term if pd.notna(cash_and_short_term) else np.nan
        enterprise_value = market_cap + net_debt if pd.notna(market_cap) and pd.notna(net_debt) else np.nan
        total_capital = (
            (0.0 if pd.isna(total_debt) else float(total_debt))
            + (0.0 if pd.isna(equity) else float(equity))
            if pd.notna(total_debt) or pd.notna(equity)
            else np.nan
        )
        invested_capital = np.nan
        if pd.notna(total_debt) or pd.notna(equity):
            invested_capital = (
                (0.0 if pd.isna(total_debt) else float(total_debt))
                + (0.0 if pd.isna(equity) else float(equity))
                - (0.0 if pd.isna(cash_and_short_term) else float(cash_and_short_term))
            )
        tax_rate = _safe_divide(tax_expense, pretax_income)
        tax_complement = 1.0 - tax_rate if pd.notna(tax_rate) else np.nan
        nopat = operating_income * tax_complement if pd.notna(operating_income) and pd.notna(tax_complement) else np.nan
        pretax_return_on_assets = _safe_divide(pretax_income, total_assets)
        ebit_return_on_assets = _safe_divide(operating_income, total_assets)
        equity_multiplier = _safe_divide(total_assets, equity)
        earnings_retention = (
            1.0 - _safe_divide(dividends_paid, net_income)
            if pd.notna(dividends_paid) and pd.notna(net_income)
            else np.nan
        )
        reinvestment_rate = (
            _safe_divide(net_income - dividends_paid, equity)
            if pd.notna(net_income) and pd.notna(dividends_paid)
            else np.nan
        )
        cash_turnover = _safe_divide(revenue, cash_and_short_term)
        receivables_turnover = _safe_divide(revenue, receivables)
        inventory_turnover = _safe_divide(cost_of_revenue, inventory)
        if pd.isna(inventory_turnover):
            inventory_turnover = _safe_divide(revenue, inventory)
        payables_turnover = _safe_divide(cost_of_revenue, payables)
        if pd.isna(payables_turnover):
            payables_turnover = _safe_divide(revenue, payables)
        current_assets_turnover = _safe_divide(revenue, current_assets)
        fixed_assets_turnover = _safe_divide(revenue, fixed_assets)
        total_assets_turnover = _safe_divide(revenue, total_assets)
        working_capital_turnover = _safe_divide(revenue, current_assets - current_liabilities)
        days_inventory_on_hand = _safe_divide(365.0, inventory_turnover)
        days_sales_outstanding = _safe_divide(365.0, receivables_turnover)
        operating_cycle = (
            days_inventory_on_hand + days_sales_outstanding
            if pd.notna(days_inventory_on_hand) and pd.notna(days_sales_outstanding)
            else np.nan
        )
        days_payables_outstanding = _safe_divide(365.0, payables_turnover)
        net_operating_cycle = (
            operating_cycle - days_payables_outstanding
            if pd.notna(operating_cycle) and pd.notna(days_payables_outstanding)
            else np.nan
        )

        rows.append(
            {
                "ticker": data.ticker,
                "frequency": frequency,
                "period": pd.Timestamp(period),
                "current_price": price,
                "market_cap": market_cap,
                "enterprise_value": enterprise_value,
                "shares_outstanding": shares,
                "eps": eps,
                "eps_recurring": eps,
                "eps_basic": _value_at_period(income, ["Basic EPS"], period),
                "eps_diluted": _value_at_period(income, ["Diluted EPS"], period),
                "revenue": revenue,
                "cost_of_revenue": cost_of_revenue,
                "gross_profit": gross_profit,
                "net_income": net_income,
                "operating_income": operating_income,
                "pretax_income": pretax_income,
                "ebitda": ebitda,
                "total_equity": equity,
                "total_assets": total_assets,
                "total_debt": total_debt,
                "long_term_debt": long_term_debt,
                "fixed_assets": fixed_assets,
                "cash_and_short_term": cash_and_short_term,
                "receivables": receivables,
                "inventory": inventory,
                "current_assets": current_assets,
                "payables": payables,
                "current_liabilities": current_liabilities,
                "free_cash_flow": free_cash_flow,
                "price_to_sales": _safe_divide(market_cap, revenue),
                "trailing_pe": _safe_divide(price, eps),
                "price_to_book": _safe_divide(price, book_value_per_share),
                "price_to_tangible_book": _safe_divide(price, tangible_book_value_per_share),
                "price_to_cash_flow": _safe_divide(market_cap, operating_cash_flow),
                "price_to_free_cash_flow": _safe_divide(market_cap, free_cash_flow),
                "dividend_yield": _safe_divide(_safe_divide(dividends_paid, shares), price),
                "enterprise_value_to_sales": _safe_divide(enterprise_value, revenue),
                "enterprise_value_to_ebit": _safe_divide(enterprise_value, operating_income),
                "enterprise_value_to_ebitda": _safe_divide(enterprise_value, ebitda),
                "gross_margin": _safe_divide(gross_profit, revenue),
                "sga_to_sales": _safe_divide(_abs_or_nan(sga_expense), revenue),
                "roe": _safe_divide(net_income, equity),
                "return_on_common_equity": _safe_divide(net_income, equity),
                "return_on_assets": _safe_divide(net_income, total_assets),
                "return_on_total_capital": _safe_divide(operating_income, total_capital),
                "return_on_invested_capital": _safe_divide(nopat, invested_capital),
                "cash_flow_return_on_invested_capital": _safe_divide(operating_cash_flow, invested_capital),
                "profit_margin": _safe_divide(net_income, revenue),
                "pretax_margin": _safe_divide(pretax_income, revenue),
                "operating_margin": _safe_divide(operating_income, revenue),
                "free_cash_flow_margin": _safe_divide(free_cash_flow, revenue),
                "free_cash_flow_conversion_ratio": _safe_divide(free_cash_flow, net_income),
                "capex_to_sales": _safe_divide(_abs_or_nan(capital_expenditure), revenue),
                "sales_per_share": _safe_divide(revenue, shares),
                "operating_income_per_share": _safe_divide(operating_income, shares),
                "dividends_per_share": _safe_divide(dividends_paid, shares),
                "dividend_payout_ratio": _safe_divide(dividends_paid, net_income),
                "cash_flow_per_share": _safe_divide(operating_cash_flow, shares),
                "free_cash_flow_per_share": _safe_divide(free_cash_flow, shares),
                "book_value_per_share": book_value_per_share,
                "tangible_book_value_per_share": tangible_book_value_per_share,
                "diluted_shares_outstanding": shares,
                "basic_shares_outstanding": shares,
                "total_shares_outstanding": shares,
                "cash_and_short_term_turnover": cash_turnover,
                "receivables_turnover": receivables_turnover,
                "current_assets_turnover": current_assets_turnover,
                "fixed_assets_turnover": fixed_assets_turnover,
                "total_assets_turnover": total_assets_turnover,
                "asset_turnover_dupont": total_assets_turnover,
                "pretax_return_on_assets": pretax_return_on_assets,
                "tax_rate_complement": tax_complement,
                "return_on_assets_dupont": _safe_divide(net_income, total_assets),
                "equity_multiplier": equity_multiplier,
                "return_on_equity_dupont": _safe_divide(net_income, equity),
                "earnings_retention": earnings_retention,
                "reinvestment_rate": reinvestment_rate,
                "ebit_return_on_assets": ebit_return_on_assets,
                "interest_as_percent_assets": _safe_divide(interest_expense, total_assets),
                "debt_to_equity": _safe_divide(total_debt, equity),
                "long_term_debt_to_total_equity": _safe_divide(long_term_debt, equity),
                "long_term_debt_to_total_capital": _safe_divide(long_term_debt, total_capital),
                "long_term_debt_to_total_assets": _safe_divide(long_term_debt, total_assets),
                "total_debt_to_total_assets": _safe_divide(total_debt, total_assets),
                "total_debt_to_total_capital": _safe_divide(total_debt, total_capital),
                "net_debt_to_total_equity": _safe_divide(net_debt, equity),
                "net_debt_to_total_capital": _safe_divide(net_debt, total_capital),
                "current_ratio": _safe_divide(current_assets, current_liabilities),
                "quick_ratio": _safe_divide(
                    (0.0 if pd.isna(cash_and_short_term) else float(cash_and_short_term))
                    + (0.0 if pd.isna(receivables) else float(receivables)),
                    current_liabilities,
                ),
                "cash_ratio": _safe_divide(cash_and_short_term, current_liabilities),
                "cash_and_short_term_to_current_assets": _safe_divide(cash_and_short_term, current_assets),
                "cfo_to_current_liabilities": _safe_divide(operating_cash_flow, current_liabilities),
                "inventory_turnover": inventory_turnover,
                "payables_turnover": payables_turnover,
                "asset_turnover": total_assets_turnover,
                "working_capital_turnover": working_capital_turnover,
                "days_inventory_on_hand": days_inventory_on_hand,
                "days_sales_outstanding": days_sales_outstanding,
                "operating_cycle": operating_cycle,
                "days_payables_outstanding": days_payables_outstanding,
                "net_operating_cycle": net_operating_cycle,
                "net_debt_to_ebitda": _safe_divide(net_debt, ebitda),
                "net_debt_to_ebitda_minus_capex": _safe_divide(net_debt, ebitda + capital_expenditure),
                "total_debt_to_ebitda": _safe_divide(total_debt, ebitda),
                "interest_coverage": _safe_divide(operating_income, interest_expense),
                "ebitda_to_interest_expense": _safe_divide(ebitda, interest_expense),
                "fixed_charge_coverage_ratio": _safe_divide(operating_income, interest_expense),
                "cfo_to_interest_expense": _safe_divide(operating_cash_flow, interest_expense),
                "cash_dividend_coverage_ratio": _safe_divide(operating_cash_flow, dividends_paid),
                "long_term_debt_to_ebitda": _safe_divide(long_term_debt, ebitda),
                "net_debt_to_ffo": _safe_divide(net_debt, operating_cash_flow),
                "long_term_debt_to_ffo": _safe_divide(long_term_debt, operating_cash_flow),
                "cfo_to_total_debt": _safe_divide(operating_cash_flow, total_debt),
                "ebitda_minus_capex_to_interest_expense": _safe_divide(ebitda + capital_expenditure, interest_expense),
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
    """
    Build a metrics table for a group of companies.

    Parameters
    ----------
    records : iterable of FundamentalData
        Raw fundamental records.

    Returns
    -------
    pandas.DataFrame
        Metrics table indexed by ticker.
    """
    rows = [build_fundamental_metrics(record) for record in records]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("ticker", drop=False)


def build_metric_history_frame(
    records: Iterable[FundamentalData],
    frequency: StatementFrequency = "quarterly",
    trailing_periods: int = 4,
) -> pd.DataFrame:
    """
    Build a period-by-period metrics table for a group of companies.

    Parameters
    ----------
    records : iterable of FundamentalData
        Raw fundamental records.
    frequency : {"annual", "quarterly"}, default "quarterly"
        Statement frequency used to build histories.
    trailing_periods : int, default 4
        Maximum number of recent periods returned per company.

    Returns
    -------
    pandas.DataFrame
        Long-format metrics table across tickers and reporting periods.
    """
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
