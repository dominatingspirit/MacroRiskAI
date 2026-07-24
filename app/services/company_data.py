"""
Fetches real, publicly-reported company financials (for listed companies)
from Alpha Vantage and maps them into the same balance_sheet /
income_statement / cash_flow shape that app/services/ratio_engine.py expects,
so a real ticker can be used anywhere the app currently accepts
hand-typed/mock statements.
"""

from typing import Any

from app.utils import api_client
from app.utils.api_client import AlphaVantageError
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _num(value: Any, default: float = 0.0) -> float:
    """Alpha Vantage returns numbers as strings, and 'None' literally for missing fields."""
    if value in (None, "None", "", "-"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _latest_report(payload: dict, period: str = "annual") -> dict:
    key = "annualReports" if period == "annual" else "quarterlyReports"
    reports = payload.get(key, [])
    if not reports:
        raise AlphaVantageError(f"No {key} found in Alpha Vantage response for this symbol")
    return reports[0]  # Alpha Vantage returns most-recent first


def fetch_company_financials(ticker: str, period: str = "annual") -> dict[str, dict]:
    """
    Returns {"income_statement": {...}, "balance_sheet": {...}, "cash_flow": {...}}
    using the most recent annual (or quarterly) report for `ticker`.
    """
    ticker = ticker.strip().upper()

    income_payload = api_client.fetch_income_statement(ticker)
    balance_payload = api_client.fetch_balance_sheet(ticker)
    cashflow_payload = api_client.fetch_cash_flow(ticker)

    inc = _latest_report(income_payload, period)
    bs = _latest_report(balance_payload, period)
    cf = _latest_report(cashflow_payload, period)

    revenue = _num(inc.get("totalRevenue"))
    cogs = _num(inc.get("costofGoodsAndServicesSold")) or _num(inc.get("costOfRevenue"))
    opex = _num(inc.get("operatingExpenses"))
    ebitda = _num(inc.get("ebitda")) or (revenue - cogs - opex)
    interest_expense = _num(inc.get("interestExpense")) or _num(inc.get("interestAndDebtExpense"))
    depreciation = _num(inc.get("depreciationAndAmortization")) or _num(inc.get("depreciation"))
    tax = _num(inc.get("incomeTaxExpense"))
    net_income = _num(inc.get("netIncome"))

    total_assets = _num(bs.get("totalAssets"))
    current_assets = _num(bs.get("totalCurrentAssets"))
    cash = _num(bs.get("cashAndCashEquivalentsAtCarryingValue")) or _num(bs.get("cashAndShortTermInvestments"))
    inventory = _num(bs.get("inventory"))
    current_liabilities = _num(bs.get("totalCurrentLiabilities"))
    total_debt = _num(bs.get("shortLongTermDebtTotal")) or (
        _num(bs.get("shortTermDebt")) + _num(bs.get("longTermDebt"))
    )
    total_equity = _num(bs.get("totalShareholderEquity"))

    operating_cf = _num(cf.get("operatingCashflow"))
    investing_cf = _num(cf.get("cashflowFromInvestment"))
    financing_cf = _num(cf.get("cashflowFromFinancing"))
    net_change_cash = _num(cf.get("changeInCashAndCashEquivalents")) or (
        operating_cf + investing_cf + financing_cf
    )

    logger.info(f"Fetched real {period} financials for {ticker} (fiscal date: {inc.get('fiscalDateEnding')})")

    return {
        "fiscal_date_ending": inc.get("fiscalDateEnding"),
        "income_statement": {
            "revenue": revenue,
            "cogs": cogs,
            "operating_expenses": opex,
            "ebitda": ebitda,
            "interest_expense": interest_expense,
            "depreciation": depreciation,
            "tax": tax,
            "net_income": net_income,
        },
        "balance_sheet": {
            "total_assets": total_assets,
            "current_assets": current_assets,
            "cash": cash,
            "inventory": inventory,
            "current_liabilities": current_liabilities,
            "total_debt": total_debt,
            "total_equity": total_equity,
        },
        "cash_flow": {
            "operating_cash_flow": operating_cf,
            "investing_cash_flow": investing_cf,
            "financing_cash_flow": financing_cf,
            "net_change_in_cash": net_change_cash,
        },
    }
