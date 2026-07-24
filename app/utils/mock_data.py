"""
Synthetic data generators.

Used so the pipeline is fully runnable offline (hackathon-friendly) whenever
the caller doesn't supply real macro history / financial statements / a real
macro API key. Swap `generate_macro_history` for a real FRED/RBI/MOSPI fetch
in production by implementing `app/services/forecasting.py::fetch_live_macro_data`.
"""

import math
import random
from datetime import date
from dateutil.relativedelta import relativedelta


def generate_macro_history(months: int = 48, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    start = date.today() - relativedelta(months=months)
    data = []
    cpi = 5.2
    oil = 80.0
    fx = 82.0
    for i in range(months):
        d = start + relativedelta(months=i)
        cpi += rng.uniform(-0.2, 0.25) + 0.02 * math.sin(i / 6)
        cpi = max(2.0, min(9.0, cpi))
        oil += rng.uniform(-3, 3.2)
        oil = max(55, min(120, oil))
        fx += rng.uniform(-0.4, 0.5)
        data.append({
            "date": d.strftime("%Y-%m"),
            "cpi": round(cpi, 2),
            "core_cpi": round(cpi - rng.uniform(0.2, 0.6), 2),
            "wpi": round(cpi + rng.uniform(2, 6), 2),
            "repo_rate": 6.5,
            "reverse_repo": 3.35,
            "oil_price": round(oil, 2),
            "exchange_rate": round(fx, 2),
            "pmi": round(rng.uniform(52, 59), 1),
            "iip": round(140 + rng.uniform(-5, 10), 1),
        })
    return data


def generate_financial_statements(seed: int = 7) -> dict:
    rng = random.Random(seed)
    revenue = round(rng.uniform(900, 1500), 1)
    cogs = round(revenue * rng.uniform(0.55, 0.68), 1)
    opex = round(revenue * rng.uniform(0.12, 0.2), 1)
    ebitda = round(revenue - cogs - opex, 1)
    interest = round(rng.uniform(15, 45), 1)
    depreciation = round(rng.uniform(20, 60), 1)
    tax = round(max(0.0, (ebitda - interest - depreciation) * 0.25), 1)
    net_income = round(ebitda - interest - depreciation - tax, 1)

    total_assets = round(revenue * rng.uniform(1.1, 1.6), 1)
    current_assets = round(total_assets * rng.uniform(0.35, 0.5), 1)
    cash = round(current_assets * rng.uniform(0.2, 0.4), 1)
    inventory = round(current_assets * rng.uniform(0.25, 0.4), 1)
    current_liabilities = round(current_assets * rng.uniform(0.4, 0.65), 1)
    total_debt = round(total_assets * rng.uniform(0.2, 0.45), 1)
    total_equity = round(total_assets - total_debt - current_liabilities, 1)

    operating_cf = round(ebitda - interest - tax + rng.uniform(-20, 20), 1)
    investing_cf = round(-rng.uniform(30, 90), 1)
    financing_cf = round(rng.uniform(-40, 40), 1)

    return {
        "income_statement": {
            "revenue": revenue,
            "cogs": cogs,
            "operating_expenses": opex,
            "ebitda": ebitda,
            "interest_expense": interest,
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
            "net_change_in_cash": round(operating_cf + investing_cf + financing_cf, 1),
        },
    }
