import pandas as pd

class FinancialParser:
    def __init__(self):
        pass

    def _load_sheet(self, excel_file, sheet_name):
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        df = df.iloc[:, :2]
        df.columns = ["Line Item", "Value"]
        df = df.dropna(subset=["Line Item"])

        section_headers = {
            "CURRENT ASSETS",
            "NON-CURRENT ASSETS",
            "CURRENT LIABILITIES",
            "NON-CURRENT LIABILITIES",
            "EQUITY",
            "Operating Activities",
            "Investing Activities",
            "Financing Activities"
        }

        df = df[~df["Line Item"].isin(section_headers)]
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce").fillna(0)
        return dict(zip(df["Line Item"], df["Value"]))

    def parse(self, excel_file):
        bs = self._load_sheet(excel_file, "1_BALANCE_SHEET")
        inc = self._load_sheet(excel_file, "2_INCOME_STATEMENT")
        cf = self._load_sheet(excel_file, "3_CASHFLOW_STATEMENT")

        cash = bs.get("Cash and Cash Equivalents", 0)
        receivables = bs.get("Trade Receivables", 0)
        inventory = bs.get("Inventories", 0)
        other_current_assets = bs.get("Other Current Assets", 0)

        current_assets = cash + receivables + inventory + other_current_assets

        ppe = bs.get("Property, Plant and Equipment", 0)
        intangible = bs.get("Intangible Assets", 0)
        other_non_current = bs.get("Other Non-current Assets", 0)

        total_assets = current_assets + ppe + intangible + other_non_current

        trade_payables = bs.get("Trade Payables", 0)
        short_term_debt = bs.get("Short-term Borrowings", 0)
        other_current_liabilities = bs.get("Other Current Liabilities", 0)

        current_liabilities = trade_payables + short_term_debt + other_current_liabilities

        long_term_debt = bs.get("Long-term Borrowings", 0)
        total_debt = short_term_debt + long_term_debt

        share_capital = bs.get("Share Capital", 0)
        other_equity = bs.get("Other Equity", 0)
        total_equity = share_capital + other_equity

        revenue = inc.get("Total Revenue", 0)
        if revenue == 0:
            revenue = inc.get("Revenue from Operations", 0) + inc.get("Other Income", 0)

        cogs = inc.get("Cost of Materials Consumed / COGS", 0)
        gross_profit = revenue - cogs
        ebitda = inc.get("EBITDA", 0)
        ebit = inc.get("EBIT", 0)
        interest = inc.get("Finance Costs", 0)
        net_income = inc.get("Profit After Tax (Net Income)", 0)

        operating_cf = cf.get("Net Cash from Operating Activities", 0)
        capex = cf.get("Capital Expenditure", 0)
        investing_cf = cf.get("Net Cash from Investing Activities", 0)
        financing_cf = cf.get("Net Cash from Financing Activities", 0)
        net_cash_change = cf.get("Net Increase in Cash & Cash Equivalents", 0)

        return {
            "balance_sheet": {
                "total_current_assets": current_assets,
                "total_current_liabilities": current_liabilities,
                "cash_and_equivalents": cash,
                "trade_receivables": receivables,
                "inventory": inventory,
                "trade_payables": trade_payables,
                "total_assets": total_assets,
                "total_debt": total_debt,
                "total_equity": total_equity
            },
            "income_statement": {
                "total_revenue": revenue,
                "cogs": cogs,
                "gross_profit": gross_profit,
                "ebitda": ebitda,
                "ebit": ebit,
                "interest_expense": interest,
                "net_income": net_income
            },
            "cash_flow": {
                "net_operating_cash": operating_cf,
                "capex": capex,
                "net_investing_cash": investing_cf,
                "net_financing_cash": financing_cf,
                "net_increase_cash": net_cash_change
            }
        }