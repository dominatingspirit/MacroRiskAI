import os
import sys
import json
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Dynamically add root project directory to path so imports resolve cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.agents.financial_parser import FinancialParser

class FinanceAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.2,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    def _get_llm_insights(self, profile: dict, financials: dict, ratios: dict, dna: dict) -> dict:
        parser = JsonOutputParser()
        
        prompt = PromptTemplate(
            template="""
            You are an expert corporate financial analyst. Review the following company profile, 
            current financials, calculated deterministic ratios, and core DNA traits.
            
            Company Profile: {profile}
            Current Financials: {financials}
            Financial Ratios: {ratios}
            Company DNA: {dna}
            
            Generate a concise, highly analytical assessment outputting EXACTLY in this JSON format:
            {{
                "strengths": ["strength 1", "strength 2"],
                "weaknesses": ["weakness 1", "weakness 2"],
                "summary": "A 2-3 sentence executive summary of the company's financial posture and situation."
            }}
            """,
            input_variables=["profile", "financials", "ratios", "dna"],
        )
        
        chain = prompt | self.llm | parser
        
        try:
            return chain.invoke({
                "profile": json.dumps(profile),
                "financials": json.dumps(financials),
                "ratios": json.dumps(ratios),
                "dna": json.dumps(dna)
            })
        except Exception as e:
            print(f"LLM Insight Generation Failed: {e}")
            return {
                "strengths": ["Stable cash reserves"],
                "weaknesses": ["High debt dependency"],
                "summary": "Financial analysis generated with fallback parameters due to LLM timeout."
            }

    def execute(self, parsed_workbook_data: Dict[str, Any], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        metadata = metadata or {}
        bs = parsed_workbook_data.get("balance_sheet", {})
        inc = parsed_workbook_data.get("income_statement", {})

        current_assets = bs.get("total_current_assets", 1.0)
        current_liabilities = bs.get("total_current_liabilities", 1.0)
        cash = bs.get("cash_and_equivalents", 0.0)
        inventory = bs.get("inventory", 0.0)
        total_assets = bs.get("total_assets", 1.0)
        total_debt = bs.get("total_debt", 0.0)
        total_equity = bs.get("total_equity", 1.0)

        revenue = inc.get("total_revenue", 1.0)
        cogs = inc.get("cogs", 0.0)
        gross_profit = inc.get("gross_profit", 0.0)
        ebitda = inc.get("ebitda", 0.0)
        net_income = inc.get("net_income", 0.0)
        interest_expense = inc.get("interest_expense", 1.0)

        company_profile = {
            "sector": metadata.get("sector", "Manufacturing"),
            "industry": metadata.get("industry", "Automobile Components"),
            "company_size": metadata.get("company_size", "Mid Cap")
        }

        current_financials = {
            "revenue": revenue,
            "ebitda": ebitda,
            "net_income": net_income,
            "cash": cash,
            "total_debt": total_debt,
            "inventory": inventory
        }

        current_ratio = round(current_assets / current_liabilities, 2) if current_liabilities > 0 else 0.0
        quick_ratio = round((current_assets - inventory) / current_liabilities, 2) if current_liabilities > 0 else 0.0
        cash_ratio = round(cash / current_liabilities, 2) if current_liabilities > 0 else 0.0
        debt_equity = round(total_debt / total_equity, 2) if total_equity > 0 else 0.0
        
        roe = round((net_income / total_equity) * 100, 2) if total_equity > 0 else 0.0
        roa = round((net_income / total_assets) * 100, 2) if total_assets > 0 else 0.0
        ebitda_margin = round((ebitda / revenue) * 100, 2) if revenue > 0 else 0.0
        gross_margin = round((gross_profit / revenue) * 100, 2) if revenue > 0 else 0.0
        
        ebitda_interest_coverage = round(ebitda / interest_expense, 2) if interest_expense > 0 else 99.9
        inventory_turnover = round(cogs / inventory, 2) if inventory > 0 else 0.0

        financial_ratios = {
            "current_ratio": current_ratio,
            "quick_ratio": quick_ratio,
            "cash_ratio": cash_ratio,
            "debt_equity": debt_equity,
            "roe": roe,
            "roa": roa,
            "ebitda_margin": ebitda_margin,
            "ebitda_interest_coverage": ebitda_interest_coverage,
            "inventory_turnover": inventory_turnover
        }

        capital_intensity = "High" if (total_assets / revenue) > 1.5 else "Low"
        pricing_power = "High" if gross_margin > 40 else "Medium" if gross_margin > 20 else "Low"
        
        if capital_intensity == "High" and pricing_power == "Low":
            inflation_sensitivity = "High"
        elif capital_intensity == "Low" and pricing_power == "High":
            inflation_sensitivity = "Low"
        else:
            inflation_sensitivity = "Moderate"

        debt_heavy = debt_equity > 1.5

        company_dna = {
            "capital_intensity": capital_intensity,
            "pricing_power": pricing_power,
            "inflation_sensitivity": inflation_sensitivity,
            "debt_heavy": debt_heavy
        }

        health_score = 50
        if current_ratio > 1.5: health_score += 10
        if cash_ratio > 0.3: health_score += 10
        if not debt_heavy: health_score += 15
        if ebitda_margin > 12: health_score += 15
        
        health_category = "Strong" if health_score >= 75 else "Stable" if health_score >= 50 else "Distressed"

        financial_health = {
            "category": health_category
        }

        insights = self._get_llm_insights(company_profile, current_financials, financial_ratios, company_dna)

        return {
            "company_profile": company_profile,
            "current_financials": current_financials,
            "financial_ratios": financial_ratios,
            "company_dna": company_dna,
            "financial_health": financial_health,
            "insights": insights
        }

if __name__ == "__main__":
    # Point to sample.xlsx in the root folder
    workbook_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sample.xlsx'))
    
    if not os.path.exists(workbook_path):
        print(f"❌ Error: Could not find 'sample.xlsx' at {workbook_path}")
    else:
        print("📊 Step 1: Parsing workbook deterministically...")
        parser = FinancialParser()
        parsed_data = parser.parse(workbook_path)

        print("🤖 Step 2: Running Finance Agent & generating LLM insights...")
        agent = FinanceAgent()
        final_output = agent.execute(parsed_data)

        print("\n✅ Final Output Schema:\n")
        print(json.dumps(final_output, indent=4))