from datetime import date
from typing import Any, Optional
from pydantic import BaseModel, Field

class Metadata(BaseModel):
    company_name: str = "ABC Ltd"
    sector: str = "Manufacturing"
    industry: str = "Steel"
    country: str = "India"
    forecast_horizon: int = 1  # Defaulting to 1 to match our current month-ahead macro forecast
    analysis_date: date = Field(default_factory=date.today)

class MacroDataPoint(BaseModel):
    """
    One monthly macro observation. 
    Updated to match the hybrid live (yfinance) and static (government) 
    data pipeline used by the MacroAgent.
    """
    date: str
    wpi: float
    repo_rate: float
    oil_price: float
    exchange_rate: float
    cpi_lag_1: float
    cpi_lag_2: float
    wpi_lag_1: float
    oil_lag_1: float

class AnalysisContext(BaseModel):
    """
    Single shared state object for the LangGraph pipeline.
    Updated to reflect the strict segregation of our 4 Agents and the Stress Engine.
    """
    metadata: Metadata = Field(default_factory=Metadata)

    # --- INPUTS ---
    raw_financials: dict[str, Any] = Field(default_factory=dict)
    macroeconomic_data: list[MacroDataPoint] = Field(default_factory=list)

    # --- AGENT 1: MACRO INTELLIGENCE ---
    macro_analysis: dict[str, Any] = Field(default_factory=dict)
    
    # --- AGENT 2: CORPORATE FINANCIAL INTELLIGENCE ---
    company_dna: dict[str, Any] = Field(default_factory=dict)
    
    # --- THE STRESS TESTING ENGINE ---
    stressed_financials: dict[str, Any] = Field(default_factory=dict)
    
    # --- AGENT 3: SCENARIO & RISK INTELLIGENCE ---
    risk_assessment: dict[str, Any] = Field(default_factory=dict)
    
    # --- AGENT 4: STRATEGIC DECISION INTELLIGENCE ---
    strategic_advice: dict[str, Any] = Field(default_factory=dict)

    errors: list[str] = Field(default_factory=list)

    def to_dashboard(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.model_dump(mode="json"),
            "macro_analysis": self.macro_analysis,
            "company_dna": self.company_dna,
            "stressed_financials": self.stressed_financials,
            "risk_assessment": self.risk_assessment,
            "strategic_advice": self.strategic_advice,
            "errors": self.errors,
        }