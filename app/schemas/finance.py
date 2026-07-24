from typing import Any, Optional
from pydantic import BaseModel
from app.schemas.common import Metadata

class AnalyzeRequest(BaseModel):
    metadata: Optional[Metadata] = None
    ticker: Optional[str] = None
    raw_financials: dict[str, Any] = {} # Consolidated from separate statements for cleaner payload

class FinancialRatios(BaseModel):
    current_ratio: float
    quick_ratio: float
    cash_ratio: float
    roe: float
    roa: float
    ebitda_margin: float
    de_ratio: float

class FinancialHealth(BaseModel):
    score: float
    category: str

class CompanyDNA(BaseModel):
    capital_intensity: str
    pricing_power: str
    inflation_sensitivity: str
    cash_rich: bool
    debt_heavy: bool
    growth_stage: str
    working_capital: Optional[str] = None

class Insights(BaseModel):
    strengths: list[str]
    weaknesses: list[str]
    summary: str

class AnalyzeResponse(BaseModel):
    financial_ratios: FinancialRatios
    financial_health: FinancialHealth
    company_dna: CompanyDNA
    insights: Insights