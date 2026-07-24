from typing import Any
from pydantic import BaseModel

class RiskRequest(BaseModel):
    stressed_financials: dict[str, Any] # Updated to match LangGraph state
    company_dna: dict[str, Any] = {}
    macro_analysis: dict[str, Any] = {}

class RiskAssessment(BaseModel):
    financial_distress_probability: float
    risk_category: str
    risk_score: float

class FeatureImportanceItem(BaseModel):
    feature: str
    shap_value: float

class RiskResponse(BaseModel):
    risk_assessment: RiskAssessment
    risk_breakdown: dict[str, float]
    feature_importance: list[FeatureImportanceItem]