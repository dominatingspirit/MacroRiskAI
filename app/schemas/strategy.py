from typing import Any, Optional
from pydantic import BaseModel

class StrategyRequest(BaseModel):
    risk_assessment: dict[str, Any]
    stressed_financials: dict[str, Any] = {}
    macro_analysis: dict[str, Any] = {}
    company_dna: dict[str, Any] = {}

class Strategy(BaseModel):
    pricing: str
    inventory: str
    financing: str
    cost_control: str
    cash_management: str

class StrategyResponse(BaseModel):
    strategy: Strategy
    executive_summary: str
    priority_actions: list[str]

class PipelineRequest(BaseModel):
    """Full end-to-end run triggering LangGraph."""
    metadata: Optional[dict[str, Any]] = None
    macroeconomic_data: Optional[list[dict[str, Any]]] = None
    ticker: Optional[str] = None
    raw_financials: dict[str, Any] = {}
    forecast_horizon: int = 1
