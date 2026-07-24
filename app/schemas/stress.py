from typing import Any, Optional
from pydantic import BaseModel

class StressTestRequest(BaseModel):
    company_dna: dict[str, Any]
    raw_financials: dict[str, Any] = {}
    macro_analysis: dict[str, Any] = {} # Replaced separate forecast/regime fields with the consolidated agent 1 output

class StressBaseline(BaseModel):
    revenue: float
    ebitda: float
    margin: float

class StressAssumptions(BaseModel):
    revenue_change: float
    cogs_change: float
    interest_cost: float

class StressTestResult(BaseModel):
    baseline: StressBaseline
    projected: StressBaseline
    assumptions: StressAssumptions

class StressTestResponse(BaseModel):
    stress_test: StressTestResult