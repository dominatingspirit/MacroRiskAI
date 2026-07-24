from typing import Optional
from pydantic import BaseModel, Field
from app.schemas.common import MacroDataPoint, Metadata

class ForecastRequest(BaseModel):
    forecast_horizon: int = 1
    metadata: Optional[Metadata] = None
    macroeconomic_data: Optional[list[MacroDataPoint]] = None  # Falls back to our live yfinance API if omitted

class ForecastPoint(BaseModel):
    month: str
    inflation: float
    lower_ci: float
    upper_ci: float

class InflationForecast(BaseModel):
    model_config = {"protected_namespaces": ()}

    forecast: list[ForecastPoint]
    model_used: str = "Delta-Stacked Ensemble (XGB+LGB+ARIMAX+VAR)"
    confidence_score: float
    model_agreement: str  

class InflationRegime(BaseModel):
    model_config = {"populate_by_name": True}

    class_: str = Field(alias="class")
    probabilistic_distribution: dict[str, int]  

class MacroSummary(BaseModel):
    inflation_trend: str      
    policy_outlook: str        
    commodity_pressure: str    

class ForecastResponse(BaseModel):
    inflation_forecast: InflationForecast
    ensemble_breakdown: dict[str, float] = {}  
    inflation_regime: InflationRegime
    macro_summary: MacroSummary