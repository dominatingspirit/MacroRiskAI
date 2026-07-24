from typing import Any
import numpy as np

_LEVEL_SCORE = {"High": 1.0, "Medium": 0.5, "Low": 0.0}


def build_risk_features(stress_test: dict[str, Any], company_dna: dict[str, Any],
                         financial_ratios: dict[str, Any], inflation_forecast: dict[str, Any]) -> np.ndarray:
    assumptions = stress_test.get("assumptions", {})
    projected = stress_test.get("projected", {})

    forecast_points = inflation_forecast.get("forecast", [])
    avg_inflation = (
        sum(p["inflation"] for p in forecast_points) / len(forecast_points)
        if forecast_points else 5.0
    )

    features = [
        financial_ratios.get("current_ratio", 1.5),
        financial_ratios.get("quick_ratio", 1.0),
        financial_ratios.get("de_ratio", 0.5),
        projected.get("margin", financial_ratios.get("ebitda_margin", 15)),
        financial_ratios.get("roe", 10),
        avg_inflation,
        assumptions.get("revenue_change", 0.0),
        assumptions.get("cogs_change", 0.0),
        _LEVEL_SCORE.get(company_dna.get("capital_intensity", "Medium"), 0.5),
        _LEVEL_SCORE.get(company_dna.get("pricing_power", "Medium"), 0.5),
    ]
    return np.array(features, dtype=float).reshape(1, -1)


def build_risk_breakdown(financial_ratios: dict[str, Any], company_dna: dict[str, Any],
                          stress_test: dict[str, Any], avg_inflation: float) -> dict[str, float]:
    liquidity = max(0, min(100, 100 - financial_ratios.get("current_ratio", 1.5) * 35))
    profitability = max(0, min(100, 100 - financial_ratios.get("ebitda_margin", 15) * 3))
    debt = max(0, min(100, financial_ratios.get("de_ratio", 0.5) * 55))
    inflation = max(0, min(100, avg_inflation * 6))
    sector = max(0, min(100, 40 if company_dna.get("inflation_sensitivity") == "High" else
                         25 if company_dna.get("inflation_sensitivity") == "Medium" else 12))

    return {
        "liquidity": round(liquidity, 1),
        "profitability": round(profitability, 1),
        "debt": round(debt, 1),
        "inflation": round(inflation, 1),
        "sector": round(sector, 1),
    }
