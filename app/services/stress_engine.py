from typing import Any

SECTOR_PASS_THROUGH = {
    # fraction of cost inflation companies in this sector can pass on via pricing
    "Manufacturing": 0.45,
    "Steel": 0.35,
    "Cement": 0.4,
    "IT Services": 0.75,
    "FMCG": 0.6,
    "Retail": 0.55,
    "Pharma": 0.65,
}

PRICING_POWER_MULTIPLIER = {"High": 1.3, "Medium": 1.0, "Low": 0.7}


def run_stress_test(company_dna: dict[str, Any], forecast: dict[str, Any],
                     sector: str, current_financials: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Deterministic financial simulation. No ML/LLM involved — pure formula-driven
    projection of revenue, COGS, opex, EBITDA and margin under the forecasted
    inflation path.
    """
    inflation_points = forecast.get("forecast", [])
    avg_inflation = (
        sum(p["inflation"] for p in inflation_points) / len(inflation_points)
        if inflation_points else 5.0
    )

    baseline = current_financials or {"revenue": 1200.0, "ebitda": 210.0, "cogs": 750.0, "opex": 240.0}
    revenue = baseline.get("revenue", 1200.0)
    cogs = baseline.get("cogs", revenue * 0.62)
    opex = baseline.get("opex", revenue * 0.2)
    ebitda = baseline.get("ebitda", revenue - cogs - opex)
    baseline_margin = round((ebitda / revenue) * 100, 1) if revenue else 0.0

    pass_through = SECTOR_PASS_THROUGH.get(sector, 0.5)
    pricing_power = company_dna.get("pricing_power", "Medium")
    pass_through *= PRICING_POWER_MULTIPLIER.get(pricing_power, 1.0)
    pass_through = min(pass_through, 0.95)

    inflation_sensitivity = company_dna.get("inflation_sensitivity", "Medium")
    sensitivity_multiplier = {"High": 1.3, "Medium": 1.0, "Low": 0.7}.get(inflation_sensitivity, 1.0)

    cogs_change_pct = round(avg_inflation * sensitivity_multiplier, 2)
    revenue_change_pct = round((avg_inflation / 100) * pass_through * 100 - (avg_inflation * 0.15), 2)
    interest_cost_pct = round(avg_inflation * 0.6, 2)

    projected_revenue = round(revenue * (1 + revenue_change_pct / 100), 1)
    projected_cogs = round(cogs * (1 + cogs_change_pct / 100), 1)
    projected_opex = round(opex * (1 + (avg_inflation * 0.5) / 100), 1)
    projected_ebitda = round(projected_revenue - projected_cogs - projected_opex, 1)
    projected_margin = round((projected_ebitda / projected_revenue) * 100, 1) if projected_revenue else 0.0

    return {
        "stress_test": {
            "baseline": {
                "revenue": round(revenue, 1),
                "ebitda": round(ebitda, 1),
                "margin": baseline_margin,
            },
            "projected": {
                "revenue": projected_revenue,
                "ebitda": projected_ebitda,
                "margin": projected_margin,
            },
            "assumptions": {
                "revenue_change": revenue_change_pct,
                "cogs_change": cogs_change_pct,
                "interest_cost": interest_cost_pct,
            },
        },
        "projected_ratios": {
            "ebitda_margin": projected_margin,
            "revenue_growth": revenue_change_pct,
            "cost_inflation": cogs_change_pct,
        },
    }
