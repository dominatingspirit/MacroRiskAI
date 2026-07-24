import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.utils.mock_data import generate_financial_statements
from app.utils.api_client import AlphaVantageError
from app.services import company_data

STATEMENTS = generate_financial_statements()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_forecast(client):
    resp = client.post("/forecast", json={"forecast_horizon": 6})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["inflation_forecast"]["forecast"]) == 6
    assert "inflation_regime" in body


def test_analyze(client):
    resp = client.post("/analyze", json={
        "balance_sheet": STATEMENTS["balance_sheet"],
        "income_statement": STATEMENTS["income_statement"],
        "cash_flow": STATEMENTS["cash_flow"],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "financial_ratios" in body
    assert "company_dna" in body


def test_stress_test(client):
    forecast_resp = client.post("/forecast", json={"forecast_horizon": 6}).json()
    analyze_resp = client.post("/analyze", json={
        "balance_sheet": STATEMENTS["balance_sheet"],
        "income_statement": STATEMENTS["income_statement"],
        "cash_flow": STATEMENTS["cash_flow"],
    }).json()

    resp = client.post("/stress-test", json={
        "company_dna": analyze_resp["company_dna"],
        "financial_ratios": analyze_resp["financial_ratios"],
        "forecast": forecast_resp["inflation_forecast"],
        "sector": "Manufacturing",
    })
    assert resp.status_code == 200
    assert "stress_test" in resp.json()


def test_analyze_without_key_and_ticker_returns_502(client):
    """No ALPHAVANTAGE_API_KEY configured in the test env -> a ticker request
    should fail loudly with a 502, not silently substitute fake data."""
    resp = client.post("/analyze", json={"ticker": "AAPL"})
    assert resp.status_code == 502
    assert "ALPHAVANTAGE_API_KEY" in resp.json()["detail"]


def test_analyze_ticker_uses_real_mapped_financials(client, monkeypatch):
    """With the HTTP layer mocked, a ticker request should flow through
    company_data's Alpha Vantage mapping into /analyze's ratio engine."""

    def fake_fetch(ticker, period="annual"):
        assert ticker == "AAPL"
        return {
            "fiscal_date_ending": "2025-09-30",
            "income_statement": {
                "revenue": 1000.0, "cogs": 600.0, "operating_expenses": 150.0,
                "ebitda": 250.0, "interest_expense": 10.0, "depreciation": 20.0,
                "tax": 40.0, "net_income": 180.0,
            },
            "balance_sheet": {
                "total_assets": 2000.0, "current_assets": 700.0, "cash": 300.0,
                "inventory": 150.0, "current_liabilities": 400.0,
                "total_debt": 500.0, "total_equity": 1100.0,
            },
            "cash_flow": {
                "operating_cash_flow": 220.0, "investing_cash_flow": -80.0,
                "financing_cash_flow": -30.0, "net_change_in_cash": 110.0,
            },
        }

    monkeypatch.setattr("app.routers.analyze.fetch_company_financials", fake_fetch)
    resp = client.post("/analyze", json={"ticker": "AAPL", "metadata": {"sector": "IT Services"}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["financial_ratios"]["current_ratio"] == round(700.0 / 400.0, 2)


def test_company_data_maps_alpha_vantage_fields(monkeypatch):
    """Unit test for the field-mapping logic against a shape matching real
    Alpha Vantage payloads (numbers as strings, occasional 'None' literals)."""

    def fake_income(symbol):
        return {"annualReports": [{
            "fiscalDateEnding": "2025-09-30", "totalRevenue": "1000", "costOfRevenue": "600",
            "operatingExpenses": "150", "ebitda": "250", "interestExpense": "10",
            "depreciationAndAmortization": "20", "incomeTaxExpense": "40", "netIncome": "180",
        }]}

    def fake_balance(symbol):
        return {"annualReports": [{
            "totalAssets": "2000", "totalCurrentAssets": "700",
            "cashAndCashEquivalentsAtCarryingValue": "300", "inventory": "150",
            "totalCurrentLiabilities": "400", "shortTermDebt": "100", "longTermDebt": "400",
            "shortLongTermDebtTotal": "None", "totalShareholderEquity": "1100",
        }]}

    def fake_cashflow(symbol):
        return {"annualReports": [{
            "operatingCashflow": "220", "cashflowFromInvestment": "-80",
            "cashflowFromFinancing": "-30", "changeInCashAndCashEquivalents": "110",
        }]}

    monkeypatch.setattr(company_data.api_client, "fetch_income_statement", fake_income)
    monkeypatch.setattr(company_data.api_client, "fetch_balance_sheet", fake_balance)
    monkeypatch.setattr(company_data.api_client, "fetch_cash_flow", fake_cashflow)

    result = company_data.fetch_company_financials("aapl")
    assert result["income_statement"]["revenue"] == 1000.0
    # shortLongTermDebtTotal was "None" -> falls back to shortTermDebt + longTermDebt
    assert result["balance_sheet"]["total_debt"] == 500.0
    assert result["cash_flow"]["net_change_in_cash"] == 110.0


def test_full_pipeline(client):
    resp = client.post("/pipeline", json={
        "metadata": {"company_name": "Test Co", "sector": "Manufacturing"},
        "balance_sheet": STATEMENTS["balance_sheet"],
        "income_statement": STATEMENTS["income_statement"],
        "cash_flow": STATEMENTS["cash_flow"],
        "forecast_horizon": 6,
    })
    assert resp.status_code == 200
    body = resp.json()
    for key in ["forecast", "financial_health", "company_dna", "stress_test", "risk_assessment", "strategy"]:
        assert key in body
