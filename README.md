# InflationGuard AI — Backend

Agentic financial intelligence platform: forecasts inflation, analyzes company
financials, stress-tests them under the forecasted macro environment, predicts
financial distress, and generates AI strategy recommendations.

## Architecture

Four LangGraph agents + one deterministic stress engine share a single state
object (`AnalysisContext`):

```
Macro Agent (MIA) → Finance Agent (CFIA) → Stress Engine → Risk Agent (SRIA) → Strategy Agent (SDIA)
```

- **Deterministic**: financial ratios, Company DNA, stress testing math — never touched by an LLM.
- **ML**: ARIMAX + VAR + XGBoost stacking for inflation; LightGBM + SHAP for distress risk.
- **LLM**: only used for interpreting ratios/risk and drafting strategy text (Anthropic by default, OpenAI supported, falls back to a deterministic mock if no key is set — so the whole API works out of the box).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY if you want real LLM text, otherwise leave blank
uvicorn app.main:app --reload
```

Docs at `http://localhost:8000/docs`.

Run tests:

```bash
pytest -q
```

Docker:

```bash
docker compose -f docker/docker-compose.yml up --build
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/forecast` | Inflation forecast + regime classification |
| POST | `/analyze` | Ratios, health score, Company DNA, insights |
| POST | `/stress-test` | Project financials under the forecasted inflation path |
| POST | `/risk` | Financial distress probability + SHAP feature importance |
| POST | `/strategy` | Executive strategy recommendations |
| POST | `/pipeline` | Runs the full LangGraph pipeline end-to-end, returns the dashboard payload, persists the run |
| GET | `/pipeline/{run_id}` | Fetch a previously saved run |
| GET | `/health` | Liveness check |

If you don't pass `macroeconomic_data` / financial statements, endpoints fall
back to realistic synthetic data (`app/utils/mock_data.py`) so everything is
runnable with zero setup.

### Real data (Alpha Vantage)

Set `MACRO_DATA_MODE=live` and `ALPHAVANTAGE_API_KEY` in `.env` (free key:
https://www.alphavantage.co/support/#api-key) to pull real data instead of
mock data:

- **Macro** (`/forecast`, `/pipeline` when `macroeconomic_data` is omitted):
  real CPI, WTI oil price, Fed funds rate, and FX (`FX_FROM_SYMBOL` →
  `FX_TO_SYMBOL`, default USD→INR) from Alpha Vantage
  (`app/services/forecasting.py::fetch_live_macro_data`). CPI is converted
  from an index level to YoY % inflation. A few fields with no free
  real-time source (core CPI, WPI, reverse repo, PMI, IIP) are derived from
  the real series via documented fixed offsets/defaults — see the constants
  at the top of `forecasting.py` — rather than randomized; swap in a real
  feed (e.g. RBI/MOSPI) for those when you have access to one.
- **Company financials** (`/analyze`, `/pipeline`): pass a `ticker` (e.g.
  `"ticker": "AAPL"`) instead of `balance_sheet`/`income_statement`/
  `cash_flow` to pull the latest real annual report via Alpha Vantage
  (`app/services/company_data.py::fetch_company_financials`).

If the API key is missing, rate-limited (free tier = 25 requests/day), or
Alpha Vantage errors, macro fetches fall back to mock data automatically and
`/analyze`+`/pipeline` return a clear `502` naming the ticker issue (bad
symbol, no data, rate limit, etc.) rather than silently faking a ticker's
financials.

## Repo layout

```
app/
├── agents/        # 4 LangGraph agent nodes
├── services/       # deterministic engines: ratios, stress test, forecasting orchestration, feature eng
├── models/         # ARIMAX, VAR, XGBoost stacker, LightGBM, SHAP wrappers
├── schemas/        # pydantic contracts + shared AnalysisContext
├── routers/        # FastAPI endpoints
├── database/        # SQLAlchemy models + session
├── utils/          # mock data, LLM client, logging
├── config/         # settings
├── graph.py        # LangGraph wiring
└── main.py         # FastAPI app
```

## Notes for the hackathon MVP

- LightGBM risk model bootstraps on a synthetic-but-plausible dataset at
  startup so `/risk` and `/pipeline` work immediately; swap in real labeled
  distress data via `LightGBMRiskModel.fit()` when available.
- XGBoost stacker falls back to a weighted ARIMAX/VAR blend when there isn't
  enough history to train a meta-model — no crash on small datasets.
- SQLite by default; set `DATABASE_URL` to Postgres for production.
