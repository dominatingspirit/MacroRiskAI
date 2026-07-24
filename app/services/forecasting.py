"""
Macro data sourcing utilities shared by the /forecast route and the
MacroAgent pipeline path.

NOTE: this module used to also contain `forecast_inflation()`, a live
per-request ARIMAX+VAR+XGBoost fit built on `ArimaxForecaster`/
`VarForecaster`/`XGBoostStacker` wrapper classes. Those classes were
replaced by offline training scripts (see app/models/arimax.py, var.py,
xgboost_stack.py) that produce the .pkl artifacts MacroAgent loads directly
(app/agents/macro_agent.py). forecast_inflation() is gone — MacroAgent is now
the single source of truth for inflation forecasting, used by both /forecast
and /pipeline.
"""
import pandas as pd

from app.config.settings import get_settings
from app.utils import api_client
from app.utils.api_client import AlphaVantageError
from app.utils.logger import get_logger
from app.utils.mock_data import generate_macro_history

logger = get_logger(__name__)

# Documented, deterministic assumptions used to fill in fields that don't have
# a free real-time source on Alpha Vantage. These are NOT randomly generated —
# they're fixed offsets applied to the real series so the shape of the data
# still moves with real inflation/rates. Replace with a real feed (e.g. RBI's
# WPI/PMI releases) when available.
CORE_CPI_OFFSET = 0.4          # core CPI typically runs slightly below headline
WPI_SPREAD_OVER_CPI = 2.0      # wholesale/producer prices vs. headline CPI
REPO_TO_FFR_SPREAD = 1.5       # RBI repo rate has historically run above the US Fed funds rate
REVERSE_REPO_SPREAD = 0.9      # RBI reverse repo sits below the repo rate
DEFAULT_PMI = 52.0             # neutral-to-expansionary; no free real-time PMI source
DEFAULT_IIP = 145.0            # last known-ish industrial production index level


def _series_to_monthly_df(payload: dict, value_key: str = "value", column: str = "value") -> pd.DataFrame:
    """Alpha Vantage time-series responses share a `{"data": [{"date": ..., "value": ...}]}` shape."""
    records = payload.get("data", [])
    if not records:
        raise AlphaVantageError(f"Alpha Vantage response had no data for {payload.get('name', 'series')}")
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df[column] = pd.to_numeric(df[value_key], errors="coerce")
    df = df.dropna(subset=[column]).sort_values("date")
    df["month"] = df["date"].dt.to_period("M")
    # collapse to monthly (last observation of the month, since some series are daily/weekly)
    monthly = df.groupby("month")[column].last()
    return monthly


def fetch_live_macro_data(months: int = 48) -> list[dict]:
    """
    Real macro data integration via Alpha Vantage (one free API key covers
    CPI, oil, Fed funds rate and FX — see app/utils/api_client.py).

    cpi/oil_price/exchange_rate/repo_rate below are genuine live series. A few
    fields (core_cpi, wpi, reverse_repo, pmi, iip) don't have a free real-time
    source, so they're derived from the real series with documented, fixed
    assumptions above rather than left as random mock noise. Swap in a real
    feed for those (e.g. RBI/MOSPI) when you have access to one.
    """
    settings = get_settings()
    if not settings.alphavantage_api_key:
        raise AlphaVantageError("ALPHAVANTAGE_API_KEY is not configured")

    cpi_monthly = _series_to_monthly_df(api_client.fetch_cpi(), column="cpi_index")
    oil_monthly = _series_to_monthly_df(api_client.fetch_wti_oil(), column="oil_price")
    ffr_monthly = _series_to_monthly_df(api_client.fetch_federal_funds_rate(), column="repo_rate")
    fx_monthly = _series_to_monthly_df(
        api_client.fetch_fx_monthly(settings.fx_from_symbol, settings.fx_to_symbol),
        column="exchange_rate",
    )

    df = pd.concat([cpi_monthly, oil_monthly, ffr_monthly, fx_monthly], axis=1).sort_index()
    df = df.ffill().dropna()
    if len(df) < 13:
        raise AlphaVantageError("Not enough overlapping history returned by Alpha Vantage to build a forecast")

    df = df.tail(months + 12)  # extra 12 months so YoY CPI can be computed for the full window

    # CPI is a level/index on Alpha Vantage; the model wants YoY % inflation.
    df["cpi"] = df["cpi_index"].pct_change(periods=12) * 100
    df = df.dropna(subset=["cpi"]).tail(months)

    # WPI is derived from CPI plus a term tied to oil-price movement (wholesale
    # prices are more commodity-sensitive than headline CPI) rather than a
    # pure constant offset — a constant offset makes wpi perfectly collinear
    # with cpi, which crashes the VAR model (singular covariance matrix).
    oil_deviation = df["oil_price"] - df["oil_price"].mean()

    records = []
    for month_period, row in df.iterrows():
        oil_dev = float(oil_deviation.loc[month_period])
        records.append({
            "date": str(month_period),
            "cpi": round(float(row["cpi"]), 2),
            "core_cpi": round(float(row["cpi"]) - CORE_CPI_OFFSET, 2),
            "wpi": round(float(row["cpi"]) + WPI_SPREAD_OVER_CPI + 0.03 * oil_dev, 2),
            "repo_rate": round(float(row["repo_rate"]) + REPO_TO_FFR_SPREAD, 2),
            "reverse_repo": round(float(row["repo_rate"]) + REPO_TO_FFR_SPREAD - REVERSE_REPO_SPREAD, 2),
            "oil_price": round(float(row["oil_price"]), 2),
            "exchange_rate": round(float(row["exchange_rate"]), 2),
            "pmi": DEFAULT_PMI,
            "iip": DEFAULT_IIP,
        })

    if not records:
        raise AlphaVantageError("Alpha Vantage returned data but none survived alignment/cleaning")
    return records


def get_macro_dataframe(records: list[dict] | None) -> pd.DataFrame:
    """
    Single source of macro history for both /forecast and /pipeline:
    caller-supplied records > live Alpha Vantage (if MACRO_DATA_MODE=live) >
    mock data. Returned df is indexed by date, sorted ascending, ready to be
    fed into MacroAgent.run_agent() (via .reset_index().to_dict("records")).
    """
    if not records:
        settings = get_settings()
        if settings.macro_data_mode == "live":
            try:
                records = fetch_live_macro_data()
            except AlphaVantageError as exc:
                logger.warning(f"Live macro fetch failed, falling back to mock data: {exc}")
                records = generate_macro_history()
        else:
            records = generate_macro_history()
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df
