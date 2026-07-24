"""
Thin HTTP client for Alpha Vantage — a single free-tier API key gives us both
macro-economic series (CPI, WTI oil, Fed funds rate, FX) and company
fundamentals (income statement, balance sheet, cash flow), which is why it
was chosen over stitching together several government data portals.

Get a free key at https://www.alphavantage.co/support/#api-key and set
ALPHAVANTAGE_API_KEY in .env. Free tier = 25 requests/day, so responses are
cached in-process for `api_cache_ttl_seconds` (default 1h) to avoid burning
through the quota on repeated calls during a session.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from app.config.settings import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_cache: dict[str, tuple[float, dict]] = {}


class AlphaVantageError(Exception):
    """Raised for any failure talking to Alpha Vantage (network, auth, rate limit)."""


def _cache_key(params: dict[str, Any]) -> str:
    return "&".join(f"{k}={v}" for k, v in sorted(params.items()) if k != "apikey")


def _request(params: dict[str, Any], timeout: int = 20) -> dict:
    settings = get_settings()
    if not settings.alphavantage_api_key:
        raise AlphaVantageError(
            "ALPHAVANTAGE_API_KEY is not set. Get a free key at "
            "https://www.alphavantage.co/support/#api-key and add it to .env, "
            "or leave MACRO_DATA_MODE=mock to keep using synthetic data."
        )

    key = _cache_key(params)
    cached = _cache.get(key)
    now = time.time()
    if cached and (now - cached[0]) < settings.api_cache_ttl_seconds:
        return cached[1]

    try:
        resp = requests.get(
            settings.alphavantage_base_url,
            params={**params, "apikey": settings.alphavantage_api_key},
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error(f"Alpha Vantage request failed for {params.get('function')}: {exc}")
        raise AlphaVantageError(f"Network error calling Alpha Vantage: {exc}") from exc

    try:
        data = resp.json()
    except ValueError as exc:
        raise AlphaVantageError("Alpha Vantage returned a non-JSON response") from exc

    # Alpha Vantage returns 200 OK even for errors/rate limits, with an
    # explanatory message under one of these keys instead of a normal payload.
    for error_key in ("Error Message", "Note", "Information"):
        if error_key in data:
            raise AlphaVantageError(f"Alpha Vantage error ({params.get('function')}): {data[error_key]}")

    _cache[key] = (now, data)
    return data


def fetch_cpi(interval: str = "monthly") -> dict:
    """US Consumer Price Index, level (index, not YoY %)."""
    return _request({"function": "CPI", "interval": interval})


def fetch_wti_oil(interval: str = "monthly") -> dict:
    """Crude oil, West Texas Intermediate, USD/barrel."""
    return _request({"function": "WTI", "interval": interval})


def fetch_federal_funds_rate(interval: str = "monthly") -> dict:
    """US Federal Funds effective rate — used as the policy-rate proxy."""
    return _request({"function": "FEDERAL_FUNDS_RATE", "interval": interval})


def fetch_fx_monthly(from_symbol: str, to_symbol: str) -> dict:
    """Monthly FX rate between two currencies (e.g. USD -> INR)."""
    return _request({"function": "FX_MONTHLY", "from_symbol": from_symbol, "to_symbol": to_symbol})


def fetch_income_statement(symbol: str) -> dict:
    return _request({"function": "INCOME_STATEMENT", "symbol": symbol})


def fetch_balance_sheet(symbol: str) -> dict:
    return _request({"function": "BALANCE_SHEET", "symbol": symbol})


def fetch_cash_flow(symbol: str) -> dict:
    return _request({"function": "CASH_FLOW", "symbol": symbol})
