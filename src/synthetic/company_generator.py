"""Synthetic company financial observation generator (development only).

Creates new *peer* companies within each sector, spanning the SAME quarters as
the real panel (no invented time periods, so macro joins cleanly). Each
synthetic company is a full quarterly series that:

* is temporally continuous — Sales follow an AR(1) growth path; margins and
  balance-sheet/cash-flow ratios drift smoothly quarter to quarter;
* respects sector scale — base size and ratios are sampled from statistics
  learned from the real data of the same sector;
* satisfies the robust accounting identities exactly:
    - Operating Profit = Sales - Expenses
    - Net Cash Flow    = CFO + CFI + CFF
    - Total Assets      = Equity + Total Liabilities
* is clearly flagged (``source = 'synthetic'``, ``is_synthetic_company = True``).

The generator never mutates real rows; it only produces new ones.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_QUARTER_TO_NUM = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


class SyntheticCompanyGenerator:
    def __init__(self, master: pd.DataFrame, config: dict[str, Any]):
        self.master = master
        self.schema = config["schema"]
        self.cfg = config["synthetic"]["companies"]
        self.seed = config["synthetic"]["random_seed"]
        self.numeric_cols = [c for c in self.schema["numeric_columns"] if c in master.columns]
        # Quarter scaffold (chronological) shared by every synthetic company.
        self.quarter_scaffold = (
            master[["Year", "Quarter", "Period", "quarter_num", "time_index"]]
            .drop_duplicates("time_index")
            .sort_values("time_index")
            .reset_index(drop=True)
        )
        self.sector_stats = self._compute_sector_stats()

    # ------------------------------------------------------------------ #
    def generate(self) -> pd.DataFrame:
        if not self.cfg.get("enabled", True):
            return pd.DataFrame(columns=self.master.columns)

        rng = np.random.default_rng(self.seed)
        n_per_sector = int(self.cfg["n_per_sector"])
        rows: list[pd.DataFrame] = []
        for sector, stats in self.sector_stats.items():
            for i in range(1, n_per_sector + 1):
                rows.append(self._generate_company(sector, i, stats, rng))
        if not rows:
            return pd.DataFrame(columns=self.master.columns)
        out = pd.concat(rows, ignore_index=True)
        return out

    # ------------------------------------------------------------------ #
    def _compute_sector_stats(self) -> dict[str, dict[str, float]]:
        """Learn per-sector scale and ratio distributions from real data."""
        stats: dict[str, dict[str, float]] = {}
        df = self.master
        for sector, g in df.groupby("Sector"):
            sales = g["Sales"].astype(float)
            op_margin = (g["Operating Profit"] / g["Sales"]).replace([np.inf, -np.inf], np.nan)
            net_margin = (g["Net Profit"] / g["Sales"]).replace([np.inf, -np.inf], np.nan)
            assets_to_sales = (g["Total Assets"] / g["Sales"]).replace([np.inf, -np.inf], np.nan)
            equity_ratio = (g["Equity"] / g["Total Assets"]).replace([np.inf, -np.inf], np.nan)
            borrow_ratio = (g["Borrowings"] / g["Total Liabilities"]).replace([np.inf, -np.inf], np.nan)
            cfo_to_op = (g["CFO"] / g["Operating Profit"]).replace([np.inf, -np.inf], np.nan)
            cfi_to_cfo = (g["CFI"] / g["CFO"]).replace([np.inf, -np.inf], np.nan)
            cff_to_cfo = (g["CFF"] / g["CFO"]).replace([np.inf, -np.inf], np.nan)
            stats[sector] = {
                "sales_log_mean": float(np.log(sales).mean()),
                "op_margin": float(np.clip(op_margin.median(), 0.02, 0.6)),
                "net_margin": float(np.clip(net_margin.median(), 0.01, 0.5)),
                "assets_to_sales": float(np.clip(assets_to_sales.median(), 0.5, 20.0)),
                "equity_ratio": float(np.clip(equity_ratio.median(), 0.1, 0.9)),
                "borrow_ratio": float(np.clip(borrow_ratio.median(), 0.05, 0.95)),
                "cfo_to_op": float(np.clip(cfo_to_op.median(), 0.5, 2.0)),
                "cfi_to_cfo": float(np.clip(cfi_to_cfo.median(), -1.5, -0.1)),
                "cff_to_cfo": float(np.clip(cff_to_cfo.median(), -1.5, -0.05)),
            }
        return stats

    def _generate_company(
        self, sector: str, idx: int, stats: dict[str, float], rng
    ) -> pd.DataFrame:
        n = len(self.quarter_scaffold)
        gcfg = self.cfg

        # --- Sales: AR(1) growth path anchored to a sampled sector scale --- #
        base_log = stats["sales_log_mean"] + rng.normal(0, gcfg["scale_log_std"])
        sales0 = float(np.exp(base_log))
        sales = np.empty(n)
        sales[0] = sales0
        g_prev = gcfg["sales_growth_mean"]
        for t in range(1, n):
            g = (
                gcfg["sales_growth_mean"]
                + gcfg["sales_growth_ar_phi"] * (g_prev - gcfg["sales_growth_mean"])
                + rng.normal(0, gcfg["sales_growth_std"])
            )
            sales[t] = max(sales[t - 1] * (1 + g), 1.0)
            g_prev = g

        # --- Smoothly drifting ratios (temporal continuity) --------------- #
        op_margin = self._drift_series(stats["op_margin"], n, rng, lo=0.01, hi=0.7)
        net_margin = np.minimum(
            self._drift_series(stats["net_margin"], n, rng, lo=0.005, hi=0.6), op_margin
        )
        assets_to_sales = self._drift_series(stats["assets_to_sales"], n, rng, lo=0.3, hi=25.0)
        equity_ratio = self._drift_series(stats["equity_ratio"], n, rng, lo=0.05, hi=0.95)
        borrow_ratio = self._drift_series(stats["borrow_ratio"], n, rng, lo=0.02, hi=0.98)
        cfo_to_op = self._drift_series(stats["cfo_to_op"], n, rng, lo=0.3, hi=2.5)
        cfi_to_cfo = self._drift_series(stats["cfi_to_cfo"], n, rng, lo=-1.5, hi=-0.05)
        cff_to_cfo = self._drift_series(stats["cff_to_cfo"], n, rng, lo=-1.5, hi=-0.02)

        # --- Derive financials, enforcing accounting identities ----------- #
        expenses = sales * (1.0 - op_margin)
        operating_profit = sales - expenses               # identity by construction
        net_profit = sales * net_margin
        total_assets = sales * assets_to_sales
        equity = total_assets * equity_ratio
        total_liabilities = total_assets - equity          # A = E + L by construction
        borrowings = total_liabilities * borrow_ratio
        cfo = operating_profit * cfo_to_op
        cfi = cfo * cfi_to_cfo
        cff = cfo * cff_to_cfo
        net_cash_flow = cfo + cfi + cff                    # identity by construction

        df = self.quarter_scaffold.copy()
        df["Company"] = f"SYN {sector} {idx:02d}"
        df["Ticker"] = f"SYN{sector[:3].upper()}{idx:02d}"
        df["Sector"] = sector
        df["source"] = "synthetic"
        df["Sales"] = np.round(sales, 2)
        df["Expenses"] = np.round(expenses, 2)
        df["Operating Profit"] = np.round(operating_profit, 2)
        df["Net Profit"] = np.round(net_profit, 2)
        df["Total Assets"] = np.round(total_assets, 2)
        df["Equity"] = np.round(equity, 2)
        df["Borrowings"] = np.round(borrowings, 2)
        df["Total Liabilities"] = np.round(total_liabilities, 2)
        df["CFO"] = np.round(cfo, 2)
        df["CFI"] = np.round(cfi, 2)
        df["CFF"] = np.round(cff, 2)
        df["Net Cash Flow"] = np.round(net_cash_flow, 2)
        return df

    def _drift_series(self, start: float, n: int, rng, *, lo: float, hi: float) -> np.ndarray:
        """A smoothly drifting positive series via small multiplicative jitter."""
        jitter = self.cfg["ratio_jitter_std"]
        vals = np.empty(n)
        vals[0] = start
        for t in range(1, n):
            vals[t] = np.clip(vals[t - 1] * (1 + rng.normal(0, jitter)), lo, hi)
        return vals
