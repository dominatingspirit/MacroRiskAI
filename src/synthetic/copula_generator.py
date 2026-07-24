"""Sector-conditional Gaussian-copula financial generator (development only).

Why this method
---------------
The legacy generator anchored every synthetic company to its sector's *median*
ratios and perturbed each ratio independently. That collapses marginal spread
and destroys the joint dependence between features, so the synthetic
correlation/covariance structure does not match the real data.

This generator instead models the **joint distribution of the independent
economic drivers** with a Gaussian copula fit per sector:

* Empirical marginals are preserved exactly (inverse-CDF mapping), so each
  driver's distribution matches the real one — including multi-modal shapes
  that arise from pooling the two source datasets.
* A shrinkage-regularized Gaussian copula preserves the rank-dependence
  (hence correlation/covariance) among drivers, which — after reconstruction —
  reproduces the correlation/covariance among the financial line items.
* Every company is a fresh copula draw evolved over the 8 quarters with
  calibrated AR(1) / mean-reverting dynamics, giving realistic quarterly
  behaviour without duplicating or interpolating real rows.
* All financials are reconstructed from the drivers via the accounting
  identities, so the identities hold exactly by construction.

Drivers (the true degrees of freedom)
-------------------------------------
    log_sales, op_margin, net_to_op, assets_to_sales, equity_ratio,
    borrow_to_liab, cfo_to_op, cfi_to_cfo, cff_to_cfo

Reconstruction (identities enforced)
------------------------------------
    Sales             = exp(log_sales)
    Operating Profit  = Sales * op_margin
    Expenses          = Sales - Operating Profit          # OP = Sales - Exp
    Net Profit        = net_to_op * Operating Profit
    Total Assets      = assets_to_sales * Sales
    Equity            = equity_ratio * Total Assets
    Total Liabilities = Total Assets - Equity              # A = E + L
    Borrowings        = borrow_to_liab * Total Liabilities
    CFO               = cfo_to_op * Operating Profit
    CFI               = cfi_to_cfo * CFO
    CFF               = cff_to_cfo * CFO
    Net Cash Flow     = CFO + CFI + CFF                    # NCF = CFO+CFI+CFF
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

# Order matters for reconstruction & correlation bookkeeping.
DRIVERS = [
    "log_sales",
    "op_margin",
    "net_to_op",
    "assets_to_sales",
    "equity_ratio",
    "borrow_to_liab",
    "cfo_to_op",
    "cfi_to_cfo",
    "cff_to_cfo",
]


def compute_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the driver variables from raw financial columns."""
    eps = 1e-9
    sales = df["Sales"].astype(float).clip(lower=eps)
    op = df["Operating Profit"].astype(float)
    ta = df["Total Assets"].astype(float).clip(lower=eps)
    tl = df["Total Liabilities"].astype(float)
    cfo = df["CFO"].astype(float)
    out = pd.DataFrame({
        "log_sales": np.log(sales),
        "op_margin": op / sales,
        "net_to_op": df["Net Profit"].astype(float) / op.replace(0, np.nan),
        "assets_to_sales": ta / sales,
        "equity_ratio": df["Equity"].astype(float) / ta,
        "borrow_to_liab": df["Borrowings"].astype(float) / tl.replace(0, np.nan),
        "cfo_to_op": cfo / op.replace(0, np.nan),
        "cfi_to_cfo": df["CFI"].astype(float) / cfo.replace(0, np.nan),
        "cff_to_cfo": df["CFF"].astype(float) / cfo.replace(0, np.nan),
    })
    return out.replace([np.inf, -np.inf], np.nan)


@dataclass
class SectorModel:
    """Fitted copula + temporal calibration for one sector."""
    sector: str
    marginals: dict[str, np.ndarray]          # sorted empirical values per driver
    active_drivers: list[str]                 # non-constant drivers (in copula)
    constant_values: dict[str, float]         # constant drivers -> value
    corr: np.ndarray                          # copula correlation (active drivers)
    chol: np.ndarray                          # Cholesky factor of corr
    growth_mu: float                          # real QoQ log-sales growth mean
    growth_sigma: float                       # real QoQ log-sales growth std
    ratio_sigma: float                        # typical real QoQ ratio pct-change std


class CopulaFinancialGenerator:
    def __init__(self, real_reference: pd.DataFrame, config: dict[str, Any]):
        self.real = real_reference.copy()
        self.schema = config["schema"]
        self.scfg = config["synthetic"]
        self.copula_cfg = self.scfg["copula"]
        self.temporal_cfg = self.scfg["temporal"]
        self.seed = self.scfg["random_seed"]
        self.numeric_cols = self.schema["numeric_columns"]
        # Quarter scaffold shared by all synthetic companies (real quarters only).
        self.quarter_scaffold = (
            self.real[["Year", "Quarter", "Period", "quarter_num", "time_index"]]
            .drop_duplicates("time_index").sort_values("time_index").reset_index(drop=True)
        )
        self.n_quarters = len(self.quarter_scaffold)
        self.sector_models: dict[str, SectorModel] = {}

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(self) -> "CopulaFinancialGenerator":
        drivers_all = compute_drivers(self.real)
        drivers_all["Sector"] = self.real["Sector"].values
        for sector, g in drivers_all.groupby("Sector"):
            self.sector_models[sector] = self._fit_sector(sector, g.drop(columns="Sector"))
        return self

    def _fit_sector(self, sector: str, d: pd.DataFrame) -> SectorModel:
        min_std = float(self.copula_cfg["min_driver_std"])
        marginals: dict[str, np.ndarray] = {}
        active: list[str] = []
        constants: dict[str, float] = {}
        for col in DRIVERS:
            vals = d[col].dropna().to_numpy()
            if len(vals) == 0:
                vals = np.array([0.0])
            marginals[col] = np.sort(vals)
            if np.std(vals) <= min_std or len(np.unique(vals)) < 3:
                constants[col] = float(np.median(vals))
            else:
                active.append(col)

        # Gaussian copula correlation on normal scores of the active drivers.
        corr = self._copula_correlation(d[active]) if active else np.zeros((0, 0))
        chol = np.linalg.cholesky(corr) if len(active) else np.zeros((0, 0))

        growth_mu, growth_sigma, ratio_sigma = self._temporal_calibration(sector)
        return SectorModel(
            sector=sector, marginals=marginals, active_drivers=active,
            constant_values=constants, corr=corr, chol=chol,
            growth_mu=growth_mu, growth_sigma=growth_sigma, ratio_sigma=ratio_sigma,
        )

    def _copula_correlation(self, d: pd.DataFrame) -> np.ndarray:
        """Correlation of Gaussian-copula normal scores, shrunk toward I."""
        # Rank -> uniform -> normal score for each column (van der Waerden).
        z = np.empty((len(d), d.shape[1]))
        for j, col in enumerate(d.columns):
            v = d[col].to_numpy(dtype=float)
            # Impute occasional NaNs with the column median before ranking.
            if np.isnan(v).any():
                v = np.where(np.isnan(v), np.nanmedian(v), v)
            ranks = pd.Series(v).rank(method="average").to_numpy()
            u = (ranks - 0.5) / len(v)
            z[:, j] = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        corr = np.corrcoef(z, rowvar=False)
        corr = np.atleast_2d(corr)
        # Shrinkage toward identity for conditioning with few rows.
        lam = float(self.copula_cfg["shrinkage"])
        corr = (1 - lam) * corr + lam * np.eye(corr.shape[0])
        # Guarantee positive definiteness.
        return self._nearest_pd(corr)

    @staticmethod
    def _nearest_pd(a: np.ndarray) -> np.ndarray:
        a = (a + a.T) / 2
        vals, vecs = np.linalg.eigh(a)
        vals = np.clip(vals, 1e-6, None)
        pd_mat = (vecs * vals) @ vecs.T
        # Renormalize to unit diagonal (a correlation matrix).
        d = np.sqrt(np.diag(pd_mat))
        return pd_mat / np.outer(d, d)

    def _temporal_calibration(self, sector: str) -> tuple[float, float, float]:
        """Estimate real QoQ dynamics for this sector from its company series."""
        sub = self.real[self.real["Sector"] == sector]
        growths: list[float] = []
        ratio_pcts: list[float] = []
        # Each (source, Company) is one contiguous quarterly series.
        group_cols = [c for c in ["source", "Company"] if c in sub.columns]
        for _, s in sub.sort_values("time_index").groupby(group_cols):
            sales = s["Sales"].to_numpy(dtype=float)
            if len(sales) >= 2:
                growths.extend(np.diff(np.log(np.clip(sales, 1e-9, None))).tolist())
            opm = (s["Operating Profit"] / s["Sales"]).to_numpy(dtype=float)
            if len(opm) >= 2:
                pct = np.diff(opm) / np.clip(opm[:-1], 1e-9, None)
                ratio_pcts.extend(pct[np.isfinite(pct)].tolist())
        growth_mu = float(np.mean(growths)) if growths else 0.01
        growth_sigma = float(np.std(growths)) if growths else 0.03
        ratio_sigma = float(np.std(ratio_pcts)) if ratio_pcts else 0.03
        # Keep calibrations in a sane range.
        return (
            float(np.clip(growth_mu, -0.05, 0.08)),
            float(np.clip(growth_sigma, 0.005, 0.15)),
            float(np.clip(ratio_sigma, 0.005, 0.12)),
        )

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def companies_per_sector(self) -> int:
        target = int(self.scfg["target_total_rows"])
        n_sectors = max(1, len(self.sector_models))
        per_sector = int(np.ceil(target / (n_sectors * self.n_quarters)))
        return max(per_sector, 1)

    def generate(self, *, noise_scale: float) -> pd.DataFrame:
        """Generate the full synthetic company-quarter panel."""
        if not self.sector_models:
            raise RuntimeError("Call fit() before generate().")
        rng = np.random.default_rng(self.seed)
        per_sector = self.companies_per_sector()
        frames: list[pd.DataFrame] = []
        for sector, model in self.sector_models.items():
            for i in range(1, per_sector + 1):
                frames.append(self._generate_company(sector, i, model, rng, noise_scale))
        out = pd.concat(frames, ignore_index=True)
        out["is_synthetic_company"] = True
        out["source"] = "synthetic"
        return out

    def _sample_anchor(self, model: SectorModel, rng) -> dict[str, float]:
        """One Gaussian-copula draw -> driver values via inverse empirical CDF."""
        anchor: dict[str, float] = dict(model.constant_values)
        k = len(model.active_drivers)
        if k:
            z = model.chol @ rng.standard_normal(k)
            u = norm.cdf(z)
            for j, col in enumerate(model.active_drivers):
                anchor[col] = float(_empirical_quantile(model.marginals[col], u[j]))
        return anchor

    def _generate_company(
        self, sector: str, idx: int, model: SectorModel, rng, noise_scale: float
    ) -> pd.DataFrame:
        n = self.n_quarters
        anchor = self._sample_anchor(model, rng)
        phi = self.temporal_cfg["sales_growth_ar_phi"]
        revert = self.temporal_cfg["mean_revert"]
        g_sigma = model.growth_sigma * noise_scale
        r_sigma = model.ratio_sigma * noise_scale

        # --- Sales: AR(1) growth around the anchor log-level -------------- #
        log_sales = np.empty(n)
        log_sales[0] = anchor["log_sales"]
        g_prev = model.growth_mu
        anchor_log = anchor["log_sales"]
        for t in range(1, n):
            g = model.growth_mu + phi * (g_prev - model.growth_mu) + rng.normal(0, g_sigma)
            # Gentle pull back toward the anchor level to preserve the marginal.
            level_pull = revert * (anchor_log - log_sales[t - 1])
            log_sales[t] = log_sales[t - 1] + g + level_pull
            g_prev = g
        sales = np.exp(log_sales)

        # --- Ratios: mean-reverting around their anchor value ------------- #
        ratio_paths = {}
        for col in DRIVERS:
            if col == "log_sales":
                continue
            base = anchor[col]
            path = np.empty(n)
            path[0] = base
            for t in range(1, n):
                shock = rng.normal(0, r_sigma) * abs(base)
                path[t] = path[t - 1] + revert * (base - path[t - 1]) + shock
            ratio_paths[col] = path

        # --- Reconstruct financials via identities ------------------------ #
        op_margin = ratio_paths["op_margin"]
        operating_profit = sales * op_margin
        expenses = sales - operating_profit
        net_profit = ratio_paths["net_to_op"] * operating_profit
        total_assets = ratio_paths["assets_to_sales"] * sales
        equity = ratio_paths["equity_ratio"] * total_assets
        total_liabilities = total_assets - equity
        borrowings = ratio_paths["borrow_to_liab"] * total_liabilities
        cfo = ratio_paths["cfo_to_op"] * operating_profit
        cfi = ratio_paths["cfi_to_cfo"] * cfo
        cff = ratio_paths["cff_to_cfo"] * cfo
        net_cash_flow = cfo + cfi + cff

        df = self.quarter_scaffold.copy()
        df["Company"] = f"SYN {sector} {idx:03d}"
        df["Ticker"] = f"SYN{sector[:3].upper()}{idx:03d}"
        df["Sector"] = sector
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


def _empirical_quantile(sorted_vals: np.ndarray, u: float) -> float:
    """Inverse empirical CDF via linear interpolation on sorted values."""
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    positions = (np.arange(len(sorted_vals)) + 0.5) / len(sorted_vals)
    return float(np.interp(u, positions, sorted_vals))
