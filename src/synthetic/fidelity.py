"""Fidelity validation: synthetic vs. real reference.

Computes the full comparison battery requested for Phase 2:
  * KS statistics (2-sample, per feature)
  * Wasserstein distances (per feature, raw and standardized by real std)
  * correlation-matrix comparison (MAE + Frobenius)
  * covariance-matrix comparison (on standardized features)
  * PCA projection (real-fit basis; explained variance + overlap metrics; plot)
  * feature distribution summaries (moments/quantiles)
  * target distribution comparison (the six modelling targets)
  * sector distribution comparison
  * quarterly distribution comparison

An acceptance verdict is derived from configurable thresholds so the
generation loop can refine automatically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

TARGETS = ["Sales", "Operating Profit", "Net Profit", "Borrowings", "Total Assets", "CFO"]


class FidelityValidator:
    def __init__(self, real: pd.DataFrame, synth: pd.DataFrame, config: dict[str, Any]):
        self.real = real.reset_index(drop=True)
        self.synth = synth.reset_index(drop=True)
        self.numeric_cols = [c for c in config["schema"]["numeric_columns"]
                             if c in real.columns and c in synth.columns]
        self.acceptance = config["synthetic"]["validation"]["acceptance"]

    # ------------------------------------------------------------------ #
    def evaluate(self, *, figure_path: Path | None = None) -> dict[str, Any]:
        ks = self._ks()
        wass = self._wasserstein()
        corr = self._correlation_comparison()
        cov = self._covariance_comparison()
        pca = self._pca_projection(figure_path)
        feat = self._feature_distributions()
        tgt = self._target_distributions()
        sector = self._categorical_distribution("Sector")
        quarter = self._categorical_distribution("Period")

        summary = {
            "median_ks": float(np.median([v["ks_stat"] for v in ks.values()])),
            "max_ks": float(np.max([v["ks_stat"] for v in ks.values()])),
            "mean_wasserstein_std": float(np.mean([v["wasserstein_std"] for v in wass.values()])),
            "corr_mae": corr["mae"],
            "corr_frobenius": corr["frobenius"],
            "cov_frobenius_standardized": cov["frobenius"],
        }
        verdict = self._acceptance(summary)
        return {
            "n_real": int(len(self.real)),
            "n_synth": int(len(self.synth)),
            "summary": summary,
            "acceptance": verdict,
            "ks_statistics": ks,
            "wasserstein": wass,
            "correlation": corr,
            "covariance": cov,
            "pca": pca,
            "feature_distributions": feat,
            "target_distributions": tgt,
            "sector_distribution": sector,
            "quarterly_distribution": quarter,
        }

    # ------------------------------------------------------------------ #
    def _ks(self) -> dict[str, Any]:
        out = {}
        for c in self.numeric_cols:
            stat, p = ks_2samp(self.real[c].dropna(), self.synth[c].dropna())
            out[c] = {"ks_stat": round(float(stat), 4), "p_value": round(float(p), 4)}
        return out

    def _wasserstein(self) -> dict[str, Any]:
        out = {}
        for c in self.numeric_cols:
            r = self.real[c].dropna().to_numpy(dtype=float)
            s = self.synth[c].dropna().to_numpy(dtype=float)
            w = wasserstein_distance(r, s)
            std = float(np.std(r)) or 1.0
            out[c] = {
                "wasserstein": round(float(w), 4),
                "wasserstein_std": round(float(w / std), 4),
            }
        return out

    def _corr(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.numeric_cols].corr().to_numpy()

    def _correlation_comparison(self) -> dict[str, Any]:
        cr = self._corr(self.real)
        cs = self._corr(self.synth)
        diff = np.abs(cr - cs)
        # Off-diagonal MAE (diagonal is always 1).
        n = cr.shape[0]
        off = ~np.eye(n, dtype=bool)
        return {
            "mae": round(float(diff[off].mean()), 4),
            "max_abs_diff": round(float(diff[off].max()), 4),
            "frobenius": round(float(np.linalg.norm(cr - cs)), 4),
            "features": self.numeric_cols,
        }

    def _covariance_comparison(self) -> dict[str, Any]:
        """Compare covariance of standardized features (scale-free)."""
        rz = _standardize(self.real[self.numeric_cols])
        sz = _standardize(self.synth[self.numeric_cols])
        cov_r = np.cov(rz, rowvar=False)
        cov_s = np.cov(sz, rowvar=False)
        return {
            "frobenius": round(float(np.linalg.norm(cov_r - cov_s)), 4),
            "max_abs_diff": round(float(np.abs(cov_r - cov_s).max()), 4),
        }

    def _pca_projection(self, figure_path: Path | None) -> dict[str, Any]:
        """Fit PCA on standardized real; project both; compare."""
        cols = self.numeric_cols
        mu = self.real[cols].mean().to_numpy()
        sd = self.real[cols].std(ddof=0).replace(0, 1).to_numpy()
        rz = (self.real[cols].to_numpy() - mu) / sd
        sz = (self.synth[cols].to_numpy() - mu) / sd
        # PCA basis from real via SVD.
        u, s, vt = np.linalg.svd(rz - rz.mean(0), full_matrices=False)
        evr = (s ** 2) / np.sum(s ** 2)
        comps = vt[:2]
        real_proj = (rz - rz.mean(0)) @ comps.T
        synth_proj = (sz - rz.mean(0)) @ comps.T
        centroid_dist = float(np.linalg.norm(real_proj.mean(0) - synth_proj.mean(0)))
        result = {
            "explained_variance_ratio_top5": [round(float(x), 4) for x in evr[:5]],
            "pc_centroid_distance": round(centroid_dist, 4),
            "real_pc_std": [round(float(real_proj[:, 0].std()), 3),
                            round(float(real_proj[:, 1].std()), 3)],
            "synth_pc_std": [round(float(synth_proj[:, 0].std()), 3),
                             round(float(synth_proj[:, 1].std()), 3)],
        }
        if figure_path is not None:
            result["figure"] = _plot_pca(real_proj, synth_proj, evr, figure_path)
        return result

    def _feature_distributions(self) -> dict[str, Any]:
        out = {}
        for c in self.numeric_cols:
            r, s = self.real[c].dropna(), self.synth[c].dropna()
            out[c] = {
                "real": _moments(r),
                "synth": _moments(s),
            }
        return out

    def _target_distributions(self) -> dict[str, Any]:
        out = {}
        for c in TARGETS:
            if c not in self.numeric_cols:
                continue
            r, s = self.real[c].dropna(), self.synth[c].dropna()
            stat, p = ks_2samp(r, s)
            std = float(np.std(r)) or 1.0
            out[c] = {
                "ks_stat": round(float(stat), 4),
                "wasserstein_std": round(float(wasserstein_distance(r, s) / std), 4),
                "real_mean": round(float(r.mean()), 2),
                "synth_mean": round(float(s.mean()), 2),
                "real_median": round(float(r.median()), 2),
                "synth_median": round(float(s.median()), 2),
            }
        return out

    def _categorical_distribution(self, col: str) -> dict[str, Any]:
        r = (self.real[col].value_counts(normalize=True) * 100).round(2)
        s = (self.synth[col].value_counts(normalize=True) * 100).round(2)
        keys = sorted(set(r.index) | set(s.index))
        table = {k: {"real_pct": float(r.get(k, 0.0)), "synth_pct": float(s.get(k, 0.0))} for k in keys}
        max_gap = max((abs(v["real_pct"] - v["synth_pct"]) for v in table.values()), default=0.0)
        return {"distribution": table, "max_pct_gap": round(float(max_gap), 3)}

    def _acceptance(self, summary: dict[str, float]) -> dict[str, Any]:
        a = self.acceptance
        checks = {
            "median_ks": summary["median_ks"] <= a["median_ks_max"],
            "max_ks": summary["max_ks"] <= a["max_ks_max"],
            "mean_wasserstein_std": summary["mean_wasserstein_std"] <= a["mean_wasserstein_std_max"],
            "corr_mae": summary["corr_mae"] <= a["corr_mae_max"],
        }
        # Composite score (lower is better) used to pick the best refine iteration.
        score = (
            summary["median_ks"]
            + summary["mean_wasserstein_std"]
            + summary["corr_mae"]
            + 0.2 * summary["max_ks"]
        )
        return {"accepted": all(checks.values()), "checks": checks, "score": round(float(score), 4)}


# ---------------------------------------------------------------------- #
def _standardize(df: pd.DataFrame) -> np.ndarray:
    mu = df.mean().to_numpy()
    sd = df.std(ddof=0).replace(0, 1).to_numpy()
    return (df.to_numpy() - mu) / sd


def _moments(s: pd.Series) -> dict[str, float]:
    return {
        "mean": round(float(s.mean()), 2),
        "std": round(float(s.std()), 2),
        "min": round(float(s.min()), 2),
        "p25": round(float(s.quantile(0.25)), 2),
        "median": round(float(s.median()), 2),
        "p75": round(float(s.quantile(0.75)), 2),
        "max": round(float(s.max()), 2),
        "skew": round(float(s.skew()), 3),
    }


def _plot_pca(real_proj, synth_proj, evr, figure_path: Path) -> str:
    """Scatter of the first two PCs, real vs synthetic. Returns relative path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    # Subsample synthetic for a readable scatter.
    rng = np.random.default_rng(0)
    if len(synth_proj) > 1500:
        idx = rng.choice(len(synth_proj), 1500, replace=False)
        sp = synth_proj[idx]
    else:
        sp = synth_proj
    ax.scatter(sp[:, 0], sp[:, 1], s=8, alpha=0.25, label=f"synthetic (n={len(synth_proj)})", color="#d1495b")
    ax.scatter(real_proj[:, 0], real_proj[:, 1], s=28, alpha=0.85, label=f"real (n={len(real_proj)})",
               color="#1f77b4", edgecolors="white", linewidths=0.4)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
    ax.set_title("PCA projection — real vs synthetic (real-fit basis)")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=110)
    plt.close(fig)
    return figure_path.name
