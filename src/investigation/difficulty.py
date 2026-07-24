"""Quantitative diagnostics comparing real vs synthetic difficulty.

All statistics are computed within each entity (source + Company), ordered by
time_index, then pooled. The six base targets are analysed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TARGETS = ["Sales", "Operating Profit", "Net Profit", "Borrowings", "Total Assets", "CFO"]
ENTITY = ["source", "Company"]
ORDER = "time_index"


# ---------------------------------------------------------------------- #
# Helpers to build within-entity lagged pairs / deltas
# ---------------------------------------------------------------------- #
def _grouped(df: pd.DataFrame):
    return df.sort_values(ENTITY + [ORDER]).groupby(ENTITY, sort=False)


def _lagged_pairs(df: pd.DataFrame, col: str, lag: int) -> tuple[np.ndarray, np.ndarray]:
    g = _grouped(df)
    cur, lagged = [], []
    for _, s in g:
        v = s[col].to_numpy(dtype=float)
        if len(v) > lag:
            cur.append(v[lag:])
            lagged.append(v[:-lag])
    if not cur:
        return np.array([]), np.array([])
    return np.concatenate(cur), np.concatenate(lagged)


def _deltas(df: pd.DataFrame, col: str) -> dict[str, np.ndarray]:
    g = _grouped(df)
    d, pct, prev = [], [], []
    for _, s in g:
        v = s[col].to_numpy(dtype=float)
        if len(v) < 2:
            continue
        dd = np.diff(v)
        pv = v[:-1]
        d.append(dd)
        prev.append(pv)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct.append(np.where(np.abs(pv) > 1e-9, dd / pv, np.nan))
    if not d:
        return {"delta": np.array([]), "pct": np.array([]), "prev": np.array([])}
    return {
        "delta": np.concatenate(d),
        "pct": np.concatenate(pct),
        "prev": np.concatenate(prev),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------- #
# 1 & 3 & 5 & 6 — autocorrelation / lag-1 variance explained
# ---------------------------------------------------------------------- #
def autocorrelation_table(real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in TARGETS:
        row: dict[str, Any] = {}
        for name, df in [("real", real), ("synth", synth)]:
            entry = {}
            for lag in [1, 2, 4]:
                cur, lagged = _lagged_pairs(df, col, lag)
                r = _corr(cur, lagged)
                entry[f"acf_lag{lag}"] = round(r, 4) if np.isfinite(r) else None
            # lag-1 variance explained = r_lag1^2 (level persistence).
            r1 = entry["acf_lag1"]
            entry["lag1_variance_explained"] = round(r1 ** 2, 4) if r1 is not None else None
            row[name] = entry
        out[col] = row
    return out


# ---------------------------------------------------------------------- #
# 2 — variance and delta distributions
# ---------------------------------------------------------------------- #
def delta_distribution_table(real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, Any]:
    from scipy.stats import ks_2samp

    out: dict[str, Any] = {}
    for col in TARGETS:
        r = _deltas(real, col)
        s = _deltas(synth, col)
        r_pct = r["pct"][np.isfinite(r["pct"])]
        s_pct = s["pct"][np.isfinite(s["pct"])]
        # KS on percentage-growth distribution (scale-free).
        if len(r_pct) > 2 and len(s_pct) > 2:
            ks = float(ks_2samp(r_pct, s_pct).statistic)
        else:
            ks = None
        out[col] = {
            "level_var_ratio_synth_over_real": _safe_ratio(np.var(synth[col]), np.var(real[col])),
            "delta_std_real": _num(np.std(r["delta"])),
            "delta_std_synth": _num(np.std(s["delta"])),
            "delta_std_ratio_synth_over_real": _safe_ratio(np.std(s["delta"]), np.std(r["delta"])),
            "pct_growth_std_real": _num(np.nanstd(r_pct)),
            "pct_growth_std_synth": _num(np.nanstd(s_pct)),
            "pct_growth_std_ratio_synth_over_real": _safe_ratio(np.nanstd(s_pct), np.nanstd(r_pct)),
            "pct_growth_ks_synth_vs_real": round(ks, 4) if ks is not None else None,
        }
    return out


# ---------------------------------------------------------------------- #
# 4 — synthetic diversity / concentration
# ---------------------------------------------------------------------- #
def diversity_analysis(real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, Any]:
    from sklearn.neighbors import NearestNeighbors

    # Entity-level centroids in standardized 6-target space (per dataset).
    def centroids(df):
        cen = df.groupby(ENTITY)[TARGETS].mean()
        return cen

    real_cen = centroids(real)
    synth_cen = centroids(synth)

    # Standardize synth centroids by their own scale for NN distances.
    z = (synth_cen - synth_cen.mean()) / synth_cen.std(ddof=0).replace(0, 1)
    zv = z.to_numpy()
    nn = NearestNeighbors(n_neighbors=2).fit(zv)
    dist, _ = nn.kneighbors(zv)
    nn_dist = dist[:, 1]  # nearest OTHER company

    # Between-company dispersion per sector: synth vs real (are synth too tight?).
    sector_ratio = {}
    real_sector = real.groupby(ENTITY)["Sector"].first()
    synth_sector = synth.groupby(ENTITY)["Sector"].first()
    for sector in sorted(synth_sector.unique()):
        r_idx = real_sector[real_sector == sector].index
        s_idx = synth_sector[synth_sector == sector].index
        if len(r_idx) < 2 or len(s_idx) < 2:
            continue
        # dispersion = mean over targets of std of company means (log scale).
        r_disp = np.mean([np.std(np.log(np.abs(real_cen.loc[r_idx, c]) + 1)) for c in TARGETS])
        s_disp = np.mean([np.std(np.log(np.abs(synth_cen.loc[s_idx, c]) + 1)) for c in TARGETS])
        sector_ratio[sector] = _safe_ratio(s_disp, r_disp)

    return {
        "n_real_entities": int(len(real_cen)),
        "n_synth_entities": int(len(synth_cen)),
        "synth_nn_distance_mean": _num(np.mean(nn_dist)),
        "synth_nn_distance_median": _num(np.median(nn_dist)),
        "synth_nn_distance_min": _num(np.min(nn_dist)),
        "synth_near_duplicate_fraction_lt_0_1": round(float(np.mean(nn_dist < 0.1)), 4),
        "between_company_dispersion_ratio_by_sector": {k: round(v, 3) for k, v in sector_ratio.items()},
        "dispersion_ratio_note": (
            "ratio = synthetic / real between-company dispersion; ~1 means synthetic "
            "companies are as diverse as real ones, <1 means over-concentrated."
        ),
    }


# ---------------------------------------------------------------------- #
# 7 — formulation comparison (levels vs delta vs pct growth)
# ---------------------------------------------------------------------- #
def formulation_comparison(processed: pd.DataFrame, metadata: dict[str, Any],
                           config: dict[str, Any]) -> dict[str, Any]:
    """Fit one model per formulation on a single chronological split.

    Uses the Phase-3 tree_impute preprocessor + ExtraTrees (a strong Phase-4
    baseline). Reports test R^2 for predicting the absolute level, the QoQ
    delta, and the percentage growth of each target — plus the naive R^2 for
    each formulation.
    """
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import r2_score
    from sklearn.pipeline import Pipeline
    from ..features.preprocessor import build_preprocessor

    numeric = metadata["numeric_features"]
    categorical = metadata["categorical_features"]
    feat = categorical + numeric
    base_map = metadata["target_base_map"]
    order = config["features"]["order_by"]
    seed = config["phase4"]["random_seed"]

    df = processed[processed["has_target"]].copy()
    times = np.sort(df[order].unique())
    split = times[-1]  # last quarter is the test fold
    tr = df[df[order] < split]
    te = df[df[order] == split]

    result: dict[str, Any] = {}
    for tcol, base in base_map.items():
        cur_tr, cur_te = tr[base].to_numpy(float), te[base].to_numpy(float)
        nxt_tr, nxt_te = tr[tcol].to_numpy(float), te[tcol].to_numpy(float)

        forms = {
            "levels": (nxt_tr, nxt_te, cur_te),                       # predict x_{t+1}; naive=x_t
            "delta": (nxt_tr - cur_tr, nxt_te - cur_te, np.zeros_like(cur_te)),   # predict Δ; naive=0
            "pct_growth": ((nxt_tr - cur_tr) / np.where(np.abs(cur_tr) > 1e-9, cur_tr, np.nan),
                           (nxt_te - cur_te) / np.where(np.abs(cur_te) > 1e-9, cur_te, np.nan),
                           np.zeros_like(cur_te)),                    # predict g; naive=0
        }
        entry = {}
        for fname, (ytr, yte, naive_pred) in forms.items():
            m = np.isfinite(ytr)
            mt = np.isfinite(yte)
            pre = build_preprocessor("tree_impute", numeric, categorical, config)
            model = Pipeline([("preprocess", pre),
                              ("model", ExtraTreesRegressor(n_estimators=200, n_jobs=-1,
                                                            random_state=seed))])
            model.fit(tr.loc[m, feat], ytr[m])
            pred = model.predict(te.loc[mt, feat])
            model_r2 = float(r2_score(yte[mt], pred)) if mt.sum() > 1 else float("nan")
            naive_r2 = float(r2_score(yte[mt], naive_pred[mt])) if mt.sum() > 1 else float("nan")
            entry[fname] = {
                "model_r2": round(model_r2, 4),
                "naive_r2": round(naive_r2, 4),
                "target_std": _num(np.nanstd(yte[mt])),
                "skill_over_naive": round(model_r2 - naive_r2, 4),
            }
        result[tcol] = entry
    return result


# ---------------------------------------------------------------------- #
def _num(v) -> float | None:
    if v is None:
        return None
    f = float(v)
    if np.isnan(f) or np.isinf(f):
        return None
    return round(f, 4)


def _safe_ratio(a, b) -> float | None:
    a, b = float(a), float(b)
    if b == 0 or np.isnan(a) or np.isnan(b):
        return None
    return round(a / b, 4)
