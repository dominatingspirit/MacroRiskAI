"""Automatic leakage diagnostics for suspiciously high performance.

Triggered when a model's mean validation R^2 exceeds ``r2_trigger``. High R^2 is
NOT automatically leakage: financial *levels* are highly persistent, so a naive
"previous quarter" forecast is itself very strong. These diagnostics separate
"the task is genuinely easy/persistent" from "a feature leaks the future":

1. **Baseline comparison** — if the naive baseline also achieves high R^2, the
   performance reflects series persistence, not feature leakage.
2. **Feature–target correlation scan** — flag any predictor whose correlation
   with the target is ~1.0 (a candidate leaked future value). Note that a lag of
   the target being highly correlated is expected persistence, not leakage.
3. **Target-permutation test** — refit on shuffled targets; R^2 must collapse to
   ~0. If it stays high, the pipeline is exploiting structure it should not.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score


def run_leakage_diagnostics(
    *,
    target_col: str,
    base_var: str,
    pipe,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    naive_r2: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    dcfg = config["leakage_diagnostics"]
    seed = config["phase4"]["random_seed"]
    rng = np.random.default_rng(seed)

    # 2) Feature-target correlation scan on training data.
    numeric = X_train.select_dtypes(include=[np.number])
    corrs = {}
    for c in numeric.columns:
        col = numeric[c].to_numpy(dtype=float)
        if np.nanstd(col) == 0:
            continue
        with np.errstate(invalid="ignore"):
            r = np.corrcoef(np.nan_to_num(col, nan=np.nanmedian(col)), y_train)[0, 1]
        if np.isfinite(r):
            corrs[c] = float(r)
    flag = float(dcfg["feature_target_corr_flag"])
    suspicious = {c: round(r, 5) for c, r in corrs.items() if abs(r) >= flag}
    top_corr = dict(sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8])
    top_corr = {k: round(v, 4) for k, v in top_corr.items()}

    # 3) Target-permutation test: shuffle y, refit, expect R^2 ~ 0.
    y_perm = y_train.copy()
    rng.shuffle(y_perm)
    perm_pipe = clone(pipe)
    perm_pipe.fit(X_train, y_perm)
    perm_pred = perm_pipe.predict(X_val)
    perm_r2 = float(r2_score(y_val, perm_pred))
    perm_tol = float(dcfg["permutation_r2_tolerance"])
    permutation_ok = perm_r2 <= perm_tol

    # Any suspicious feature that is NOT simply a lag/rolling of the target base
    # variable is a genuine concern; lags of the target are legitimate persistence.
    base_token = base_var.replace(" ", "_").lower()
    genuine_concern = {
        c: r for c, r in suspicious.items()
        if base_var.lower() not in c.lower() and base_token not in c.lower()
    }

    verdict = _verdict(naive_r2, permutation_ok, genuine_concern, config)
    return {
        "triggered": True,
        "naive_baseline_r2": round(float(naive_r2), 4),
        "baseline_also_high": naive_r2 >= dcfg["r2_trigger"] - 0.05,
        "permutation_r2": round(perm_r2, 4),
        "permutation_test_passed": permutation_ok,
        "suspicious_features_corr_ge_flag": suspicious,
        "genuine_concern_features": {k: round(v, 5) for k, v in genuine_concern.items()},
        "top_feature_target_correlations": top_corr,
        "verdict": verdict,
    }


def _verdict(naive_r2, permutation_ok, genuine_concern, config) -> dict[str, Any]:
    dcfg = config["leakage_diagnostics"]
    baseline_high = naive_r2 >= dcfg["r2_trigger"] - 0.05
    if not permutation_ok:
        return {"leakage_suspected": True,
                "reason": "Model retains high R^2 on permuted targets — structural leakage."}
    if genuine_concern:
        return {"leakage_suspected": True,
                "reason": f"Non-target features near-perfectly correlated with target: "
                          f"{list(genuine_concern)}."}
    if baseline_high:
        return {"leakage_suspected": False,
                "reason": "High R^2 explained by series persistence — the naive baseline is "
                          "also high and the permutation test passed. Accept."}
    return {"leakage_suspected": False,
            "reason": "Permutation test passed and no non-target feature is degenerate; "
                      "high R^2 appears legitimate. Accept."}
