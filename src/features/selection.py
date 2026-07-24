"""STEP 5 — feature-space analysis and conservative selection.

Analyses the engineered feature space for multicollinearity, variance inflation
(VIF), redundancy, constant features, and highly-correlated pairs.

Policy (per the brief): remove ONLY features that demonstrably reduce model
quality. Since no models are trained in this phase, the only demonstrably
harmful features are:

* **constant / zero-variance** columns (carry no information), and
* **exact-duplicate** columns (pure redundancy — keep one representative).

High correlation and high VIF are **reported but retained**: they typically
reflect genuine accounting relationships, tree models are robust to them, and
linear models handle them via regularization. Nothing is dropped merely for
being correlated.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class FeatureSelector:
    def __init__(self, df: pd.DataFrame, feature_columns: list[str], config: dict[str, Any]):
        self.df = df
        self.features = [c for c in feature_columns if c in df.columns]
        self.cfg = config["feature_selection"]
        # Analyse on trainable rows only (target present), matching model input.
        self.mask = df["has_target"] if "has_target" in df.columns else pd.Series(True, index=df.index)

    def analyse(self) -> dict[str, Any]:
        X = self.df.loc[self.mask, self.features]
        numeric = [c for c in self.features if pd.api.types.is_numeric_dtype(X[c])]

        constants = self._constant(X, numeric)
        duplicates = self._duplicates(X, numeric)
        high_corr = self._high_correlation(X, numeric)
        vif = self._vif(X, numeric)

        # Build the drop list per policy.
        drop: list[str] = []
        if self.cfg.get("drop_constant", True):
            drop += constants
        if self.cfg.get("drop_exact_duplicates", True):
            # keep first of each duplicate group, drop the rest
            for grp in duplicates:
                drop += grp[1:]
        if self.cfg.get("auto_drop_high_correlation", False):
            drop += [p["b"] for p in high_corr]  # only if explicitly enabled
        drop = sorted(set(drop))

        retained = [c for c in self.features if c not in drop]
        return {
            "n_features_in": len(self.features),
            "constant_features": constants,
            "exact_duplicate_groups": duplicates,
            "high_correlation_pairs": high_corr,
            "vif": vif,
            "dropped_features": drop,
            "drop_reasons": self._drop_reasons(constants, duplicates, drop),
            "retained_features": retained,
            "n_features_out": len(retained),
            "policy": {
                "drop_constant": self.cfg.get("drop_constant", True),
                "drop_exact_duplicates": self.cfg.get("drop_exact_duplicates", True),
                "auto_drop_high_correlation": self.cfg.get("auto_drop_high_correlation", False),
                "note": ("High-correlation / high-VIF features are RETAINED — they carry "
                         "accounting meaning; trees are robust and linear models regularize."),
            },
        }

    # ------------------------------------------------------------------ #
    def _constant(self, X: pd.DataFrame, numeric: list[str]) -> list[str]:
        out = []
        for c in numeric:
            col = X[c].dropna()
            if col.nunique() <= 1:
                out.append(c)
        return out

    def _duplicates(self, X: pd.DataFrame, numeric: list[str]) -> list[list[str]]:
        """Group columns that are value-identical (ignoring NaN alignment)."""
        groups: list[list[str]] = []
        seen: set[str] = set()
        cols = numeric
        for i in range(len(cols)):
            if cols[i] in seen:
                continue
            grp = [cols[i]]
            a = X[cols[i]].to_numpy(dtype="float64")
            for j in range(i + 1, len(cols)):
                if cols[j] in seen:
                    continue
                b = X[cols[j]].to_numpy(dtype="float64")
                if np.allclose(np.nan_to_num(a), np.nan_to_num(b), rtol=1e-9, atol=1e-6):
                    grp.append(cols[j])
                    seen.add(cols[j])
            if len(grp) > 1:
                seen.update(grp)
                groups.append(grp)
        return groups

    def _high_correlation(self, X: pd.DataFrame, numeric: list[str]) -> list[dict[str, Any]]:
        thr = float(self.cfg["high_corr_threshold"])
        corr = X[numeric].corr().abs()
        pairs = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = corr.iloc[i, j]
                if pd.notna(r) and r >= thr:
                    pairs.append({"a": cols[i], "b": cols[j], "abs_r": round(float(r), 4)})
        pairs.sort(key=lambda d: d["abs_r"], reverse=True)
        return pairs

    def _vif(self, X: pd.DataFrame, numeric: list[str]) -> dict[str, Any]:
        """VIF via R^2 of each feature regressed on the others (least squares).

        Reported for the top offenders only; computation guards against
        singular designs from perfectly collinear accounting identities.
        """
        data = X[numeric].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) <= len(numeric) + 1:
            return {"note": "Too few complete rows for reliable VIF.", "top": {}}
        vals = data.to_numpy(dtype="float64")
        # Standardize to improve conditioning.
        mu = vals.mean(0)
        sd = vals.std(0)
        sd[sd == 0] = 1.0
        z = (vals - mu) / sd
        n, k = z.shape
        vif = {}
        for i in range(k):
            y = z[:, i]
            others = np.delete(z, i, axis=1)
            design = np.column_stack([np.ones(n), others])
            try:
                beta, *_ = np.linalg.lstsq(design, y, rcond=None)
                resid = y - design @ beta
                ss_res = float(resid @ resid)
                ss_tot = float(((y - y.mean()) ** 2).sum())
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                vif[numeric[i]] = float("inf") if r2 >= 1.0 else round(1.0 / (1.0 - r2), 2)
            except np.linalg.LinAlgError:
                vif[numeric[i]] = float("inf")
        thr = float(self.cfg["vif_report_threshold"])
        severe = {c: (v if v != float("inf") else "inf") for c, v in vif.items()
                  if v == float("inf") or v >= thr}
        return {
            "threshold": thr,
            "n_features_above_threshold": len(severe),
            "severe_features": dict(sorted(
                severe.items(),
                key=lambda kv: (float("inf") if kv[1] == "inf" else kv[1]),
                reverse=True)),
        }

    def _drop_reasons(self, constants, duplicates, drop) -> dict[str, str]:
        reasons = {}
        for c in constants:
            reasons[c] = "constant (zero variance)"
        for grp in duplicates:
            for c in grp[1:]:
                reasons[c] = f"exact duplicate of {grp[0]}"
        return {c: reasons.get(c, "policy") for c in drop}
