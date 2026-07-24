"""STEP 1 — per-dataset audit.

Produces a structured, data-driven audit for a single dataset covering every
item requested in the Phase 1 brief:

* observations, companies, sectors, quarters, coverage, time span
* feature names and dtypes
* missing values, duplicate rows, duplicate company-quarter records
* descriptive statistics, outliers (IQR), distribution summaries
* correlation matrix + multicollinearity (VIF)
* accounting-identity consistency
* quarterly continuity + panel balance

The auditor only *observes and reports*. It never mutates or repairs the data;
repair decisions belong to the master-builder and are made explicitly.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .loader import LoadedDataset


class DatasetAuditor:
    """Compute a full audit report for one :class:`LoadedDataset`."""

    def __init__(self, dataset: LoadedDataset, config: dict[str, Any]):
        self.ds = dataset
        self.df = dataset.frame
        self.schema = config["schema"]
        self.accounting = config["accounting_identities"]
        self.numeric_cols = [c for c in self.schema["numeric_columns"] if c in self.df.columns]
        self.panel_key = self.schema["panel_key"]

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def audit(self) -> dict[str, Any]:
        """Run every check and return one nested report dict."""
        return {
            "dataset": self.ds.name,
            "source_file": self.ds.path.name,
            "file_format": self.ds.file_format,
            "raw_columns": self.ds.raw_columns,
            "renamed_columns": self.ds.renamed_columns,
            "shape": {"rows": int(self.df.shape[0]), "columns": int(self.df.shape[1])},
            "overview": self._overview(),
            "features": self._features(),
            "missing_values": self._missing_values(),
            "duplicates": self._duplicates(),
            "descriptive_statistics": self._descriptive_stats(),
            "outliers": self._outliers(),
            "distributions": self._distributions(),
            "correlation_matrix": self._correlation_matrix(),
            "multicollinearity": self._multicollinearity(),
            "accounting_consistency": self._accounting_consistency(),
            "quarterly_continuity": self._quarterly_continuity(),
            "panel_balance": self._panel_balance(),
        }

    # ------------------------------------------------------------------ #
    # Individual checks
    # ------------------------------------------------------------------ #
    def _overview(self) -> dict[str, Any]:
        df = self.df
        periods = df.sort_values("time_index")["Period"].unique().tolist()
        return {
            "total_observations": int(len(df)),
            "n_companies": int(df["Company"].nunique()),
            "companies": sorted(df["Company"].unique().tolist()),
            "n_sectors": int(df["Sector"].nunique()),
            "sectors": sorted(df["Sector"].unique().tolist()),
            "n_distinct_quarters": int(df["Period"].nunique()),
            "quarter_coverage": periods,
            "time_span": {
                "start": periods[0] if periods else None,
                "end": periods[-1] if periods else None,
                "year_min": int(df["Year"].min()),
                "year_max": int(df["Year"].max()),
            },
            "companies_per_sector": (
                df.drop_duplicates("Company").groupby("Sector")["Company"].count().to_dict()
            ),
        }

    def _features(self) -> dict[str, Any]:
        return {
            "identifier_columns": [c for c in self.schema["identifier_columns"] if c in self.df.columns],
            "numeric_columns": self.numeric_cols,
            "engineered_columns": [c for c in ["quarter_num", "time_index"] if c in self.df.columns],
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
        }

    def _missing_values(self) -> dict[str, Any]:
        na = self.df.isna().sum()
        per_col = {col: int(n) for col, n in na.items() if n > 0}
        return {
            "total_missing_cells": int(na.sum()),
            "columns_with_missing": per_col,
            "rows_with_any_missing": int(self.df.isna().any(axis=1).sum()),
        }

    def _duplicates(self) -> dict[str, Any]:
        df = self.df
        full_dup = int(df.duplicated().sum())
        key_dup_mask = df.duplicated(subset=self.panel_key, keep=False)
        dup_subset = df.loc[key_dup_mask, self.panel_key].drop_duplicates().astype(str)
        dup_keys = [" | ".join(row) for row in dup_subset.itertuples(index=False, name=None)]
        return {
            "full_row_duplicates": full_dup,
            "panel_key": self.panel_key,
            "duplicate_company_quarter_records": int(key_dup_mask.sum()),
            "duplicate_company_quarter_keys": dup_keys,
        }

    def _descriptive_stats(self) -> dict[str, Any]:
        desc = self.df[self.numeric_cols].describe().T
        # add skew/kurtosis for distribution characterization
        desc["skew"] = self.df[self.numeric_cols].skew()
        desc["kurtosis"] = self.df[self.numeric_cols].kurtosis()
        return {col: {k: _num(v) for k, v in row.items()} for col, row in desc.iterrows()}

    def _outliers(self) -> dict[str, Any]:
        """IQR-based outlier count per numeric column (1.5 * IQR fences)."""
        result: dict[str, Any] = {}
        for col in self.numeric_cols:
            s = self.df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (s < low) | (s > high)
            result[col] = {
                "count": int(mask.sum()),
                "lower_fence": _num(low),
                "upper_fence": _num(high),
                "min": _num(s.min()),
                "max": _num(s.max()),
            }
        result["_total_outlier_cells"] = int(sum(v["count"] for v in result.values() if isinstance(v, dict)))
        return result

    def _distributions(self) -> dict[str, Any]:
        """Compact distribution summary: quartiles + shape flags per column."""
        out: dict[str, Any] = {}
        for col in self.numeric_cols:
            s = self.df[col].dropna()
            skew = float(s.skew())
            out[col] = {
                "mean": _num(s.mean()),
                "median": _num(s.median()),
                "std": _num(s.std()),
                "min": _num(s.min()),
                "p25": _num(s.quantile(0.25)),
                "p50": _num(s.quantile(0.50)),
                "p75": _num(s.quantile(0.75)),
                "max": _num(s.max()),
                "skew": skew,
                "shape": _skew_label(skew),
                "has_negative": bool((s < 0).any()),
            }
        return out

    def _correlation_matrix(self) -> dict[str, Any]:
        corr = self.df[self.numeric_cols].corr(method="pearson")
        # highlight strongly correlated pairs (|r| >= 0.9), excluding self-pairs
        pairs = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = corr.iloc[i, j]
                if pd.notna(r) and abs(r) >= 0.9:
                    pairs.append({"a": cols[i], "b": cols[j], "r": _num(r)})
        pairs.sort(key=lambda d: abs(d["r"]), reverse=True)
        return {
            "matrix": {c: {c2: _num(corr.loc[c, c2]) for c2 in cols} for c in cols},
            "high_correlation_pairs": pairs,
        }

    def _multicollinearity(self) -> dict[str, Any]:
        """Variance Inflation Factor per numeric feature.

        Computed without statsmodels: VIF_i = 1 / (1 - R_i^2), where R_i^2 is
        from an OLS regression of feature *i* on all other features (with an
        intercept), solved via least squares.
        """
        X = self.df[self.numeric_cols].dropna()
        vif: dict[str, Any] = {}
        if len(X) <= len(self.numeric_cols):
            return {"note": "Too few complete rows to compute VIF reliably.", "vif": {}}

        values = X.values.astype(float)
        n, k = values.shape
        for i in range(k):
            y = values[:, i]
            others = np.delete(values, i, axis=1)
            design = np.column_stack([np.ones(n), others])
            # Least-squares fit; guard singular designs.
            try:
                beta, *_ = np.linalg.lstsq(design, y, rcond=None)
                y_hat = design @ beta
                ss_res = float(np.sum((y - y_hat) ** 2))
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                vif_val = float("inf") if r2 >= 1.0 else 1.0 / (1.0 - r2)
            except np.linalg.LinAlgError:  # pragma: no cover - defensive
                vif_val = float("inf")
            vif[self.numeric_cols[i]] = _num(vif_val)

        severe = {c: v for c, v in vif.items() if v == float("inf") or (isinstance(v, float) and v >= 10)}
        return {
            "vif": vif,
            "interpretation": "VIF > 10 indicates severe multicollinearity; > 5 moderate.",
            "severe_features": severe,
        }

    def _accounting_consistency(self) -> dict[str, Any]:
        """Check each configured accounting identity within a relative tolerance."""
        tol = float(self.accounting["relative_tolerance"])
        results: dict[str, Any] = {}
        for identity in self.accounting["identities"]:
            lhs = self.df[identity["lhs"]].astype(float)
            rhs = pd.Series(0.0, index=self.df.index)
            for col in identity["rhs_add"]:
                rhs = rhs + self.df[col].astype(float)
            for col in identity["rhs_sub"]:
                rhs = rhs - self.df[col].astype(float)

            abs_diff = (lhs - rhs).abs()
            denom = lhs.abs().replace(0, np.nan)
            rel_diff = abs_diff / denom
            holds = rel_diff <= tol
            n_total = int(holds.notna().sum())
            n_holds = int(holds.sum())
            results[identity["name"]] = {
                "formula": identity["formula"],
                "tolerance": tol,
                "rows_checked": n_total,
                "rows_satisfied": n_holds,
                "rows_violated": int(n_total - n_holds),
                "pct_satisfied": round(100.0 * n_holds / n_total, 2) if n_total else None,
                "max_relative_diff": _num(rel_diff.max()),
                "mean_relative_diff": _num(rel_diff.mean()),
            }
        return results

    def _quarterly_continuity(self) -> dict[str, Any]:
        """Verify each company has a gap-free run of consecutive quarters."""
        issues: list[dict[str, Any]] = []
        for company, grp in self.df.groupby("Company"):
            idx = sorted(grp["time_index"].tolist())
            gaps = [
                {"after_time_index": idx[i], "gap_size": idx[i + 1] - idx[i]}
                for i in range(len(idx) - 1)
                if idx[i + 1] - idx[i] != 1
            ]
            dup = len(idx) != len(set(idx))
            if gaps or dup:
                issues.append({"company": company, "gaps": gaps, "duplicate_time_index": dup})
        return {
            "companies_checked": int(self.df["Company"].nunique()),
            "companies_with_issues": len(issues),
            "issues": issues,
            "is_continuous": len(issues) == 0,
        }

    def _panel_balance(self) -> dict[str, Any]:
        """Check whether every company covers the same set of quarters."""
        counts = self.df.groupby("Company")["time_index"].nunique()
        quarter_sets = self.df.groupby("Company")["Period"].apply(lambda s: frozenset(s))
        unique_sets = set(quarter_sets)
        balanced = counts.nunique() == 1 and len(unique_sets) == 1
        return {
            "quarters_per_company": {k: int(v) for k, v in counts.items()},
            "min_quarters": int(counts.min()),
            "max_quarters": int(counts.max()),
            "is_balanced": bool(balanced),
            "expected_rows_if_balanced": int(counts.max() * self.df["Company"].nunique()),
        }


# ---------------------------------------------------------------------- #
# Small helpers
# ---------------------------------------------------------------------- #
def _num(v: Any) -> Any:
    """Convert numpy/pandas scalars to JSON-friendly Python numbers."""
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        f = float(v)
        if np.isnan(f):
            return None
        if np.isinf(f):
            return "inf" if f > 0 else "-inf"
        return round(f, 6)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v


def _skew_label(skew: float) -> str:
    if abs(skew) < 0.5:
        return "approximately symmetric"
    if skew >= 0.5:
        return "right-skewed"
    return "left-skewed"
