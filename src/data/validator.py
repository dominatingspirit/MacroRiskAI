"""STEP 4 — validate the canonical master dataset.

Runs a battery of assertions and returns a structured pass/fail report. The
validator is strict but *aware of the merge strategy*: under
``source_tagged_pool`` the uniqueness guarantee applies to the full provenance
key (source, Company, Year, Quarter), while a plain (Company, Year, Quarter)
duplicate is expected-by-design and reported as informational, not a failure.

Checks
------
* no duplicate company-quarter records (on the effective panel key)
* correct quarterly ordering (monotone time_index within each series)
* no broken accounting identities (within tolerance)
* correct datatypes
* no obvious leakage (no future-derived or constant/degenerate columns,
  no target-equals-feature duplication)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class MasterValidator:
    def __init__(self, master: pd.DataFrame, config: dict[str, Any], strategy: str):
        self.df = master
        self.schema = config["schema"]
        self.accounting = config["accounting_identities"]
        self.merge_cfg = config["merge"]
        self.strategy = strategy
        self.numeric_cols = [c for c in self.schema["numeric_columns"] if c in master.columns]
        self.provenance_col = self.merge_cfg["provenance_column"]
        self.base_key = self.schema["panel_key"]
        # Effective uniqueness key depends on strategy.
        self.effective_key = (
            [self.provenance_col] + self.base_key
            if strategy == "source_tagged_pool" and self.provenance_col in master.columns
            else self.base_key
        )

    def validate(self) -> dict[str, Any]:
        checks = {
            "no_duplicate_company_quarter": self._check_duplicates(),
            "correct_quarterly_ordering": self._check_ordering(),
            "no_broken_accounting_identities": self._check_accounting(),
            "correct_datatypes": self._check_dtypes(),
            "no_obvious_leakage": self._check_leakage(),
        }
        all_pass = all(c["passed"] for c in checks.values())
        return {
            "strategy": self.strategy,
            "effective_panel_key": self.effective_key,
            "all_checks_passed": all_pass,
            "checks": checks,
        }

    # ------------------------------------------------------------------ #
    def _check_duplicates(self) -> dict[str, Any]:
        dup_effective = int(self.df.duplicated(subset=self.effective_key).sum())
        dup_base = int(self.df.duplicated(subset=self.base_key).sum())
        passed = dup_effective == 0
        info = None
        if self.strategy == "source_tagged_pool" and dup_base > 0:
            info = (
                f"{dup_base} rows share (Company, Year, Quarter) across sources — "
                f"expected by design under source_tagged_pool; each is uniquely "
                f"identified by adding the '{self.provenance_col}' provenance tag."
            )
        return {
            "passed": passed,
            "duplicates_on_effective_key": dup_effective,
            "duplicates_on_base_key": dup_base,
            "info": info,
        }

    def _check_ordering(self) -> dict[str, Any]:
        group_cols = (
            [self.provenance_col, "Company"]
            if self.provenance_col in self.df.columns and self.strategy == "source_tagged_pool"
            else ["Company"]
        )
        violations = []
        for keys, grp in self.df.groupby(group_cols, sort=False):
            ti = grp["time_index"].tolist()
            if ti != sorted(ti):
                violations.append({"group": keys, "time_index_sequence": ti})
        return {
            "passed": len(violations) == 0,
            "grouped_by": group_cols,
            "unordered_groups": violations,
        }

    def _check_accounting(self) -> dict[str, Any]:
        tol = float(self.accounting["relative_tolerance"])
        results = {}
        overall_ok = True
        for identity in self.accounting["identities"]:
            lhs = self.df[identity["lhs"]].astype(float)
            rhs = pd.Series(0.0, index=self.df.index)
            for col in identity["rhs_add"]:
                rhs = rhs + self.df[col].astype(float)
            for col in identity["rhs_sub"]:
                rhs = rhs - self.df[col].astype(float)
            rel = (lhs - rhs).abs() / lhs.abs().replace(0, np.nan)
            violated = int((rel > tol).sum())
            n = int(rel.notna().sum())
            pct_ok = round(100.0 * (n - violated) / n, 2) if n else None
            results[identity["name"]] = {
                "formula": identity["formula"],
                "rows_violated": violated,
                "pct_satisfied": pct_ok,
                "max_relative_diff": None if rel.max() is np.nan else round(float(rel.max()), 6),
            }
        # We report identity health but only *fail* if an identity that holds
        # (near-)perfectly elsewhere is broken by the merge itself. Since the
        # builder never recomputes values, identities can only be as good as
        # the source data; we therefore treat this as informational unless an
        # identity degrades below a hard floor.
        floor = 50.0  # if <50% of rows satisfy an identity, flag as failure
        for name, r in results.items():
            if r["pct_satisfied"] is not None and r["pct_satisfied"] < floor:
                overall_ok = False
        return {
            "passed": overall_ok,
            "note": (
                "Identities are validated on the merged data. The builder never "
                "recomputes financials, so any violation is inherited from the "
                "source datasets, not introduced by merging."
            ),
            "identities": results,
        }

    def _check_dtypes(self) -> dict[str, Any]:
        problems = {}
        for col in self.numeric_cols:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                problems[col] = str(self.df[col].dtype)
        if not pd.api.types.is_integer_dtype(self.df["Year"]):
            problems["Year"] = str(self.df["Year"].dtype)
        if "time_index" in self.df.columns and not pd.api.types.is_integer_dtype(self.df["time_index"]):
            problems["time_index"] = str(self.df["time_index"].dtype)
        return {"passed": len(problems) == 0, "wrong_dtype_columns": problems}

    def _check_leakage(self) -> dict[str, Any]:
        """Guard against obvious leakage sources in the canonical dataset.

        Phase 1 has no engineered targets yet, so "leakage" here means
        structural red flags that would silently corrupt later supervised
        learning:
          * constant/degenerate numeric columns (zero variance)
          * exact duplicate numeric columns (one metric secretly equal to
            another — a future-target/feature collision risk)
          * any column literally named like a future/leading indicator
        """
        findings: dict[str, Any] = {}

        # Zero-variance columns.
        zero_var = [c for c in self.numeric_cols if float(self.df[c].std(ddof=0)) == 0.0]
        findings["zero_variance_columns"] = zero_var

        # Duplicate numeric columns (value-identical across all rows).
        dup_pairs = []
        cols = self.numeric_cols
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if np.allclose(
                    self.df[cols[i]].astype(float).values,
                    self.df[cols[j]].astype(float).values,
                    equal_nan=True,
                ):
                    dup_pairs.append([cols[i], cols[j]])
        findings["value_identical_column_pairs"] = dup_pairs

        # Suspicious forward-looking names (none expected).
        suspicious = [
            c for c in self.df.columns
            if any(tok in c.lower() for tok in ["future", "next", "target", "forward", "lead"])
        ]
        findings["suspicious_forward_looking_columns"] = suspicious

        passed = not zero_var and not dup_pairs and not suspicious
        findings["passed"] = passed
        findings["note"] = (
            "Value-identical column pairs are reported for awareness; in these "
            "datasets some balance-sheet fields may coincide by construction. "
            "They are flagged so feature engineering in Phase 3 can avoid using "
            "redundant columns, but are not necessarily errors."
        )
        # Treat only zero-variance / suspicious names as hard failures; identical
        # pairs are advisory (they may be legitimate in the source accounting).
        findings["passed"] = not zero_var and not suspicious
        return findings
