"""Validation for the Phase 2 training dataset.

Confirms the augmentation and macro merge did not corrupt the data:

* accounting identities still hold on the synthetic company rows,
* the macro block is complete (every company-quarter has macro, no nulls),
* the synthetic macro is economically coherent
  (Reverse_Repo < Repo, gradual Repo moves, bounded oil/FX quarterly moves,
   positive CPI-index vs WPI relationship),
* required provenance flags are present.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class Phase2Validator:
    def __init__(self, training: pd.DataFrame, macro_panel: pd.DataFrame, config: dict[str, Any]):
        self.df = training
        self.macro = macro_panel.sort_values("Year").reset_index(drop=True)
        # ensure chronological macro order via time_index if available
        self.config = config
        self.schema = config["schema"]
        self.accounting = config["accounting_identities"]
        self.macro_cfg = config["macro"]
        self.variables = self.macro_cfg["variables"]

    def validate(self) -> dict[str, Any]:
        checks = {
            "synthetic_accounting_identities": self._synthetic_identities(),
            "macro_join_complete": self._macro_join_complete(),
            "macro_coherence": self._macro_coherence(),
            "provenance_flags_present": self._flags_present(),
        }
        all_pass = all(c["passed"] for c in checks.values())
        return {"all_checks_passed": all_pass, "checks": checks}

    # ------------------------------------------------------------------ #
    def _synthetic_identities(self) -> dict[str, Any]:
        synth = self.df[self.df["is_synthetic_company"]]
        tol = float(self.accounting["relative_tolerance"])
        results = {}
        overall = True
        if synth.empty:
            return {"passed": True, "note": "No synthetic rows to check.", "identities": {}}
        for identity in self.accounting["identities"]:
            lhs = synth[identity["lhs"]].astype(float)
            rhs = pd.Series(0.0, index=synth.index)
            for c in identity["rhs_add"]:
                rhs = rhs + synth[c].astype(float)
            for c in identity["rhs_sub"]:
                rhs = rhs - synth[c].astype(float)
            rel = (lhs - rhs).abs() / lhs.abs().replace(0, np.nan)
            violated = int((rel > tol).sum())
            results[identity["name"]] = {
                "formula": identity["formula"],
                "rows_violated": violated,
                "max_relative_diff": round(float(rel.max()), 8) if rel.notna().any() else None,
            }
            if violated > 0:
                overall = False
        return {"passed": overall, "identities": results}

    def _macro_join_complete(self) -> dict[str, Any]:
        null_counts = {v: int(self.df[v].isna().sum()) for v in self.variables}
        total_nulls = sum(null_counts.values())
        return {
            "passed": total_nulls == 0,
            "macro_null_cells": total_nulls,
            "null_by_variable": {k: v for k, v in null_counts.items() if v > 0},
        }

    def _macro_coherence(self) -> dict[str, Any]:
        m = self.macro.copy()
        # Re-derive chronological order from Year+Quarter to be safe.
        q_num = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
        m["_ti"] = m["Year"] * 4 + m["Quarter"].map(q_num)
        m = m.sort_values("_ti").reset_index(drop=True)

        findings: dict[str, Any] = {}

        # Reverse repo strictly below repo.
        rev_ok = bool((m["Reverse_Repo_Rate"] < m["Repo_Rate"]).all())
        findings["reverse_repo_below_repo"] = rev_ok

        # Repo moves gradually (no jump > 0.5 between quarters).
        repo_jump = float(m["Repo_Rate"].diff().abs().max())
        findings["max_repo_quarterly_move"] = round(repo_jump, 3)
        repo_ok = repo_jump <= 0.5

        # Oil and FX bounded quarter-over-quarter.
        oil_pct = float((m["oil_price"].pct_change().abs().max()))
        fx_pct = float((m["exchange_rate"].pct_change().abs().max()))
        findings["max_oil_quarterly_pct_move"] = round(oil_pct, 4)
        findings["max_fx_quarterly_pct_move"] = round(fx_pct, 4)
        oil_ok = oil_pct <= 0.20
        fx_ok = fx_pct <= 0.10

        # Positive CPI-index vs WPI relationship (both trend with prices).
        if len(m) >= 3:
            corr = float(np.corrcoef(m["CPI_Combined_Index"], m["WPI"])[0, 1])
        else:
            corr = float("nan")
        findings["cpi_index_wpi_correlation"] = None if np.isnan(corr) else round(corr, 4)
        cpi_wpi_ok = np.isnan(corr) or corr > 0

        # Ranges sane.
        ranges_ok = bool(
            (m["Repo_Rate"].between(3, 10).all())
            and (m["oil_price"].between(40, 130).all())
            and (m["exchange_rate"].between(70, 95).all())
            and (m["CPI_Inflation_Rate"].between(0, 15).all())
        )
        findings["ranges_sane"] = ranges_ok

        findings["passed"] = bool(rev_ok and repo_ok and oil_ok and fx_ok and cpi_wpi_ok and ranges_ok)
        return findings

    def _flags_present(self) -> dict[str, Any]:
        needed = [
            "is_synthetic_company",
            self.macro_cfg["provenance"]["source_column"],
            self.macro_cfg["provenance"]["synthetic_flag_column"],
        ]
        missing = [c for c in needed if c not in self.df.columns]
        return {"passed": len(missing) == 0, "missing_flags": missing}
