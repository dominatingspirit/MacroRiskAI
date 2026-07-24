"""STEP 1 — data quality inspection of the Phase 2 training dataset.

Reports (never repairs): missing values, duplicate observations, invalid
numeric values (e.g. negatives where impossible), infinities, inconsistent
categorical values, and datatype consistency.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class DataQualityInspector:
    def __init__(self, df: pd.DataFrame, config: dict[str, Any]):
        self.df = df
        self.schema = config["schema"]
        self.feat = config["features"]
        self.numeric_cols = [c for c in self.schema["numeric_columns"] if c in df.columns]
        self.macro_cols = [c for c in self.feat["macro_vars"] if c in df.columns]
        self.panel_key = self.feat["entity_group"] + [config["features"]["order_by"]]

    def inspect(self) -> dict[str, Any]:
        return {
            "shape": {"rows": int(self.df.shape[0]), "columns": int(self.df.shape[1])},
            "dtypes": {c: str(t) for c, t in self.df.dtypes.items()},
            "missing_values": self._missing(),
            "duplicates": self._duplicates(),
            "invalid_numeric": self._invalid_numeric(),
            "infinite_values": self._infinite(),
            "categorical_consistency": self._categorical(),
            "datatype_consistency": self._datatypes(),
        }

    def _missing(self) -> dict[str, Any]:
        na = self.df.isna().sum()
        return {
            "total_missing_cells": int(na.sum()),
            "columns_with_missing": {c: int(n) for c, n in na.items() if n > 0},
        }

    def _duplicates(self) -> dict[str, Any]:
        full = int(self.df.duplicated().sum())
        key_cols = [c for c in self.panel_key if c in self.df.columns]
        key_dupes = int(self.df.duplicated(subset=key_cols).sum()) if key_cols else None
        return {
            "full_row_duplicates": full,
            "panel_key": key_cols,
            "duplicate_entity_quarter_records": key_dupes,
        }

    def _invalid_numeric(self) -> dict[str, Any]:
        """Flag economically impossible values (non-repairing)."""
        # Sales, Expenses, Total Assets, Equity should be non-negative.
        nonneg = [c for c in ["Sales", "Expenses", "Operating Profit", "Total Assets",
                              "Equity", "Total Liabilities", "Borrowings"] if c in self.df.columns]
        negatives = {c: int((self.df[c] < 0).sum()) for c in nonneg if (self.df[c] < 0).any()}
        # NaN-producing coercions already surfaced by _missing; here flag non-finite.
        return {
            "columns_checked_for_negativity": nonneg,
            "unexpected_negatives": negatives,
        }

    def _infinite(self) -> dict[str, Any]:
        inf_counts = {}
        for c in self.numeric_cols + self.macro_cols:
            col = self.df[c]
            n = int(np.isinf(col.to_numpy(dtype="float64", na_value=np.nan)).sum())
            if n:
                inf_counts[c] = n
        return {"columns_with_infinities": inf_counts, "total": int(sum(inf_counts.values()))}

    def _categorical(self) -> dict[str, Any]:
        cats = {}
        for c in ["Sector", "Quarter", "source", "macro_source"]:
            if c in self.df.columns:
                cats[c] = sorted(map(str, self.df[c].unique().tolist()))
        # Sanity: sector count, quarter labels.
        issues = []
        if "Quarter" in self.df.columns:
            bad = set(self.df["Quarter"].unique()) - {"Q1", "Q2", "Q3", "Q4"}
            if bad:
                issues.append(f"Unexpected Quarter labels: {sorted(bad)}")
        return {"unique_values": cats, "issues": issues}

    def _datatypes(self) -> dict[str, Any]:
        problems = {}
        for c in self.numeric_cols + self.macro_cols:
            if not pd.api.types.is_numeric_dtype(self.df[c]):
                problems[c] = str(self.df[c].dtype)
        for c in ["Year", "time_index", "quarter_num"]:
            if c in self.df.columns and not pd.api.types.is_integer_dtype(self.df[c]):
                problems[c] = str(self.df[c].dtype)
        return {"wrong_dtype_columns": problems, "all_consistent": len(problems) == 0}
