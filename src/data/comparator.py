"""STEP 2 — compare the two datasets and recommend a merge strategy.

The comparison is entirely data-driven. It determines the relationship
between the datasets (complementary / overlapping / partially overlapping) by
examining their panel keys and, where keys coincide, comparing the numeric
values. It then recommends the *safest* merge strategy given what it finds —
it never blindly assumes the datasets can be stacked.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .loader import LoadedDataset


class DatasetComparator:
    """Compare exactly two loaded datasets."""

    def __init__(self, a: LoadedDataset, b: LoadedDataset, config: dict[str, Any]):
        self.a = a
        self.b = b
        self.schema = config["schema"]
        self.merge_cfg = config["merge"]
        self.numeric_cols = [
            c for c in self.schema["numeric_columns"]
            if c in a.frame.columns and c in b.frame.columns
        ]
        self.panel_key = self.schema["panel_key"]

    def compare(self) -> dict[str, Any]:
        schema_diff = self._schema_diff()
        dtype_diff = self._dtype_diff()
        overlap = self._key_overlap()
        conflicts = self._value_conflicts(overlap["shared_keys"])
        relationship = self._classify_relationship(overlap)
        recommendation = self._recommend_merge(relationship, overlap, conflicts)
        return {
            "datasets": [self.a.name, self.b.name],
            "schema_differences": schema_diff,
            "datatype_differences": dtype_diff,
            "key_overlap": {
                "n_keys_a": overlap["n_keys_a"],
                "n_keys_b": overlap["n_keys_b"],
                "n_shared_keys": overlap["n_shared"],
                "n_only_in_a": len(overlap["only_in_a"]),
                "n_only_in_b": len(overlap["only_in_b"]),
                "only_in_a": overlap["only_in_a"],
                "only_in_b": overlap["only_in_b"],
            },
            "value_conflicts": conflicts,
            "relationship": relationship,
            "recommended_merge_strategy": recommendation,
        }

    # ------------------------------------------------------------------ #
    def _schema_diff(self) -> dict[str, Any]:
        # Compare *raw* headers (pre-normalization) and canonical columns.
        raw_a, raw_b = set(self.a.raw_columns), set(self.b.raw_columns)
        canon_a = set(self.a.frame.columns)
        canon_b = set(self.b.frame.columns)
        return {
            "raw_headers_identical": raw_a == raw_b,
            "raw_only_in_a": sorted(raw_a - raw_b),
            "raw_only_in_b": sorted(raw_b - raw_a),
            "canonical_columns_identical": canon_a == canon_b,
            "canonical_only_in_a": sorted(canon_a - canon_b),
            "canonical_only_in_b": sorted(canon_b - canon_a),
            "note": (
                "Raw headers differ only by unit suffixes; both map to an "
                "identical canonical schema."
                if canon_a == canon_b and raw_a != raw_b
                else None
            ),
        }

    def _dtype_diff(self) -> dict[str, Any]:
        diffs = {}
        common = set(self.a.frame.columns) & set(self.b.frame.columns)
        for col in sorted(common):
            ta, tb = str(self.a.frame[col].dtype), str(self.b.frame[col].dtype)
            if ta != tb:
                diffs[col] = {"a": ta, "b": tb}
        return {"differing_columns": diffs, "identical": len(diffs) == 0}

    def _key_index(self, df: pd.DataFrame) -> pd.DataFrame:
        idx = df.set_index(self.panel_key)
        return idx

    def _key_overlap(self) -> dict[str, Any]:
        keys_a = set(map(tuple, self.a.frame[self.panel_key].astype(str).values))
        keys_b = set(map(tuple, self.b.frame[self.panel_key].astype(str).values))
        shared = keys_a & keys_b
        return {
            "n_keys_a": len(keys_a),
            "n_keys_b": len(keys_b),
            "n_shared": len(shared),
            "shared_keys": sorted(shared),
            "only_in_a": [" | ".join(k) for k in sorted(keys_a - keys_b)],
            "only_in_b": [" | ".join(k) for k in sorted(keys_b - keys_a)],
        }

    def _value_conflicts(self, shared_keys: list[tuple]) -> dict[str, Any]:
        """For keys present in both datasets, measure value agreement."""
        if not shared_keys:
            return {"shared_keys": 0, "note": "No shared keys; nothing to compare."}

        a_idx = self.a.frame.set_index(self.panel_key)
        b_idx = self.b.frame.set_index(self.panel_key)

        # Align on the (string-cast) shared keys.
        a_idx.index = a_idx.index.map(lambda t: tuple(map(str, t)))
        b_idx.index = b_idx.index.map(lambda t: tuple(map(str, t)))

        tol = float(self.merge_cfg["conflict_relative_tolerance"])
        per_col_conflicts: dict[str, int] = {c: 0 for c in self.numeric_cols}
        identical_rows = 0
        example_conflicts: list[dict[str, Any]] = []
        max_rel_by_col: dict[str, float] = {c: 0.0 for c in self.numeric_cols}

        for key in shared_keys:
            row_a = a_idx.loc[key, self.numeric_cols].astype(float)
            row_b = b_idx.loc[key, self.numeric_cols].astype(float)
            rel = (row_a - row_b).abs() / row_a.abs().replace(0, np.nan)
            row_conflicts = (rel > tol).fillna(False)
            for col in self.numeric_cols:
                if bool(row_conflicts[col]):
                    per_col_conflicts[col] += 1
                if pd.notna(rel[col]):
                    max_rel_by_col[col] = max(max_rel_by_col[col], float(rel[col]))
            if not row_conflicts.any():
                identical_rows += 1
            elif len(example_conflicts) < 3:
                worst_col = rel.astype(float).idxmax()
                example_conflicts.append({
                    "key": " | ".join(key),
                    "column": worst_col,
                    "value_a": round(float(row_a[worst_col]), 4),
                    "value_b": round(float(row_b[worst_col]), 4),
                    "relative_diff": round(float(rel[worst_col]), 4),
                })

        n_shared = len(shared_keys)
        conflicting_rows = n_shared - identical_rows
        return {
            "shared_keys": n_shared,
            "identical_rows": identical_rows,
            "conflicting_rows": conflicting_rows,
            "pct_conflicting": round(100.0 * conflicting_rows / n_shared, 2),
            "conflicts_per_column": per_col_conflicts,
            "max_relative_diff_per_column": {c: round(v, 4) for c, v in max_rel_by_col.items()},
            "example_conflicts": example_conflicts,
        }

    def _classify_relationship(self, overlap: dict[str, Any]) -> dict[str, Any]:
        shared = overlap["n_shared"]
        only_a = len(overlap["only_in_a"])
        only_b = len(overlap["only_in_b"])
        if shared == 0:
            label = "complementary"
            desc = "The datasets cover disjoint company-quarter keys."
        elif only_a == 0 and only_b == 0:
            label = "overlapping"
            desc = "The datasets cover an identical set of company-quarter keys."
        else:
            label = "partially_overlapping"
            desc = "The datasets share some keys but each has unique keys too."
        return {"label": label, "description": desc}

    def _recommend_merge(
        self, relationship: dict[str, Any], overlap: dict[str, Any], conflicts: dict[str, Any]
    ) -> dict[str, Any]:
        """Recommend the safest merge strategy given the observed structure."""
        label = relationship["label"]
        configured = self.merge_cfg["strategy"]

        if label == "complementary":
            rec = "reconcile_mean"  # no conflicts possible; a plain union is safe
            rationale = (
                "Keys are disjoint, so a keyed union produces one row per "
                "company-quarter with no conflicts. Provenance can still be tagged."
            )
        elif label == "overlapping":
            pct = conflicts.get("pct_conflicting", 0)
            if pct and pct > 0:
                rec = "source_tagged_pool"
                rationale = (
                    f"The two datasets describe the SAME company-quarter keys but "
                    f"{pct}% of shared rows carry conflicting values. Silently "
                    f"stacking would create ambiguous duplicate keys, and picking "
                    f"one source discards real signal. The safest option that "
                    f"preserves all information is to keep both and tag provenance "
                    f"with a `source` column, making the panel key "
                    f"(source, Company, Year, Quarter). 'prefer_real' is the more "
                    f"conservative single-record alternative."
                )
            else:
                rec = "prefer_real"
                rationale = "Shared keys agree within tolerance; one authoritative copy suffices."
        else:  # partially_overlapping
            rec = "source_tagged_pool"
            rationale = (
                "Datasets partially overlap; tagging provenance preserves both the "
                "shared and the unique keys without fabricating or dropping records."
            )

        return {
            "recommended": rec,
            "configured": configured,
            "matches_config": rec == configured,
            "rationale": rationale,
            "warning": (
                "Do NOT blindly stack: shared keys with conflicting values would "
                "otherwise become undetected duplicate company-quarter records."
            ),
        }
