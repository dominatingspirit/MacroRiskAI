"""STEP 3 — build the canonical master quarterly dataset.

Given the two loaded datasets and a merge strategy, produce a single canonical
dataset that:

* preserves chronological ordering (via ``time_index``)
* preserves company identities and sectors
* preserves accounting relationships (values are never recomputed)
* removes exact duplicates
* maintains temporal integrity (sorted, gap-aware)

Three strategies are supported (see ``config.yaml``):

``source_tagged_pool``
    Concatenate both datasets and add a ``source`` provenance column. The
    effective panel key becomes (source, Company, Year, Quarter). This keeps
    every observation while making duplicated company-quarter keys explicit
    and attributable, rather than silently ambiguous.

``prefer_real``
    One row per (Company, Year, Quarter). On conflict, keep the ``real``
    dataset's values (treated as authoritative). Conservative; halves rows.

``reconcile_mean``
    One row per (Company, Year, Quarter). Numeric columns are averaged across
    whichever sources supply the key; identifiers taken from the first source.

Note on macroeconomic variables: the brief asks to "preserve macroeconomic
variables". The inspected datasets contain **no** macro columns — only
company-level financials — so there are none to preserve. The builder passes
through whatever columns exist and records this fact; it never fabricates
macro data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .loader import LoadedDataset


@dataclass
class MasterBuildResult:
    frame: pd.DataFrame
    strategy: str
    decisions: dict[str, Any]


class MasterDatasetBuilder:
    """Construct the canonical master dataset from N loaded sources."""

    def __init__(self, datasets: dict[str, LoadedDataset], config: dict[str, Any]):
        self.datasets = datasets
        self.schema = config["schema"]
        self.merge_cfg = config["merge"]
        self.numeric_cols = self.schema["numeric_columns"]
        self.id_cols = self.schema["identifier_columns"]
        self.panel_key = self.schema["panel_key"]
        self.provenance_col = self.merge_cfg["provenance_column"]

    # ------------------------------------------------------------------ #
    def build(self, strategy: str | None = None) -> MasterBuildResult:
        strategy = strategy or self.merge_cfg["strategy"]
        decisions: dict[str, Any] = {"strategy": strategy}

        # Tag each source with provenance up front (used by all strategies).
        tagged = []
        for name, ds in self.datasets.items():
            df = ds.frame.copy()
            df[self.provenance_col] = name
            tagged.append(df)
        combined = pd.concat(tagged, ignore_index=True)

        # Remove exact full-row duplicates (identical across every column,
        # including provenance) — these carry no information.
        pre = len(combined)
        combined = combined.drop_duplicates().reset_index(drop=True)
        decisions["exact_duplicate_rows_removed"] = int(pre - len(combined))

        if strategy == "source_tagged_pool":
            master = self._source_tagged_pool(combined, decisions)
        elif strategy == "prefer_real":
            master = self._collapse(combined, decisions, how="prefer_real")
        elif strategy == "reconcile_mean":
            master = self._collapse(combined, decisions, how="reconcile_mean")
        else:
            raise ValueError(f"Unknown merge strategy: {strategy!r}")

        master = self._finalize_ordering(master, decisions)
        decisions["macroeconomic_columns_present"] = self._macro_columns()
        decisions["final_rows"] = int(len(master))
        decisions["final_columns"] = list(master.columns)
        return MasterBuildResult(frame=master, strategy=strategy, decisions=decisions)

    # ------------------------------------------------------------------ #
    def _source_tagged_pool(self, combined: pd.DataFrame, decisions: dict[str, Any]) -> pd.DataFrame:
        """Keep all rows; panel key is (source, Company, Year, Quarter)."""
        full_key = [self.provenance_col] + self.panel_key
        dup = combined.duplicated(subset=full_key).sum()
        decisions["strategy_detail"] = (
            "Both sources retained with provenance tag; effective panel key is "
            f"({', '.join(full_key)})."
        )
        decisions["duplicate_full_key_rows"] = int(dup)
        if dup:
            # Should not happen for a balanced panel; surface loudly if it does.
            decisions["warning"] = (
                f"{dup} rows share the full provenance key — investigate source integrity."
            )
        return combined

    def _collapse(self, combined: pd.DataFrame, decisions: dict[str, Any], *, how: str) -> pd.DataFrame:
        """Produce one row per (Company, Year, Quarter)."""
        rows = []
        conflict_count = 0
        for _, grp in combined.groupby(self.panel_key, sort=False):
            base = grp.iloc[0].copy()
            if len(grp) > 1:
                if how == "reconcile_mean":
                    for col in self.numeric_cols:
                        base[col] = grp[col].astype(float).mean()
                elif how == "prefer_real":
                    if "real" in set(grp[self.provenance_col]):
                        base = grp.loc[grp[self.provenance_col] == "real"].iloc[0].copy()
                # detect whether the collapsed group actually disagreed
                if grp[self.numeric_cols].astype(float).nunique().gt(1).any():
                    conflict_count += 1
            base[self.provenance_col] = "+".join(sorted(grp[self.provenance_col].unique()))
            rows.append(base)
        decisions["strategy_detail"] = (
            f"Collapsed to one row per (Company, Year, Quarter) using '{how}'."
        )
        decisions["company_quarters_with_conflicts_collapsed"] = int(conflict_count)
        return pd.DataFrame(rows).reset_index(drop=True)

    def _finalize_ordering(self, df: pd.DataFrame, decisions: dict[str, Any]) -> pd.DataFrame:
        """Sort chronologically within company and order columns cleanly."""
        sort_cols = [self.provenance_col, "Company", "time_index"] \
            if self.provenance_col in df.columns else ["Company", "time_index"]
        # Only sort by provenance first for the pooled strategy so each source's
        # series stays internally contiguous and chronological.
        df = df.sort_values(sort_cols).reset_index(drop=True)

        # Column order: identifiers, engineered time cols, provenance, numerics.
        ordered = (
            [c for c in self.id_cols if c in df.columns]
            + [c for c in ["quarter_num", "time_index"] if c in df.columns]
            + [self.provenance_col]
            + [c for c in self.numeric_cols if c in df.columns]
        )
        ordered += [c for c in df.columns if c not in ordered]  # any extras
        decisions["sorted_by"] = sort_cols
        return df[ordered]

    def _macro_columns(self) -> list[str]:
        """Identify any macroeconomic columns (none expected in these inputs)."""
        known = set(self.id_cols) | set(self.numeric_cols) | {
            "quarter_num", "time_index", self.provenance_col,
        }
        any_frame = next(iter(self.datasets.values())).frame
        extras = [c for c in any_frame.columns if c not in known]
        return extras
