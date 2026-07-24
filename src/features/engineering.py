"""STEP 3 & 4 — leakage-safe feature engineering and target construction.

Every derived column is computed **within each entity** (source + Company),
ordered by ``time_index``, so no information ever crosses company boundaries or
flows backward in time. The three leakage-safe patterns used:

* **Lags** — ``group.shift(L)`` references quarter t-L (history only).
* **Rolling** — ``group.shift(1).rolling(w)`` uses quarters t-1 … t-w only;
  the current quarter is explicitly excluded via the shift.
* **Growth** — QoQ/YoY use x_t vs a past quarter (both known at t, valid as
  predictors); lagged growth uses only past quarters.

Current-quarter raw values are retained as predictors: standing at quarter t to
predict t+1, the value at t is known and legitimately usable.

Targets are ``group.shift(-horizon)`` — the value at t+1 — and are NaN for the
last quarter of every entity (no future to observe).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class EngineeringResult:
    frame: pd.DataFrame
    feature_columns: list[str] = field(default_factory=list)
    target_columns: list[str] = field(default_factory=list)
    engineered_groups: dict[str, list[str]] = field(default_factory=dict)
    provenance: dict[str, dict[str, Any]] = field(default_factory=dict)


class FeatureEngineer:
    def __init__(self, df: pd.DataFrame, config: dict[str, Any]):
        self.cfg = config["features"]
        self.tcfg = config["targets"]
        self.entity = self.cfg["entity_group"]
        self.order_by = self.cfg["order_by"]
        # Sort once; all group ops rely on this ordering.
        self.df = df.sort_values(self.entity + [self.order_by]).reset_index(drop=True)
        self._g = self.df.groupby(self.entity, sort=False)
        # New columns are accumulated here and concatenated once, avoiding the
        # DataFrame fragmentation that repeated inserts would cause.
        self._new: dict[str, pd.Series] = {}
        # Track how each engineered column was produced (for leakage validation).
        self.provenance: dict[str, dict[str, Any]] = {}
        self.groups: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ #
    def _add(self, name: str, series) -> None:
        """Stage a new column (as values aligned to self.df's index)."""
        self._new[name] = np.asarray(series)

    def build(self) -> EngineeringResult:
        self._temporal_seasonal()
        self._lags()
        self._macro_lags()
        self._rolling()
        self._growth()
        self._interactions()
        target_cols = self._targets()

        # Materialize all engineered columns in a single concat (no fragmentation).
        self.df = pd.concat(
            [self.df, pd.DataFrame(self._new, index=self.df.index)], axis=1
        )

        # Replace infinities produced by ratios/growth with NaN (reported later).
        engineered = [c for cols in self.groups.values() for c in cols] + target_cols
        self.df[engineered] = self.df[engineered].replace([np.inf, -np.inf], np.nan)
        # De-fragment once, then append the derived has_target flag.
        self.df = self.df.copy()
        self.df["has_target"] = self.df[target_cols].notna().all(axis=1)

        feature_cols = self._resolve_feature_columns()
        return EngineeringResult(
            frame=self.df,
            feature_columns=feature_cols,
            target_columns=target_cols,
            engineered_groups=self.groups,
            provenance=self.provenance,
        )

    # ------------------------------------------------------------------ #
    def _register(self, group: str, name: str, meta: dict[str, Any]) -> None:
        self.groups.setdefault(group, []).append(name)
        self.provenance[name] = meta

    def _temporal_seasonal(self) -> None:
        if self.cfg["seasonal"].get("quarter_onehot", True):
            dummies = pd.get_dummies(self.df["Quarter"], prefix="Quarter").astype(int)
            for c in dummies.columns:
                self._add(c, dummies[c].values)
                self._register("seasonal", c, {"kind": "seasonal_onehot", "source": "Quarter",
                                               "references": "current"})
        if self.cfg["seasonal"].get("fiscal_year_end_flag", True):
            self._add("is_Q4", (self.df["Quarter"] == "Q4").astype(int).values)
            self._register("seasonal", "is_Q4", {"kind": "seasonal_flag", "source": "Quarter",
                                                 "references": "current"})

    def _lags(self) -> None:
        for var in self.cfg["lag_vars"]:
            for L in self.cfg["lags"]:
                name = f"{var}_lag{L}"
                self._add(name, self._g[var].shift(L))
                self._register("lag", name, {"kind": "lag", "base": var, "lag": L,
                                             "references": f"t-{L}"})

    def _macro_lags(self) -> None:
        for var in self.cfg["macro_vars"]:
            for L in self.cfg["macro_lags"]:
                name = f"{var}_lag{L}"
                self._add(name, self._g[var].shift(L))
                self._register("macro_lag", name, {"kind": "macro_lag", "base": var, "lag": L,
                                                   "references": f"t-{L}"})

    def _rolling(self) -> None:
        rcfg = self.cfg["rolling"]
        mp = int(rcfg.get("min_periods", 2))
        for var in rcfg["vars"]:
            for w in rcfg["windows"]:
                shifted = self._g[var].shift(1)  # exclude current quarter
                grp = shifted.groupby(self.df[self.entity[0]].astype(str) + "|" +
                                      self.df[self.entity[1]].astype(str))
                for stat in rcfg["stats"]:
                    name = f"{var}_roll{w}_{stat}"
                    if stat == "mean":
                        val = grp.transform(lambda s: s.rolling(w, min_periods=mp).mean())
                    elif stat == "median":
                        val = grp.transform(lambda s: s.rolling(w, min_periods=mp).median())
                    elif stat == "std":
                        val = grp.transform(lambda s: s.rolling(w, min_periods=mp).std())
                    elif stat == "growth":
                        # growth over the window, ending at t-1 (all past).
                        val = grp.transform(lambda s: s.pct_change(w))
                    else:
                        continue
                    self._add(name, val.values)
                    self._register("rolling", name, {"kind": "rolling", "base": var, "window": w,
                                                     "stat": stat, "shift": 1,
                                                     "references": f"t-1..t-{w}"})

    def _growth(self) -> None:
        gcfg = self.cfg["growth"]
        for var in gcfg["vars"]:
            if gcfg.get("qoq", True):
                name = f"{var}_qoq_growth"
                self._add(name, self._g[var].pct_change(1))
                self._register("growth", name, {"kind": "growth_qoq", "base": var,
                                                "references": "t vs t-1"})
            if gcfg.get("yoy", True):
                name = f"{var}_yoy_growth"
                self._add(name, self._g[var].pct_change(4))
                self._register("growth", name, {"kind": "growth_yoy", "base": var,
                                                "references": "t vs t-4"})
            if gcfg.get("include_lagged_growth", True):
                name = f"{var}_qoq_growth_lag1"
                lagged = self._g[var].pct_change(1).groupby(
                    [self.df[self.entity[0]], self.df[self.entity[1]]]).shift(1)
                self._add(name, lagged)
                self._register("growth", name, {"kind": "growth_qoq_lagged", "base": var,
                                                "references": "t-1 vs t-2"})

    def _interactions(self) -> None:
        for spec in self.cfg["interactions"]:
            a, b, name = spec["a"], spec["b"], spec["name"]
            if a in self.df.columns and b in self.df.columns:
                self._add(name, (self.df[a].astype(float) * self.df[b].astype(float)).values)
                self._register("interaction", name, {"kind": "interaction", "a": a, "b": b,
                                                     "references": "current x current"})

    def _targets(self) -> list[str]:
        horizon = int(self.tcfg["horizon"])
        prefix = self.tcfg["prefix"]
        cols: list[str] = []
        for var in self.tcfg["variables"]:
            name = f"{prefix}{var}".replace(" ", "_")
            self._add(name, self._g[var].shift(-horizon))
            self.provenance[name] = {"kind": "target", "base": var, "horizon": horizon,
                                     "references": f"t+{horizon}"}
            cols.append(name)
        return cols

    # ------------------------------------------------------------------ #
    def _resolve_feature_columns(self) -> list[str]:
        """All predictor columns: current raw financial+macro + engineered."""
        raw_predictors = list(self.cfg["financial_vars"]) + list(self.cfg["macro_vars"])
        engineered = [c for grp, cols in self.groups.items() for c in cols]
        # Deduplicate while preserving order.
        seen: set[str] = set()
        out: list[str] = []
        for c in raw_predictors + engineered:
            if c in self.df.columns and c not in seen:
                seen.add(c)
                out.append(c)
        return out
