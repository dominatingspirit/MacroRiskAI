"""Naive forecasting baselines (leakage-safe, entity-aware).

Every ML model must beat these. All are computed within each entity
(source+Company) using only information available at or before quarter t, to
forecast the base variable at t+1:

* **Previous Quarter (Naive)** — forecast t+1 with the value at t.
* **Seasonal Naive**           — forecast t+1 with the value at t-3 (same
  quarter of the previous year).
* **Historical Mean**          — forecast t+1 with the expanding mean of the
  base variable through quarter t.

These reference only past/current quarters, never the future.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class BaselineForecaster:
    def __init__(self, df: pd.DataFrame, config: dict[str, Any]):
        self.cfg = config["features"]
        self.entity = self.cfg["entity_group"]
        self.order_by = self.cfg["order_by"]
        self.df = df.sort_values(self.entity + [self.order_by])
        self._g = self.df.groupby(self.entity, sort=False)

    def predictions(self, base_var: str) -> pd.DataFrame:
        """Return baseline predictions for target = base_var at t+1, per row."""
        g = self._g[base_var]
        out = pd.DataFrame(index=self.df.index)
        # Previous-quarter naive: forecast(t+1) = value(t).
        out["naive_prev_quarter"] = self.df[base_var].to_numpy()
        # Seasonal naive: forecast(t+1) = value(t-3) = same quarter last year.
        out["seasonal_naive"] = g.shift(3).to_numpy()
        # Historical mean: expanding mean through t (inclusive), per entity.
        out["historical_mean"] = g.transform(
            lambda s: s.expanding(min_periods=1).mean()
        ).to_numpy()
        return out


BASELINE_NAMES = ["naive_prev_quarter", "seasonal_naive", "historical_mean"]
