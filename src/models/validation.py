"""Walk-forward (expanding-window) validation over quarters.

The dataset is a panel of entities (source+Company) each observed over the same
set of quarters. Folds are defined purely by ``time_index`` so ordering is never
violated and no shuffling occurs:

    fold k validates on rows at quarter ``v_k`` and trains on all rows with
    time_index <= v_k - 1 (expanding window).

Because a feature row at quarter t carries the target for t+1, training rows
(time <= v_k - 1) have targets no later than quarter v_k, and the validation
target is at v_k + 1 — so no future information leaks into training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Fold:
    index: int
    val_time: int
    train_idx: np.ndarray
    val_idx: np.ndarray


def make_walk_forward_folds(df: pd.DataFrame, config: dict[str, Any]) -> list[Fold]:
    """Build expanding-window folds on trainable rows only."""
    vcfg = config["validation"]
    order_by = config["features"]["order_by"]
    mask = df["has_target"].to_numpy()
    times = np.sort(df.loc[mask, order_by].unique())

    n_folds = int(vcfg["n_folds"])
    min_train_q = int(vcfg["min_train_quarters"])

    # Validation quarters = the last n_folds available feature-quarters that
    # still leave at least `min_train_quarters` quarters for training.
    candidate_val_times = times[min_train_q:]
    val_times = candidate_val_times[-n_folds:]

    folds: list[Fold] = []
    idx_all = np.arange(len(df))
    for i, vt in enumerate(val_times):
        train_mask = mask & (df[order_by].to_numpy() <= vt - 1)
        val_mask = mask & (df[order_by].to_numpy() == vt)
        folds.append(Fold(
            index=i,
            val_time=int(vt),
            train_idx=idx_all[train_mask],
            val_idx=idx_all[val_mask],
        ))
    return folds
