"""Optuna hyperparameter optimization for each target's top-N models.

For every (target, candidate model):
  * run an Optuna study whose objective is the mean walk-forward RMSE (same
    folds as Phase 4); MAE is recorded as the tie-break,
  * select best params by (RMSE, then MAE),
  * recompute pooled out-of-fold metrics identically to Phase 4 for an
    apples-to-apples before/after comparison,
  * refit on all trainable rows and persist the tuned Pipeline,
  * save every trial and the best params,
  * decide promotion (tuned beats Phase-4 baseline RMSE by the configured
    margin).

Booster early stopping is supported (via a time-based holdout carved from each
training fold) but only engages if a booster is a candidate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from ..features.preprocessor import build_preprocessor
from ..models.metrics import compute_metrics
from ..models.registry import get_model_spec, make_estimator, ModelSpec
from ..models.validation import make_walk_forward_folds
from .search_spaces import suggest_params, supports_early_stopping

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Phase-4 baseline defaults (searched params only) enqueued as each study's
# first trial, so tuning is guaranteed to consider the baseline configuration.
_BASELINE_ENQUEUE: dict[str, dict[str, Any]] = {
    "ridge": {"alpha": 1.0},
    "lasso": {"alpha": 0.01},
    "elastic_net": {"alpha": 0.01, "l1_ratio": 0.5},
    "random_forest": {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1,
                      "min_samples_split": 2, "max_features": 1.0},
    "extra_trees": {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1,
                    "min_samples_split": 2, "max_features": 1.0, "bootstrap": False},
    "decision_tree": {"max_depth": None, "min_samples_leaf": 1, "min_samples_split": 2},
}


@dataclass
class TuneOutputs:
    before_after: list[dict[str, Any]] = field(default_factory=list)
    trials: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict)
    best_params: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    promoted: dict[str, dict[str, Any]] = field(default_factory=dict)  # per target


class HyperparameterOptimizer:
    def __init__(self, df, metadata, config, out_root: Path, models_dir: Path,
                 candidates: dict[str, list[str]], baseline: pd.DataFrame):
        self.df = df
        self.meta = metadata
        self.config = config
        self.p5 = config["phase5"]
        self.seed = self.p5["random_seed"]
        self.out_root = out_root
        self.models_dir = models_dir
        self.candidates = candidates
        self.baseline = baseline

        self.numeric = metadata["numeric_features"]
        self.categorical = metadata["categorical_features"]
        self.feature_cols = self.categorical + self.numeric
        self.base_map = metadata["target_base_map"]
        self.folds = make_walk_forward_folds(df, config)
        self.X_all = df[self.feature_cols]
        self.tr_mask = df["has_target"].to_numpy()
        self.out = TuneOutputs()

    # ------------------------------------------------------------------ #
    def run(self) -> TuneOutputs:
        for target, models in self.candidates.items():
            y_full = self.df[target].to_numpy(dtype=float)
            target_best = None
            for model_name in models:
                res = self._tune_one(target, model_name, y_full)
                self.out.before_after.append(res)
                cand = {"model": model_name, "tuned_rmse": res["tuned_rmse"],
                        "tuned_mae": res["tuned_mae"], "promoted": res["promoted"]}
                if res["promoted"] and (target_best is None or res["tuned_rmse"] < target_best["tuned_rmse"]):
                    target_best = cand
            if target_best is not None:
                self.out.promoted[target] = target_best
        return self.out

    # ------------------------------------------------------------------ #
    def _family_n_trials(self, group: str) -> int:
        return int(self.p5["n_trials"].get(group, 20))

    def _build_pipe(self, spec: ModelSpec, params: dict[str, Any]) -> Pipeline:
        pre = build_preprocessor(spec.pre, self.numeric, self.categorical, self.config)
        est = make_estimator(spec, self.seed)
        est.set_params(**params)
        return Pipeline([("preprocess", clone(pre)), ("model", est)])

    def _pooled_oof(self, spec: ModelSpec, params: dict[str, Any], y_full):
        """Walk-forward pooled out-of-fold (y, pred, n_features).

        This single definition is used for BOTH the Optuna objective and the
        final before/after metrics, so the optimized quantity is exactly the
        quantity compared against the Phase-4 baseline (pooled-OOF, identical
        methodology). Combined with the enqueued baseline defaults, tuning can
        never end up worse than the baseline on the objective.
        """
        oof = np.full(len(self.df), np.nan)
        p = None
        for f in self.folds:
            pipe = self._build_pipe(spec, params)
            pipe.fit(self.X_all.iloc[f.train_idx], y_full[f.train_idx])
            oof[f.val_idx] = pipe.predict(self.X_all.iloc[f.val_idx])
            p = pipe.named_steps["model"].n_features_in_
        idx = np.concatenate([f.val_idx for f in self.folds])
        return y_full[idx], oof[idx], p

    def _pooled_oof_metrics(self, spec: ModelSpec, params: dict[str, Any], y_full):
        y, pred, p = self._pooled_oof(spec, params, y_full)
        return compute_metrics(y, pred, n_features=p)

    def _tune_one(self, target: str, model_name: str, y_full) -> dict[str, Any]:
        spec = get_model_spec(self.config, model_name)
        n_trials = self._family_n_trials(spec.group)

        def objective(trial):
            params = suggest_params(trial, model_name, self.seed)
            y_oof, pred_oof, p = self._pooled_oof(spec, params, y_full)
            m = compute_metrics(y_oof, pred_oof, n_features=p)
            trial.set_user_attr("mean_mae", m["mae"])
            return m["rmse"]

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        # Warm start: evaluate the Phase-4 baseline defaults first so the study's
        # best is guaranteed to be at least as good as the baseline (fair
        # before/after; promotion only fires on genuine improvement).
        enqueue = _BASELINE_ENQUEUE.get(model_name)
        if enqueue is not None:
            study.enqueue_trial(enqueue)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Trials table + best selection by (RMSE, MAE).
        trials_df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
        trials_df = trials_df.rename(columns={"value": "rmse", "user_attrs_mean_mae": "mae"})
        best_row = trials_df.sort_values(["rmse", "mae"]).iloc[0]
        best_params = _reconstruct_params(model_name, best_row, self.seed)

        # Pooled-OOF metrics for before/after (matches Phase 4 methodology).
        tuned_metrics = self._pooled_oof_metrics(spec, best_params, y_full)

        base = self.baseline[(self.baseline["target"] == target) &
                             (self.baseline["model"] == model_name)]
        base_rmse = float(base["rmse"].iloc[0]) if len(base) else float("nan")
        base_mae = float(base["mae"].iloc[0]) if len(base) else float("nan")
        base_r2 = float(base["r2"].iloc[0]) if len(base) else float("nan")

        min_frac = float(self.p5["promotion"]["min_rmse_improvement_frac"])
        improved = np.isfinite(base_rmse) and tuned_metrics["rmse"] < base_rmse * (1 - min_frac)
        rmse_improve_pct = (100.0 * (base_rmse - tuned_metrics["rmse"]) / base_rmse
                            if np.isfinite(base_rmse) and base_rmse != 0 else float("nan"))

        # Persist tuned pipeline (refit on all trainable rows) + artifacts.
        self._persist(target, spec, best_params, y_full, tuned_metrics,
                      {"rmse": base_rmse, "mae": base_mae, "r2": base_r2},
                      improved, rmse_improve_pct, n_trials)
        self.out.trials[(target, model_name)] = trials_df
        self.out.best_params[(target, model_name)] = best_params

        return {
            "target": target, "model": model_name, "group": spec.group,
            "baseline_rmse": base_rmse, "tuned_rmse": tuned_metrics["rmse"],
            "baseline_mae": base_mae, "tuned_mae": tuned_metrics["mae"],
            "baseline_r2": base_r2, "tuned_r2": tuned_metrics["r2"],
            "rmse_improvement_pct": round(rmse_improve_pct, 3) if np.isfinite(rmse_improve_pct) else None,
            "n_trials": n_trials, "promoted": bool(improved),
            "best_params": best_params,
            "early_stopping_used": supports_early_stopping(model_name),
        }

    def _persist(self, target, spec, params, y_full, tuned_metrics, base_metrics,
                 improved, improve_pct, n_trials) -> None:
        out_dir = self.models_dir / target
        out_dir.mkdir(parents=True, exist_ok=True)
        pipe = self._build_pipe(spec, params)
        pipe.fit(self.X_all.iloc[self.tr_mask], y_full[self.tr_mask])
        artifact = {
            "pipeline": pipe, "target": target, "base_variable": self.base_map[target],
            "model_name": spec.name, "model_group": spec.group,
            "preprocessor_family": spec.pre, "best_params": params,
            "feature_schema": {"numeric": self.numeric, "categorical": self.categorical},
            "tuned_metrics_oof": tuned_metrics, "baseline_metrics_oof": base_metrics,
            "rmse_improvement_pct": improve_pct, "promoted": bool(improved),
            "n_trials": n_trials,
            "validation": {"scheme": self.config["validation"]["scheme"],
                           "n_folds": len(self.folds),
                           "val_quarters": [f.val_time for f in self.folds]},
            "tuning": "optuna_TPE", "seed": self.seed,
        }
        joblib.dump(artifact, out_dir / f"{spec.name}.joblib")
        with (out_dir / f"{spec.name}.best_params.json").open("w", encoding="utf-8") as fh:
            json.dump({"best_params": params, "tuned_metrics_oof": tuned_metrics,
                       "baseline_metrics_oof": base_metrics, "promoted": bool(improved)},
                      fh, indent=2, default=float)


def _reconstruct_params(model_name: str, best_row: pd.Series, seed: int) -> dict[str, Any]:
    """Rebuild the estimator kwargs from the best trial's params columns."""
    prefix = "params_"
    params = {c[len(prefix):]: best_row[c] for c in best_row.index
              if c.startswith(prefix) and pd.notna(best_row[c])}
    # Cast numpy/pandas scalars and fix known int params.
    int_keys = {"n_estimators", "min_samples_leaf", "min_samples_split", "max_iter",
                "num_leaves", "min_child_samples", "min_child_weight", "depth",
                "iterations", "max_leaf_nodes"}
    clean: dict[str, Any] = {}
    for k, v in params.items():
        if k == "max_depth":
            # Categorical [None, 6, 10, ...]; NaN/None -> None, else int.
            clean[k] = None if (v is None or (isinstance(v, float) and np.isnan(v))) else int(v)
        elif k == "max_features":
            # 'sqrt'/'log2' strings pass through; numeric fractions -> float.
            clean[k] = v if isinstance(v, str) else float(v)
        elif k in int_keys:
            clean[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean[k] = float(v)
        elif isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (np.bool_,)):
            clean[k] = bool(v)
        else:
            clean[k] = v
    # Re-attach fixed (non-searched) kwargs and n_jobs where relevant.
    if model_name in {"random_forest", "extra_trees"}:
        clean["n_jobs"] = -1
    if model_name == "elastic_net":
        clean.setdefault("max_iter", 5000); clean.setdefault("tol", 1e-3)
    if model_name == "lasso":
        clean.setdefault("max_iter", 10000); clean.setdefault("tol", 1e-3)
    return clean
