"""Core baseline benchmarking engine.

For every target independently:
  * run walk-forward CV for each model (fit → predict → per-fold metrics),
  * pool out-of-fold predictions for headline metrics + residual diagnostics,
  * compute stability (mean/std of RMSE and R^2 across folds),
  * refit on all trainable rows, persist the whole Pipeline + metadata,
  * extract feature importance / standardized coefficients,
  * derive OOF-residual prediction intervals (stored separately),
  * run leakage diagnostics when mean R^2 exceeds the configured trigger.

Baselines (naive / seasonal-naive / historical-mean) are scored on the same
out-of-fold rows. A multi-output Random Forest is scored as a benchmark only.

Everything heavy (fitting, plotting, persistence) happens inside the loop so no
more than one fitted pipeline is held in memory at a time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from .baselines import BASELINE_NAMES, BaselineForecaster
from .importance import extract_importance
from .leakage_diag import run_leakage_diagnostics
from .metrics import compute_metrics
from .plots import prediction_plot, residual_diagnostics
from .registry import (
    ModelSpec,
    build_pipeline,
    get_benchmark_spec,
    iter_model_specs,
    make_estimator,
)
from .validation import make_walk_forward_folds
from ..features.preprocessor import build_preprocessor


@dataclass
class BenchmarkOutputs:
    results: list[dict[str, Any]] = field(default_factory=list)        # one row per (target, model/baseline)
    importances: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict)
    intervals: list[dict[str, Any]] = field(default_factory=list)
    leakage: list[dict[str, Any]] = field(default_factory=list)
    oof: dict[tuple[str, str], dict[str, np.ndarray]] = field(default_factory=dict)


class BaselineBenchmark:
    def __init__(self, df: pd.DataFrame, metadata: dict[str, Any], config: dict[str, Any],
                 out_root: Path, models_dir: Path):
        self.df = df
        self.meta = metadata
        self.config = config
        self.seed = config["phase4"]["random_seed"]
        self.out_root = out_root
        self.models_dir = models_dir

        self.numeric = metadata["numeric_features"]
        self.categorical = metadata["categorical_features"]
        self.feature_cols = self.categorical + self.numeric
        self.targets = metadata["target_columns"]
        self.base_map = metadata["target_base_map"]

        self.folds = make_walk_forward_folds(df, config)
        self.X_all = df[self.feature_cols]
        self.baseline_preds = BaselineForecaster(df, config)
        self.out = BenchmarkOutputs()

    # ------------------------------------------------------------------ #
    def run(self) -> BenchmarkOutputs:
        self._run_baselines_and_models()
        self._run_multioutput_benchmark()
        return self.out

    # ------------------------------------------------------------------ #
    def _oof_indices(self) -> np.ndarray:
        return np.concatenate([f.val_idx for f in self.folds])

    def _run_baselines_and_models(self) -> None:
        specs = list(iter_model_specs(self.config))
        for target in self.targets:
            base_var = self.base_map[target]
            y_full = self.df[target].to_numpy(dtype=float)

            # ---- baselines (scored on pooled OOF rows) ---------------- #
            base_pred_df = self.baseline_preds.predictions(base_var)
            oof_idx = self._oof_indices()
            y_oof = y_full[oof_idx]
            naive_r2 = None
            for bname in BASELINE_NAMES:
                pred = base_pred_df[bname].to_numpy(dtype=float)[oof_idx]
                m = self._masked_metrics(y_oof, pred, n_features=1)
                self.out.results.append(self._row(target, bname, "baseline", m,
                                                   stability=None, beats=None))
                if bname == "naive_prev_quarter":
                    naive_r2 = m["r2"]

            best_baseline_rmse = min(
                r["rmse"] for r in self.out.results
                if r["target"] == target and r["group"] == "baseline"
            )

            # ---- ML models ------------------------------------------- #
            for spec in specs:
                self._evaluate_model(target, base_var, spec, y_full,
                                     best_baseline_rmse, naive_r2)

    def _evaluate_model(self, target, base_var, spec: ModelSpec, y_full,
                        best_baseline_rmse, naive_r2) -> None:
        fold_rmse, fold_r2 = [], []
        oof_pred = np.full(len(self.df), np.nan)
        # Keep one fold's train/val for the leakage permutation test.
        last_split = None

        for f in self.folds:
            pipe = build_pipeline(spec, self.numeric, self.categorical, self.config, self.seed)
            Xtr, ytr = self.X_all.iloc[f.train_idx], y_full[f.train_idx]
            Xva, yva = self.X_all.iloc[f.val_idx], y_full[f.val_idx]
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xva)
            oof_pred[f.val_idx] = pred
            p = pipe.named_steps["model"].n_features_in_
            m = compute_metrics(yva, pred, n_features=p)
            fold_rmse.append(m["rmse"])
            fold_r2.append(m["r2"])
            last_split = (pipe, Xtr, ytr, Xva, yva)

        oof_idx = self._oof_indices()
        y_oof = y_full[oof_idx]
        pred_oof = oof_pred[oof_idx]
        p = last_split[0].named_steps["model"].n_features_in_
        metrics = compute_metrics(y_oof, pred_oof, n_features=p)

        stability = {
            "mean_rmse": float(np.mean(fold_rmse)),
            "std_rmse": float(np.std(fold_rmse)),
            "mean_r2": float(np.mean(fold_r2)),
            "std_r2": float(np.std(fold_r2)),
        }
        beats = {
            "beats_naive": bool(metrics["rmse"] < best_baseline_rmse),
            "best_baseline_rmse": float(best_baseline_rmse),
        }
        self.out.results.append(self._row(target, spec.name, spec.group, metrics,
                                           stability=stability, beats=beats))
        self.out.oof[(target, spec.name)] = {"y": y_oof, "pred": pred_oof}

        # ---- leakage diagnostics if suspiciously high R^2 ------------- #
        if stability["mean_r2"] > float(self.config["leakage_diagnostics"]["r2_trigger"]):
            pipe, Xtr, ytr, Xva, yva = last_split
            diag = run_leakage_diagnostics(
                target_col=target, base_var=base_var, pipe=pipe,
                X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva,
                naive_r2=naive_r2 if naive_r2 is not None else 0.0, config=self.config)
            diag.update({"target": target, "model": spec.name,
                         "mean_val_r2": round(stability["mean_r2"], 4)})
            self.out.leakage.append(diag)

        # ---- refit on all trainable rows, persist, importance --------- #
        tr_mask = self.df["has_target"].to_numpy()
        pipe_full = build_pipeline(spec, self.numeric, self.categorical, self.config, self.seed)
        pipe_full.fit(self.X_all.iloc[tr_mask], y_full[tr_mask])
        self._persist(target, spec, pipe_full, metrics, stability, beats)

        imp = extract_importance(pipe_full, spec.group)
        if imp is not None:
            self.out.importances[(target, spec.name)] = imp
            imp_path = self.out_root / "feature_importance" / target / f"{spec.name}.csv"
            imp_path.parent.mkdir(parents=True, exist_ok=True)
            imp.to_csv(imp_path, index=False)

        # ---- prediction intervals from OOF residuals (stored only) ---- #
        self._intervals(target, spec.name, y_oof, pred_oof)

        # ---- plots ---------------------------------------------------- #
        rp = self.out_root / "residual_plots" / target / f"{spec.name}.png"
        residual_diagnostics(y_oof, pred_oof, f"{target} — {spec.name} (OOF)", rp)
        pp = self.out_root / "prediction_plots" / target / f"{spec.name}.png"
        prediction_plot(y_oof, pred_oof, f"{target} — {spec.name} (OOF)", pp)

    def _run_multioutput_benchmark(self) -> None:
        """One RF predicting all six targets jointly; scored per target."""
        spec = get_benchmark_spec(self.config)
        Y = self.df[self.targets].to_numpy(dtype=float)
        oof_pred = np.full((len(self.df), len(self.targets)), np.nan)
        for f in self.folds:
            pre = build_preprocessor(spec.pre, self.numeric, self.categorical, self.config)
            est = make_estimator(spec, self.seed)
            from sklearn.pipeline import Pipeline
            pipe = Pipeline([("preprocess", clone(pre)), ("model", est)])
            pipe.fit(self.X_all.iloc[f.train_idx], Y[f.train_idx])
            oof_pred[f.val_idx] = pipe.predict(self.X_all.iloc[f.val_idx])

        oof_idx = self._oof_indices()
        for j, target in enumerate(self.targets):
            y_oof = Y[oof_idx, j]
            pred_oof = oof_pred[oof_idx, j]
            base_rmses = [r["rmse"] for r in self.out.results
                          if r["target"] == target and r["group"] == "baseline"]
            best_baseline_rmse = min(base_rmses)
            metrics = compute_metrics(y_oof, pred_oof, n_features=len(self.numeric))
            beats = {"beats_naive": bool(metrics["rmse"] < best_baseline_rmse),
                     "best_baseline_rmse": float(best_baseline_rmse)}
            self.out.results.append(self._row(target, spec.name, "benchmark", metrics,
                                              stability=None, beats=beats))
            self.out.oof[(target, spec.name)] = {"y": y_oof, "pred": pred_oof}
            rp = self.out_root / "residual_plots" / target / f"{spec.name}.png"
            residual_diagnostics(y_oof, pred_oof, f"{target} — {spec.name} (benchmark, OOF)", rp)

    # ------------------------------------------------------------------ #
    def _intervals(self, target, model, y_oof, pred_oof) -> None:
        cov = float(self.config["uncertainty"]["interval"])
        resid = pred_oof - y_oof
        lo_q = np.nanpercentile(resid, (1 - cov) / 2 * 100)
        hi_q = np.nanpercentile(resid, (1 + cov) / 2 * 100)
        lower = pred_oof - hi_q
        upper = pred_oof - lo_q
        emp = float(np.mean((y_oof >= lower) & (y_oof <= upper)))
        self.out.intervals.append({
            "target": target, "model": model, "coverage_nominal": cov,
            "resid_q_low": round(float(lo_q), 3), "resid_q_high": round(float(hi_q), 3),
            "empirical_coverage": round(emp, 4),
        })

    def _persist(self, target, spec: ModelSpec, pipe, metrics, stability, beats) -> None:
        out_dir = self.models_dir / target
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact = {
            "pipeline": pipe,
            "target": target,
            "base_variable": self.base_map[target],
            "model_name": spec.name,
            "model_group": spec.group,
            "preprocessor_family": spec.pre,
            "feature_schema": {"numeric": self.numeric, "categorical": self.categorical},
            "metrics_oof": metrics,
            "stability": stability,
            "beats_baseline": beats,
            "validation": {"scheme": self.config["validation"]["scheme"],
                           "n_folds": len(self.folds),
                           "val_quarters": [f.val_time for f in self.folds]},
            "config_snapshot": {"models": {spec.name: dict(spec.params)},
                                "seed": self.seed},
        }
        joblib.dump(artifact, out_dir / f"{spec.name}.joblib")
        with (out_dir / f"{spec.name}.metrics.json").open("w", encoding="utf-8") as fh:
            json.dump({"metrics_oof": metrics, "stability": stability, "beats": beats},
                      fh, indent=2, default=float)

    def _masked_metrics(self, y, pred, *, n_features) -> dict[str, float]:
        mask = ~np.isnan(pred)
        if mask.sum() == 0:
            from .metrics import compute_metrics as _cm
            return _cm(y, np.zeros_like(y), n_features=n_features)
        return compute_metrics(y[mask], pred[mask], n_features=n_features)

    def _row(self, target, model, group, metrics, *, stability, beats) -> dict[str, Any]:
        row = {"target": target, "model": model, "group": group}
        row.update(metrics)
        if stability:
            row.update({f"stability_{k}": v for k, v in stability.items()})
        else:
            row.update({"stability_mean_rmse": metrics["rmse"], "stability_std_rmse": np.nan,
                        "stability_mean_r2": metrics["r2"], "stability_std_r2": np.nan})
        row["beats_naive"] = beats["beats_naive"] if beats else None
        return row
