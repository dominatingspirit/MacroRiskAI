"""Phase 4 orchestrator — baseline model development, benchmarking, evaluation.

Trains independent baseline pipelines for each of the six targets using
leakage-safe walk-forward validation, benchmarks them against naive baselines,
computes full metrics + residual diagnostics + stability + feature importance +
prediction intervals, runs leakage diagnostics on suspiciously high R², persists
every model, and produces leaderboards and reports.

No hyperparameter tuning, no SHAP, no inference/deployment code.

Run:  python run_phase4.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

# The earliest walk-forward fold legitimately has all-NaN year-lag columns
# (no prior year exists yet); the imputer skips them. This is expected.
warnings.filterwarnings("ignore", message="Skipping features without any observed values")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.impute._base")

from src.models.evaluate import BaselineBenchmark
from src.models.importance import extract_importance  # noqa: F401 (kept for symmetry)
from src.models.phase4_report import (
    build_leaderboards,
    render_metrics_summary,
    render_per_target_report,
)
from src.models.plots import importance_plot
from src.utils.io import ensure_dir, load_config, resolve_path, write_text


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    config = load_config()
    print("=" * 72)
    print("MacroRisk AI — PHASE 4: Baseline Benchmarking (no tuning, no SHAP)")
    print("=" * 72)

    p4 = config["phase4"]
    df = pd.read_csv(resolve_path(p4["input_dataset"]))
    with resolve_path(p4["feature_metadata"]).open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    print(f"[load] {p4['input_dataset']} ({df.shape[0]} rows), "
          f"{len(metadata['numeric_features'])} numeric + "
          f"{len(metadata['categorical_features'])} categorical features")

    out_root = ensure_dir(resolve_path(p4["output_root"]))
    models_dir = ensure_dir(resolve_path(p4["models_dir"]))
    config["phase4"]["_targets"] = metadata["target_columns"]

    # ---- run the full benchmark -------------------------------------- #
    print(f"\n[benchmark] {len(metadata['target_columns'])} targets × models via "
          f"{config['validation']['n_folds']}-fold walk-forward ...")
    engine = BaselineBenchmark(df, metadata, config, out_root, models_dir)
    outputs = engine.run()
    print(f"[benchmark] produced {len(outputs.results)} result rows; "
          f"leakage triggers: {len(outputs.leakage)}")

    # ---- results table ----------------------------------------------- #
    results = pd.DataFrame(outputs.results)
    results.to_csv(out_root / "baseline_results.csv", index=False)

    # ---- leaderboards ------------------------------------------------ #
    per_target, overall, top_by_target = build_leaderboards(results, config)
    per_target.to_csv(out_root / "leaderboard.csv", index=False)
    overall.to_csv(out_root / "overall_leaderboard.csv", index=False)

    # ---- prediction intervals + leakage json ------------------------- #
    pd.DataFrame(outputs.intervals).to_csv(out_root / "prediction_intervals.csv", index=False)
    with (out_root / "leakage_diagnostics.json").open("w", encoding="utf-8") as fh:
        json.dump(outputs.leakage, fh, indent=2, default=float)

    # ---- reports ----------------------------------------------------- #
    summary = render_metrics_summary(results, per_target, overall, top_by_target,
                                     outputs.leakage, config)
    write_text(summary, out_root / "metrics_summary.md")

    per_dir = ensure_dir(out_root / "per_target_reports")
    for target in metadata["target_columns"]:
        rep = render_per_target_report(target, results, per_target, top_by_target,
                                       outputs.importances, outputs.leakage, config)
        write_text(rep, per_dir / f"{target}.md")
        # importance plot for the best model of this target
        best = per_target[per_target["target"] == target].iloc[0]["model"]
        imp = outputs.importances.get((target, best))
        if imp is not None:
            importance_plot(imp, f"{target} — {best} (top 20)",
                            out_root / "feature_importance" / target / f"{best}_top20.png")

    # ---- console summary --------------------------------------------- #
    print("\n=== Best model per target ===")
    for target in metadata["target_columns"]:
        g = per_target[per_target["target"] == target].iloc[0]
        print(f"  {target:24s} -> {g['model']:22s} RMSE={g['rmse']:.1f} "
              f"R2={g['r2']:.4f} beats_naive={g['beats_naive']}")

    print(f"\n[write] {out_root}/baseline_results.csv, leaderboard.csv, overall_leaderboard.csv")
    print(f"[write] {out_root}/metrics_summary.md, per_target_reports/, plots, {models_dir}/")
    if outputs.leakage:
        print(f"\n[leakage] {len(outputs.leakage)} model(s) triggered diagnostics — see "
              f"metrics_summary.md / leakage_diagnostics.json")
    print("\nPHASE 4 COMPLETE — baselines trained & benchmarked. No tuning performed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
