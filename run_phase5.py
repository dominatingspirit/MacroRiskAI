"""Phase 5 orchestrator — Hyperparameter Optimization (Optuna).

Tunes each target's Phase-4 top-N candidate models with leakage-safe
walk-forward validation, saves all trials/params/pipelines, compares against the
Phase-4 baseline, and promotes only genuinely-improved models. Does not modify
the synthetic dataset, feature engineering, or target formulation.

Run:  python run_phase5.py
"""
from __future__ import annotations

import json
import sys
import warnings

import pandas as pd

from src.tuning.optimize import HyperparameterOptimizer
from src.tuning.phase5_report import render_per_target, render_summary
from src.utils.io import ensure_dir, load_config, resolve_path, write_json, write_text

warnings.filterwarnings("ignore", message="Skipping features without any observed values")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.impute._base")
warnings.filterwarnings("ignore", message=".*did not converge.*")


def load_candidates(leaderboard_path, top_n: int) -> dict[str, list[str]]:
    """Each target's top-N Phase-4 candidate models (excludes benchmark)."""
    lb = pd.read_csv(leaderboard_path)
    lb = lb[lb["group"] != "benchmark"]
    out: dict[str, list[str]] = {}
    for target, g in lb.groupby("target"):
        out[target] = g.sort_values("leaderboard_position").head(top_n)["model"].tolist()
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    config = load_config()
    print("=" * 72)
    print("MacroRisk AI — PHASE 5: Hyperparameter Optimization (Optuna)")
    print("Levels formulation unchanged; only Phase-4 top-N models tuned.")
    print("=" * 72)

    p5 = config["phase5"]
    df = pd.read_csv(resolve_path(p5["input_dataset"]))
    with resolve_path(p5["feature_metadata"]).open(encoding="utf-8") as fh:
        metadata = json.load(fh)
    baseline = pd.read_csv(resolve_path(p5["phase4_results"]))
    candidates = load_candidates(resolve_path(p5["phase4_leaderboard"]), int(p5["candidates_top_n"]))

    print("[candidates] tuning per target:")
    for t, ms in candidates.items():
        print(f"   {t:24s} -> {ms}")

    out_root = ensure_dir(resolve_path(p5["output_root"]))
    models_dir = ensure_dir(resolve_path(p5["models_dir"]))
    trials_dir = ensure_dir(out_root / "trials")

    opt = HyperparameterOptimizer(df, metadata, config, out_root, models_dir, candidates, baseline)
    print("\n[optimize] running Optuna studies ...")
    outputs = opt.run()

    # Save trials + before/after.
    for (target, model), tdf in outputs.trials.items():
        tdf.to_csv(trials_dir / f"{target}__{model}.csv", index=False)
    ba = pd.DataFrame([{k: v for k, v in r.items() if k != "best_params"}
                       for r in outputs.before_after])
    ba.to_csv(out_root / "before_after.csv", index=False)
    write_json(outputs.promoted, out_root / "promoted_models.json")

    # Reports.
    write_text(render_summary(outputs.before_after, outputs.promoted, candidates, config),
               out_root / "phase5_summary.md")
    per_dir = ensure_dir(out_root / "per_target_reports")
    for target in candidates:
        write_text(render_per_target(target, outputs.before_after, outputs.trials,
                                     outputs.best_params, config),
                   per_dir / f"{target}.md")

    # Console summary.
    print("\n=== Before -> After (pooled OOF RMSE) ===")
    for r in outputs.before_after:
        imp = f"{r['rmse_improvement_pct']:+.2f}%" if r["rmse_improvement_pct"] is not None else "  n/a"
        print(f"  {r['target']:24s} {r['model']:16s} {r['baseline_rmse']:>11.1f} -> "
              f"{r['tuned_rmse']:>11.1f}  ({imp})  promoted={r['promoted']}")
    n_prom = len(outputs.promoted)
    print(f"\n[promoted] {n_prom}/{len(candidates)} targets got an improved tuned model.")
    print(f"[write] {out_root}/before_after.csv, phase5_summary.md, per_target_reports/, trials/")
    print(f"[write] {models_dir}/ (tuned pipelines)")
    print("\nPHASE 5 COMPLETE — tuning done. Phase-4 artifacts untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
