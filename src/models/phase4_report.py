"""Leaderboard construction and Phase 4 reporting."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Rows that are reference baselines (not ranked as candidate models).
BASELINE_GROUP = "baseline"
BENCHMARK_GROUP = "benchmark"


def build_leaderboards(results: pd.DataFrame, config: dict[str, Any]):
    """Return (per_target_leaderboard, overall_leaderboard, top_n_by_target)."""
    top_n = int(config["leaderboard"]["top_n"])
    # Rank ML models + benchmark; exclude naive baselines from ranking.
    ranked = results[results["group"] != BASELINE_GROUP].copy()

    per_target_frames = []
    top_by_target: dict[str, list[str]] = {}
    for target, g in ranked.groupby("target"):
        g = g.copy()
        g["rank_rmse"] = g["rmse"].rank(ascending=True, method="min")
        g["rank_r2"] = g["r2"].rank(ascending=False, method="min")
        g["rank_mae"] = g["mae"].rank(ascending=True, method="min")
        stab = g["stability_std_rmse"].fillna(g["stability_std_rmse"].max())
        g["rank_stability"] = stab.rank(ascending=True, method="min")
        g["avg_rank"] = g[["rank_rmse", "rank_r2", "rank_mae", "rank_stability"]].mean(axis=1)
        g = g.sort_values(["avg_rank", "rmse"]).reset_index(drop=True)
        g["leaderboard_position"] = np.arange(1, len(g) + 1)
        per_target_frames.append(g)
        # Top-N candidate models for Phase 5 = best single-target models
        # (exclude the multi-output benchmark from promotion).
        candidates = g[g["group"] != BENCHMARK_GROUP]["model"].tolist()
        top_by_target[target] = candidates[:top_n]

    per_target = pd.concat(per_target_frames, ignore_index=True)

    # Overall leaderboard: average across targets per model.
    overall = (
        per_target.groupby("model")
        .agg(avg_rank=("avg_rank", "mean"),
             avg_rmse=("rmse", "mean"),
             avg_r2=("r2", "mean"),
             avg_mae=("mae", "mean"),
             group=("group", "first"),
             n_targets=("target", "nunique"))
        .reset_index()
        .sort_values("avg_rank")
        .reset_index(drop=True)
    )
    overall["overall_position"] = np.arange(1, len(overall) + 1)
    return per_target, overall, top_by_target


def render_metrics_summary(results, per_target, overall, top_by_target, leakage, config) -> str:
    p: list[str] = []
    p.append("# MacroRisk AI — Phase 4: Baseline Benchmarking Summary\n")
    p.append("_Baseline models only. No hyperparameter tuning, no SHAP, no deployment._\n")

    v = config["validation"]
    p.append(f"- Validation: **{v['scheme']}**, {v['n_folds']} expanding-window folds "
             f"(never shuffled; company boundaries + quarterly order preserved).")
    p.append("- Each of the six targets has its own independent Pipeline "
             "(preprocessor + estimator). Multi-output RF is a benchmark only.\n")

    p.append("## Overall leaderboard (averaged across targets)")
    p.append("| Model | Avg rank | Avg RMSE | Avg R² | Avg MAE | Group |")
    p.append("|---|--:|--:|--:|--:|---|")
    for _, r in overall.iterrows():
        p.append(f"| {r['model']} | {r['avg_rank']:.2f} | {r['avg_rmse']:,.1f} | "
                 f"{r['avg_r2']:.4f} | {r['avg_mae']:,.1f} | {r['group']} |")
    p.append("")

    p.append("## Best model per target (vs. best naive baseline)")
    p.append("| Target | Best model | RMSE | MAE | R² | Beats naive | Top-3 candidates |")
    p.append("|---|---|--:|--:|--:|:--:|---|")
    for target in config["phase4"]["_targets"]:
        g = per_target[per_target["target"] == target].iloc[0]
        beats = "✅" if g["beats_naive"] else "❌"
        p.append(f"| {target} | {g['model']} | {g['rmse']:,.1f} | {g['mae']:,.1f} | "
                 f"{g['r2']:.4f} | {beats} | {', '.join(top_by_target[target])} |")
    p.append("")

    if leakage:
        suspected = [d for d in leakage if d["verdict"]["leakage_suspected"]]
        n_trig = len(leakage)
        n_cleared = n_trig - len(suspected)
        p.append("## ⚠️ Leakage diagnostics (auto-triggered by R² > "
                 f"{config['leakage_diagnostics']['r2_trigger']})")
        p.append(f"- **{n_trig}** model/target combinations exceeded the R² trigger and were "
                 f"auto-diagnosed before acceptance.")
        p.append(f"- **{n_cleared}** cleared (target-permutation test passed **and** no "
                 f"non-target feature is degenerate → high R² reflects series persistence).")
        p.append(f"- **{len(suspected)}** with leakage suspected.")
        p.append("")
        p.append("Naive-baseline R² per target (context — a strong naive baseline confirms "
                 "persistence, not leakage):")
        naive_by_t = {}
        for d in leakage:
            naive_by_t.setdefault(d["target"], d["naive_baseline_r2"])
        p.append("| Target | Naive R² | Typical model mean R² |")
        p.append("|---|--:|--:|")
        for t, nv in naive_by_t.items():
            mr = max(d["mean_val_r2"] for d in leakage if d["target"] == t)
            p.append(f"| {t} | {nv} | {mr} |")
        p.append("")
        if suspected:
            p.append("### Combinations with leakage SUSPECTED")
            p.append("| Target | Model | Mean R² | Perm R² | Reason |")
            p.append("|---|---|--:|--:|---|")
            for d in suspected:
                p.append(f"| {d['target']} | {d['model']} | {d['mean_val_r2']} | "
                         f"{d['permutation_r2']} | {d['verdict']['reason']} |")
        else:
            p.append("_No leakage suspected: every triggered model collapsed to ~0 R² on "
                     "permuted targets, and no non-target feature is near-perfectly correlated "
                     "with the target. The high R² is explained by the persistence of financial "
                     "levels (the naive baseline is itself strong). Full detail in "
                     "`leakage_diagnostics.json`._")
        p.append("")

    p.append("## Notes")
    p.append("- Baselines (naive / seasonal-naive / historical-mean) are included in "
             "`baseline_results.csv` for every target.")
    p.append("- Prediction intervals are stored in `prediction_intervals.csv` (not used for ranking).")
    p.append("- Models are **not** eliminated for being slower.")
    p.append("")
    return "\n".join(p)


def render_per_target_report(target, results, per_target, top_by_target,
                             importances, leakage, config) -> str:
    p: list[str] = []
    p.append(f"# Phase 4 — Target report: `{target}`\n")
    board = per_target[per_target["target"] == target]
    base = results[(results["target"] == target) & (results["group"] == "baseline")]

    p.append("## Leaderboard")
    p.append("| # | Model | Group | RMSE | MAE | R² | Adj R² | MAPE | SMAPE | Stab σ(RMSE) | Beats naive |")
    p.append("|--:|---|---|--:|--:|--:|--:|--:|--:|--:|:--:|")
    for _, r in board.iterrows():
        beats = "✅" if r["beats_naive"] else "❌"
        p.append(f"| {int(r['leaderboard_position'])} | {r['model']} | {r['group']} | "
                 f"{r['rmse']:,.1f} | {r['mae']:,.1f} | {r['r2']:.4f} | {r['adjusted_r2']:.4f} | "
                 f"{r['mape']:.2f} | {r['smape']:.2f} | {r['stability_std_rmse'] if pd.notna(r['stability_std_rmse']) else float('nan'):,.1f} | {beats} |")
    p.append("")

    p.append("## Baselines")
    p.append("| Baseline | RMSE | MAE | R² |")
    p.append("|---|--:|--:|--:|")
    for _, r in base.iterrows():
        p.append(f"| {r['model']} | {r['rmse']:,.1f} | {r['mae']:,.1f} | {r['r2']:.4f} |")
    p.append("")

    p.append(f"## Top-3 candidates for Phase 5: {', '.join(top_by_target[target])}\n")

    best = board.iloc[0]["model"]
    imp = importances.get((target, best))
    if imp is not None:
        p.append(f"## Top features — best model (`{best}`)")
        p.append("| Feature | Importance |")
        p.append("|---|--:|")
        for _, r in imp.head(15).iterrows():
            p.append(f"| {r['feature']} | {r['importance']:.5f} |")
        p.append("")

    tl = [d for d in leakage if d["target"] == target]
    if tl:
        p.append("## Leakage diagnostics")
        for d in tl:
            v_ = d["verdict"]
            p.append(f"- `{d['model']}` (mean R²={d['mean_val_r2']}): "
                     f"permutation R²={d['permutation_r2']} "
                     f"(passed={d['permutation_test_passed']}); "
                     f"leakage_suspected={v_['leakage_suspected']} — {v_['reason']}")
        p.append("")
    p.append("Diagnostic plots: `residual_plots/`, `prediction_plots/`, `feature_importance/`.\n")
    return "\n".join(p)
