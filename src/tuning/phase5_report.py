"""Phase 5 reporting — before/after comparison and promotion summary."""
from __future__ import annotations

from typing import Any


def render_summary(before_after, promoted, candidates, config) -> str:
    p: list[str] = []
    p.append("# MacroRisk AI — Phase 5: Hyperparameter Optimization Summary\n")
    p.append("_Optuna (TPE) tuning of each target's Phase-4 top-N candidates. Same "
             "leakage-safe walk-forward folds. Objective: RMSE (MAE tie-break). Target "
             "formulation, features, and synthetic data unchanged._\n")

    tr = config["phase5"]["n_trials"]
    p.append(f"- Trial budget: linear={tr['linear']}, tree={tr['tree']}, boosting={tr['boosting']} "
             f"per (target, model).")
    p.append(f"- Early stopping (XGB/LightGBM/CatBoost): implemented; **not engaged** — no "
             f"booster was among any target's top-N candidates in Phase 4.")
    p.append("- Promotion rule: tuned pooled-OOF RMSE must beat the Phase-4 baseline RMSE by "
             f"≥ {config['phase5']['promotion']['min_rmse_improvement_frac']*100:.1f}%.\n")

    p.append("## Before → After (pooled out-of-fold, identical to Phase 4)")
    p.append("| Target | Model | Base RMSE | Tuned RMSE | ΔRMSE % | Base MAE | Tuned MAE | Base R² | Tuned R² | Promoted |")
    p.append("|---|---|--:|--:|--:|--:|--:|--:|--:|:--:|")
    for r in before_after:
        imp = f"{r['rmse_improvement_pct']:+.2f}%" if r["rmse_improvement_pct"] is not None else "—"
        p.append(f"| {r['target']} | {r['model']} | {r['baseline_rmse']:,.1f} | {r['tuned_rmse']:,.1f} | "
                 f"{imp} | {r['baseline_mae']:,.1f} | {r['tuned_mae']:,.1f} | {r['baseline_r2']:.4f} | "
                 f"{r['tuned_r2']:.4f} | {'✅' if r['promoted'] else '❌'} |")
    p.append("")

    p.append("## Promoted tuned model per target")
    p.append("| Target | Promoted model | Tuned RMSE | Tuned MAE | Notes |")
    p.append("|---|---|--:|--:|---|")
    for target in candidates:
        if target in promoted:
            b = promoted[target]
            p.append(f"| {target} | {b['model']} | {b['tuned_rmse']:,.1f} | {b['tuned_mae']:,.1f} | "
                     f"improved over baseline |")
        else:
            p.append(f"| {target} | — (baseline retained) | — | — | no tuned candidate beat baseline |")
    p.append("")

    n_prom = sum(1 for t in candidates if t in promoted)
    p.append(f"## Outcome: {n_prom} of {len(candidates)} targets received a promoted tuned model.")
    p.append("Tuned pipelines for every (target, candidate) are saved under "
             "`saved_tuned_models/` (promotion flag inside each artifact). All Optuna trials "
             "are in `reports/phase5/trials/`. Phase-4 artifacts are untouched.\n")

    p.append("### Important context")
    p.append("- As established in Phase 4.5, the **levels** formulation makes the naive "
             "previous-quarter forecast extremely strong; tuning operates within that regime. "
             "Gains here are incremental. The larger opportunity (modelling percentage growth) "
             "was deliberately **not** applied in this phase per the requirement to leave target "
             "formulation unchanged.")
    p.append("")
    return "\n".join(p)


def render_per_target(target, before_after, trials, best_params, config) -> str:
    p: list[str] = []
    p.append(f"# Phase 5 — Tuning report: `{target}`\n")
    rows = [r for r in before_after if r["target"] == target]
    p.append("## Before → After")
    p.append("| Model | Base RMSE | Tuned RMSE | ΔRMSE % | Tuned MAE | Tuned R² | Promoted |")
    p.append("|---|--:|--:|--:|--:|--:|:--:|")
    for r in rows:
        imp = f"{r['rmse_improvement_pct']:+.2f}%" if r["rmse_improvement_pct"] is not None else "—"
        p.append(f"| {r['model']} | {r['baseline_rmse']:,.1f} | {r['tuned_rmse']:,.1f} | {imp} | "
                 f"{r['tuned_mae']:,.1f} | {r['tuned_r2']:.4f} | {'✅' if r['promoted'] else '❌'} |")
    p.append("")
    for r in rows:
        p.append(f"### `{r['model']}` best params")
        p.append("```json")
        import json
        p.append(json.dumps(r["best_params"], indent=2, default=str))
        p.append("```")
        key = (target, r["model"])
        if key in trials:
            t = trials[key]
            p.append(f"- Trials: {len(t)} (best RMSE {t['rmse'].min():,.2f}). "
                     f"Full trial log: `trials/{target}__{r['model']}.csv`.")
        p.append("")
    return "\n".join(p)
