"""Render Phase 3 reports: feature engineering (incl. quality + selection) and
leakage validation."""
from __future__ import annotations

from typing import Any


def _ok(v: bool) -> str:
    return "✅ PASS" if v else "❌ FAIL"


# ---------------------------------------------------------------------- #
def render_feature_report(
    quality: dict[str, Any],
    eng_summary: dict[str, Any],
    selection: dict[str, Any],
    config: dict[str, Any],
) -> str:
    p: list[str] = []
    p.append("# MacroRisk AI — Phase 3: Feature Engineering Report\n")
    p.append("_Preprocessing + feature engineering only. No models trained, no tuning, "
             "no evaluation._\n")

    # STEP 1 — quality
    p.append("## STEP 1 — Data quality inspection")
    p.append(f"- Shape: {quality['shape']['rows']} rows × {quality['shape']['columns']} columns")
    mv = quality["missing_values"]
    p.append(f"- Missing cells: **{mv['total_missing_cells']}**"
             + (f" (columns: {list(mv['columns_with_missing'])[:8]}{'…' if len(mv['columns_with_missing'])>8 else ''})"
                if mv["columns_with_missing"] else ""))
    du = quality["duplicates"]
    p.append(f"- Full-row duplicates: **{du['full_row_duplicates']}**; "
             f"duplicate entity-quarter records: **{du['duplicate_entity_quarter_records']}**")
    inv = quality["invalid_numeric"]
    p.append(f"- Unexpected negatives: {inv['unexpected_negatives'] or 'none'}")
    inf = quality["infinite_values"]
    p.append(f"- Infinite values in input: **{inf['total']}**")
    cc = quality["categorical_consistency"]
    p.append(f"- Categorical issues: {cc['issues'] or 'none'} "
             f"(sectors: {cc['unique_values'].get('Sector')})")
    dt = quality["datatype_consistency"]
    p.append(f"- Datatype consistency: {'all consistent' if dt['all_consistent'] else dt['wrong_dtype_columns']}")
    p.append("")

    # STEP 2 — preprocessing policy
    p.append("## STEP 2 — Preprocessing policy")
    p.append("- **Scaling is not global.** Two serializable preprocessors are produced:")
    p.append("  - `tree`: Sector one-hot + numeric passthrough (**no scaling**, NaN-tolerant).")
    p.append("  - `linear`: Sector one-hot + median imputation + StandardScaler.")
    p.append("- Both are saved **unfitted** in `preprocessing_pipeline.joblib` and must be fit "
             "on the training split of each CV fold (leakage-safe).")
    p.append("")

    # STEP 3 — feature engineering
    p.append("## STEP 3 — Feature engineering")
    groups = eng_summary["group_counts"]
    p.append(f"- Total engineered predictor columns: **{eng_summary['n_features']}**")
    p.append("- By family:")
    for grp, n in groups.items():
        p.append(f"  - {grp}: {n}")
    fcfg = config["features"]
    p.append(f"- Lags: {fcfg['lags']} on {len(fcfg['lag_vars'])} financial vars; "
             f"macro lags {fcfg['macro_lags']} on {len(fcfg['macro_vars'])} macro vars.")
    rc = fcfg["rolling"]
    p.append(f"- Rolling (shift(1), windows {rc['windows']}, stats {rc['stats']}) on "
             f"{len(rc['vars'])} target vars — **past observations only**.")
    p.append(f"- Growth: QoQ + YoY + lagged-QoQ on {len(fcfg['growth']['vars'])} vars.")
    p.append(f"- Seasonal: quarter one-hot + is_Q4.")
    p.append(f"- Interactions (justified only): "
             f"{', '.join(s['name'] for s in fcfg['interactions'])}.")
    p.append("- All derived within (source, Company) groups ordered by time_index; "
             "current-quarter raw values retained as valid predictors.")
    p.append("")

    # STEP 4 — targets
    p.append("## STEP 4 — Target construction")
    p.append(f"- Horizon: **t+{config['targets']['horizon']}** (one-quarter-ahead).")
    p.append(f"- Targets: {', '.join(eng_summary['target_columns'])}")
    p.append("- Each target is the entity's value at t+1 (NaN at each entity's last quarter).")
    p.append("- **Growth % is NOT a target** — derived post-prediction.")
    p.append(f"- Trainable rows (all six targets present): **{eng_summary['trainable_rows']}** "
             f"of {eng_summary['total_rows']}.")
    p.append("")

    # STEP 5 — feature selection
    p.append("## STEP 5 — Feature selection analysis")
    p.append(f"- Features analysed: {selection['n_features_in']} → retained: "
             f"**{selection['n_features_out']}**")
    p.append(f"- Constant (zero-variance) removed: {selection['constant_features'] or 'none'}")
    dup_groups = selection["exact_duplicate_groups"]
    if dup_groups:
        p.append(f"- Exact-duplicate groups (kept first, dropped rest): {dup_groups}")
    else:
        p.append("- Exact-duplicate columns: none")
    p.append(f"- Dropped in total: {selection['dropped_features'] or 'none'}")
    vif = selection["vif"]
    if "severe_features" in vif:
        top = list(vif["severe_features"].items())[:10]
        p.append(f"- VIF ≥ {vif['threshold']}: {vif['n_features_above_threshold']} features "
                 f"(top: {top})")
    hc = selection["high_correlation_pairs"]
    p.append(f"- High-correlation pairs (|r| ≥ {config['feature_selection']['high_corr_threshold']}): "
             f"{len(hc)} (e.g. {[ (x['a'],x['b'],x['abs_r']) for x in hc[:5]]})")
    p.append(f"- **Policy:** {selection['policy']['note']}")
    p.append("  Multicollinearity/VIF are reported but **retained** — not dropped for "
             "correlation alone, as they encode genuine accounting relationships.")
    p.append("")

    p.append("## Conclusion")
    p.append("Feature engineering and preprocessing are complete and leakage-safe "
             "(see the separate leakage validation report). **No models were trained.**")
    p.append("")
    return "\n".join(p)


# ---------------------------------------------------------------------- #
def render_leakage_report(leakage: dict[str, Any]) -> str:
    p: list[str] = []
    p.append("# MacroRisk AI — Phase 3: Leakage Validation Report\n")
    p.append(f"**Overall: {_ok(leakage['all_checks_passed'])}**\n")
    c = leakage["checks"]
    p.append("| Check | Result | Detail |")
    p.append("|---|:--:|---|")
    p.append(f"| Quarterly ordering preserved | {_ok(c['quarterly_ordering_preserved']['passed'])} | "
             f"unordered entities: {c['quarterly_ordering_preserved']['n_unordered']} |")
    p.append(f"| Company boundaries preserved | {_ok(c['company_boundaries_preserved']['passed'])} | "
             f"entities with cross-boundary bleed: {c['company_boundaries_preserved'].get('entities_with_bleed', 0)} |")
    p.append(f"| Lag features reference history | {_ok(c['lag_features_reference_history']['passed'])} | "
             f"mismatches: {c['lag_features_reference_history']['mismatches'] or 'none'} |")
    p.append(f"| Rolling uses history only | {_ok(c['rolling_uses_history_only']['passed'])} | "
             f"mismatches: {c['rolling_uses_history_only']['mismatches'] or 'none'} |")
    p.append(f"| Target alignment correct | {_ok(c['target_alignment_correct']['passed'])} | "
             f"mismatched: {c['target_alignment_correct']['mismatched_targets'] or 'none'}; "
             f"non-null last-quarter targets: {c['target_alignment_correct']['entities_with_nonnull_last_target']} |")
    p.append(f"| No future reference in metadata | {_ok(c['no_future_reference_in_metadata']['passed'])} | "
             f"offenders: {c['no_future_reference_in_metadata']['future_referencing_features'] or 'none'} |")
    p.append("")
    p.append("### Method")
    p.append("Checks are **empirical**: a random sample of lag/rolling/target columns is "
             "re-derived from the raw series and compared value-by-value (NaN positions must "
             "match). Lags must equal `group.shift(L)`, rolling stats must equal "
             "`group.shift(1).rolling(w)` (current quarter excluded), and targets must equal "
             "`group.shift(-horizon)`. Company boundaries are verified by requiring lag-1 to be "
             "NaN at each entity's first quarter, and ordering by monotone `time_index` per entity.")
    p.append("")
    return "\n".join(p)
