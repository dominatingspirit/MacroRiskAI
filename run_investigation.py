"""Phase 4.5 — task-difficulty investigation runner.

Computes the diagnostics and writes a JSON of raw findings for interpretation.
No model tuning, no synthetic regeneration here (that decision follows the
findings).
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from src.investigation.difficulty import (
    autocorrelation_table,
    delta_distribution_table,
    diversity_analysis,
    formulation_comparison,
)
from src.utils.io import ensure_dir, load_config, resolve_path, write_json


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    config = load_config()
    root = resolve_path(config["paths"]["project_root"])

    real = pd.read_csv(root / config["paths"]["master_dataset"])
    synth = pd.read_csv(root / "synthetic_financial_dataset.csv", comment="#")
    processed = pd.read_csv(resolve_path(config["phase4"]["input_dataset"]))
    with resolve_path(config["phase4"]["feature_metadata"]).open(encoding="utf-8") as fh:
        metadata = json.load(fh)

    print("[investigate] autocorrelation ...")
    acf = autocorrelation_table(real, synth)
    print("[investigate] delta distributions ...")
    deltas = delta_distribution_table(real, synth)
    print("[investigate] diversity ...")
    diversity = diversity_analysis(real, synth)
    print("[investigate] formulation comparison (levels vs delta vs pct) ...")
    forms = formulation_comparison(processed, metadata, config)

    findings = {
        "autocorrelation": acf,
        "delta_distributions": deltas,
        "diversity": diversity,
        "formulation_comparison": forms,
    }
    out_dir = ensure_dir(resolve_path("reports/phase4_5"))
    write_json(findings, out_dir / "difficulty_findings.json")
    print(f"[write] {out_dir / 'difficulty_findings.json'}")

    # Console highlights for quick interpretation.
    print("\n--- lag-1 autocorrelation (levels) real vs synth ---")
    for c, r in acf.items():
        print(f"  {c:18s} real={r['real']['acf_lag1']}  synth={r['synth']['acf_lag1']}  "
              f"lag1_R2 real={r['real']['lag1_variance_explained']} synth={r['synth']['lag1_variance_explained']}")
    print("\n--- pct-growth std (QoQ) real vs synth (ratio) ---")
    for c, d in deltas.items():
        print(f"  {c:18s} real={d['pct_growth_std_real']} synth={d['pct_growth_std_synth']} "
              f"ratio={d['pct_growth_std_ratio_synth_over_real']} KS={d['pct_growth_ks_synth_vs_real']}")
    print("\n--- formulation R2 (synth, single split) ---")
    for c, e in forms.items():
        print(f"  {c}")
        for f, v in e.items():
            print(f"     {f:11s} model_R2={v['model_r2']:>8}  naive_R2={v['naive_r2']:>8}  "
                  f"skill={v['skill_over_naive']}")
    print("\n--- diversity ---")
    print(f"  synth entities: {diversity['n_synth_entities']}, "
          f"NN dist mean={diversity['synth_nn_distance_mean']}, "
          f"near-dup frac={diversity['synth_near_duplicate_fraction_lt_0_1']}")
    print(f"  dispersion ratio by sector: {diversity['between_company_dispersion_ratio_by_sector']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
