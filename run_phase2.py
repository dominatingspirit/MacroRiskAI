"""Phase 2 orchestrator — Synthetic Augmentation & Macro Integration.

DEVELOPMENT ONLY. Generates temporary synthetic company observations and a
temporary synthetic macroeconomic dataset, merges the macro variables onto the
augmented company panel, validates the result, and writes a training-ready
dataset. No models are trained.

Pipeline:
    1. load the pristine real master (Phase 1 output)
    2. generate synthetic peer companies (flagged)
    3. generate synthetic macro via the swappable provider (flagged temporary)
    4. merge macro onto the augmented panel by (Year, Quarter)
    5. validate identities, macro completeness, macro coherence, flags
    6. write outputs + report

Outputs:
    data/synthetic/synthetic_companies_TEMPORARY.csv
    data/synthetic/synthetic_macro_TEMPORARY.csv
    data/processed/phase2_training_dataset.csv
    reports/phase2/phase2_report.md
    reports/phase2/phase2_results.json

Run:  python run_phase2.py
"""
from __future__ import annotations

import sys

import pandas as pd

from src.data.phase2_report import render as render_phase2
from src.data.phase2_validate import Phase2Validator
from src.data.training_assembler import TrainingAssembler
from src.utils.io import (
    ensure_dir,
    load_config,
    resolve_path,
    write_json,
    write_text,
)

# Header line prepended to every temporary synthetic CSV so the file is
# self-documenting even outside this repo.
_TEMP_BANNER = (
    "# DEVELOPMENT-ONLY SYNTHETIC PLACEHOLDER — NOT REAL DATA. "
    "For pipeline validation only. Safe to delete/replace.\n"
)


def _write_csv_with_banner(df: pd.DataFrame, path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(_TEMP_BANNER)
        df.to_csv(fh, index=False)


def main() -> int:
    config = load_config()
    print("=" * 70)
    print("MacroRisk AI — PHASE 2: Synthetic Augmentation & Macro Integration")
    print("DEVELOPMENT ONLY — no models trained in this phase.")
    print("=" * 70)

    paths = config["paths"]
    root = resolve_path(paths["project_root"])

    # --- 1. load pristine real master ---------------------------------- #
    master_path = root / paths["master_dataset"]
    if not master_path.exists():
        print(f"! Master dataset not found: {master_path}. Run Phase 1 first.")
        return 1
    master = pd.read_csv(master_path)
    print(f"[load] master <- {master_path.name} ({len(master)} real rows)")

    # --- 2-4. assemble (synthetic companies + macro provider + merge) -- #
    print("\n[assemble] generating synthetic companies + macro, merging ...")
    result = TrainingAssembler(master, config).assemble()
    d = result.decisions
    print(f"  - synthetic companies: {d['synthetic_companies_generated']} "
          f"({d['synthetic_rows_generated']} rows)")
    print(f"  - macro provider: {d['macro_provider']} "
          f"(synthetic={d['macro_is_synthetic']}, {d['macro_quarters']} quarters)")
    print(f"  - training rows after merge: {d['rows_after_macro_merge']}")

    # --- 5. validate --------------------------------------------------- #
    print("\n[validate] checking identities, macro completeness & coherence ...")
    validation = Phase2Validator(result.frame, result.macro_panel, config).validate()
    for name, c in validation["checks"].items():
        print(f"  - {name}: {'PASS' if c['passed'] else 'FAIL'}")
    print(f"  => ALL CHECKS PASSED: {validation['all_checks_passed']}")

    # --- 6. write outputs ---------------------------------------------- #
    synth_dir = ensure_dir(resolve_path(paths["synthetic_dir"]))
    processed_dir = ensure_dir(resolve_path(paths["processed_dir"]))
    reports_dir = ensure_dir(resolve_path(paths["reports_phase2_dir"]))

    if len(result.synthetic_companies):
        _write_csv_with_banner(result.synthetic_companies, synth_dir / paths["synthetic_companies_file"])
    _write_csv_with_banner(result.macro_panel, synth_dir / paths["synthetic_macro_file"])

    training_path = processed_dir / paths["training_dataset"]
    result.frame.to_csv(training_path, index=False)
    print(f"\n[write] training dataset -> {training_path} "
          f"({result.frame.shape[0]} rows × {result.frame.shape[1]} cols)")

    md = render_phase2(result.decisions, result.macro_panel, validation, config)
    write_text(md, reports_dir / "phase2_report.md")
    write_json(
        {"assemble_decisions": result.decisions, "validation": validation},
        reports_dir / "phase2_results.json",
    )
    print(f"[reports] {reports_dir / 'phase2_report.md'}")

    print("\nPHASE 2 COMPLETE — synthetic data + macro merged. No models trained.")
    return 0 if validation["all_checks_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
