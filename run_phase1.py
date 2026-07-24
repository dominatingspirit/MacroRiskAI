"""Phase 1 orchestrator — Data Discovery & Canonical Dataset.

Runs the full Phase 1 pipeline end to end:

    STEP 1  audit each dataset independently
    STEP 2  compare the datasets and recommend a merge strategy
    STEP 3  build the canonical master dataset
    STEP 4  validate the master dataset

Outputs
-------
* master_quarterly_dataset.csv         (project root — the canonical dataset)
* data/interim/master_quarterly_dataset.csv  (working copy)
* reports/phase1/phase1_audit_report.md      (human-readable audit)
* reports/phase1/phase1_results.json          (machine-readable results)

The script trains no models and generates no synthetic data. Run with:

    python run_phase1.py
"""
from __future__ import annotations

import sys

from src.data.auditor import DatasetAuditor
from src.data.comparator import DatasetComparator
from src.data.loader import load_all
from src.data.master_builder import MasterDatasetBuilder
from src.data.report import render_full_report
from src.data.validator import MasterValidator
from src.utils.io import (
    ensure_dir,
    load_config,
    resolve_path,
    write_json,
    write_text,
)


def main() -> int:
    config = load_config()
    print("=" * 70)
    print("MacroRisk AI — PHASE 1: Data Discovery & Canonical Dataset")
    print("=" * 70)

    # --- Load ---------------------------------------------------------- #
    datasets = load_all(config)
    for name, ds in datasets.items():
        print(f"[load] {name:5s} <- {ds.path.name} ({ds.file_format}, {ds.n_rows} rows)")

    # --- STEP 1: audit each dataset ------------------------------------ #
    print("\n[STEP 1] Auditing datasets ...")
    audits = {}
    for name, ds in datasets.items():
        audits[name] = DatasetAuditor(ds, config).audit()
        ov = audits[name]["overview"]
        print(f"  - {name}: {ov['total_observations']} obs, "
              f"{ov['n_companies']} companies, {ov['n_sectors']} sectors, "
              f"balanced={audits[name]['panel_balance']['is_balanced']}")

    # --- STEP 2: compare ----------------------------------------------- #
    print("\n[STEP 2] Comparing datasets ...")
    names = list(datasets)
    if len(names) != 2:
        print(f"  ! Expected exactly 2 datasets, found {len(names)}: {names}")
        return 1
    comparator = DatasetComparator(datasets[names[0]], datasets[names[1]], config)
    comparison = comparator.compare()
    rec = comparison["recommended_merge_strategy"]
    print(f"  - relationship: {comparison['relationship']['label']}")
    print(f"  - recommended merge: {rec['recommended']} (configured: {rec['configured']})")

    # --- STEP 3: build master ------------------------------------------ #
    print("\n[STEP 3] Building master dataset ...")
    builder = MasterDatasetBuilder(datasets, config)
    result = builder.build()
    master = result.frame
    print(f"  - strategy: {result.strategy}")
    print(f"  - master shape: {master.shape[0]} rows × {master.shape[1]} cols")

    # persist canonical dataset at project root and interim copy
    root = resolve_path(config["paths"]["project_root"])
    master_path = root / config["paths"]["master_dataset"]
    master.to_csv(master_path, index=False)
    interim_dir = ensure_dir(resolve_path(config["paths"]["interim_dir"]))
    master.to_csv(interim_dir / config["paths"]["master_dataset"], index=False)
    print(f"  - written: {master_path}")

    # --- STEP 4: validate ---------------------------------------------- #
    print("\n[STEP 4] Validating master dataset ...")
    validation = MasterValidator(master, config, result.strategy).validate()
    for cname, c in validation["checks"].items():
        print(f"  - {cname}: {'PASS' if c['passed'] else 'FAIL'}")
    print(f"  => ALL CHECKS PASSED: {validation['all_checks_passed']}")

    # --- Reports ------------------------------------------------------- #
    reports_dir = ensure_dir(resolve_path(config["paths"]["reports_dir"]))
    md = render_full_report(audits, comparison, result.decisions, validation)
    md_path = reports_dir / "phase1_audit_report.md"
    write_text(md, md_path)

    json_path = reports_dir / "phase1_results.json"
    write_json(
        {
            "audits": audits,
            "comparison": comparison,
            "build_decisions": result.decisions,
            "validation": validation,
        },
        json_path,
    )
    print(f"\n[reports] {md_path}")
    print(f"[reports] {json_path}")

    print("\nPHASE 1 COMPLETE — no synthetic data generated, no models trained.")
    return 0 if validation["all_checks_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
