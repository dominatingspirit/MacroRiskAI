"""Phase 3 orchestrator — Preprocessing, Feature Engineering, Targets.

Reads the single source of truth
    data/processed/updated_phase2_training_dataset.csv
and produces a leakage-safe, model-ready feature space plus reusable
preprocessing artifacts. Trains NO models, performs NO tuning or evaluation.

Outputs:
    data/processed/processed_training_dataset.csv
    artifacts/preprocessing_pipeline.joblib
    artifacts/feature_metadata.json
    reports/phase3/feature_engineering_report.md
    reports/phase3/leakage_validation_report.md
    reports/phase3/quality_report.json        (STEP 1 machine-readable)
    reports/phase3/feature_selection_report.md (STEP 5 standalone)

Run:  python run_phase3.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone

import joblib
import pandas as pd

from src.features.engineering import FeatureEngineer
from src.features.leakage import LeakageValidator
from src.features.phase3_report import render_feature_report, render_leakage_report
from src.features.preprocessor import build_all_preprocessors
from src.features.quality import DataQualityInspector
from src.features.selection import FeatureSelector
from src.utils.io import ensure_dir, load_config, resolve_path, write_json, write_text


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    config = load_config()
    print("=" * 72)
    print("MacroRisk AI — PHASE 3: Preprocessing, Feature Engineering, Targets")
    print("No models trained in this phase.")
    print("=" * 72)

    p3 = config["paths_phase3"]
    input_path = resolve_path(config["phase3"]["input_dataset"])
    if not input_path.exists():
        print(f"! Input not found: {input_path}. Run Phase 2 first.")
        return 1
    df = pd.read_csv(input_path)
    print(f"[load] {input_path.name} ({df.shape[0]} rows × {df.shape[1]} cols)")

    # --- STEP 1: quality ---------------------------------------------- #
    print("\n[STEP 1] Data quality inspection ...")
    quality = DataQualityInspector(df, config).inspect()
    print(f"  - missing cells: {quality['missing_values']['total_missing_cells']}; "
          f"dupes: {quality['duplicates']['full_row_duplicates']}; "
          f"infinities: {quality['infinite_values']['total']}")

    # --- STEP 3 & 4: feature engineering + targets -------------------- #
    print("\n[STEP 3/4] Feature engineering + target construction ...")
    eng = FeatureEngineer(df, config).build()
    frame = eng.frame
    trainable = int(frame["has_target"].sum())
    print(f"  - engineered predictors: {len(eng.feature_columns)}; targets: {len(eng.target_columns)}")
    print(f"  - trainable rows (all targets present): {trainable} of {len(frame)}")

    # --- STEP 5: feature selection ------------------------------------ #
    print("\n[STEP 5] Feature selection analysis ...")
    selection = FeatureSelector(frame, eng.feature_columns, config).analyse()
    final_features = selection["retained_features"]
    print(f"  - features {selection['n_features_in']} -> {selection['n_features_out']} "
          f"(dropped: {selection['dropped_features'] or 'none'})")

    # --- STEP 6: leakage validation ----------------------------------- #
    print("\n[STEP 6] Leakage validation ...")
    leakage = LeakageValidator(eng, config).validate()
    for name, c in leakage["checks"].items():
        print(f"  - {name}: {'PASS' if c['passed'] else 'FAIL'}")
    print(f"  => ALL LEAKAGE CHECKS PASSED: {leakage['all_checks_passed']}")

    # --- STEP 2 & 7: preprocessors + artifacts ------------------------ #
    print("\n[STEP 2/7] Building preprocessors + writing artifacts ...")
    categorical = [c for c in config["features"]["categorical_features"] if c in frame.columns]
    numeric_features = [c for c in final_features if c not in categorical]
    bundle = build_all_preprocessors(numeric_features, categorical, config)

    # Assemble the processed dataset: identifiers + retained features + targets.
    identifiers = [c for c in config["features"]["identifier_columns"] if c in frame.columns]
    keep_cols = (
        identifiers
        + categorical
        + numeric_features
        + eng.target_columns
        + ["has_target"]
    )
    keep_cols = list(dict.fromkeys(keep_cols))  # dedupe, preserve order
    processed = frame[keep_cols].copy()

    processed_path = resolve_path(p3["processed_dataset"])
    ensure_dir(processed_path.parent)
    processed.to_csv(processed_path, index=False)

    pipe_path = resolve_path(p3["pipeline_file"])
    ensure_dir(pipe_path.parent)
    joblib.dump(bundle, pipe_path)

    # feature_metadata.json — the contract every model will consume.
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_dataset": str(input_path.name),
        "processed_dataset": processed_path.name,
        "n_rows": int(len(processed)),
        "n_trainable_rows": trainable,
        "entity_group": config["features"]["entity_group"],
        "order_by": config["features"]["order_by"],
        "identifier_columns": identifiers,
        "categorical_features": categorical,
        "numeric_features": numeric_features,
        "all_feature_columns": categorical + numeric_features,
        "target_columns": eng.target_columns,
        "target_horizon": config["targets"]["horizon"],
        "target_base_map": {f"{config['targets']['prefix']}{v}".replace(" ", "_"): v
                            for v in config["targets"]["variables"]},
        "has_target_flag": "has_target",
        "preprocessing": {
            "families": list(config["preprocessing"]["model_families"].keys()),
            "scaling_is_global": False,
            "tree_family": "unscaled, NaN-tolerant",
            "linear_family": "median-imputed + StandardScaler",
            "fit_instructions": bundle["fit_instructions"],
        },
        "feature_provenance": eng.provenance,
        "dropped_features": selection["dropped_features"],
        "leakage_validation_passed": leakage["all_checks_passed"],
    }
    meta_path = resolve_path(p3["feature_metadata"])
    write_json(metadata, meta_path)

    # Reports.
    reports_dir = ensure_dir(resolve_path(p3["reports_dir"]))
    eng_summary = {
        "n_features": len(eng.feature_columns),
        "group_counts": {g: len(cols) for g, cols in eng.engineered_groups.items()},
        "target_columns": eng.target_columns,
        "trainable_rows": trainable,
        "total_rows": int(len(frame)),
    }
    write_text(render_feature_report(quality, eng_summary, selection, config),
               reports_dir / "feature_engineering_report.md")
    write_text(render_leakage_report(leakage),
               reports_dir / "leakage_validation_report.md")
    write_json(quality, reports_dir / "quality_report.json")
    write_json({"feature_selection": selection},
               reports_dir / "feature_selection_results.json")

    print(f"\n[write] {processed_path}  ({processed.shape[0]} rows × {processed.shape[1]} cols)")
    print(f"[write] {pipe_path}")
    print(f"[write] {meta_path}")
    print(f"[write] {reports_dir / 'feature_engineering_report.md'}")
    print(f"[write] {reports_dir / 'leakage_validation_report.md'}")

    ok = leakage["all_checks_passed"]
    print(f"\nPHASE 3 COMPLETE — leakage-safe={ok}. No models trained.")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
