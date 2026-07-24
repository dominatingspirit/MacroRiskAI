"""Phase 2 (revised) — large-scale synthetic financial dataset with fidelity gates.

DEVELOPMENT ONLY. Replaces the small sector-simulation dataset with a
copula-based generator that produces ~4,000-5,000 statistically plausible new
company-quarter records, validates them against the real reference across the
full fidelity battery, and automatically refines until acceptance.

Outputs (per the request):
    synthetic_financial_dataset.csv            (synthetic company-quarters)
    data/processed/updated_phase2_training_dataset.csv   (real + synthetic + macro)
    reports/phase2/updated_validation_report.md
    reports/phase2/figures/pca_projection.png
    reports/phase2/updated_validation_results.json

No preprocessing or model training is performed.

Run:  python run_synthetic_v2.py
"""
from __future__ import annotations

import sys

import pandas as pd

from src.data.phase2_validate import Phase2Validator
from src.data.training_assembler import TrainingAssembler
from src.synthetic.copula_generator import CopulaFinancialGenerator
from src.synthetic.fidelity import FidelityValidator
from src.synthetic.fidelity_report import render as render_fidelity
from src.utils.io import (
    ensure_dir,
    load_config,
    resolve_path,
    write_json,
    write_text,
)

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
    # Ensure console can handle any non-ASCII in status output on Windows.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    config = load_config()
    print("=" * 72)
    print("MacroRisk AI — PHASE 2 (REVISED): Large-scale synthetic financial data")
    print("DEVELOPMENT ONLY — no models trained in this phase.")
    print("=" * 72)

    paths = config["paths"]
    root = resolve_path(paths["project_root"])
    reports_dir = ensure_dir(resolve_path(paths["reports_phase2_dir"]))
    figures_dir = ensure_dir(reports_dir / "figures")

    # --- load real reference (all non-synthetic observations) ---------- #
    master_path = root / paths["master_dataset"]
    if not master_path.exists():
        print(f"! Master dataset not found: {master_path}. Run Phase 1 first.")
        return 1
    real = pd.read_csv(master_path)
    print(f"[load] real reference <- {master_path.name} ({len(real)} rows)")

    # --- fit copula generator ------------------------------------------ #
    print("\n[fit] fitting sector-conditional Gaussian copula on economic drivers ...")
    gen = CopulaFinancialGenerator(real, config).fit()
    per_sector = gen.companies_per_sector()
    print(f"  - sectors: {len(gen.sector_models)}, companies/sector: {per_sector}, "
          f"quarters: {gen.n_quarters}")

    # --- refine loop: generate -> validate -> shrink noise ------------- #
    vcfg = config["synthetic"]["validation"]
    tcfg = config["synthetic"]["temporal"]
    max_iter = int(vcfg["max_refine_iterations"])
    noise = float(tcfg["noise_scale_init"])
    floor = float(tcfg["noise_scale_floor"])
    decay = float(tcfg["noise_scale_decay"])

    history: list[dict] = []
    best = None  # (score, synth_df, eval_dict, iteration, noise)
    print("\n[refine] generating and validating ...")
    for it in range(1, max_iter + 1):
        synth = gen.generate(noise_scale=noise)
        fig_path = figures_dir / "pca_projection.png"
        ev = FidelityValidator(real, synth, config).evaluate(figure_path=fig_path)
        ev_meta = {"iteration": it, "noise_scale": noise,
                   "summary": ev["summary"], "acceptance": ev["acceptance"]}
        history.append(ev_meta)
        s = ev["summary"]
        acc = ev["acceptance"]
        print(f"  iter {it}: noise={noise:.3f} medKS={s['median_ks']:.3f} "
              f"maxKS={s['max_ks']:.3f} W/std={s['mean_wasserstein_std']:.3f} "
              f"corrMAE={s['corr_mae']:.3f} score={acc['score']:.3f} "
              f"accepted={acc['accepted']}")

        if best is None or acc["score"] < best[0]:
            best = (acc["score"], synth.copy(), ev, it, noise)
        if acc["accepted"]:
            break
        noise = max(floor, noise * decay)

    _, best_synth, best_eval, best_it, best_noise = best
    print(f"\n[refine] selected iteration {best_it} (noise={best_noise:.3f}, "
          f"accepted={best_eval['acceptance']['accepted']})")

    # --- assemble updated training dataset (real + synth + macro) ------ #
    print("\n[assemble] merging macro onto real + synthetic panel ...")
    result = TrainingAssembler(real, config, synthetic_companies=best_synth).assemble()
    training = result.frame

    # Phase-2 structural validation still applies (identities/macro/flags).
    struct = Phase2Validator(training, result.macro_panel, config).validate()
    print(f"  - structural validation passed: {struct['all_checks_passed']}")

    # --- write outputs ------------------------------------------------- #
    # synthetic_financial_dataset.csv at project root (as requested)
    _write_csv_with_banner(best_synth, root / "synthetic_financial_dataset.csv")
    # keep a copy in data/synthetic too
    synth_dir = ensure_dir(resolve_path(paths["synthetic_dir"]))
    _write_csv_with_banner(best_synth, synth_dir / "synthetic_financial_dataset.csv")

    training_path = resolve_path(paths["processed_dir"]) / "updated_phase2_training_dataset.csv"
    ensure_dir(training_path.parent)
    training.to_csv(training_path, index=False)

    gen_meta = {
        "companies_per_sector": per_sector,
        "n_sectors": len(gen.sector_models),
        "n_companies": per_sector * len(gen.sector_models),
        "n_quarters": gen.n_quarters,
        "n_synth_rows": int(len(best_synth)),
        "selected_iteration": best_it,
        "selected_noise_scale": best_noise,
    }
    md = render_fidelity(config["synthetic"]["method"], history, best_eval, gen_meta, config)
    write_text(md, reports_dir / "updated_validation_report.md")
    write_json(
        {
            "generation": gen_meta,
            "refine_history": history,
            "final_evaluation": best_eval,
            "structural_validation": struct,
        },
        reports_dir / "updated_validation_results.json",
    )

    print(f"\n[write] synthetic_financial_dataset.csv ({len(best_synth)} rows)")
    print(f"[write] {training_path} ({training.shape[0]} rows × {training.shape[1]} cols)")
    print(f"[write] {reports_dir / 'updated_validation_report.md'}")
    print(f"[write] {figures_dir / 'pca_projection.png'}")

    accepted = best_eval["acceptance"]["accepted"] and struct["all_checks_passed"]
    print(f"\nPHASE 2 (REVISED) COMPLETE — synthetic accepted={best_eval['acceptance']['accepted']}, "
          f"structural={struct['all_checks_passed']}. No models trained.")
    return 0 if accepted else 2


if __name__ == "__main__":
    sys.exit(main())
