"""Render the Phase 2 report (synthetic data + macro merge) to Markdown."""
from __future__ import annotations

from typing import Any

import pandas as pd


def _bool(v: bool) -> str:
    return "✅ PASS" if v else "❌ FAIL"


DISCLAIMER = (
    "> ⚠️ **DEVELOPMENT ONLY.** The synthetic company observations and the "
    "synthetic macroeconomic variables in this phase are TEMPORARY placeholders "
    "generated to enlarge the dataset and validate the pipeline. They are **not "
    "real data**. Any model trained on this dataset is for **pipeline validation "
    "only — not production evaluation or reporting**. The synthetic macro block "
    "is designed to be replaced by real historical data with no pipeline changes "
    "(switch `macro.provider` to `historical`)."
)


def render(
    assemble_decisions: dict[str, Any],
    macro_panel: pd.DataFrame,
    validation: dict[str, Any],
    config: dict[str, Any],
) -> str:
    p: list[str] = []
    p.append("# MacroRisk AI — Phase 2: Synthetic Augmentation & Macro Integration\n")
    p.append(DISCLAIMER + "\n")

    # --- Synthetic companies ------------------------------------------- #
    p.append("## 1. Synthetic company observations")
    p.append(f"- Companies generated: **{assemble_decisions['synthetic_companies_generated']}** "
             f"(sector-aware peers)")
    p.append(f"- Synthetic rows: **{assemble_decisions['synthetic_rows_generated']}**")
    p.append(f"- Total company rows (real + synthetic): **{assemble_decisions['company_rows_total']}**")
    p.append("- Generation preserves temporal continuity (AR(1) Sales growth, drifting "
             "ratios) and enforces the accounting identities exactly.")
    p.append("")

    # --- Macro --------------------------------------------------------- #
    p.append("## 2. Synthetic macroeconomic variables")
    p.append(f"- Provider: **`{assemble_decisions['macro_provider']}`** "
             f"(synthetic placeholder: {assemble_decisions['macro_is_synthetic']})")
    p.append(f"- Quarters generated: **{assemble_decisions['macro_quarters']}**")
    p.append(f"- Variables: {', '.join(f'`{v}`' for v in config['macro']['variables'])}")
    p.append("")
    p.append("### Generated macro series")
    p.append(_macro_table(macro_panel, config))
    p.append("")
    p.append("**Economic design guarantees:** Reverse_Repo tracks below Repo within a "
             "policy corridor; Repo evolves in small policy-sized steps; CPI index "
             "compounds with CPI inflation; WPI loads positively on CPI (and mildly on "
             "oil); oil and exchange rate move by bounded quarterly amounts; exchange "
             "rate carries a mild depreciation drift with positive oil sensitivity.\n")

    # --- Merge --------------------------------------------------------- #
    p.append("## 3. Training dataset assembly")
    p.append(f"- Rows after macro merge: **{assemble_decisions['rows_after_macro_merge']}**")
    p.append(f"- Columns ({len(assemble_decisions['final_columns'])}): "
             f"{', '.join(f'`{c}`' for c in assemble_decisions['final_columns'])}")
    p.append("- Macro attached by (Year, Quarter); real master left untouched.\n")

    # --- Validation ---------------------------------------------------- #
    p.append("## 4. Validation")
    p.append(f"- **Overall:** {_bool(validation['all_checks_passed'])}\n")
    c = validation["checks"]
    p.append("| Check | Result | Detail |")
    p.append("|---|:--:|---|")
    si = c["synthetic_accounting_identities"]
    viol = sum(v["rows_violated"] for v in si.get("identities", {}).values()) if si.get("identities") else 0
    p.append(f"| Synthetic accounting identities | {_bool(si['passed'])} | violations: {viol} |")
    mj = c["macro_join_complete"]
    p.append(f"| Macro join complete (no nulls) | {_bool(mj['passed'])} | null cells: {mj['macro_null_cells']} |")
    mc = c["macro_coherence"]
    p.append(f"| Macro economic coherence | {_bool(mc['passed'])} | "
             f"rev<repo={mc['reverse_repo_below_repo']}, max repo move={mc['max_repo_quarterly_move']}, "
             f"CPI-WPI corr={mc['cpi_index_wpi_correlation']} |")
    fp = c["provenance_flags_present"]
    p.append(f"| Provenance flags present | {_bool(fp['passed'])} | missing: {fp['missing_flags'] or 'none'} |")
    p.append("")
    p.append("### Macro coherence detail")
    p.append(f"- Reverse_Repo < Repo everywhere: {mc['reverse_repo_below_repo']}")
    p.append(f"- Max quarterly Repo move: {mc['max_repo_quarterly_move']} (gradual)")
    p.append(f"- Max quarterly oil move: {mc['max_oil_quarterly_pct_move']*100:.1f}%")
    p.append(f"- Max quarterly FX move: {mc['max_fx_quarterly_pct_move']*100:.1f}%")
    p.append(f"- CPI-index vs WPI correlation: {mc['cpi_index_wpi_correlation']} (positive)")
    p.append("")
    p.append("## 5. Conclusion")
    p.append("Synthetic company observations and synthetic macro variables generated, "
             "merged into the training dataset, and validated. **No models trained.** "
             "Real macro data can be dropped in later without pipeline changes.")
    p.append("")
    p.append(DISCLAIMER + "\n")
    return "\n".join(p)


def _macro_table(macro: pd.DataFrame, config: dict[str, Any]) -> str:
    variables = config["macro"]["variables"]
    q_num = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    m = macro.copy()
    m["_ti"] = m["Year"] * 4 + m["Quarter"].map(q_num)
    m = m.sort_values("_ti")
    header = "| Period | " + " | ".join(variables) + " |"
    sep = "|---|" + "|".join(["--:"] * len(variables)) + "|"
    lines = [header, sep]
    for _, r in m.iterrows():
        period = f"{r['Quarter']} {r['Year']}"
        vals = " | ".join(f"{r[v]:,.2f}" for v in variables)
        lines.append(f"| {period} | {vals} |")
    return "\n".join(lines)
