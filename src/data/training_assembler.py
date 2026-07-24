"""Assemble the Phase 2 training dataset.

Combines three ingredients into one training-ready quarterly panel:

1. the pristine real master (mock + real, from Phase 1),
2. synthetic peer companies (development-only augmentation),
3. macroeconomic variables from the configured (swappable) macro provider,

joined on (Year, Quarter). The real master file is never modified; the result
is written to a separate, clearly-flagged training dataset.

Flags added
-----------
``is_synthetic_company``  : True for generated company rows, False for real.
``macro_is_synthetic``    : True while the macro provider is the placeholder.
``macro_source``          : provenance label for the macro block.

Because macro is attached purely by (Year, Quarter) through the provider
interface, swapping in real macro data later requires no change here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..synthetic.company_generator import SyntheticCompanyGenerator
from ..synthetic.macro_provider import get_macro_provider


@dataclass
class AssembleResult:
    frame: pd.DataFrame
    macro_panel: pd.DataFrame
    synthetic_companies: pd.DataFrame
    decisions: dict[str, Any] = field(default_factory=dict)


class TrainingAssembler:
    def __init__(
        self,
        master: pd.DataFrame,
        config: dict[str, Any],
        synthetic_companies: pd.DataFrame | None = None,
    ):
        self.master = master.copy()
        self.config = config
        self.schema = config["schema"]
        self.macro_cfg = config["macro"]
        self.join_key = self.macro_cfg["join_key"]
        # Optionally inject a pre-generated synthetic panel (e.g. the copula
        # generator's output). If None, fall back to the legacy generator.
        self.synthetic_companies = synthetic_companies

    def assemble(self) -> AssembleResult:
        decisions: dict[str, Any] = {}

        # --- 1. real rows (flagged) ------------------------------------- #
        real = self.master.copy()
        real["is_synthetic_company"] = False

        # --- 2. synthetic company rows ---------------------------------- #
        if self.synthetic_companies is not None:
            synth = self.synthetic_companies.copy()
        else:
            synth = SyntheticCompanyGenerator(self.master, self.config).generate()
        if len(synth):
            synth["is_synthetic_company"] = True
        decisions["synthetic_companies_generated"] = int(
            synth["Company"].nunique() if len(synth) else 0
        )
        decisions["synthetic_rows_generated"] = int(len(synth))

        # Align columns and concatenate.
        combined = pd.concat([real, synth], ignore_index=True)
        combined = combined.sort_values(["is_synthetic_company", "source", "Company", "time_index"]) \
                           .reset_index(drop=True)
        decisions["company_rows_total"] = int(len(combined))

        # --- 3. macro via provider (swappable) -------------------------- #
        provider = get_macro_provider(self.config)
        quarters = (
            combined[["Year", "Quarter", "time_index"]]
            .drop_duplicates("time_index")
            .sort_values("time_index")
            .reset_index(drop=True)
        )
        macro_panel = provider.get_macro_panel(quarters)
        decisions["macro_provider"] = self.macro_cfg["provider"]
        decisions["macro_is_synthetic"] = bool(macro_panel[
            self.macro_cfg["provenance"]["synthetic_flag_column"]
        ].iloc[0])
        decisions["macro_quarters"] = int(len(macro_panel))

        # --- merge macro onto the company panel ------------------------- #
        merged = combined.merge(macro_panel, on=self.join_key, how="left", validate="many_to_one")
        decisions["rows_after_macro_merge"] = int(len(merged))
        decisions["final_columns"] = list(merged.columns)

        return AssembleResult(
            frame=merged,
            macro_panel=macro_panel,
            synthetic_companies=synth,
            decisions=decisions,
        )
