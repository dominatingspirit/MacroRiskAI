# MacroRisk AI — ML Backend

Machine-learning backend that predicts future quarterly financial performance
of companies from historical quarterly financials. Backend only — no frontend,
API, dashboard, deployment, or database.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Data discovery & canonical dataset | ✅ complete |
| 2+ | Preprocessing, features, targets, models, evaluation | not started |

## Phase 1 — Data Discovery & Canonical Dataset

Turns the two raw quarterly datasets in the project root into one validated,
canonical `master_quarterly_dataset.csv`.

### Run

```bash
pip install -r requirements.txt
python run_phase1.py
```

### What it does

1. **Loads** both datasets with automatic CSV/Excel detection and normalizes
   headers to a canonical schema (strips unit suffixes like `(INR Cr)`).
2. **Audits** each dataset independently — observations, entities, coverage,
   dtypes, missing values, duplicates, descriptive stats, outliers,
   distributions, correlation, multicollinearity (VIF), accounting-identity
   consistency, quarterly continuity, and panel balance.
3. **Compares** the datasets, classifies their relationship
   (complementary / overlapping / partially overlapping), quantifies value
   conflicts, and **recommends the safest merge strategy** (never blindly stacks).
4. **Builds** the canonical master dataset using the configured strategy,
   preserving identities, sectors, chronology, and accounting relationships.
5. **Validates** the master: no duplicate company-quarter records (on the
   effective panel key), correct ordering, intact accounting identities,
   correct dtypes, and no obvious leakage.

### Key findings

- Both datasets: **80 obs each** — a balanced panel of **10 companies × 8
  quarters (Q1 2024–Q4 2025)**, 5 sectors, zero missing values.
- The datasets are **overlapping**: identical company-quarter keys, but **100%
  of shared rows carry conflicting values** (different value scenarios for the
  same entity/period). → merged via `source_tagged_pool` (provenance-tagged),
  yielding **160 rows** with effective key `(source, Company, Year, Quarter)`.
- **Accounting identities:** `Operating Profit = Sales − Expenses` and
  `Net Cash Flow = CFO + CFI + CFF` hold in both. The balance-sheet identity
  `Total Assets = Equity + Total Liabilities` holds in `mock` (100%) but is
  **broken in `real` (5%)** — a data-quality flag carried forward, never "fixed".
- **No macroeconomic columns** exist in the inputs (company financials only).

### Outputs

- `master_quarterly_dataset.csv` — canonical dataset (project root)
- `data/interim/master_quarterly_dataset.csv` — working copy
- `reports/phase1/phase1_audit_report.md` — full human-readable audit
- `reports/phase1/phase1_results.json` — machine-readable results

## Configuration

All behavior is driven by `config/config.yaml` (dataset locations, canonical
schema, accounting identities, merge strategy). Nothing about the data is
hardcoded in source.

## Layout

```
config/            config.yaml (single source of truth)
src/
  utils/io.py      config + IO helpers
  data/
    loader.py      format detection + schema normalization
    auditor.py     STEP 1 — per-dataset audit
    comparator.py  STEP 2 — comparison + merge recommendation
    master_builder.py  STEP 3 — canonical dataset construction
    validator.py   STEP 4 — master validation
    report.py      Markdown report rendering
run_phase1.py      Phase 1 orchestrator
data/, reports/    generated artifacts
```
