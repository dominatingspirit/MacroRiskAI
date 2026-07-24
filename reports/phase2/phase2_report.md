# MacroRisk AI — Phase 2: Synthetic Augmentation & Macro Integration

> ⚠️ **DEVELOPMENT ONLY.** The synthetic company observations and the synthetic macroeconomic variables in this phase are TEMPORARY placeholders generated to enlarge the dataset and validate the pipeline. They are **not real data**. Any model trained on this dataset is for **pipeline validation only — not production evaluation or reporting**. The synthetic macro block is designed to be replaced by real historical data with no pipeline changes (switch `macro.provider` to `historical`).

## 1. Synthetic company observations
- Companies generated: **50** (sector-aware peers)
- Synthetic rows: **400**
- Total company rows (real + synthetic): **560**
- Generation preserves temporal continuity (AR(1) Sales growth, drifting ratios) and enforces the accounting identities exactly.

## 2. Synthetic macroeconomic variables
- Provider: **`synthetic`** (synthetic placeholder: True)
- Quarters generated: **8**
- Variables: `CPI_Combined_Index`, `CPI_Inflation_Rate`, `WPI`, `Repo_Rate`, `Reverse_Repo_Rate`, `oil_price`, `exchange_rate`

### Generated macro series
| Period | CPI_Combined_Index | CPI_Inflation_Rate | WPI | Repo_Rate | Reverse_Repo_Rate | oil_price | exchange_rate |
|---|--:|--:|--:|--:|--:|--:|--:|
| Q1 2024 | 172.00 | 5.00 | 152.00 | 6.50 | 5.84 | 80.00 | 83.00 |
| Q2 2024 | 174.00 | 4.66 | 153.20 | 6.55 | 5.90 | 83.91 | 83.25 |
| Q3 2024 | 176.15 | 4.94 | 154.57 | 6.45 | 5.77 | 83.39 | 83.26 |
| Q4 2024 | 178.17 | 4.58 | 155.56 | 6.55 | 5.93 | 81.96 | 83.25 |
| Q1 2025 | 180.44 | 5.10 | 156.66 | 6.65 | 6.02 | 80.81 | 83.46 |
| Q2 2025 | 182.71 | 5.04 | 157.79 | 6.45 | 5.80 | 82.53 | 83.77 |
| Q3 2025 | 184.98 | 4.95 | 159.06 | 6.30 | 5.68 | 83.74 | 84.04 |
| Q4 2025 | 187.15 | 4.70 | 160.35 | 6.30 | 5.66 | 85.12 | 84.13 |

**Economic design guarantees:** Reverse_Repo tracks below Repo within a policy corridor; Repo evolves in small policy-sized steps; CPI index compounds with CPI inflation; WPI loads positively on CPI (and mildly on oil); oil and exchange rate move by bounded quarterly amounts; exchange rate carries a mild depreciation drift with positive oil sensitivity.

## 3. Training dataset assembly
- Rows after macro merge: **560**
- Columns (31): `Company`, `Ticker`, `Sector`, `Year`, `Quarter`, `Period`, `quarter_num`, `time_index`, `source`, `Sales`, `Expenses`, `Operating Profit`, `Net Profit`, `Total Assets`, `Equity`, `Borrowings`, `Total Liabilities`, `CFO`, `CFI`, `CFF`, `Net Cash Flow`, `is_synthetic_company`, `CPI_Combined_Index`, `CPI_Inflation_Rate`, `WPI`, `Repo_Rate`, `Reverse_Repo_Rate`, `oil_price`, `exchange_rate`, `macro_source`, `macro_is_synthetic`
- Macro attached by (Year, Quarter); real master left untouched.

## 4. Validation
- **Overall:** ✅ PASS

| Check | Result | Detail |
|---|:--:|---|
| Synthetic accounting identities | ✅ PASS | violations: 0 |
| Macro join complete (no nulls) | ✅ PASS | null cells: 0 |
| Macro economic coherence | ✅ PASS | rev<repo=True, max repo move=0.2, CPI-WPI corr=0.9992 |
| Provenance flags present | ✅ PASS | missing: none |

### Macro coherence detail
- Reverse_Repo < Repo everywhere: True
- Max quarterly Repo move: 0.2 (gradual)
- Max quarterly oil move: 4.9%
- Max quarterly FX move: 0.4%
- CPI-index vs WPI correlation: 0.9992 (positive)

## 5. Conclusion
Synthetic company observations and synthetic macro variables generated, merged into the training dataset, and validated. **No models trained.** Real macro data can be dropped in later without pipeline changes.

> ⚠️ **DEVELOPMENT ONLY.** The synthetic company observations and the synthetic macroeconomic variables in this phase are TEMPORARY placeholders generated to enlarge the dataset and validate the pipeline. They are **not real data**. Any model trained on this dataset is for **pipeline validation only — not production evaluation or reporting**. The synthetic macro block is designed to be replaced by real historical data with no pipeline changes (switch `macro.provider` to `historical`).
