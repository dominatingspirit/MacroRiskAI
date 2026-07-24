# MacroRisk AI — Phase 3: Leakage Validation Report

**Overall: ✅ PASS**

| Check | Result | Detail |
|---|:--:|---|
| Quarterly ordering preserved | ✅ PASS | unordered entities: 0 |
| Company boundaries preserved | ✅ PASS | entities with cross-boundary bleed: 0 |
| Lag features reference history | ✅ PASS | mismatches: none |
| Rolling uses history only | ✅ PASS | mismatches: none |
| Target alignment correct | ✅ PASS | mismatched: none; non-null last-quarter targets: 0 |
| No future reference in metadata | ✅ PASS | offenders: none |

### Method
Checks are **empirical**: a random sample of lag/rolling/target columns is re-derived from the raw series and compared value-by-value (NaN positions must match). Lags must equal `group.shift(L)`, rolling stats must equal `group.shift(1).rolling(w)` (current quarter excluded), and targets must equal `group.shift(-horizon)`. Company boundaries are verified by requiring lag-1 to be NaN at each entity's first quarter, and ordering by monotone `time_index` per entity.
