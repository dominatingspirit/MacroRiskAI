# MacroRisk AI — Phase 3: Feature Engineering Report

_Preprocessing + feature engineering only. No models trained, no tuning, no evaluation._

## STEP 1 — Data quality inspection
- Shape: 4680 rows × 31 columns
- Missing cells: **0**
- Full-row duplicates: **0**; duplicate entity-quarter records: **0**
- Unexpected negatives: none
- Infinite values in input: **0**
- Categorical issues: none (sectors: ['Auto', 'Banking', 'Energy', 'FMCG', 'Tech'])
- Datatype consistency: all consistent

## STEP 2 — Preprocessing policy
- **Scaling is not global.** Two serializable preprocessors are produced:
  - `tree`: Sector one-hot + numeric passthrough (**no scaling**, NaN-tolerant).
  - `linear`: Sector one-hot + median imputation + StandardScaler.
- Both are saved **unfitted** in `preprocessing_pipeline.joblib` and must be fit on the training split of each CV fold (leakage-safe).

## STEP 3 — Feature engineering
- Total engineered predictor columns: **121**
- By family:
  - seasonal: 5
  - lag: 36
  - macro_lag: 14
  - rolling: 24
  - growth: 18
  - interaction: 5
- Lags: [1, 2, 4] on 12 financial vars; macro lags [1, 4] on 7 macro vars.
- Rolling (shift(1), windows [4], stats ['mean', 'median', 'std', 'growth']) on 6 target vars — **past observations only**.
- Growth: QoQ + YoY + lagged-QoQ on 6 vars.
- Seasonal: quarter one-hot + is_Q4.
- Interactions (justified only): repo_x_borrowings, repo_x_total_assets, oil_x_sales, cpi_infl_x_sales, fx_x_sales.
- All derived within (source, Company) groups ordered by time_index; current-quarter raw values retained as valid predictors.

## STEP 4 — Target construction
- Horizon: **t+1** (one-quarter-ahead).
- Targets: target_Sales, target_Operating_Profit, target_Net_Profit, target_Borrowings, target_Total_Assets, target_CFO
- Each target is the entity's value at t+1 (NaN at each entity's last quarter).
- **Growth % is NOT a target** — derived post-prediction.
- Trainable rows (all six targets present): **4095** of 4680.

## STEP 5 — Feature selection analysis
- Features analysed: 121 → retained: **120**
- Constant (zero-variance) removed: none
- Exact-duplicate groups (kept first, dropped rest): [['Quarter_Q4', 'is_Q4']]
- Dropped in total: ['is_Q4']
- VIF ≥ 10.0: 98 features (top: [('Sales', 'inf'), ('CPI_Combined_Index', 'inf'), ('CPI_Inflation_Rate', 'inf'), ('WPI', 'inf'), ('Repo_Rate', 'inf'), ('Reverse_Repo_Rate', 'inf'), ('oil_price', 'inf'), ('exchange_rate', 'inf'), ('Quarter_Q2', 'inf'), ('Quarter_Q3', 'inf')])
- High-correlation pairs (|r| ≥ 0.95): 392 (e.g. [('Sales', 'fx_x_sales', 1.0), ('WPI', 'WPI_lag4', 1.0), ('Quarter_Q4', 'is_Q4', 1.0), ('Repo_Rate_lag1', 'Repo_Rate_lag4', 1.0), ('Sales_roll4_mean', 'Sales_roll4_median', 1.0)])
- **Policy:** High-correlation / high-VIF features are RETAINED — they carry accounting meaning; trees are robust and linear models regularize.
  Multicollinearity/VIF are reported but **retained** — not dropped for correlation alone, as they encode genuine accounting relationships.

## Conclusion
Feature engineering and preprocessing are complete and leakage-safe (see the separate leakage validation report). **No models were trained.**
