# Phase 4 — Target report: `target_Sales`

## Leaderboard
| # | Model | Group | RMSE | MAE | R² | Adj R² | MAPE | SMAPE | Stab σ(RMSE) | Beats naive |
|--:|---|---|--:|--:|--:|--:|--:|--:|--:|:--:|
| 1 | ridge | linear | 3,159.2 | 2,004.9 | 0.9984 | 0.9983 | 2.28 | 2.28 | 235.2 | ✅ |
| 2 | elastic_net | linear | 3,258.6 | 2,100.3 | 0.9983 | 0.9982 | 2.43 | 2.43 | 172.7 | ✅ |
| 3 | extra_trees | tree | 3,275.0 | 2,001.4 | 0.9983 | 0.9982 | 2.02 | 2.00 | 267.8 | ❌ |
| 4 | hist_gradient_boosting | tree | 3,485.1 | 2,061.6 | 0.9981 | 0.9979 | 2.05 | 2.04 | 184.6 | ❌ |
| 5 | random_forest | tree | 3,401.0 | 2,081.5 | 0.9982 | 0.9980 | 2.08 | 2.07 | 262.4 | ❌ |
| 6 | lightgbm | boosting | 3,561.3 | 2,118.0 | 0.9980 | 0.9978 | 2.12 | 2.11 | 211.5 | ❌ |
| 7 | xgboost | boosting | 3,659.4 | 2,191.9 | 0.9979 | 0.9977 | 2.26 | 2.25 | 269.9 | ❌ |
| 8 | lasso | linear | 4,400.0 | 3,433.3 | 0.9969 | 0.9967 | 6.18 | 6.55 | 1,212.8 | ❌ |
| 9 | decision_tree | tree | 4,558.3 | 2,810.0 | 0.9967 | 0.9965 | 2.84 | 2.82 | 381.1 | ❌ |
| 10 | catboost | boosting | 5,757.1 | 3,797.9 | 0.9948 | 0.9944 | 6.00 | 5.60 | 1,984.1 | ❌ |
| 11 | multioutput_random_forest | benchmark | 13,892.0 | 6,936.4 | 0.9695 | 0.9673 | 6.39 | 6.20 | nan | ❌ |
| 12 | linear_regression | linear | 30,423,640.4 | 23,932,377.0 | -146308.7889 | -157535.7524 | 50991.05 | 199.11 | 18,784,009.0 | ❌ |

## Baselines
| Baseline | RMSE | MAE | R² |
|---|--:|--:|--:|
| naive_prev_quarter | 3,262.6 | 2,014.3 | 0.9983 |
| seasonal_naive | 8,665.9 | 5,425.1 | 0.9881 |
| historical_mean | 7,187.9 | 4,517.1 | 0.9918 |

## Top-3 candidates for Phase 5: ridge, elastic_net, extra_trees

## Top features — best model (`ridge`)
| Feature | Importance |
|---|--:|
| Expenses | 21289.28015 |
| Sales | 19324.98726 |
| fx_x_sales | 18120.81381 |
| oil_x_sales | 10676.21887 |
| cpi_infl_x_sales | 6422.82226 |
| Sales_roll4_mean | 5325.57723 |
| Expenses_lag2 | 4289.46802 |
| Sales_lag2 | 3733.73363 |
| Sales_roll4_median | 2996.84745 |
| Total Assets_lag1 | 2570.66452 |
| Borrowings_roll4_median | 2160.77966 |
| Operating Profit | 2156.93225 |
| Equity_lag1 | 1959.93955 |
| Borrowings_roll4_mean | 1957.80847 |
| repo_x_total_assets | 1952.10485 |

## Leakage diagnostics
- `ridge` (mean R²=0.9984): permutation R²=0.0305 (passed=True); leakage_suspected=False — High R^2 explained by series persistence — the naive baseline is also high and the permutation test passed. Accept.
- `lasso` (mean R²=0.997): permutation R²=0.0136 (passed=True); leakage_suspected=False — High R^2 explained by series persistence — the naive baseline is also high and the permutation test passed. Accept.
- `elastic_net` (mean R²=0.9983): permutation R²=0.0327 (passed=True); leakage_suspected=False — High R^2 explained by series persistence — the naive baseline is also high and the permutation test passed. Accept.
- `decision_tree` (mean R²=0.9967): permutation R²=-0.9495 (passed=True); leakage_suspected=False — High R^2 explained by series persistence — the naive baseline is also high and the permutation test passed. Accept.
- `random_forest` (mean R²=0.9982): permutation R²=0.1179 (passed=True); leakage_suspected=False — High R^2 explained by series persistence — the naive baseline is also high and the permutation test passed. Accept.
- `extra_trees` (mean R²=0.9983): permutation R²=0.1221 (passed=True); leakage_suspected=False — High R^2 explained by series persistence — the naive baseline is also high and the permutation test passed. Accept.
- `hist_gradient_boosting` (mean R²=0.9981): permutation R²=0.0262 (passed=True); leakage_suspected=False — High R^2 explained by series persistence — the naive baseline is also high and the permutation test passed. Accept.
- `xgboost` (mean R²=0.9979): permutation R²=-0.0852 (passed=True); leakage_suspected=False — High R^2 explained by series persistence — the naive baseline is also high and the permutation test passed. Accept.
- `lightgbm` (mean R²=0.998): permutation R²=-0.0033 (passed=True); leakage_suspected=False — High R^2 explained by series persistence — the naive baseline is also high and the permutation test passed. Accept.
- `catboost` (mean R²=0.9948): permutation R²=0.0707 (passed=True); leakage_suspected=False — High R^2 explained by series persistence — the naive baseline is also high and the permutation test passed. Accept.

Diagnostic plots: `residual_plots/`, `prediction_plots/`, `feature_importance/`.
