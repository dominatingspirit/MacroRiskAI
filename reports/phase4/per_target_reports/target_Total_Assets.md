# Phase 4 — Target report: `target_Total_Assets`

## Leaderboard
| # | Model | Group | RMSE | MAE | R² | Adj R² | MAPE | SMAPE | Stab σ(RMSE) | Beats naive |
|--:|---|---|--:|--:|--:|--:|--:|--:|--:|:--:|
| 1 | elastic_net | linear | 98,097.3 | 50,523.9 | 0.9893 | 0.9885 | 10.10 | 10.32 | 2,464.5 | ❌ |
| 2 | extra_trees | tree | 98,520.4 | 45,440.6 | 0.9892 | 0.9884 | 6.56 | 6.51 | 4,118.9 | ❌ |
| 3 | random_forest | tree | 99,653.5 | 46,032.5 | 0.9889 | 0.9881 | 6.58 | 6.52 | 4,246.7 | ❌ |
| 4 | hist_gradient_boosting | tree | 103,532.9 | 47,371.6 | 0.9881 | 0.9872 | 6.78 | 6.72 | 2,976.3 | ❌ |
| 5 | ridge | linear | 100,030.7 | 51,884.9 | 0.9889 | 0.9880 | 10.69 | 10.99 | 4,147.7 | ❌ |
| 6 | multioutput_random_forest | benchmark | 100,454.5 | 46,053.4 | 0.9888 | 0.9879 | 6.58 | 6.51 | nan | ❌ |
| 7 | lightgbm | boosting | 105,759.5 | 49,630.6 | 0.9876 | 0.9866 | 7.10 | 7.04 | 5,103.7 | ❌ |
| 8 | lasso | linear | 102,836.2 | 58,994.8 | 0.9882 | 0.9873 | 17.94 | 16.68 | 5,166.5 | ❌ |
| 9 | xgboost | boosting | 112,542.0 | 52,503.0 | 0.9859 | 0.9848 | 7.57 | 7.53 | 3,826.8 | ❌ |
| 10 | decision_tree | tree | 125,079.8 | 61,197.1 | 0.9826 | 0.9813 | 9.33 | 9.27 | 3,762.8 | ❌ |
| 11 | catboost | boosting | 123,371.3 | 62,615.4 | 0.9831 | 0.9818 | 14.26 | 12.86 | 9,473.5 | ❌ |
| 12 | linear_regression | linear | 1,890,195,022.7 | 1,415,445,518.3 | -3975669.9895 | -4280739.8935 | 736895.32 | 199.42 | 1,252,735,411.1 | ❌ |

## Baselines
| Baseline | RMSE | MAE | R² |
|---|--:|--:|--:|
| naive_prev_quarter | 94,337.3 | 43,536.5 | 0.9901 |
| seasonal_naive | 153,774.7 | 74,686.6 | 0.9737 |
| historical_mean | 119,336.2 | 58,405.0 | 0.9842 |

## Top-3 candidates for Phase 5: elastic_net, extra_trees, random_forest

## Top features — best model (`elastic_net`)
| Feature | Importance |
|---|--:|
| repo_x_total_assets | 282749.67693 |
| Total Assets | 280183.90128 |
| Equity | 149156.86606 |
| Total Liabilities | 137717.72006 |
| Total Assets_lag1 | 83602.45029 |
| Borrowings_lag1 | 59528.08434 |
| Equity_lag2 | 55073.62703 |
| repo_x_borrowings | 39762.48148 |
| Total Liabilities_lag4 | 39066.34136 |
| Borrowings | 34808.70461 |
| Total Assets_roll4_mean | 34059.84882 |
| Net Profit_roll4_median | 32686.93423 |
| Total Liabilities_lag2 | 31889.06527 |
| Total Assets_roll4_median | 25411.99161 |
| Total Assets_lag4 | 23897.21888 |

## Leakage diagnostics
- `ridge` (mean R²=0.9889): permutation R²=-0.0958 (passed=True); leakage_suspected=False — High R^2 explained by series persistence — the naive baseline is also high and the permutation test passed. Accept.
- `lasso` (mean R²=0.9882): permutation R²=-0.1722 (passed=True); leakage_suspected=False — High R^2 explained by series persistence — the naive baseline is also high and the permutation test passed. Accept.
- `elastic_net` (mean R²=0.9893): permutation R²=-0.0586 (passed=True); leakage_suspected=False — High R^2 explained by series persistence — the naive baseline is also high and the permutation test passed. Accept.
- `decision_tree` (mean R²=0.9826): permutation R²=-2.2291 (passed=True); leakage_suspected=False — High R^2 explained by series persistence — the naive baseline is also high and the permutation test passed. Accept.
- `random_forest` (mean R²=0.9889): permutation R²=-0.0938 (passed=True); leakage_suspected=False — High R^2 explained by series persistence — the naive baseline is also high and the permutation test passed. Accept.
- `extra_trees` (mean R²=0.9892): permutation R²=-0.0528 (passed=True); leakage_suspected=False — High R^2 explained by series persistence — the naive baseline is also high and the permutation test passed. Accept.
- `hist_gradient_boosting` (mean R²=0.9881): permutation R²=-0.0932 (passed=True); leakage_suspected=False — High R^2 explained by series persistence — the naive baseline is also high and the permutation test passed. Accept.
- `xgboost` (mean R²=0.9859): permutation R²=-0.3058 (passed=True); leakage_suspected=False — High R^2 explained by series persistence — the naive baseline is also high and the permutation test passed. Accept.
- `lightgbm` (mean R²=0.9876): permutation R²=-0.1803 (passed=True); leakage_suspected=False — High R^2 explained by series persistence — the naive baseline is also high and the permutation test passed. Accept.
- `catboost` (mean R²=0.9831): permutation R²=-0.1257 (passed=True); leakage_suspected=False — High R^2 explained by series persistence — the naive baseline is also high and the permutation test passed. Accept.

Diagnostic plots: `residual_plots/`, `prediction_plots/`, `feature_importance/`.
