# Phase 4 — Target report: `target_Borrowings`

## Leaderboard
| # | Model | Group | RMSE | MAE | R² | Adj R² | MAPE | SMAPE | Stab σ(RMSE) | Beats naive |
|--:|---|---|--:|--:|--:|--:|--:|--:|--:|:--:|
| 1 | extra_trees | tree | 61,158.4 | 25,036.8 | 0.9775 | 0.9758 | 10.90 | 10.67 | 3,826.8 | ❌ |
| 2 | random_forest | tree | 61,337.6 | 25,273.4 | 0.9774 | 0.9757 | 10.98 | 10.77 | 2,890.4 | ❌ |
| 3 | elastic_net | linear | 56,054.9 | 27,278.8 | 0.9811 | 0.9797 | 19.04 | 19.98 | 5,300.7 | ❌ |
| 4 | ridge | linear | 58,405.7 | 28,565.0 | 0.9795 | 0.9780 | 20.22 | 21.09 | 5,269.6 | ❌ |
| 5 | xgboost | boosting | 64,855.0 | 26,793.5 | 0.9748 | 0.9728 | 12.23 | 12.02 | 3,317.8 | ❌ |
| 6 | hist_gradient_boosting | tree | 68,092.5 | 27,024.3 | 0.9722 | 0.9700 | 11.63 | 11.36 | 3,392.8 | ❌ |
| 7 | lightgbm | boosting | 68,250.0 | 27,321.2 | 0.9720 | 0.9699 | 11.83 | 11.62 | 2,134.7 | ❌ |
| 8 | multioutput_random_forest | benchmark | 63,508.1 | 27,418.6 | 0.9758 | 0.9740 | 12.87 | 12.28 | nan | ❌ |
| 9 | catboost | boosting | 70,777.1 | 31,114.3 | 0.9699 | 0.9676 | 20.63 | 17.91 | 2,185.8 | ❌ |
| 10 | lasso | linear | 69,407.4 | 45,305.6 | 0.9711 | 0.9689 | 63.27 | 52.22 | 8,126.0 | ❌ |
| 11 | decision_tree | tree | 83,906.4 | 33,960.3 | 0.9577 | 0.9545 | 14.95 | 14.91 | 8,855.6 | ❌ |
| 12 | linear_regression | linear | 59,044,220.5 | 50,825,511.6 | -20926.8491 | -22532.7307 | 92388.11 | 200.00 | 30,045,737.7 | ❌ |

## Baselines
| Baseline | RMSE | MAE | R² |
|---|--:|--:|--:|
| naive_prev_quarter | 54,046.6 | 22,941.4 | 0.9825 |
| seasonal_naive | 91,560.8 | 38,437.7 | 0.9497 |
| historical_mean | 69,057.7 | 29,271.1 | 0.9714 |

## Top-3 candidates for Phase 5: extra_trees, random_forest, elastic_net

## Top features — best model (`extra_trees`)
| Feature | Importance |
|---|--:|
| Borrowings | 0.28741 |
| repo_x_borrowings | 0.28138 |
| repo_x_total_assets | 0.12844 |
| Total Liabilities | 0.11331 |
| Total Assets | 0.09281 |
| Total Liabilities_lag1 | 0.02199 |
| Borrowings_lag1 | 0.01986 |
| Equity | 0.01548 |
| Borrowings_lag2 | 0.00650 |
| Total Assets_lag1 | 0.00620 |
| Borrowings_roll4_median | 0.00222 |
| Borrowings_roll4_mean | 0.00178 |
| Sector_Energy | 0.00143 |
| oil_x_sales | 0.00116 |
| Sector_Banking | 0.00115 |

## Leakage diagnostics
- `elastic_net` (mean R²=0.9811): permutation R²=-0.1123 (passed=True); leakage_suspected=False — High R^2 explained by series persistence — the naive baseline is also high and the permutation test passed. Accept.

Diagnostic plots: `residual_plots/`, `prediction_plots/`, `feature_importance/`.
