# Phase 4 — Target report: `target_Net_Profit`

## Leaderboard
| # | Model | Group | RMSE | MAE | R² | Adj R² | MAPE | SMAPE | Stab σ(RMSE) | Beats naive |
|--:|---|---|--:|--:|--:|--:|--:|--:|--:|:--:|
| 1 | extra_trees | tree | 1,928.9 | 1,179.7 | 0.9489 | 0.9449 | 9.18 | 8.89 | 80.4 | ❌ |
| 2 | random_forest | tree | 1,926.5 | 1,188.7 | 0.9490 | 0.9451 | 9.27 | 9.01 | 95.8 | ❌ |
| 3 | elastic_net | linear | 1,866.1 | 1,189.1 | 0.9521 | 0.9485 | 10.01 | 9.73 | 118.9 | ❌ |
| 4 | ridge | linear | 1,896.3 | 1,206.2 | 0.9506 | 0.9468 | 10.19 | 9.89 | 106.2 | ❌ |
| 5 | hist_gradient_boosting | tree | 2,028.3 | 1,231.7 | 0.9435 | 0.9391 | 9.57 | 9.33 | 101.6 | ❌ |
| 6 | lightgbm | boosting | 2,043.1 | 1,261.8 | 0.9426 | 0.9382 | 9.94 | 9.72 | 96.8 | ❌ |
| 7 | xgboost | boosting | 2,103.9 | 1,306.7 | 0.9392 | 0.9345 | 10.29 | 10.05 | 85.5 | ❌ |
| 8 | catboost | boosting | 2,082.4 | 1,326.7 | 0.9404 | 0.9358 | 11.71 | 11.13 | 101.6 | ❌ |
| 9 | lasso | linear | 2,404.7 | 1,772.6 | 0.9205 | 0.9144 | 19.09 | 20.32 | 406.4 | ❌ |
| 10 | decision_tree | tree | 2,688.3 | 1,704.9 | 0.9007 | 0.8931 | 13.19 | 12.86 | 121.7 | ❌ |
| 11 | multioutput_random_forest | benchmark | 3,427.6 | 2,156.5 | 0.8385 | 0.8267 | 18.88 | 17.02 | nan | ❌ |
| 12 | linear_regression | linear | 6,722,752.3 | 5,821,375.6 | -621113.1227 | -668773.8135 | 80063.46 | 200.00 | 3,362,575.7 | ❌ |

## Baselines
| Baseline | RMSE | MAE | R² |
|---|--:|--:|--:|
| naive_prev_quarter | 1,844.1 | 1,141.3 | 0.9533 |
| seasonal_naive | 3,212.6 | 2,010.5 | 0.8582 |
| historical_mean | 2,463.9 | 1,545.2 | 0.9166 |

## Top-3 candidates for Phase 5: extra_trees, random_forest, elastic_net

## Top features — best model (`extra_trees`)
| Feature | Importance |
|---|--:|
| Net Profit | 0.35481 |
| Operating Profit | 0.19506 |
| Net Profit_lag1 | 0.07769 |
| CFO | 0.04760 |
| Operating Profit_lag1 | 0.04165 |
| CFI | 0.03966 |
| CFF | 0.03413 |
| fx_x_sales | 0.03233 |
| Sales | 0.02616 |
| cpi_infl_x_sales | 0.02047 |
| oil_x_sales | 0.01923 |
| Net Profit_roll4_median | 0.01359 |
| Net Profit_roll4_mean | 0.01323 |
| Expenses | 0.01214 |
| repo_x_total_assets | 0.00597 |

Diagnostic plots: `residual_plots/`, `prediction_plots/`, `feature_importance/`.
