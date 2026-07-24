# Phase 4 — Target report: `target_Operating_Profit`

## Leaderboard
| # | Model | Group | RMSE | MAE | R² | Adj R² | MAPE | SMAPE | Stab σ(RMSE) | Beats naive |
|--:|---|---|--:|--:|--:|--:|--:|--:|--:|:--:|
| 1 | ridge | linear | 1,731.0 | 1,132.2 | 0.9748 | 0.9729 | 7.14 | 7.05 | 70.9 | ❌ |
| 2 | elastic_net | linear | 1,723.2 | 1,139.0 | 0.9750 | 0.9731 | 7.35 | 7.26 | 79.5 | ❌ |
| 3 | extra_trees | tree | 1,816.7 | 1,150.2 | 0.9722 | 0.9701 | 6.66 | 6.54 | 73.1 | ❌ |
| 4 | random_forest | tree | 1,880.4 | 1,178.8 | 0.9703 | 0.9680 | 6.78 | 6.68 | 69.2 | ❌ |
| 5 | hist_gradient_boosting | tree | 2,003.1 | 1,218.0 | 0.9663 | 0.9637 | 6.90 | 6.80 | 65.8 | ❌ |
| 6 | lasso | linear | 1,865.9 | 1,269.7 | 0.9707 | 0.9685 | 8.71 | 8.47 | 120.8 | ❌ |
| 7 | lightgbm | boosting | 2,020.1 | 1,241.8 | 0.9657 | 0.9631 | 7.17 | 7.04 | 88.9 | ❌ |
| 8 | xgboost | boosting | 2,072.4 | 1,304.3 | 0.9639 | 0.9611 | 7.60 | 7.48 | 123.1 | ❌ |
| 9 | catboost | boosting | 2,078.1 | 1,343.0 | 0.9637 | 0.9609 | 9.04 | 8.69 | 107.5 | ❌ |
| 10 | decision_tree | tree | 2,690.8 | 1,678.2 | 0.9391 | 0.9344 | 9.58 | 9.52 | 86.6 | ❌ |
| 11 | multioutput_random_forest | benchmark | 4,009.9 | 2,456.0 | 0.8648 | 0.8549 | 15.68 | 14.46 | nan | ❌ |
| 12 | linear_regression | linear | 2,480,083.3 | 2,071,204.5 | -51717.1297 | -55685.6786 | 20647.99 | 191.22 | 1,363,998.4 | ❌ |

## Baselines
| Baseline | RMSE | MAE | R² |
|---|--:|--:|--:|
| naive_prev_quarter | 1,695.2 | 1,087.2 | 0.9758 |
| seasonal_naive | 3,156.3 | 2,036.9 | 0.9162 |
| historical_mean | 2,469.3 | 1,556.8 | 0.9487 |

## Top-3 candidates for Phase 5: ridge, elastic_net, extra_trees

## Top features — best model (`ridge`)
| Feature | Importance |
|---|--:|
| Operating Profit | 9946.65621 |
| Net Profit_roll4_mean | 1308.83245 |
| Borrowings_roll4_median | 988.26725 |
| Borrowings_roll4_mean | 945.17954 |
| Sales_roll4_mean | 862.31961 |
| Net Profit_lag2 | 833.14601 |
| cpi_infl_x_sales | 715.32981 |
| Expenses | 711.36381 |
| Sales | 695.37641 |
| Total Liabilities | 681.98538 |
| Expenses_lag2 | 568.65397 |
| Sales_lag2 | 563.64989 |
| Total Assets_lag4 | 502.00234 |
| fx_x_sales | 501.20686 |
| Net Profit_roll4_median | 492.08383 |

Diagnostic plots: `residual_plots/`, `prediction_plots/`, `feature_importance/`.
