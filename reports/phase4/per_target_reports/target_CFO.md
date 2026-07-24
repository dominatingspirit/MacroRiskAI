# Phase 4 — Target report: `target_CFO`

## Leaderboard
| # | Model | Group | RMSE | MAE | R² | Adj R² | MAPE | SMAPE | Stab σ(RMSE) | Beats naive |
|--:|---|---|--:|--:|--:|--:|--:|--:|--:|:--:|
| 1 | elastic_net | linear | 3,636.0 | 2,268.1 | 0.9721 | 0.9699 | 10.01 | 9.79 | 216.0 | ❌ |
| 2 | ridge | linear | 3,682.4 | 2,296.9 | 0.9714 | 0.9692 | 10.18 | 9.89 | 225.7 | ❌ |
| 3 | extra_trees | tree | 3,746.0 | 2,238.1 | 0.9704 | 0.9681 | 9.11 | 8.90 | 278.2 | ❌ |
| 4 | random_forest | tree | 3,766.3 | 2,271.0 | 0.9700 | 0.9677 | 9.24 | 9.04 | 253.8 | ❌ |
| 5 | hist_gradient_boosting | tree | 3,978.1 | 2,336.4 | 0.9666 | 0.9640 | 9.32 | 9.13 | 462.5 | ❌ |
| 6 | lightgbm | boosting | 4,064.4 | 2,407.8 | 0.9651 | 0.9624 | 9.65 | 9.46 | 423.3 | ❌ |
| 7 | xgboost | boosting | 4,135.0 | 2,440.5 | 0.9639 | 0.9611 | 10.06 | 9.86 | 423.2 | ❌ |
| 8 | decision_tree | tree | 5,581.6 | 3,336.7 | 0.9342 | 0.9291 | 13.40 | 13.06 | 112.1 | ❌ |
| 9 | lasso | linear | 4,367.2 | 3,173.8 | 0.9597 | 0.9566 | 19.84 | 23.53 | 336.6 | ❌ |
| 10 | catboost | boosting | 4,243.6 | 2,610.7 | 0.9620 | 0.9590 | 12.41 | 11.80 | 466.4 | ❌ |
| 11 | multioutput_random_forest | benchmark | 7,414.3 | 4,234.8 | 0.8839 | 0.8754 | 17.61 | 15.98 | nan | ❌ |
| 12 | linear_regression | linear | 6,268,547.5 | 6,036,848.6 | -82997.0049 | -89365.7898 | 53320.60 | 199.61 | 1,688,446.3 | ❌ |

## Baselines
| Baseline | RMSE | MAE | R² |
|---|--:|--:|--:|
| naive_prev_quarter | 3,591.7 | 2,149.9 | 0.9728 |
| seasonal_naive | 6,458.4 | 3,883.0 | 0.9119 |
| historical_mean | 5,026.9 | 2,979.6 | 0.9466 |

## Top-3 candidates for Phase 5: elastic_net, ridge, extra_trees

## Top features — best model (`elastic_net`)
| Feature | Importance |
|---|--:|
| CFI | 6839.39365 |
| CFO | 5706.96321 |
| Net Cash Flow | 3830.95298 |
| CFF | 2991.10803 |
| CFI_lag1 | 1046.23943 |
| Total Liabilities_lag1 | 1043.36684 |
| Net Profit_lag2 | 700.13998 |
| Operating Profit_lag4 | 675.52279 |
| Operating Profit_roll4_median | 619.31784 |
| repo_x_total_assets | 613.49845 |
| Borrowings_lag2 | 603.66521 |
| CFF_lag2 | 584.56945 |
| Operating Profit_lag1 | 576.81708 |
| Borrowings_roll4_mean | 566.95236 |
| Equity_lag1 | 565.35966 |

Diagnostic plots: `residual_plots/`, `prediction_plots/`, `feature_importance/`.
