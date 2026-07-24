# MacroRisk AI — Phase 4: Baseline Benchmarking Summary

_Baseline models only. No hyperparameter tuning, no SHAP, no deployment._

- Validation: **walk_forward**, 3 expanding-window folds (never shuffled; company boundaries + quarterly order preserved).
- Each of the six targets has its own independent Pipeline (preprocessor + estimator). Multi-output RF is a benchmark only.

## Overall leaderboard (averaged across targets)
| Model | Avg rank | Avg RMSE | Avg R² | Avg MAE | Group |
|---|--:|--:|--:|--:|---|
| elastic_net | 2.58 | 27,439.4 | 0.9780 | 14,083.2 | linear |
| extra_trees | 2.96 | 28,407.6 | 0.9761 | 12,841.1 | tree |
| ridge | 3.42 | 28,150.9 | 0.9773 | 14,515.0 | linear |
| random_forest | 3.62 | 28,660.9 | 0.9756 | 13,004.3 | tree |
| hist_gradient_boosting | 5.00 | 30,520.0 | 0.9724 | 13,540.6 | tree |
| lightgbm | 6.17 | 30,949.8 | 0.9718 | 13,996.9 | boosting |
| xgboost | 6.83 | 31,561.3 | 0.9709 | 14,423.3 | boosting |
| lasso | 8.21 | 30,880.2 | 0.9679 | 18,991.6 | linear |
| catboost | 8.71 | 34,718.3 | 0.9690 | 17,134.7 | boosting |
| decision_tree | 9.00 | 37,417.5 | 0.9518 | 17,447.9 | tree |
| multioutput_random_forest | 9.50 | 32,117.7 | 0.9202 | 14,875.9 | benchmark |
| linear_regression | 11.75 | 332,522,377.8 | -816455.4808 | 250,688,805.9 | linear |

## Best model per target (vs. best naive baseline)
| Target | Best model | RMSE | MAE | R² | Beats naive | Top-3 candidates |
|---|---|--:|--:|--:|:--:|---|
| target_Sales | ridge | 3,159.2 | 2,004.9 | 0.9984 | ✅ | ridge, elastic_net, extra_trees |
| target_Operating_Profit | ridge | 1,731.0 | 1,132.2 | 0.9748 | ❌ | ridge, elastic_net, extra_trees |
| target_Net_Profit | extra_trees | 1,928.9 | 1,179.7 | 0.9489 | ❌ | extra_trees, random_forest, elastic_net |
| target_Borrowings | extra_trees | 61,158.4 | 25,036.8 | 0.9775 | ❌ | extra_trees, random_forest, elastic_net |
| target_Total_Assets | elastic_net | 98,097.3 | 50,523.9 | 0.9893 | ❌ | elastic_net, extra_trees, random_forest |
| target_CFO | elastic_net | 3,636.0 | 2,268.1 | 0.9721 | ❌ | elastic_net, ridge, extra_trees |

## ⚠️ Leakage diagnostics (auto-triggered by R² > 0.98)
- **21** model/target combinations exceeded the R² trigger and were auto-diagnosed before acceptance.
- **21** cleared (target-permutation test passed **and** no non-target feature is degenerate → high R² reflects series persistence).
- **0** with leakage suspected.

Naive-baseline R² per target (context — a strong naive baseline confirms persistence, not leakage):
| Target | Naive R² | Typical model mean R² |
|---|--:|--:|
| target_Sales | 0.9983 | 0.9984 |
| target_Borrowings | 0.9825 | 0.9811 |
| target_Total_Assets | 0.9901 | 0.9893 |

_No leakage suspected: every triggered model collapsed to ~0 R² on permuted targets, and no non-target feature is near-perfectly correlated with the target. The high R² is explained by the persistence of financial levels (the naive baseline is itself strong). Full detail in `leakage_diagnostics.json`._

## Notes
- Baselines (naive / seasonal-naive / historical-mean) are included in `baseline_results.csv` for every target.
- Prediction intervals are stored in `prediction_intervals.csv` (not used for ranking).
- Models are **not** eliminated for being slower.
