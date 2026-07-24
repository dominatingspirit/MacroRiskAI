"""Phase 5 — hyperparameter optimization (Optuna).

Tunes only each target's Phase-4 top-N candidate models, using the same
leakage-safe walk-forward validation. Does not change the target formulation,
feature engineering, or synthetic dataset.
"""
