"""Optuna search spaces per model + estimator construction from sampled params.

Only the models that appear in Phase-4 top-N are exercised (ridge, elastic_net,
extra_trees, random_forest). Booster spaces + early stopping are defined for
completeness and engage automatically if a booster is ever a candidate.
"""
from __future__ import annotations

from typing import Any

# Models that support iterative early stopping.
EARLY_STOPPING_MODELS = {"xgboost", "lightgbm", "catboost"}


def suggest_params(trial, model_name: str, seed: int) -> dict[str, Any]:
    """Return a sampled hyperparameter dict for the given model."""
    if model_name == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}

    if model_name == "lasso":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 1e1, log=True),
                "max_iter": 10000, "tol": 1e-3}

    if model_name == "elastic_net":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 1e1, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
                "max_iter": 5000, "tol": 1e-3}

    if model_name in {"random_forest", "extra_trees"}:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 400, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 6, 10, 16, 24]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7, 1.0]),
            "n_jobs": -1,
        }
        if model_name == "random_forest":
            params["bootstrap"] = True
        else:  # extra_trees
            params["bootstrap"] = trial.suggest_categorical("bootstrap", [False, True])
        return params

    if model_name == "decision_tree":
        return {"max_depth": trial.suggest_categorical("max_depth", [None, 6, 10, 16, 24]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 30)}

    if model_name == "hist_gradient_boosting":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 1.0, log=True)}

    if model_name == "xgboost":
        return {"n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "tree_method": "hist", "n_jobs": -1}

    if model_name == "lightgbm":
        return {"n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "n_jobs": -1, "verbose": -1}

    if model_name == "catboost":
        return {"iterations": trial.suggest_int("iterations", 200, 800, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "verbose": 0}

    raise KeyError(f"No search space defined for model '{model_name}'.")


def supports_early_stopping(model_name: str) -> bool:
    return model_name in EARLY_STOPPING_MODELS
