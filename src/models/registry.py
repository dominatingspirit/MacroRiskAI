"""Model registry + pipeline factory.

Instantiates estimators from the config (baseline defaults only) and wraps each
in a single sklearn ``Pipeline`` with the appropriate Phase-3 preprocessor:

* ``linear``      → scaled + imputed (LinearRegression/Ridge/Lasso/ElasticNet)
* ``tree``        → unscaled, NaN-native (HistGB/XGBoost/LightGBM/CatBoost)
* ``tree_impute`` → unscaled + imputed (sklearn DecisionTree/RandomForest/ExtraTrees)

The preprocessor logic itself is reused from Phase 3 (``build_preprocessor``);
nothing is duplicated here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from ..features.preprocessor import build_preprocessor

# Lazily import the boosting libraries so a missing one degrades gracefully.
_ESTIMATORS: dict[str, Any] = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
}


def _load_boosting() -> None:
    """Register boosting estimators on first use."""
    if "XGBRegressor" not in _ESTIMATORS:
        from xgboost import XGBRegressor
        _ESTIMATORS["XGBRegressor"] = XGBRegressor
    if "LGBMRegressor" not in _ESTIMATORS:
        from lightgbm import LGBMRegressor
        _ESTIMATORS["LGBMRegressor"] = LGBMRegressor
    if "CatBoostRegressor" not in _ESTIMATORS:
        from catboost import CatBoostRegressor
        _ESTIMATORS["CatBoostRegressor"] = CatBoostRegressor


# Estimators whose native param name for the RNG seed differs.
_SEED_KW = {
    "LinearRegression": None,   # deterministic, no seed
    "CatBoostRegressor": "random_seed",
}


@dataclass
class ModelSpec:
    name: str
    group: str          # linear | tree | boosting | benchmark
    estimator_name: str
    pre: str            # preprocessor family
    params: dict[str, Any]


def iter_model_specs(config: dict[str, Any], *, include_benchmark: bool = True):
    """Yield ModelSpec for every configured single-target model.

    The multi-output benchmark is handled separately by the orchestrator.
    """
    groups = config["models"]
    for group in ["linear", "tree", "boosting"]:
        for name, spec in groups.get(group, {}).items():
            yield ModelSpec(name, group, spec["estimator"], spec["pre"], dict(spec.get("params", {})))


def get_model_spec(config: dict[str, Any], name: str) -> ModelSpec:
    """Look up a single model's spec by name across all groups."""
    for group in ["linear", "tree", "boosting", "benchmark"]:
        grp = config["models"].get(group, {})
        if name in grp:
            spec = grp[name]
            return ModelSpec(name, group, spec["estimator"], spec["pre"], dict(spec.get("params", {})))
    raise KeyError(f"Model '{name}' not found in config['models'].")


def get_benchmark_spec(config: dict[str, Any]) -> ModelSpec:
    spec = config["models"]["benchmark"]["multioutput_random_forest"]
    return ModelSpec("multioutput_random_forest", "benchmark", spec["estimator"],
                     spec["pre"], dict(spec.get("params", {})))


def make_estimator(spec: ModelSpec, seed: int):
    """Instantiate a fresh estimator with baseline params + seed where supported."""
    if spec.estimator_name in {"XGBRegressor", "LGBMRegressor", "CatBoostRegressor"}:
        _load_boosting()
    cls = _ESTIMATORS[spec.estimator_name]
    params = dict(spec.params)
    seed_kw = _SEED_KW.get(spec.estimator_name, "random_state")
    if seed_kw is not None:
        params.setdefault(seed_kw, seed)
    return cls(**params)


def build_pipeline(
    spec: ModelSpec,
    numeric_features: list[str],
    categorical_features: list[str],
    config: dict[str, Any],
    seed: int,
) -> Pipeline:
    """Single Pipeline = Phase-3 preprocessor (by family) + estimator."""
    pre = build_preprocessor(spec.pre, numeric_features, categorical_features, config)
    est = make_estimator(spec, seed)
    return Pipeline([("preprocess", clone(pre)), ("model", est)])
