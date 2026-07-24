"""STEP 2 & 7 — build reusable, serializable preprocessing pipelines.

Scaling is NOT applied globally. Two preprocessor variants are produced via a
factory so each model family gets appropriate treatment:

* ``tree``   — Sector one-hot encoding, numeric passthrough. No scaling, no
  imputation (HistGradientBoosting/XGBoost/LightGBM/CatBoost handle NaN or
  split on it; Random Forest / Extra Trees can use a light imputer if desired).
* ``linear`` — Sector one-hot encoding, median imputation, StandardScaler on
  numeric features (Linear/Ridge/Lasso require complete, scaled inputs).

Both are returned as **unfitted** scikit-learn ``ColumnTransformer`` objects.
Leaving them unfitted is deliberate and leakage-safe: they must be fit on the
TRAINING portion of each CV fold in Phase 4/5 (and, for final inference, on all
training data) — never on validation/test rows. They serialize cleanly with
joblib and are fully reusable at inference time.
"""
from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _one_hot() -> OneHotEncoder:
    # handle_unknown='ignore' keeps inference robust to unseen categories.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    kind: str,
    numeric_features: list[str],
    categorical_features: list[str],
    config: dict[str, Any],
) -> ColumnTransformer:
    """Construct an unfitted preprocessor for the given model family."""
    fam = config["preprocessing"]["model_families"][kind]
    impute = bool(fam.get("impute", False))
    scale = bool(fam.get("scale", False))

    steps: list[tuple[str, Any]] = []
    if impute:
        strategy = config["preprocessing"]["imputation"]["numeric_strategy"]
        steps.append(("impute", SimpleImputer(strategy=strategy)))
    if scale:
        steps.append(("scale", StandardScaler()))
    numeric_pipe: Any = Pipeline(steps) if steps else "passthrough"

    cat_pipe = Pipeline([("onehot", _one_hot())])

    transformers = [("numeric", numeric_pipe, numeric_features)]
    if categorical_features:
        transformers.append(("categorical", cat_pipe, categorical_features))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_all_preprocessors(
    numeric_features: list[str],
    categorical_features: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return both family preprocessors plus the column manifest."""
    families = list(config["preprocessing"]["model_families"].keys())
    preprocessors = {
        fam: build_preprocessor(fam, numeric_features, categorical_features, config)
        for fam in families
    }
    return {
        "preprocessors": preprocessors,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "fit_instructions": (
            "Unfitted by design. Fit on the TRAINING split of each CV fold only "
            "(leakage-safe); for final inference fit on all training rows, then "
            "transform the live feature row."
        ),
    }
