"""Feature importance / coefficient extraction from a fitted Pipeline.

* Tree/boosting models → native ``feature_importances_`` (impurity/gain).
* Linear models → standardized coefficients. Because the linear preprocessor
  already StandardScales numeric inputs, the raw ``coef_`` are already on a
  comparable (standardized) scale; we report them directly with magnitude rank.

Returns a DataFrame [feature, importance] or None if unsupported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def _feature_names(pipe: Pipeline) -> list[str]:
    pre = pipe.named_steps["preprocess"]
    try:
        return list(pre.get_feature_names_out())
    except Exception:  # pragma: no cover - defensive
        return [f"f{i}" for i in range(pipe.named_steps["model"].n_features_in_)]


def extract_importance(pipe: Pipeline, group: str) -> pd.DataFrame | None:
    model = pipe.named_steps["model"]
    names = _feature_names(pipe)

    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
        kind = "native_importance"
    elif hasattr(model, "coef_"):
        imp = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
        kind = "standardized_coefficient_abs"
    else:
        return None

    n = min(len(names), len(imp))
    df = pd.DataFrame({"feature": names[:n], "importance": imp[:n], "kind": kind})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)
