import numpy as np
from app.models.agent1_models.agent1_lightgbm_model import FEATURE_NAMES


class ShapExplainer:
    """
    Wraps shap.TreeExplainer for the LightGBM risk model. Import of `shap` is
    lazy since it's a heavy optional dependency — the model_used and
    feature_importance still work fine without it.
    """

    def __init__(self, model):
        self.model = model
        self._explainer = None

    def _get_explainer(self):
        if self._explainer is None:
            import shap
            self._explainer = shap.TreeExplainer(self.model.model)
        return self._explainer

    def explain(self, X: np.ndarray) -> list[dict]:
        try:
            explainer = self._get_explainer()
            shap_values = explainer.shap_values(X)
            values = shap_values[1] if isinstance(shap_values, list) else shap_values
            row = values[0] if values.ndim > 1 else values
            pairs = sorted(zip(FEATURE_NAMES, row), key=lambda p: -abs(p[1]))
            return [{"feature": name, "shap_value": round(float(v), 4)} for name, v in pairs]
        except Exception:
            # graceful fallback to model-level feature importance if SHAP isn't available
            return self.model.feature_importance()
