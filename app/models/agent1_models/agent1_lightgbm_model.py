"""
Agent 3 (SRIA) financial-distress classifier — unrelated to the LightGBM
*inflation* model MacroAgent uses (that one lives in
app/models/train_lightgbm_inflation.py + lightgbm_inflation_model.pkl).
Same library, different target: this one predicts P(financial distress)
from stress-test output + ratios + Company DNA + inflation forecast.
"""
import numpy as np
from lightgbm import LGBMClassifier

# Order must match app/services/feature_engineering.py::build_risk_features
FEATURE_NAMES = [
    "current_ratio",
    "quick_ratio",
    "de_ratio",
    "projected_margin",
    "roe",
    "avg_inflation",
    "revenue_change",
    "cogs_change",
    "capital_intensity_score",
    "pricing_power_score",
]


def _bootstrap_dataset(n_samples: int = 800, seed: int = 42):
    """
    Synthetic-but-plausible distress-risk training set (hackathon MVP —
    see README): labels come from a deterministic, financially-sensible
    scoring rule (weak liquidity + high leverage + thin margins + high
    inflation exposure + low pricing power => higher distress probability)
    plus noise, so LightGBM learns a realistic decision boundary rather than
    pure noise. Swap in real labeled distress outcomes via
    LightGBMRiskModel.fit(X, y) when available.
    """
    rng = np.random.RandomState(seed)
    current_ratio = rng.uniform(0.5, 3.0, n_samples)
    quick_ratio = current_ratio * rng.uniform(0.5, 0.9, n_samples)
    de_ratio = rng.uniform(0.1, 3.0, n_samples)
    projected_margin = rng.uniform(-5, 30, n_samples)
    roe = rng.uniform(-10, 35, n_samples)
    avg_inflation = rng.uniform(2, 10, n_samples)
    revenue_change = rng.uniform(-15, 15, n_samples)
    cogs_change = rng.uniform(-10, 20, n_samples)
    capital_intensity_score = rng.choice([0.0, 0.5, 1.0], n_samples)
    pricing_power_score = rng.choice([0.0, 0.5, 1.0], n_samples)

    X = np.column_stack([
        current_ratio, quick_ratio, de_ratio, projected_margin, roe,
        avg_inflation, revenue_change, cogs_change,
        capital_intensity_score, pricing_power_score,
    ])

    distress_score = (
        -1.6 * (current_ratio - 1.0)
        + 1.4 * de_ratio
        - 0.09 * projected_margin
        - 0.04 * roe
        + 0.18 * avg_inflation
        - 0.05 * revenue_change
        + 0.06 * cogs_change
        + 0.8 * (1 - pricing_power_score)
        + 0.4 * capital_intensity_score
        + rng.normal(0, 1.0, n_samples)
    )
    y = (distress_score > np.median(distress_score)).astype(int)
    return X, y


class LightGBMRiskModel:
    """Bootstraps on a synthetic dataset at construction time so /risk and
    /pipeline work immediately with zero setup. Call .fit(X, y) with real
    labeled distress data to replace the bootstrap model when available."""

    def __init__(self):
        self.model = LGBMClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            random_state=42, verbose=-1,
        )
        X, y = _bootstrap_dataset()
        self.model.fit(X, y)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns P(distress) per row."""
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self) -> list[dict]:
        """Fallback used by ShapExplainer.explain() if `shap` isn't installed."""
        importances = self.model.feature_importances_
        total = importances.sum() or 1
        pairs = sorted(zip(FEATURE_NAMES, importances), key=lambda p: -p[1])
        return [{"feature": name, "shap_value": round(float(v) / float(total), 4)} for name, v in pairs]
