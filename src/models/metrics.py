"""Regression metrics — primary and secondary.

Primary:   RMSE, MAE, R^2, Adjusted R^2, MAPE, SMAPE
Secondary: Explained Variance, Median Absolute Error, Max Error, Mean Bias Error

MAPE/SMAPE guard against division by zero by masking (MAPE) or using the
symmetric denominator (SMAPE).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)


def _rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def _mape(y, yhat) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.abs(y) > 1e-9
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100.0)


def _smape(y, yhat) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    denom = (np.abs(y) + np.abs(yhat)) / 2.0
    mask = denom > 1e-9
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y[mask] - yhat[mask]) / denom[mask]) * 100.0)


def _adjusted_r2(r2: float, n: int, p: int) -> float:
    if n - p - 1 <= 0:
        return float("nan")
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1))


def compute_metrics(y, yhat, *, n_features: int) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    n = len(y)
    r2 = float(r2_score(y, yhat)) if n > 1 else float("nan")
    return {
        "rmse": _rmse(y, yhat),
        "mae": float(mean_absolute_error(y, yhat)),
        "r2": r2,
        "adjusted_r2": _adjusted_r2(r2, n, n_features),
        "mape": _mape(y, yhat),
        "smape": _smape(y, yhat),
        "explained_variance": float(explained_variance_score(y, yhat)),
        "median_abs_error": float(median_absolute_error(y, yhat)),
        "max_error": float(max_error(y, yhat)),
        "mean_bias_error": float(np.mean(yhat - y)),
    }


PRIMARY = ["rmse", "mae", "r2", "adjusted_r2", "mape", "smape"]
SECONDARY = ["explained_variance", "median_abs_error", "max_error", "mean_bias_error"]
