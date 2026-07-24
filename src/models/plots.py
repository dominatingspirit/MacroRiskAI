"""Diagnostic plotting (matplotlib, Agg backend).

* ``residual_diagnostics`` — a 2x2 figure per model/target: residual histogram,
  residual-vs-prediction, predicted-vs-actual, and a Normal QQ plot.
* ``prediction_plot``      — standalone predicted-vs-actual.
* ``importance_plot``      — top-N feature importances / |coefficients|.

All operate on out-of-fold (pooled walk-forward) predictions.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402


def residual_diagnostics(y, yhat, title: str, path: Path) -> None:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    resid = yhat - y
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(title, fontsize=12)

    # Residual histogram
    ax[0, 0].hist(resid, bins=40, color="#4c78a8", alpha=0.85)
    ax[0, 0].axvline(0, color="k", lw=1, ls="--")
    ax[0, 0].set_title(f"Residual histogram (mean={resid.mean():.1f}, sd={resid.std():.1f})")
    ax[0, 0].set_xlabel("residual (pred - actual)")

    # Residual vs prediction
    ax[0, 1].scatter(yhat, resid, s=6, alpha=0.25, color="#e45756")
    ax[0, 1].axhline(0, color="k", lw=1, ls="--")
    ax[0, 1].set_title("Residual vs prediction")
    ax[0, 1].set_xlabel("predicted")
    ax[0, 1].set_ylabel("residual")

    # Predicted vs actual
    lo = float(min(y.min(), yhat.min()))
    hi = float(max(y.max(), yhat.max()))
    ax[1, 0].scatter(y, yhat, s=6, alpha=0.25, color="#54a24b")
    ax[1, 0].plot([lo, hi], [lo, hi], "k--", lw=1)
    ax[1, 0].set_title("Predicted vs actual")
    ax[1, 0].set_xlabel("actual")
    ax[1, 0].set_ylabel("predicted")

    # QQ plot of residuals
    stats.probplot(resid, dist="norm", plot=ax[1, 1])
    ax[1, 1].set_title("Normal QQ plot (residuals)")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=100)
    plt.close(fig)


def prediction_plot(y, yhat, title: str, path: Path) -> None:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    lo = float(min(y.min(), yhat.min()))
    hi = float(max(y.max(), yhat.max()))
    ax.scatter(y, yhat, s=7, alpha=0.3, color="#4c78a8")
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def importance_plot(imp: pd.DataFrame, title: str, path: Path, top_n: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top = imp.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top))))
    ax.barh(top["feature"], top["importance"], color="#7c4dbe")
    ax.set_title(title)
    ax.set_xlabel(imp["kind"].iloc[0] if "kind" in imp else "importance")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
