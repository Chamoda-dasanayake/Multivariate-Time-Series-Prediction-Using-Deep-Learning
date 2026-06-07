"""
evaluate.py
===========

Regression metrics and diagnostic plots shared by the baseline and deep-learning
pipelines. Keeping these in one place guarantees every model is scored identically.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend so plots save without a display
import matplotlib.pyplot as plt
import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, MAPE and R^2 for predictions in original (Wh) units."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # Guard against division by zero in MAPE.
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def plot_predicted_vs_actual(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: str | Path, title: str, n: int = 500
) -> None:
    """Line plot of the first ``n`` actual vs predicted values."""
    y_true = np.asarray(y_true).ravel()[:n]
    y_pred = np.asarray(y_pred).ravel()[:n]

    plt.figure(figsize=(13, 4))
    plt.plot(y_true, label="Actual", linewidth=1.2)
    plt.plot(y_pred, label="Predicted", linewidth=1.2, alpha=0.8)
    plt.title(title)
    plt.xlabel("Time step (test set)")
    plt.ylabel("Appliances (Wh)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, out_path: str | Path, title: str) -> None:
    """Residual scatter + histogram to inspect error structure."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    resid = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].scatter(y_pred, resid, s=6, alpha=0.3)
    axes[0].axhline(0, color="red", linewidth=1)
    axes[0].set_title(f"{title} - residuals vs predicted")
    axes[0].set_xlabel("Predicted (Wh)")
    axes[0].set_ylabel("Residual (Wh)")

    axes[1].hist(resid, bins=60)
    axes[1].set_title(f"{title} - residual distribution")
    axes[1].set_xlabel("Residual (Wh)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_loss_curve(history: dict[str, list[float]], out_path: str | Path, title: str) -> None:
    """Plot train vs validation loss across epochs."""
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss (scaled)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_metric_comparison(results: dict[str, dict[str, float]], out_path: str | Path) -> None:
    """Bar chart comparing MAE / RMSE across all models."""
    models = list(results.keys())
    mae = [results[m]["MAE"] for m in models]
    rmse = [results[m]["RMSE"] for m in models]

    x = np.arange(len(models))
    width = 0.38
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, mae, width, label="MAE")
    plt.bar(x + width / 2, rmse, width, label="RMSE")
    plt.xticks(x, models, rotation=15)
    plt.ylabel("Error (Wh)")
    plt.title("Model comparison - MAE & RMSE (lower is better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
