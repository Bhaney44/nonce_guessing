"""
evaluate.py
-----------
Evaluation metrics for nonce prediction.

Key metrics:
  - MAE, RMSE (raw prediction error)
  - Within-range accuracy: what % of predictions fall within X% of true nonce
  - Range narrowing factor: how much does the model narrow the search space vs random?
  - Percentile calibration: if we use the model's prediction as a starting point
    and search outward, how many hashes do we save on average?

The "range narrowing factor" is the most important metric:
  - If the model perfectly predicted the nonce, factor = 2^32 (we'd search 0 extra)
  - If the model is no better than random, factor = 1.0
  - Any factor > 1.0 means the model is useful
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_std: float) -> dict:
    """
    Compute evaluation metrics.

    y_true, y_pred: nonce values in original scale (0 to 2^32)
    y_std: training std used for normalization (for context)
    """
    errors = np.abs(y_true - y_pred)
    nonce_range = 2**32  # Full 32-bit nonce space

    # Basic regression metrics
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    median_ae = float(np.median(errors))

    # Within-range accuracy at various tolerances
    tolerances = {
        "1%":   nonce_range * 0.01,
        "5%":   nonce_range * 0.05,
        "10%":  nonce_range * 0.10,
        "25%":  nonce_range * 0.25,
        "50%":  nonce_range * 0.50,
    }
    within = {k: float(np.mean(errors < v)) for k, v in tolerances.items()}

    # Range narrowing: if we search a window of size 2*error around prediction,
    # how many nonces do we skip vs searching all 2^32?
    median_search_radius = float(np.median(errors))
    search_window = 2 * median_search_radius
    narrowing_factor = nonce_range / max(search_window, 1)

    # Baseline: random guess has expected error of 2^32 / 4 = 2^30
    random_mae = nonce_range / 4
    improvement_vs_random = random_mae / max(mae, 1)

    return {
        "mae": mae,
        "rmse": rmse,
        "median_ae": median_ae,
        "within_range": within,
        "median_search_radius": median_search_radius,
        "search_window_size": search_window,
        "narrowing_factor": narrowing_factor,
        "random_baseline_mae": random_mae,
        "improvement_vs_random": improvement_vs_random,
        "n_samples": len(y_true),
    }


def print_metrics(metrics: dict, model_name: str):
    print(f"\n{'═'*55}")
    print(f"  Results: {model_name}")
    print(f"{'═'*55}")
    print(f"  MAE:                  {metrics['mae']:>15,.0f}")
    print(f"  RMSE:                 {metrics['rmse']:>15,.0f}")
    print(f"  Median AE:            {metrics['median_ae']:>15,.0f}")
    print(f"  Random baseline MAE:  {metrics['random_baseline_mae']:>15,.0f}")
    print(f"  Improvement vs rand:  {metrics['improvement_vs_random']:>14.3f}x")
    print(f"  Narrowing factor:     {metrics['narrowing_factor']:>14.3f}x")
    print(f"\n  Within-range accuracy:")
    for tol, acc in metrics["within_range"].items():
        bar = "█" * int(acc * 30)
        print(f"    {tol:>5s}:  {acc:.3f}  {bar}")
    print(f"{'═'*55}")


def plot_all_results(
    results: dict,           # {model_name: {"metrics": ..., "history": ..., "y_true": ..., "y_pred": ...}}
    output_path: str = "results/analysis.png",
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    n_models = len(results)
    fig = plt.figure(figsize=(18, 12), facecolor="#0d1117")
    fig.suptitle("Bitcoin Nonce Predictor — Model Analysis", color="#e6edf3",
                 fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, n_models + 1, figure=fig, hspace=0.5, wspace=0.4)

    bg = "#0d1117"
    text_c = "#e6edf3"
    grid_c = "#21262d"
    colors = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff"]

    model_names = list(results.keys())

    # ── Row 0: Training loss curves ────────────────────────────────────────
    ax_loss = fig.add_subplot(gs[0, :])
    ax_loss.set_facecolor(bg)
    for i, (name, res) in enumerate(results.items()):
        h = res["history"]
        epochs = h["epochs"]
        ax_loss.plot(epochs, h["train_loss"], color=colors[i], label=f"{name} train", lw=1.5)
        ax_loss.plot(epochs, h["val_loss"], color=colors[i], lw=1.5, ls="--", alpha=0.6, label=f"{name} val")
    ax_loss.set_title("Training & Validation Loss (Huber)", color=text_c)
    ax_loss.set_xlabel("Epoch", color=text_c)
    ax_loss.set_ylabel("Loss", color=text_c)
    ax_loss.tick_params(colors=text_c)
    ax_loss.spines[:].set_color(grid_c)
    ax_loss.legend(fontsize=8, labelcolor=text_c, facecolor="#161b22", edgecolor=grid_c)
    ax_loss.grid(True, color=grid_c, alpha=0.6)

    # ── Row 1: Improvement vs random ──────────────────────────────────────
    ax_impr = fig.add_subplot(gs[1, 0])
    ax_impr.set_facecolor(bg)
    names = ["Random"] + model_names
    improvements = [1.0] + [results[n]["metrics"]["improvement_vs_random"] for n in model_names]
    bar_colors = ["#484f58"] + colors[:n_models]
    bars = ax_impr.bar(names, improvements, color=bar_colors, alpha=0.85, width=0.5)
    ax_impr.axhline(1.0, color="#484f58", ls="--", lw=1)
    for bar, val in zip(bars, improvements):
        ax_impr.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.3f}x", ha="center", va="bottom", color=text_c, fontsize=9)
    ax_impr.set_title("Improvement vs Random Baseline", color=text_c, fontsize=10)
    ax_impr.set_ylabel("MAE improvement", color=text_c)
    ax_impr.tick_params(colors=text_c)
    ax_impr.spines[:].set_color(grid_c)
    ax_impr.grid(True, axis="y", color=grid_c, alpha=0.5)

    # ── Row 1: Narrowing factor ────────────────────────────────────────────
    ax_narr = fig.add_subplot(gs[1, 1])
    ax_narr.set_facecolor(bg)
    narrowing = [results[n]["metrics"]["narrowing_factor"] for n in model_names]
    ax_narr.bar(model_names, narrowing, color=colors[:n_models], alpha=0.85, width=0.5)
    for i, (name, val) in enumerate(zip(model_names, narrowing)):
        ax_narr.text(i, val + max(narrowing)*0.01, f"{val:.1f}x",
                     ha="center", va="bottom", color=text_c, fontsize=9)
    ax_narr.set_title("Search Space Narrowing Factor", color=text_c, fontsize=10)
    ax_narr.set_ylabel("2³² / search_window", color=text_c)
    ax_narr.tick_params(colors=text_c)
    ax_narr.spines[:].set_color(grid_c)
    ax_narr.grid(True, axis="y", color=grid_c, alpha=0.5)

    # ── Row 1: Within-range accuracy ─────────────────────────────────────
    ax_acc = fig.add_subplot(gs[1, 2:] if n_models <= 2 else gs[1, 2])
    ax_acc.set_facecolor(bg)
    tols = list(list(results.values())[0]["metrics"]["within_range"].keys())
    x = np.arange(len(tols))
    width = 0.25
    for i, name in enumerate(model_names):
        accs = list(results[name]["metrics"]["within_range"].values())
        ax_acc.bar(x + i * width, accs, width, label=name, color=colors[i], alpha=0.85)
    ax_acc.set_title("Within-Range Accuracy", color=text_c, fontsize=10)
    ax_acc.set_xticks(x + width * (n_models - 1) / 2)
    ax_acc.set_xticklabels(tols)
    ax_acc.set_ylabel("Fraction correct", color=text_c)
    ax_acc.tick_params(colors=text_c)
    ax_acc.spines[:].set_color(grid_c)
    ax_acc.legend(fontsize=8, labelcolor=text_c, facecolor="#161b22", edgecolor=grid_c)
    ax_acc.grid(True, axis="y", color=grid_c, alpha=0.5)

    # ── Row 2: Prediction scatter plots ─────────────────────────────────
    for i, (name, res) in enumerate(results.items()):
        ax = fig.add_subplot(gs[2, i])
        ax.set_facecolor(bg)
        y_true = res["y_true"][:500]
        y_pred = res["y_pred"][:500]
        ax.scatter(y_true, y_pred, alpha=0.3, s=8, color=colors[i])
        # Perfect prediction line
        mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], color="#484f58", lw=1, ls="--")
        ax.set_title(f"{name}: Predicted vs Actual", color=text_c, fontsize=10)
        ax.set_xlabel("True Nonce", color=text_c)
        ax.set_ylabel("Predicted Nonce", color=text_c)
        ax.tick_params(colors=text_c)
        ax.spines[:].set_color(grid_c)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nPlot saved: {output_path}")
    plt.show()
