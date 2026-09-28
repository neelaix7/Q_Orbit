"""Plotting helpers for QU-AI-GUARD (matplotlib; no seaborn required)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "#0b1026",
        "axes.facecolor": "#0b1026",
        "axes.edgecolor": "#33406e",
        "axes.labelcolor": "#cfe0ff",
        "xtick.color": "#9fb3d8",
        "ytick.color": "#9fb3d8",
        "text.color": "#cfe0ff",
        "font.size": 11,
    })


def plot_elevation_profile(pass_df, ax=None) -> plt.Figure:
    """Elevation vs time over a pass."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 3), dpi=110)
    ax.plot(pass_df["t_sec"] / 60.0, pass_df["elevation_deg"],
            color="#4fc3f7", lw=2)
    ax.axhline(15, color="#ffd54f", ls="--", lw=1, label="min usable elevation")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Elevation (deg)")
    ax.set_title("Satellite pass geometry"); ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    return fig


def plot_qber_series(qber_clean, qber_attack=None, ax=None) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(7, 3), dpi=110)
    t = np.arange(len(qber_clean))
    ax.plot(t, qber_clean, color="#26a69a", lw=2, label="clean (atmospheric noise)")
    if qber_attack is not None:
        ax.plot(t, qber_attack, color="#ef5350", lw=2, ls="--",
                label="Eve attack active")
    ax.axhline(0.11, color="#ffd54f", ls=":", lw=1, label="typical QBER threshold")
    ax.set_xlabel("Sample"); ax.set_ylabel("QBER")
    ax.set_title("QBER time series"); ax.legend(frameon=False); ax.grid(alpha=0.2)
    return fig


def plot_key_rate_comparison(bps_fixed, bps_tuned, ax=None) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    labels = ["Fixed config", "AI-tuned"]
    vals = [float(np.mean(bps_fixed)) / 1e3, float(np.mean(bps_tuned)) / 1e3]
    colors = ["#64748b", "#4fc3f7"]
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02 * max(vals, default=1),
                f"{v:.1f} kbps", ha="center", color="#cfe0ff", fontweight="bold")
    ax.set_ylabel("Mean secure key rate (kbps)")
    ax.set_title("AI auto-tune vs fixed QKD config")
    ax.grid(axis="y", alpha=0.2)
    return fig


def plot_roc(fpr, tpr, auc, ax=None) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=110)
    ax.plot(fpr, tpr, color="#4fc3f7", lw=2.2, label=f"AI detector (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#64748b", ls="--", lw=1, label="naive / chance")
    ax.set_xlabel("False alarm rate"); ax.set_ylabel("Detection rate")
    ax.set_title("Eve detector ROC"); ax.legend(frameon=False); ax.grid(alpha=0.2)
    return fig


def plot_confusion(cm, class_names=("Clean", "Eve"), ax=None) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(4.6, 4), dpi=110)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="magma", colorbar=False)
    ax.set_title("Eve detector confusion matrix")
    return fig


def plot_attack_sweep(strengths, detection_rates, ax=None) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=110)
    ax.plot(strengths, detection_rates, color="#ba68c8", marker="o", lw=2)
    ax.set_xlabel("Attack strength (fraction of pulses)"); ax.set_ylabel("Detection rate")
    ax.set_title("Attack strength vs detection rate"); ax.grid(alpha=0.2)
    return fig


def save_fig(fig: plt.Figure, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
