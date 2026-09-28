# viz.py - plotting helpers
from __future__ import annotations
from typing import Dict, List, Any
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_light_curve(curve: np.ndarray, title: str = "Light Curve", show: bool = False):
    """Plot a single light curve."""
    plt.figure(figsize=(8, 3))
    plt.plot(curve, color="#4472C7", linewidth=1.5)
    plt.title(title, fontsize=14)
    plt.xlabel("Time step (5s cadence)", fontsize=10)
    plt.ylabel("Normalized brightness", fontsize=10)
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    if show:
        plt.show()


def plot_comparison_chart(
    results: Dict[str, Dict[str, float]],
    output_path: str = "results/figures/comparison.png",
):
    """Plot comparison bar chart across models."""
    plt.figure(figsize=(10, 6))
    
    model_names = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in model_names]
    f1_scores = [results[m]["f1_macro"] for m in model_names]
    
    x_pos = np.arange(len(model_names))
    plt.bar(x_pos - 0.2, accuracies, width=0.4, label="Accuracy", color="#4472C7")
    plt.bar(x_pos + 0.2, f1_scores, width=0.4, label="F1-Macro", color="#ED7D31")
    
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("Model Comparison: Accuracy and F1-Macro")
    plt.xticks(x_pos, model_names, rotation=15, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison chart to {output_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str = "results/figures/confusion_matrix.png",
):
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    plt.matshow(cm, cmap="Blues", fignum=None)
    plt.colorbar(label="Count")
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=15, ha="right")
    plt.yticks(tick_marks, class_names)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center", 
                color="black" if cm[i, j] < cm.max() / 2 else "white",
            )
    
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def plot_noise_sweep(
    noise_results: Dict[float, Dict[str, float]],
    output_path: str = "results/figures/noise_sweep.png",
):
    """Plot how model accuracy varies with noise level."""
    plt.figure(figsize=(10, 6))
    
    noise_levels = sorted(noise_results.keys())
    accuracies = []
    f1_scores = []
    
    for nl in noise_levels:
        accuracies.append(noise_results[nl]["accuracy"])
        f1_scores.append(noise_results[nl]["f1_macro"])
    
    plt.plot(noise_levels, accuracies, "o-", label="Accuracy", color="#4472C7", linewidth=2)
    plt.plot(noise_levels, f1_scores, "s-", label="F1-Macro", color="#ED7D31", linewidth=2)
    
    plt.xscale("log")
    plt.xlabel("Noise standard deviation (log scale)")
    plt.ylabel("Score")
    plt.title("Model Performance vs Noise Level")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved noise sweep plot to {output_path}")


def plot_learning_curve(
    train_scores: List[float],
    val_scores: List[float],
    output_path: str = "results/figures/learning_curve.png",
):
    """Plot training vs validation learning curve."""
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_scores) + 1)
    plt.plot(epochs, train_scores, "o-", label="Training score", color="#4472C7")
    plt.plot(epochs, val_scores, "s-", label="Validation score", color="#ED7D31")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved learning curve to {output_path}")


def plotly_bar_chart(
    labels: List[str],
    values: List[float],
    title: str,
    output_path: str,
) -> None:
    """Create a Plotly bar chart."""
    fig = go.Figure(data=[
        go.Bar(y=labels, x=values, orientation="h", marker_color="#4472C7")
    ])
    fig.update_layout(
        title=title,
        xaxis_title="Value",
        yaxis_title="Class",
        template="plotly_white",
        height=400,
    )
    fig.write_image(output_path) if output_path else fig.show()