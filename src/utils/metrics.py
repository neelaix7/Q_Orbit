# metrics.py - accuracy, F1, confusion matrix, per-class report
from __future__ import annotations
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
    labels: List[int] = None,
) -> Dict[str, Any]:
    """Compute classification metrics for multi-class classification."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    accuracy = float(accuracy_score(y_true, y_pred))
    
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    
    per_class = {}
    for i, label in enumerate(labels):
        label_str = str(label)
        if label_str in report:
            per_class[f"class_{label}"] = {
                "precision": float(report[label_str]["precision"]),
                "recall": float(report[label_str]["recall"]),
                "f1": float(report[label_str]["f1-score"]),
                "support": int(report[label_str]["support"]),
            }
        else:
            per_class[f"class_{label}"] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": 0,
            }
    
    avg_metrics = {}
    for avg_method in ["macro", "micro", "weighted"]:
        if avg_method in report:
            avg_metrics[avg_method] = {
                "precision": float(report[avg_method]["precision"]),
                "recall": float(report[avg_method]["recall"]),
                "f1": float(report[avg_method]["f1-score"]),
            }
    
    f1_score = float(report["macro avg"]["f1-score"]) if "macro avg" in report else None

    per_class_f1 = [float(report[str(label)]["f1-score"]) for label in labels]
    
    result = {
        "accuracy": accuracy,
        "f1_macro": float(f1_score) if f1_score is not None else None,
        "per_class": per_class,
        "per_class_f1": per_class_f1,
    }
    
    if average is not None and average in avg_metrics:
        result["f1"] = per_class
    
    return result


def save_metrics_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    output_path: str = "results/reports/evaluation_report.md",
):
    """Generate a markdown evaluation report and save to output_path."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(max(max(y_true), max(y_pred)) + 1)]
    
    accuracy = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    report_date = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Evaluation Report\n\n")
        f.write(f"**Generated**: {report_date}\n")
        f.write(f"**Accuracy**: {accuracy:.4f}\n\n")
        
        f.write("## Classification Report\n\n")
        f.write("| Class | Precision | Recall | F1 | Support |\n")
        f.write("|-------|-----------|--------|-----|---------|\n")
        for i, c in enumerate(class_names):
            p = float(report[c]["precision"])
            r = float(report[c]["recall"])
            f1 = float(report[c]["f1-score"])
            s = int(report[c]["support"])
            f.write(f"| {c} | {p:.4f} | {r:.4f} | {f1:.4f} | {s} |\n")
        f.write("\n")
        
        f.write("## Confusion Matrix\n\n")
        f.write("| True -> Pred Down")
        for c in class_names:
            f.write(f" | {c}")
        f.write(" |\n")
        f.write("|" + "---|" * (len(class_names) + 1) + "\n")
        for i, c in enumerate(class_names):
            row_str = f"| {c}"
            for j in range(len(class_names)):
                row_str += f" | {cm[i, j]}"
            row_str += " |"
            f.write(row_str + "\n")
        f.write("\n")
        
        f.write("## Per-Class F1 Scores\n\n")
        f.write("| Class | F1 Score | Precision | Recall | Support |\n")
        f.write("|-------|----------|-----------|--------|---------|\n")
        for i, c in enumerate(class_names):
            p = float(report[c]["precision"])
            r = float(report[c]["recall"])
            f1 = float(report[c]["f1-score"])
            s = int(report[c]["support"])
            f.write(f"| {c} | {f1:.4f} | {p:.4f} | {r:.4f} | {s} |\n")