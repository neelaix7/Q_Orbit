"""Metrics for QU-AI-GUARD evaluation (honest numbers, no inflation).

Covers QBER, secure-key-rate, classifier accuracy (ROC / F1 / confusion /
false-alarm rate) and a small helper to serialise results as JSON + Markdown.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)


def binary_entropy(q: float) -> float:
    q = np.clip(q, 1e-9, 1.0 - 1e-9)
    return float(-q * math.log2(q) - (1 - q) * math.log2(1 - q))


def secure_fraction(qber: float, ec_eff: float = 1.15, margin: float = 0.02) -> float:
    """BB84 asymptotic secure-key fraction: 1 - f*EC*h2(QBER) - margin."""
    return max(0.0, 1.0 - ec_eff * binary_entropy(qber) - margin)


def mean_qber(qber_list: list[float]) -> float:
    return float(np.mean(qber_list)) if qber_list else 0.0


def mean_secure_bps(bps_list: list[float]) -> float:
    return float(np.mean(bps_list)) if bps_list else 0.0


def classifier_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                       threshold: float = 0.5) -> dict:
    """Full binary-classifier metrics at a given operating threshold."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    fpr, tpr, th = roc_curve(y_true, y_prob)
    return {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "false_alarm_rate": _false_alarm(y_true, y_pred),
        "threshold": float(threshold),
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
    }


def _false_alarm(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """False alarm rate = FP / (FP + TN) among negative (clean) samples."""
    neg = np.sum(y_true == 0)
    if neg == 0:
        return 0.0
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    return float(fp / neg)


def key_rate_improvement(bps_fixed: list[float], bps_tuned: list[float]) -> float:
    """Relative secure-key-rate gain of AI-tuned vs fixed config (%)."""
    f = float(np.mean(bps_fixed))
    t = float(np.mean(bps_tuned))
    if f <= 0:
        return 0.0
    return 100.0 * (t - f) / f


def save_metrics(metrics: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)


def write_report(report_md: str, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_md)
