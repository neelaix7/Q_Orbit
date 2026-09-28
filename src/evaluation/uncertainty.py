# uncertainty.py — scientifically meaningful uncertainty
from __future__ import annotations
import numpy as np

def uncertainty_metrics(proba: np.ndarray) -> dict:
    """
    proba: (5,) softmax
    Returns entropy, margin, calibrated label
    """
    proba = np.asarray(proba, dtype=float).ravel()
    proba = np.nan_to_num(proba, nan=0.0, posinf=0.0, neginf=0.0)
    if proba.size == 0:
        return {
            "probability": 0.0, "predicted_class": -1, "entropy": 0.0,
            "norm_entropy": 1.0, "margin": 0.0,
            "uncertainty_label": "High", "uncertainty_score": 1.0,
        }
    proba = proba / proba.sum() if proba.sum()>0 else np.full_like(proba, 1.0/len(proba))
    max_p = float(np.max(proba))
    sorted_p = np.sort(proba)
    margin = float(sorted_p[-1] - sorted_p[-2]) if len(sorted_p)>=2 else max_p
    entropy = float(-np.sum(proba*np.log(proba+1e-9)))
    # Normalize entropy 0-1 (max log 5 ≈1.609); guard the single-class case
    max_entropy = float(np.log(len(proba))) if len(proba) > 1 else 1.0
    norm_ent = float(np.clip(entropy / max_entropy, 0.0, 1.0)) if max_entropy > 0 else 0.0
    if norm_ent < 0.4 and margin > 0.3 and max_p > 0.7:
        label = "Low"
    elif norm_ent < 0.7 and margin > 0.15:
        label = "Moderate"
    else:
        label = "High"
    # Simple calibration stub: ECE proxy not computed here
    return {
        "probability": max_p,
        "predicted_class": int(np.argmax(proba)),
        "entropy": entropy,
        "norm_entropy": float(norm_ent),
        "margin": margin,
        "uncertainty_label": label,  # Low/Moderate/High
        "uncertainty_score": float(norm_ent)  # 0 certain, 1 uncertain
    }
