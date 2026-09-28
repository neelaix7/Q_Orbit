# robustness_score.py — normalized robustness metric, documented
from __future__ import annotations
import numpy as np, json, os

def robustness_score(clean_acc: float, degraded_accs: list[float]) -> float:
    """
    Robustness Score RS = mean( acc_degraded / acc_clean )
    where degraded_accs are accuracies under each degradation condition (excluding clean).
    Range [0,1]; 1 = perfectly robust (no drop), 0 = completely brittle.
    If clean_acc==0, returns 0.
    Also defined as 1 - mean_drop where drop = 1 - acc_deg/acc_clean.
    """
    if clean_acc <= 0: return 0.0
    if len(degraded_accs) == 0: return 0.0
    ratios = [float(a / clean_acc) for a in degraded_accs]
    return float(np.mean(ratios))

def compute_all_robustness(robustness_json_path: str) -> dict:
    """
    Reads robustness_noise.json etc and computes RS per model.
    Expects list of dicts with keys noise/Classical_CNN etc and clean entry first (0.0)
    """
    data = json.load(open(robustness_json_path))
    clean = data[0]
    models = [k for k in clean.keys() if k not in ("noise","observation_fraction","missing_fraction")]
    scores = {}
    for m in models:
        degraded = [d[m] for d in data[1:]]
        scores[m] = robustness_score(clean[m], degraded)
    return scores
