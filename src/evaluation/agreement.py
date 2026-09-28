# agreement.py — Model Agreement Engine
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import jensenshannon

def model_agreement(preds: dict[str,int], probas: dict[str,np.ndarray]) -> dict:
    """
    preds: {"classical":0, "quantum":1, "hybrid":0, "adaptive":0}
    probas: {"classical": (5,), "quantum": (5,), ...}
    Returns agreement metrics
    """
    classes = list(preds.values())
    uniq, counts = np.unique(classes, return_counts=True)
    max_count = int(np.max(counts))
    n = len(classes)
    if max_count == n: consensus = "Model consensus detected."
    elif max_count >= 2: consensus = "Model disagreement detected (2/3 agree)."
    else: consensus = "Model disagreement detected (all differ)."
    # probability divergence: avg pairwise JS distance
    keys = list(probas.keys())
    js_vals = []
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            p, q = probas[keys[i]], probas[keys[j]]
            js = jensenshannon(p, q, base=np.e)
            js_vals.append(float(js) if np.isfinite(js) else 0.0)
    avg_js = float(np.mean(js_vals)) if js_vals else 0.0
    # confidence per model
    confs = {k: float(np.max(v)) for k,v in probas.items()}
    # entropy
    ents = {k: float(-np.sum(v*np.log(v+1e-9))) for k,v in probas.items()}
    return {
        "agreement_count": f"{max_count}/{n}",
        "consensus": consensus,
        "avg_js_divergence": avg_js,
        "confidences": confs,
        "entropies": ents,
        "predictions": preds,
        "probas": {k:v.tolist() for k,v in probas.items()}
    }
