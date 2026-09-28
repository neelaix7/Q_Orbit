# degradation.py — reproducible degradation pipeline for robustness experiments
from __future__ import annotations
import numpy as np

def add_gaussian_noise(curves: np.ndarray, std: float, seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = curves + rng.normal(0, std, curves.shape)
    return np.clip(noisy, 0, 1)

def truncate_observation(curves: np.ndarray, fraction: float, seed=42) -> np.ndarray:
    """Simulate short observation window: keep first fraction*256 samples, pad remainder by repeating last value or zeros.
    fraction in (0,1]. For fair comparison we truncate + zero-pad to keep shape 256."""
    n = curves.shape[1]
    keep = int(min(max(1, int(n * fraction)), n))
    out = np.zeros_like(curves)
    out[:, :keep] = curves[:, :keep]
    # pad with last observed value (more physical than zeros)
    for i in range(len(curves)):
        out[i, keep:] = curves[i, keep-1] if keep>0 else 0
    return out

def add_missing(curves: np.ndarray, fraction: float, seed=42) -> np.ndarray:
    """Randomly drop fraction of samples and linearly interpolate (mimics sensor dropout)."""
    rng = np.random.default_rng(seed)
    out = curves.copy()
    n = curves.shape[1]
    for i in range(len(curves)):
        n_drop = int(n * fraction)
        if n_drop==0:
            continue
        idx = rng.choice(n, n_drop, replace=False)
        out[i, idx] = np.nan
        mask = ~np.isnan(out[i])
        if mask.sum() > 1:
            xp = np.where(mask)[0]
            fp = out[i, mask]
            out[i] = np.interp(np.arange(n), xp, fp)
    return out

def confidence(probs: np.ndarray) -> float:
    return float(np.max(probs))

def signal_quality(curve: np.ndarray) -> str:
    std = float(np.std(curve))
    if std < 0.08:
        return "low (flat/noisy)"
    if std < 0.18:
        return "medium"
    return "high"
