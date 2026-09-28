# signal_quality.py — Signal Quality Analyzer: Light Curve → quality metrics
from __future__ import annotations
import numpy as np

def analyze_signal_quality(curve: np.ndarray, sampling_interval: float = 2.82) -> dict:
    """
    720s window / 256 samples → dt≈2.82s. Returns signal-quality proxies, NOT claimed physical truths.
    """
    curve = np.asarray(curve, dtype=float).ravel()
    if curve.size == 0:
        return {
            "noise_estimate": 0.0, "signal_std": 0.0, "missing_pct": 1.0,
            "completeness": 0.0, "observation_coverage": 0.0,
            "periodicity_score": 0.0, "spectral_complexity": 0.0,
            "signal_quality_score": 0.0, "quality_label": "Low",
        }
    missing_pct = float(np.isnan(curve).sum() / curve.size)
    c = np.nan_to_num(curve, nan=0.0, posinf=1.0, neginf=0.0)
    n = len(c)
    # Noise estimate: residual after median filter (3-point) or std of diff
    if n >= 3:
        smoothed = np.convolve(c, np.ones(3)/3, mode='same')
        resid = c - smoothed
        noise_est = float(np.std(resid))
    else:
        noise_est = float(np.std(c))
    signal_std = float(np.std(c))
    # Missing (measured on the raw curve before sanitizing, so dropouts are counted)
    # Observation coverage: fraction of non-zero? Use fraction where curve not constant-pad (if truncated, last-value pad creates flat tail)
    # Estimate completeness as 1 - flat-tail fraction
    diff = np.abs(np.diff(c))
    flat_tail = 0
    if n > 10:
        # trailing flat segment where diff==0
        tail = np.where(diff < 1e-6)[0]
        # count from last non-flat backwards
        if len(tail) and tail[-1] == n-2:
            # estimate
            flat_tail = int(np.sum(diff[-int(n*0.5):] < 1e-6))
    completeness = float(1.0 - flat_tail / max(n,1))
    completeness = np.clip(completeness, 0, 1)
    # Periodicity score: ratio of dominant FFT peak to mean spectral power
    try:
        w = np.hanning(n) if n>=4 else np.ones(n)
        spec = np.abs(np.fft.rfft(c*w))
        freqs = np.fft.rfftfreq(n, d=sampling_interval)
        if len(spec) > 2:
            peak = float(np.max(spec[1:]))
            mean_spec = float(np.mean(spec[1:]) + 1e-9)
            periodicity = float(np.clip(peak / (mean_spec*5), 0, 1))  # normalize
        else:
            periodicity = 0.0
    except Exception: periodicity = 0.0
    # Spectral complexity: fft_entropy proxy normalized
    try:
        mag = spec / (np.sum(spec)+1e-9)
        entropy = float(-np.sum(mag * np.log(mag+1e-9)))
        # max entropy log(N/2), normalize
        max_ent = np.log(len(spec)+1e-9)
        complexity = float(np.clip(entropy / max_ent, 0, 1)) if max_ent>0 else 0.5
    except Exception: complexity = 0.5
    # Composite quality score (higher = better): weighted
    # Low noise, high std, high completeness, high periodicity → high quality
    quality = float(np.clip(0.35*(1 - np.clip(noise_est/0.15,0,1)) + 0.25*np.clip(signal_std/0.3,0,1) + 0.20*completeness + 0.20*periodicity, 0, 1))
    return {
        "noise_estimate": noise_est,
        "signal_std": signal_std,
        "missing_pct": missing_pct,
        "completeness": float(completeness),
        "observation_coverage": float(completeness),
        "periodicity_score": float(periodicity),
        "spectral_complexity": float(complexity),
        "signal_quality_score": quality,  # 0-1
        "quality_label": "High" if quality>0.66 else "Medium" if quality>0.33 else "Low"
    }
