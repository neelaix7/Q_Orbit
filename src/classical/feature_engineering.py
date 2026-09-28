# feature_engineering.py - traditional light-curve features (FFT, moments, period)
from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
from scipy import signal, stats


def extract_time_domain_features(curve: np.ndarray) -> Dict[str, float]:
    """Extract time-domain features from a normalized light curve."""
    c = np.asarray(curve, dtype=float).ravel()
    if c.size == 0:
        return {
            "mean": 0.0, "std": 0.0, "variance": 0.0, "median": 0.0,
            "skewness": 0.0, "kurtosis": 0.0, "peak_to_peak": 0.0,
            "amplitude": 0.0, "energy": 0.0, "above_median_fraction": 0.0,
            "min_value": 0.0,
        }
    c = np.nan_to_num(c, nan=0.0, posinf=1.0, neginf=0.0)

    mean_c = float(np.mean(c))
    std_c = float(np.std(c))
    var_c = float(np.var(c))
    median_c = float(np.median(c))
    
    # Moments are undefined for a constant signal; scipy emits a
    # "precision loss / catastrophic cancellation" warning and returns NaN.
    if var_c > 1e-12:
        skew_c = float(stats.skew(c))
        kurt_c = float(stats.kurtosis(c))
    else:
        skew_c = 0.0
        kurt_c = 0.0
    skew_c = 0.0 if not np.isfinite(skew_c) else skew_c
    kurt_c = 0.0 if not np.isfinite(kurt_c) else kurt_c
    
    peak_to_peak = float(np.ptp(c))
    amplitude = float(np.max(c))
    energy = float(np.sum(c ** 2))
    
    above_median = float(np.sum(c > median_c) / len(c))
    min_val = float(np.min(c))
    
    return {
        "mean": mean_c,
        "std": std_c,
        "variance": var_c,
        "median": median_c,
        "skewness": skew_c,
        "kurtosis": kurt_c,
        "peak_to_peak": peak_to_peak,
        "amplitude": amplitude,
        "energy": energy,
        "above_median_fraction": above_median,
        "min_value": min_val,
    }


def extract_frequency_domain_features(curve: np.ndarray,
                                      sampling_interval: float = 5.0) -> Dict[str, float]:
    """Extract frequency-domain features using FFT."""
    c = curve.astype(float)
    
    fft = np.fft.fft(c)
    fft_freq = np.fft.fftfreq(len(c), d=sampling_interval)
    
    magnitude = np.abs(fft)
    
    pos_mask = fft_freq > 0
    pos_freq = fft_freq[pos_mask]
    pos_mag = magnitude[pos_mask]
    
    dominant_freq = 0.0
    dominant_mag = 0.0
    harmonic_energy = 0.0
    n_harmonics = 0
    fft_entropy = 0.0
    
    if len(pos_freq) > 1:
        skip = max(1, int(1.0 / (sampling_interval * len(c) / 720.0)))
        freq_subset = pos_freq[skip:]
        mag_subset = pos_mag[skip:]
        
        if len(freq_subset) > 0:
            peak_idx = np.argmax(mag_subset)
            dominant_freq = float(freq_subset[peak_idx])
            dominant_mag = float(mag_subset[peak_idx])
        
        # Harmonic energy (sum of magnitudes squared beyond fundamental)
        if 'mag_subset' in dir() and len(mag_subset) > 0:
            threshold = dominant_mag * 0.1 if dominant_mag > 0 else 0
            harmonic_energy = float(np.sum(mag_subset ** 2))
            n_harmonics = int(np.sum(mag_subset > threshold))
    
    # FFT entropy
    norm_mag = magnitude / (np.sum(magnitude) + 1e-10)
    fft_entropy = float(-np.sum(norm_mag * np.log(norm_mag + 1e-10)))
    
    return {
        "dominant_freq": float(dominant_freq),
        "dominant_magnitude": float(dominant_mag),
        "harmonic_energy": float(harmonic_energy),
        "fft_entropy": float(fft_entropy),
        "n_harmonics": int(n_harmonics),
    }


def extract_temporal_pattern_features(curve: np.ndarray) -> Dict[str, float]:
    """Extract features describing the temporal pattern of the light curve."""
    c = curve.astype(float)
    
    from scipy.signal import find_peaks
    
    # Flash count: number of local maxima above a threshold
    threshold = np.percentile(c, 75) + 3 * np.std(c)
    if threshold >= np.max(c):
        threshold = np.max(c) * 0.9
    
    peaks, _ = find_peaks(c, height=threshold)
    flash_count = int(len(peaks))
    
    # Rise and fall times
    min_idx = int(np.argmin(c))
    max_idx = int(np.argmax(c))
    n = len(c)
    
    if max_idx >= min_idx:
        rise_time = (max_idx - min_idx) / n * 720.0
        fall_time = (n - max_idx + min_idx) / n * 720.0
    else:
        rise_time = (max_idx + n - min_idx) / n * 720.0
        fall_time = (min_idx + n - max_idx) / n * 720.0
    
    # Eclipse fraction
    bright_threshold = float(np.percentile(c, 20))
    eclipse_fraction = float(np.sum(c < bright_threshold) / n)
    
    # Autocorrelation period estimate
    autocorr = np.correlate(c - np.mean(c), c - np.mean(c), mode='full')[n-1:]
    # A constant signal has zero autocorrelation energy -> guard the normalization.
    if autocorr[0] > 1e-12:
        autocorr = autocorr / autocorr[0]
    else:
        autocorr = np.zeros_like(autocorr)
    autocorr = np.nan_to_num(autocorr, nan=0.0, posinf=0.0, neginf=0.0)
    period_estimate = 0.0
    for lag in range(1, min(50, n // 4)):
        if abs(autocorr[lag]) < 0.5:
            period_estimate = float(lag)
            break
    
    return {
        "flash_count": float(flash_count),
        "rise_time": float(rise_time),
        "fall_time": float(fall_time),
        "eclipse_fraction": float(eclipse_fraction),
        "period_estimate": float(period_estimate),
    }


def extract_all_features(curve: np.ndarray,
                         sampling_interval: float = 5.0) -> np.ndarray:
    """Extract all features from a light curve and return as a flat feature vector."""
    curve = np.asarray(curve, dtype=float).ravel()
    if curve.size < 2:
        # FFT / autocorrelation are undefined below 2 samples — return a neutral vector
        # so callers always receive a well-formed, finite 21-dim feature vector.
        return np.zeros(21, dtype=float)
    # Sanitize before any FFT/correlation: NaN or inf would otherwise propagate
    # into the frequency domain and yield a non-finite feature vector.
    curve = np.nan_to_num(curve, nan=0.0, posinf=1.0, neginf=0.0)
    time_feats = extract_time_domain_features(curve)
    freq_feats = extract_frequency_domain_features(curve, sampling_interval)
    pattern_feats = extract_temporal_pattern_features(curve)
    
    feature_names = (
        ["mean", "std", "variance", "median", "skewness", "kurtosis",
         "peak_to_peak", "amplitude", "energy", "above_median_fraction",
         "min_value"]
        + ["dominant_freq", "dominant_magnitude", "harmonic_energy", "fft_entropy", "n_harmonics"]
        + ["flash_count", "rise_time", "fall_time", "eclipse_fraction", "period_estimate"]
    )
    
    features = []
    for name in feature_names:
        # Get from each category - time features first, then freq, then pattern
        if name in time_feats:
            features.append(time_feats[name])
        elif name in freq_feats:
            features.append(freq_feats[name])
        elif name in pattern_feats:
            features.append(pattern_feats[name])
        else:
            features.append(0.0)
    
    return np.array(features, dtype=float)