# physics_informed.py — 21 baseline organized as physics-informed signal features (no claim of direct physical measurement)
from __future__ import annotations
import numpy as np
from src.classical.feature_engineering import extract_all_features

# Keep 21 baseline features but reorganize and label as proxies
# TIME DOMAIN (statistical brightness proxies)
TIME_DOMAIN = ["mean","std","variance","median","skewness","kurtosis","peak_to_peak","amplitude","energy","above_median_fraction","min_value"]
# FREQUENCY DOMAIN (spectral)
FREQUENCY_DOMAIN = ["dominant_freq","dominant_magnitude","harmonic_energy","fft_entropy","n_harmonics"]
# TEMPORAL / PHYSICAL PROXIES (signal proxies, not direct physical quantities)
PHYSICAL_PROXIES = ["flash_count","rise_time","fall_time","eclipse_fraction","period_estimate"]

GROUPS = {"time_domain": TIME_DOMAIN, "frequency_domain": FREQUENCY_DOMAIN, "physical_proxies": PHYSICAL_PROXIES}
ALL_ORDERED = TIME_DOMAIN + FREQUENCY_DOMAIN + PHYSICAL_PROXIES  # =21 in fixed order matching feature_engineering.py

FEATURE_DESCRIPTIONS = {
    "mean": "Mean brightness — proxy for albedo",
    "std": "Variability — proxy for tumble amplitude",
    "variance": "Variance — dispersion proxy",
    "median": "Median — robust brightness",
    "skewness": "Asymmetry proxy",
    "kurtosis": "Peakiness proxy",
    "peak_to_peak": "Range — max-min",
    "amplitude": "Peak brightness",
    "energy": "Sum of squares — integrated flux proxy",
    "above_median_fraction": "Duty cycle proxy",
    "min_value": "Minimum — eclipse depth proxy",
    "dominant_freq": "FFT peak frequency — rotation rate proxy",
    "dominant_magnitude": "Peak spectral power",
    "harmonic_energy": "Harmonic power — non-sinusoidal proxy",
    "fft_entropy": "Spectral complexity",
    "n_harmonics": "Count of significant harmonics",
    "flash_count": "Flash frequency — specular glint proxy",
    "rise_time": "Rise behavior — facet orientation proxy",
    "fall_time": "Fall behavior — shadow proxy",
    "eclipse_fraction": "Eclipse-related characteristic — fraction below 20th pct",
    "period_estimate": "Autocorr lag — crude period proxy",
}

def extract_physics_informed(curve: np.ndarray, sampling_interval: float = 5.0) -> dict:
    """Returns dict with grouped physics-informed features and flat vector (21). All are signal proxies."""
    vec = extract_all_features(curve, sampling_interval=sampling_interval)
    # map order to names
    named = {name: float(vec[i]) for i, name in enumerate(ALL_ORDERED)}
    groups = {g: {k: named[k] for k in members} for g, members in GROUPS.items()}
    return {"vector": vec, "named": named, "groups": groups, "descriptions": FEATURE_DESCRIPTIONS}
