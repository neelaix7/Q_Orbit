# accelerators.py — Python ↔ C++ bridge with fallback + benchmark helpers
from __future__ import annotations
import time
import numpy as np
from typing import Tuple

try:
    from src.cpp import qorbit_cpp
    HAS_CPP = qorbit_cpp is not None
except Exception:
    HAS_CPP = False
    qorbit_cpp = None

# Python fallbacks (identical API)
def python_extract_time_features(curves: np.ndarray) -> np.ndarray:
    # curves: (N, L)
    from src.classical.feature_engineering import extract_time_domain_features
    N = curves.shape[0]
    out = np.zeros((N, 11), dtype=float)
    for i in range(N):
        d = extract_time_domain_features(curves[i])
        out[i] = [d[k] for k in ["mean","std","variance","median","skewness","kurtosis","peak_to_peak","amplitude","energy","above_median_fraction","min_value"]]
    return out

def python_generate_synthetic(N: int, L: int = 256, noise_std: float = 0.02, seed: int = 42) -> np.ndarray:
    from src.simulator.lightcurve_generator import generate_single_light_curve
    # Use the Python generator loop (slow but correct) — we set global RNG seed separately
    import numpy as np
    # For fair timing, use vectorized quick loop with our simplified physics not full simulator to isolate feature extraction
    # We'll call generate_single_light_curve N times
    curves = []
    for _ in range(N):
        # pick random class 0..4
        cls = np.random.randint(0,5)
        c, _ = generate_single_light_curve(cls, n_samples=L, noise_std=noise_std)
        curves.append(c)
    return np.array(curves, dtype=float)

def extract_time_features(curves: np.ndarray, use_cpp: bool = True) -> np.ndarray:
    if use_cpp and HAS_CPP:
        # pybind expects double array
        arr = np.ascontiguousarray(curves, dtype=np.float64)
        return np.array(qorbit_cpp.extract_time_features(arr), dtype=np.float64)
    else:
        return python_extract_time_features(curves)

def generate_synthetic(N: int, L: int = 256, noise_std: float = 0.02, seed: int = 42, use_cpp: bool = True) -> np.ndarray:
    if use_cpp and HAS_CPP:
        return np.array(qorbit_cpp.generate_synthetic(N, L, noise_std, seed), dtype=np.float64)
    else:
        return python_generate_synthetic(N, L, noise_std, seed)

def add_gaussian_noise(curves: np.ndarray, std: float = 0.05, seed: int = 42, use_cpp: bool = True) -> np.ndarray:
    if use_cpp and HAS_CPP:
        arr = np.ascontiguousarray(curves, dtype=np.float64)
        qorbit_cpp.add_gaussian_noise(arr, float(std), int(seed))
        return arr
    else:
        rng = np.random.default_rng(seed)
        noisy = curves + rng.normal(0, std, curves.shape)
        return np.clip(noisy, 0, 1)

def benchmark_python_vs_cpp(N: int = 2000, L: int = 256, seed: int = 42) -> dict:
    """Benchmark Python vs C++ for feature extraction + generation + noise.
    Returns dict with times and speedup.
    """
    rng_curves = np.random.default_rng(seed)
    # generate test curves via python (fast random for timing fairness)
    curves = rng_curves.random((N, L))

    results = {"N": N, "L": L, "seed": seed, "has_cpp": HAS_CPP}

    # --- feature extraction Python ---
    t0 = time.perf_counter()
    py_feats = python_extract_time_features(curves)
    t1 = time.perf_counter()
    py_time = (t1 - t0) * 1000
    results["python_time_features_ms"] = py_time

    # --- feature extraction C++ ---
    if HAS_CPP:
        t0 = time.perf_counter()
        cpp_feats = extract_time_features(curves, use_cpp=True)
        t1 = time.perf_counter()
        cpp_time = (t1 - t0) * 1000
        results["cpp_time_features_ms"] = cpp_time
        results["speedup_features"] = py_time / max(cpp_time, 1e-9)
        # correctness check
        results["max_abs_diff_features"] = float(np.max(np.abs(py_feats - cpp_feats)))
    else:
        results["cpp_time_features_ms"] = None
        results["speedup_features"] = None

    # --- synthetic generation ---
    # Python gen (use simplified quick loop to avoid 60s full simulator; we time only 200 curves for Python)
    N_gen = min(N, 200)
    t0 = time.perf_counter()
    _ = python_generate_synthetic(N_gen, L, 0.02, seed)
    t1 = time.perf_counter()
    py_gen_ms = (t1 - t0) * 1000
    py_gen_per = py_gen_ms / N_gen
    # extrapolate to N
    results["python_gen_ms_200"] = py_gen_ms
    results["python_gen_ms_extrapolated_2000"] = py_gen_per * N

    if HAS_CPP:
        t0 = time.perf_counter()
        _ = generate_synthetic(N, L, 0.02, seed, use_cpp=True)
        t1 = time.perf_counter()
        cpp_gen_ms = (t1 - t0) * 1000
        results["cpp_gen_ms"] = cpp_gen_ms
        results["speedup_generation"] = (py_gen_per * N) / max(cpp_gen_ms, 1e-9)
    else:
        results["cpp_gen_ms"] = None
        results["speedup_generation"] = None

    # --- noise ---
    curves2 = curves.copy()
    t0 = time.perf_counter()
    _ = add_gaussian_noise(curves2, 0.05, seed, use_cpp=False)
    t1 = time.perf_counter()
    py_noise_ms = (t1 - t0) * 1000
    results["python_noise_ms"] = py_noise_ms
    if HAS_CPP:
        curves3 = curves.copy()
        t0 = time.perf_counter()
        _ = add_gaussian_noise(curves3, 0.05, seed, use_cpp=True)
        t1 = time.perf_counter()
        cpp_noise_ms = (t1 - t0) * 1000
        results["cpp_noise_ms"] = cpp_noise_ms
        results["speedup_noise"] = py_noise_ms / max(cpp_noise_ms, 1e-9)
    else:
        results["cpp_noise_ms"] = None
        results["speedup_noise"] = None

    return results
