# lightcurve_generator.py - generates realistic light curves for all 5 classes
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, List
import numpy as np

# Module-level RNG, seeded for reproducibility
_RNG = np.random.default_rng(SEED if 'SEED' in dir() else 42)

from .config import SEED
_RNG = np.random.default_rng(SEED)

from .physics import (
    ObjectState,
    compute_tumble_orientation,
    specular_reflectance,
    diffuse_reflectance,
    compute_light_curve_from_faces,
)
from .config import (
    N_SAMPLES,
    SAMPLE_INTERVAL_SEC,
    TOTAL_DURATION_SEC,
    N_TIME_STEPS,
    EARTH_SHADOW_CONE_ANGLE,
    TUMBLE_PERIOD_RANGE,
    ASPECT_RATIO_RANGE,
    REFLECTIVITY_RANGE,
)


@dataclass
class ClassParams:
    """Tumble and optical parameters per object class."""
    tumble_period: Tuple[float, float]
    aspect_ratio: Tuple[float, float]
    reflectivity: Tuple[float, float]
    n_faces: int


CLASS_PARAMS: Dict[int, ClassParams] = {
    0: ClassParams(tumble_period=(60, 90), aspect_ratio=(1.2, 2.5), reflectivity=(0.3, 0.5), n_faces=4),
    1: ClassParams(tumble_period=(150, 280), aspect_ratio=(1.5, 4.0), reflectivity=(0.1, 0.3), n_faces=6),
    2: ClassParams(tumble_period=(350, 500), aspect_ratio=(4.0, 8.0), reflectivity=(0.1, 0.25), n_faces=8),
    3: ClassParams(tumble_period=(30, 50), aspect_ratio=(0.5, 1.8), reflectivity=(0.05, 0.15), n_faces=3),
    4: ClassParams(tumble_period=(350, 500), aspect_ratio=(2.0, 5.0), reflectivity=(0.05, 0.12), n_faces=6),
}


def sample_class_params(cls: int) -> ClassParams:
    """Sample tumble/optical parameters for a given class."""
    cp = CLASS_PARAMS[cls]
    tp = float(_RNG.uniform(cp.tumble_period[0], cp.tumble_period[1]))
    ar = float(_RNG.uniform(cp.aspect_ratio[0], cp.aspect_ratio[1]))
    rf = float(_RNG.uniform(cp.reflectivity[0], cp.reflectivity[1]))
    return ClassParams(tumble_period=tp, aspect_ratio=ar, reflectivity=rf, n_faces=cp.n_faces)


def generate_sun_observer_geometry(times: np.ndarray,
                                   orbit_alt_km: float = 400.0,
                                   earth_radius_km: float = 6371.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate simulated sun and observer unit vectors over time."""
    N = len(times)
    # Sun: slow drift over the observation window (~0.1 deg over 2 hours)
    sun_angle = 2.0 * np.pi * (720.0 / 86400.0) * times / TOTAL_DURATION_SEC
    sun_x = np.cos(sun_angle)
    sun_y = np.sin(sun_angle)
    sun_z = np.full(N, 0.8)
    sun_dirs = np.column_stack([sun_x, sun_y, sun_z])
    sun_dirs = sun_dirs / np.linalg.norm(sun_dirs, axis=1, keepdims=True)

    omega = np.sqrt(3.986e5 / (earth_radius_km + orbit_alt_km) ** 3)
    obs_angle = omega * times
    obs_x = np.cos(obs_angle)
    obs_y = np.sin(obs_angle)
    obs_z = np.full(N, 0.0)
    obs_dirs = np.column_stack([obs_x, obs_y, obs_z])
    obs_dirs = obs_dirs / np.linalg.norm(obs_dirs, axis=1, keepdims=True)

    return sun_dirs, obs_dirs


def generate_single_light_curve(cls: int,
                                n_samples: int = N_SAMPLES,
                                add_noise: bool = True,
                                noise_std: float = 0.02,
                                add_dropouts: bool = True) -> Tuple[np.ndarray, int]:
    """Generate one light curve for a given class."""
    params = sample_class_params(cls)
    total_duration = TOTAL_DURATION_SEC
    sample_interval = total_duration / (n_samples - 1)
    times = np.linspace(0, total_duration, n_samples)
    sun_dirs, obs_dirs = generate_sun_observer_geometry(times)

    tp = params.tumble_period
    pp = max(tp * 3.0, 200.0 + _RNG.uniform(50, 150))
    spin_axis = _RNG.normal(0, 1, 3)
    spin_axis = spin_axis / np.linalg.norm(spin_axis)
    phase = _RNG.uniform(0, 2 * np.pi)

    state = ObjectState(
        semi_major_axis_km=400.0,
        eccentricity=0.001,
        inclination_deg=_RNG.uniform(0, 360),
        raan_deg=_RNG.uniform(0, 360),
        arg_of_perigee_deg=_RNG.uniform(0, 360),
        true_anomaly_deg=_RNG.uniform(0, 360),
        tumble_period_s=tp,
        precession_period_s=pp,
        spin_axis=spin_axis,
        tumble_phase=phase,
        aspect_ratio=params.aspect_ratio,
        reflectivity=params.reflectivity,
    )

    n_raw = max(N_TIME_STEPS, 100)
    raw_times = np.linspace(0, total_duration, n_raw)
    raw_sun, raw_obs = generate_sun_observer_geometry(raw_times)
    curve = compute_light_curve_from_faces(raw_times, state, raw_sun, raw_obs, n_faces=params.n_faces)

    if len(curve) != n_samples:
        from scipy.interpolate import interp1d
        if len(curve) < n_samples:
            f = interp1d(np.linspace(0, total_duration, len(curve)), curve,
                         kind='linear', fill_value=0.0, bounds_error=False)
        else:
            f = interp1d(np.linspace(0, total_duration, len(curve)), curve,
                         kind='linear', bounds_error=False)
        curve = f(np.linspace(0, total_duration, n_samples))

    if add_noise:
        noise = _RNG.normal(0, noise_std, n_samples)
        curve = curve + noise

    if add_dropouts:
        n_dropouts = max(1, int(0.05 * n_samples))
        dropout_indices = _RNG.choice(n_samples, min(n_dropouts, n_samples - 1), replace=False)
        curve[dropout_indices] = np.nan
        valid_mask = ~np.isnan(curve)
        if valid_mask.any() and valid_mask.sum() > 1:
            f_fill = interp1d(
                np.where(valid_mask)[0], curve[valid_mask],
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )
            curve = f_fill(np.arange(n_samples))

    c_min, c_max = curve.min(), curve.max()
    if c_max > c_min:
        curve = (curve - c_min) / (c_max - c_min)
    else:
        curve = np.zeros(n_samples)

    curve = np.clip(curve, 0, 1)
    return curve, cls


def generate_dataset(n_per_class: int = 2000,
                     noise_std: float = 0.02,
                     seed: int = 42,
                     save_path: str = "data/synthetic/lightcurves.npz") -> Dict[str, np.ndarray]:
    """Generate the full synthetic dataset for all 5 classes."""
    rng = np.random.default_rng(seed)
    curves = []
    labels = []

    for cls in range(5):
        for i in range(n_per_class):
            curve, label = generate_single_light_curve(cls, noise_std=noise_std)
            curves.append(curve)
            labels.append(label)

    curves = np.array(curves, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    np.savez_compressed(save_path, curves=curves, labels=labels)
    print(f"Saved dataset to {save_path}: {curves.shape} curves, {len(np.unique(labels))} classes")
    return {"curves": curves, "labels": labels}