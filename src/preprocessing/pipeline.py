# pipeline.py -Reusable preprocessing for Q-ORBIT Phase 2
# Same pipeline for CLASSICAL / PURE QUANTUM / HYBRID. No leakage. Config-driven. Saves params.
from __future__ import annotations
import os, json, joblib
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

CONFIG_PATH = Path("configs/qorbit_config.json")

# ---------- Curve-level preprocessing (per light curve) ----------
class CurvePreprocessor:
    """Handles missing, resampling, and per-curve normalization without destroying physical signal.
    - Missing: linear interpolate (preserves shape, unlike fill-zero)
    - Resampling: linear interp to fixed 256 (if needed)
    - Normalization: per-curve min-max to [0,1] already done in generator; this re-applies safely (no cross-curve leakage)
    Noise handling is deliberately NOT done here (kept for robustness sweeps).
    """
    def __init__(self, target_len: int = 256):
        self.target_len = target_len

    def handle_missing(self, curve: np.ndarray) -> np.ndarray:
        c = curve.astype(float).copy()
        if np.isnan(c).any():
            mask = ~np.isnan(c)
            if mask.sum() >= 2:
                # linear, extrapolate ends
                f = interp1d(np.where(mask)[0], c[mask], kind="linear", bounds_error=False, fill_value="extrapolate")
                c = f(np.arange(len(c)))
            elif mask.sum() == 1:
                c[:] = c[mask][0]
            else:
                c[:] = 0.0
        return c

    def resample(self, curve: np.ndarray) -> np.ndarray:
        if len(curve) == self.target_len:
            return curve
        # linear resample preserving physical time axis
        x_old = np.linspace(0, 1, len(curve))
        x_new = np.linspace(0, 1, self.target_len)
        f = interp1d(x_old, curve, kind="linear", bounds_error=False, fill_value="extrapolate")
        return f(x_new).astype(float)

    def normalize_per_curve(self, curve: np.ndarray) -> np.ndarray:
        cmin, cmax = float(curve.min()), float(curve.max())
        if cmax > cmin:
            return (curve - cmin) / (cmax - cmin)
        return np.zeros_like(curve)

    def process(self, curve: np.ndarray) -> np.ndarray:
        c = self.handle_missing(curve)
        c = self.resample(c)
        # Note: generator already min-max normalized; re-normalize only if drifted (e.g. after resample)
        # Keep as 0-1 to preserve physical relative brightness shape
        if c.min() < -0.01 or c.max() > 1.01:
            c = np.clip(c, 0, 1)
        return c.astype(np.float32)

    def process_batch(self, curves: np.ndarray) -> np.ndarray:
        return np.stack([self.process(c) for c in curves]).astype(np.float32)

# ---------- Feature-level standardization (fit on TRAIN only) ----------
class FeatureStandardizer:
    """Z-score standardization. Fit on train features only, apply to val/test/inference. Saved to disk."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X_train: np.ndarray):
        self.scaler.fit(X_train)
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.fitted, "Call fit() on train first"
        return self.scaler.transform(X)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"mean": self.scaler.mean_, "scale": self.scaler.scale_, "var": self.scaler.var_}, path)

    @classmethod
    def load(cls, path: str) -> "FeatureStandardizer":
        d = joblib.load(path)
        obj = cls()
        # reconstruct scaler
        obj.scaler.mean_ = d["mean"]
        obj.scaler.scale_ = d["scale"]
        obj.scaler.var_ = d["var"]
        obj.scaler.n_features_in_ = len(d["mean"])
        obj.fitted = True
        return obj

    def params(self):
        return {"mean": self.scaler.mean_.tolist() if self.fitted else None, "scale": self.scaler.scale_.tolist() if self.fitted else None}

# ---------- Full pipeline combining both ----------
@dataclass
class PreprocessingPipeline:
    target_len: int = 256
    save_dir: str = "data/processed"

    def __post_init__(self):
        self.curve_proc = CurvePreprocessor(target_len=self.target_len)
        self.feat_std = FeatureStandardizer()

    def fit(self, curves_train: np.ndarray, features_train: np.ndarray):
        """Fit feature standardizer on train features only. Curve proc is stateless."""
        self.feat_std.fit(features_train)
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.feat_std.save(f"{self.save_dir}/feature_scaler.joblib")
        # also save to models/ for dashboard
        self.feat_std.save("models/feature_scaler.joblib")
        with open(f"{self.save_dir}/preprocessing_params.json", "w", encoding="utf-8") as f:
            json.dump({"target_len": self.target_len, "feature_standardizer": self.feat_std.params(), "note": "All models must use this scaler; fit on train only, no test leakage"}, f, indent=2)
        return self

    def transform_curves(self, curves: np.ndarray) -> np.ndarray:
        return self.curve_proc.process_batch(curves)

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        return self.feat_std.transform(X)

    @staticmethod
    def load_scaler(path="models/feature_scaler.joblib") -> FeatureStandardizer:
        return FeatureStandardizer.load(path)
