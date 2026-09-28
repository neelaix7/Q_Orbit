"""AI part 1 — link / parameter optimisation.

Predicts QBER and secure key rate for a given (link conditions, QKD config) and
*recommends* the best QKD parameter configuration for an upcoming pass.

Model: a single regressor (gradient boosting) mapping
    [link features | QKD config params] -> [expected QBER, expected secure bps].
At inference time the predictor scans a candidate parameter grid, predicts the
secure key rate for each candidate, and returns the best configuration.  This is
exactly the "AI auto-tune" behaviour demonstrated live in the Streamlit app.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..quantum.params import QKDParams, parameter_grid

# Feature columns used for link-condition prediction.
LINK_FEATURES = [
    "max_elev", "mean_elev", "min_elev", "pass_duration_s", "mean_link_q",
    "mean_atten_db", "night_frac", "sat_idx",
]
CONFIG_FEATURES = ["mu", "dark", "eta", "pz"]


def link_features_from_pass(pass_df: pd.DataFrame, sat_idx: int = 0,
                            night_threshold_deg: float = -5.0) -> np.ndarray:
    """Compute the fixed link-condition feature vector from a pass table.

    Parameters
    ----------
    pass_df : DataFrame with columns elevation_deg, mjd, link_quality,
              plus a rough attenuation column (computed from elevation).
    """
    el = pass_df["elevation_deg"].to_numpy()
    q = pass_df.get("link_quality")
    q = q.to_numpy() if q is not None else np.clip(el / 45.0, 0, 1) ** 0.6
    mjd = pass_df["mjd"].to_numpy()

    # Night fraction from solar elevation at the pass times.
    from ..orbit.link import solar_elevation_deg
    sun = np.array([solar_elevation_deg(float(m), 13.0, 80.0) for m in mjd])
    night_frac = float(np.mean(sun < -2.0))

    dt = np.diff(mjd).mean() * 86400.0 if len(mjd) > 1 else 10.0
    atten = 0.35 / np.sin(np.maximum(np.deg2rad(el), np.deg2rad(2.0)))
    return np.array([
        float(el.max()),
        float(el.mean()),
        float(el.min()),
        float(len(el) * dt),
        float(q.mean()),
        float(atten.mean()),
        float(night_frac),
        float(sat_idx),
    ])


def make_training_frame(rows: list[dict]) -> pd.DataFrame:
    """Turn list of per-sample dicts into a DataFrame ready for training.

    Each dict should contain all LINK_FEATURES + CONFIG_FEATURES plus the two
    targets `qber` and `secure_bps`.
    """
    df = pd.DataFrame(rows)
    required = LINK_FEATURES + CONFIG_FEATURES + ["qber", "secure_bps"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in training frame: {missing}")
    return df


class LinkPredictor:
    """Regressor + recommender for QKD parameter auto-tuning."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model: MultiOutputRegressor | None = None
        self.scaler = StandardScaler()
        self._xcols: list[str] = LINK_FEATURES + CONFIG_FEATURES

    def fit(self, X: np.ndarray, y_qber: np.ndarray, y_bps: np.ndarray) -> "LinkPredictor":
        """Fit on design matrix X (link features + config) and two targets."""
        Xs = self.scaler.fit_transform(X)
        Y = np.column_stack([np.asarray(y_qber), np.asarray(y_bps)])
        base = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.08, max_depth=4,
            random_state=self.seed, subsample=0.9,
        )
        self.model = MultiOutputRegressor(base)
        self.model.fit(Xs, Y)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (predicted_qber, predicted_secure_bps)."""
        if self.model is None:
            raise RuntimeError("LinkPredictor not fitted.")
        Xs = self.scaler.transform(np.asarray(X, dtype=float))
        Y = np.asarray(self.model.predict(Xs))
        return Y[:, 0], Y[:, 1]

    def recommend(self, link_feats: np.ndarray,
                  grid: list[QKDParams] | None = None) -> tuple[QKDParams, float, float]:
        """Pick the best QKD config for the given link conditions.

        Returns (best_params, predicted_qber, predicted_secure_bps).
        """
        if grid is None:
            grid = parameter_grid()
        rows = []
        for p in grid:
            rows.append(list(link_feats) + list(p.as_array()))
        X = np.array(rows, dtype=float)
        qber_pred, bps_pred = self.predict(X)
        best = int(np.argmax(bps_pred))
        return grid[best], float(qber_pred[best]), float(bps_pred[best])

    def save(self, path):
        dump({"model": self.model, "scaler": self.scaler, "xcols": self._xcols,
              "seed": self.seed}, path)

    @classmethod
    def load(cls, path) -> "LinkPredictor":
        obj = load(path)
        inst = cls(seed=obj["seed"])
        inst.model = obj["model"]
        inst.scaler = obj["scaler"]
        inst._xcols = obj["xcols"]
        return inst
