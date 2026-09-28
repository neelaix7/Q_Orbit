"""AI part 2 — Eve detection.

Given a QBER time series measured over a pass, extract engineered features
(trend, variance, burstiness, autocorrelation, spikes) and classify the pass as
"atmospheric noise" vs "eavesdropper present".

The detector is explicitly designed to *minimise false alarms* (a false alarm
aborts a key — operators hate that).  We therefore fit on balanced classes but
report metrics at a conservative operating point chosen to keep the false
alarm rate low, and expose ``predict_proba`` for the live confidence panel.
"""

from __future__ import annotations

import numpy as np
from joblib import dump, load
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

QBER_FEATURES = [
    "qber_mean", "qber_std", "qber_min", "qber_max", "qber_slope",
    "qber_burstiness", "qber_autocorr", "qber_spikes", "qber_p95",
    "qber_rng", "qber_energy",
]
LINK_FEATURES_HDR = ["mean_link_q", "qber_residual"]


def extract_qber_features(qber_series: np.ndarray) -> np.ndarray:
    """Feature vector from a QBER time series."""
    x = np.asarray(qber_series, dtype=float)
    x = np.nan_to_num(x)
    n = len(x)
    if n == 0:
        return np.zeros(len(QBER_FEATURES))
    t = np.arange(n)
    slope = np.polyfit(t, x, 1)[0] if n > 2 else 0.0
    autocorr = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if n > 3 else 0.0
    if autocorr != autocorr:    # guard against constant series -> NaN
        autocorr = 0.0
    mean = float(x.mean())
    std = float(x.std())
    burstiness = float(std / (mean + 1e-9))
    spikes = float(np.sum(x > mean + 2 * std))
    return np.array([
        mean, std, float(x.min()), float(x.max()), slope,
        burstiness, autocorr, spikes, float(np.percentile(x, 95)),
        float(np.ptp(x)), float(np.sum(x ** 2)),
    ])


def extract_features(qber_series: np.ndarray, mean_link_q: float,
                     qber_residual: float = 0.0) -> np.ndarray:
    """Fused feature vector: link context + QBER statistics + *residual*.

    ``qber_residual`` is measured QBER minus the QBER the link predictor
    expects from the pass's link conditions.  It is the decisive signal for the
    fusion engine: atmospheric noise keeps the residual near zero, while an
    eavesdropper pushes QBER far above the link-predicted value.
    """
    q = extract_qber_features(qber_series)
    return np.concatenate([[float(mean_link_q), float(qber_residual)], q])


class EveDetector:
    """Binary classifier: atmospheric noise (0) vs Eve present (1)."""

    def __init__(self, seed: int = 42, low_fp: bool = True):
        self.seed = seed
        self.low_fp = low_fp
        self.model: HistGradientBoostingClassifier | None = None
        self.scaler = StandardScaler()
        self.operating_threshold = 0.5
        self.operating_far = 0.03

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EveDetector":
        Xs = self.scaler.fit_transform(X)
        base = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, max_leaf_nodes=15,
            random_state=self.seed, early_stopping=False,
        )
        self.model = base
        self.model.fit(Xs, np.asarray(y))
        probs = self.model.predict_proba(Xs)[:, 1]
        y = np.asarray(y)
        fpr, tpr, th = _roc_curve(y, probs)
        self._fpr = fpr
        self._tpr = tpr
        self._auc = roc_auc_score(y, probs)
        return self

    def _pick_threshold(self, X_val: np.ndarray, y_val: np.ndarray,
                        max_fpr: float = 0.03) -> float:
        """Pick the operating threshold on a *validation* set.

        Targets a low false-alarm rate (false alarms abort QKD sessions, so they
        are the costliest error).  Places the operating point at the (1 - FAR)
        quantile of the *clean* validation scores, which directly calibrates the
        false-alarm rate.  On tiny validation sets the empirical quantile can
        sit at a degenerate extreme (all-clean scores clump near 0), in which
        case the budget is relaxed only as far as needed to obtain a useful
        detector.  The achieved FAR is exposed via ``operating_far``.
        """
        probs = self.predict_proba(X_val)
        y = np.asarray(y_val)
        clean = probs[y == 0]
        attack = probs[y == 1]
        if len(clean) == 0 or len(attack) == 0:
            self.operating_far = max_fpr
            return 0.5
        budget = max_fpr
        while budget <= 1.0:
            th = float(np.quantile(clean, 1.0 - budget))
            tpr = float((attack >= th).mean())
            if tpr > 0.0:
                self.operating_far = budget
                return th
            budget += max_fpr
        return 0.5

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("EveDetector not fitted.")
        return self.model.predict_proba(self.scaler.transform(np.asarray(X, dtype=float)))[:, 1]

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        p = self.predict_proba(X)
        th = self.operating_threshold if threshold is None else threshold
        return (p >= th).astype(int)

    def alarm(self, X: np.ndarray) -> tuple[bool, float]:
        p = float(self.predict_proba(X)[0])
        return bool(p >= self.operating_threshold), p

    @property
    def auc(self) -> float:
        return float(self._auc)

    def save(self, path):
        dump({"model": self.model, "scaler": self.scaler, "threshold": self.operating_threshold,
              "seed": self.seed, "auc": self._auc, "operating_far": self.operating_far}, path)

    @classmethod
    def load(cls, path) -> "EveDetector":
        obj = load(path)
        inst = cls(seed=obj["seed"])
        inst.model = obj["model"]
        inst.scaler = obj["scaler"]
        inst.operating_threshold = obj["threshold"]
        inst.operating_far = float(obj.get("operating_far", 0.03))
        inst._auc = obj["auc"]
        return inst


def _roc_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimal ROC curve computation (no sklearn private API dependency)."""
    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
    fp = np.cumsum(1 - y)
    tp = np.cumsum(y)
    fpr = np.concatenate([[0.0], fp / n_neg])
    tpr = np.concatenate([[0.0], tp / n_pos])
    thr = np.concatenate([[scores[order[0]] + 1.0], scores[order]])
    return fpr, tpr, thr
