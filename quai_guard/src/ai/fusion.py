"""Simulated fusion engine: one guard that ties link-prediction + Eve detection.

The ``GuardEngine`` is the production-facing object the Streamlit app talks to.
It owns:
* the LinkPredictor (to auto-tune QKD parameters per pass), and
* the EveDetector (to flag eavesdropping in real time from the QBER stream).

A pass is processed sample-by-sample: for each time slice the engine runs the
QKD round with the recommended (or fixed) parameters, feeds the growing QBER
series into the Eve detector, and returns an updated threat verdict plus the
expected secure key rate.  This is the "AI-guarded" part of QU-AI-GUARD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ai.eve_detector import EveDetector, extract_features
from ..ai.link_predictor import LINK_FEATURES, LinkPredictor, link_features_from_pass
from ..quantum.params import QKDParams


@dataclass
class SampleVerdict:
    """Verdict for one QKD sample along a pass."""

    time_s: float
    qber: float
    secure_bps: float
    raw_bps: float
    alarm: bool
    confidence: float
    threat: str          # "SECURE" | "WATCH" | "INTRUSION"
    secure_key_hex: str
    secure_bits: int


class GuardEngine:
    """Combines LinkPredictor + EveDetector for a live, guarded QKD pass."""

    def __init__(self, link_predictor: LinkPredictor | None = None,
                 eve_detector: EveDetector | None = None):
        self.link_predictor = link_predictor
        self.eve_detector = eve_detector
        self.qber_series: list[float] = []
        self.alarm_votes: list[bool] = []
        self.fixed_config: QKDParams | None = None
        self.auto_tune: bool = True
        self.link_feats: np.ndarray | None = None
        self.expected_qber: float = 0.0
        self._ever_intruded: bool = False
        self._min_samples: int = 20

    def reset(self) -> None:
        self.qber_series = []
        self.alarm_votes = []
        self._ever_intruded = False

    def start_pass(self, pass_df, sat_idx: int = 0, auto_tune: bool = True,
                   fixed_config: QKDParams | None = None) -> QKDParams:
        """Called before processing a pass. Returns the config to use."""
        self.auto_tune = auto_tune
        self.fixed_config = fixed_config
        self.link_feats = link_features_from_pass(pass_df, sat_idx=sat_idx)
        if auto_tune and self.link_predictor is not None:
            best, qber_p, bps_p = self.link_predictor.recommend(self.link_feats)
            self.current_config = best
            self.expected_qber = float(qber_p)
        else:
            self.current_config = fixed_config or QKDParams()
        return self.current_config

    def update(self, qber: float, secure_bps: float, raw_bps: float,
               secure_key_hex: str, secure_bits: int, time_s: float,
               mean_link_q: float = 0.5) -> SampleVerdict:
        """Feed one sample's QKD result and return the fused verdict."""
        self.qber_series.append(float(qber))
        alarm, conf = False, 0.0
        threat = "SECURE"
        n = len(self.qber_series)
        # The detector was trained on *pass-level* windows (~30 samples); only
        # classify on a near-complete pass so the live features match training.
        if self.eve_detector is not None and n >= self._min_samples:
            residual = float(np.mean(self.qber_series)) - self.expected_qber
            feats = extract_features(np.array(self.qber_series), mean_link_q, residual)
            conf = float(self.eve_detector.predict_proba(feats[None, :])[0])
            alarm = conf >= self.eve_detector.operating_threshold
            threat = "INTRUSION" if alarm else ("WATCH" if conf > 0.5 else "SECURE")
        # Sticky verdict: once an intrusion is confirmed, the session is flagged.
        if getattr(self, "_ever_intruded", False):
            alarm, threat = True, "INTRUSION"
        if threat == "INTRUSION":
            self._ever_intruded = True
        return SampleVerdict(time_s=time_s, qber=float(qber), secure_bps=float(secure_bps),
                             raw_bps=float(raw_bps), alarm=alarm, confidence=float(conf),
                             threat=threat, secure_key_hex=secure_key_hex,
                             secure_bits=secure_bits)
