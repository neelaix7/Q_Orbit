"""Auto-tunable QKD parameters.

These are the knobs an operator (or the AI link-predictor) can tune per pass.
They map into concrete channel-noise numbers in :mod:`src.orbit.link.LinkModel`
and into the BB84 simulator in :mod:`src.quantum.bb84`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QKDParams:
    """A QKD parameter configuration.

    Attributes
    ----------
    laser_intensity : mean photon number per pulse (mu)
    dark_count_rate : per-detector dark-count rate [Hz]
    detector_eta    : detection efficiency in [0, 1]
    basis_bias      : probability of choosing the Z basis (rectilinear)
    """

    laser_intensity: float = 0.5
    dark_count_rate: float = 100.0
    detector_eta: float = 0.5
    basis_bias: float = 0.5

    def as_array(self) -> np.ndarray:
        return np.array([self.laser_intensity, self.dark_count_rate,
                         self.detector_eta, self.basis_bias], dtype=float)

    @classmethod
    def from_array(cls, a: np.ndarray) -> "QKDParams":
        a = np.asarray(a, dtype=float)
        return cls(laser_intensity=float(a[0]), dark_count_rate=float(a[1]),
                   detector_eta=float(a[2]), basis_bias=float(a[3]))

    def clamp(self) -> "QKDParams":
        """Clamp parameters to physically sensible ranges."""
        return QKDParams(
            laser_intensity=float(np.clip(self.laser_intensity, 0.05, 1.0)),
            dark_count_rate=float(np.clip(self.dark_count_rate, 10.0, 1000.0)),
            detector_eta=float(np.clip(self.detector_eta, 0.05, 0.99)),
            basis_bias=float(np.clip(self.basis_bias, 0.25, 0.75)),
        )

    def __repr__(self) -> str:
        return (f"QKDParams(mu={self.laser_intensity:.2f}, "
                f"dark={self.dark_count_rate:.0f} Hz, "
                f"eta={self.detector_eta:.2f}, pZ={self.basis_bias:.2f})")


DEFAULT_PARAMS = QKDParams()


def parameter_grid(dark_rates: tuple[float, ...] = (30.0, 100.0, 300.0),
                   intensities: tuple[float, ...] = (0.2, 0.5, 0.8),
                   etas: tuple[float, ...] = (0.3, 0.6, 0.9)) -> list[QKDParams]:
    """A small candidate grid the AI tunes over (kept tiny so the demo is fast)."""
    grid = []
    for mu in intensities:
        for dark in dark_rates:
            for eta in etas:
                grid.append(QKDParams(laser_intensity=mu, dark_count_rate=dark,
                                      detector_eta=eta, basis_bias=0.5))
    return grid


def grid_dark_rates(n: int = 4) -> list[float]:
    return [float(x) for x in np.geomspace(30.0, 400.0, n)]
