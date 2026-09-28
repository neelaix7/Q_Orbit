"""Channel / link modelling for the satellite-to-ground QKD channel.

Physical effects modelled (all approximations, clearly documented):
* Free-space path attenuation that grows as the satellite approaches the horizon
  (standard `csc(elevation)` atmospheric-slab approximation with a zenith
  optical depth).
* Day/night background-light budget: solar elevation at the ground station
  controls the optical background photon flux.
* Detector dark counts and detection efficiency.
* A scalar "link quality" in [0, 1] used by the quantum layer to scale noise.

Everything is deterministic given the input elevation profile and time, so the
whole pipeline is reproducible with a fixed random seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def solar_elevation_deg(mjd: float, gs_lat_deg: float, gs_lon_deg: float) -> float:
    """Very approximate solar elevation (deg) at a ground site (for day/night)."""
    # Days since J2000.
    d = mjd - 51544.5
    g = np.deg2rad((357.529 + 0.98560028 * d) % 360.0)          # mean anomaly
    L = np.deg2rad((280.459 + 0.98564736 * d) % 360.0) + g      # mean longitude
    decl = np.deg2rad(23.439) * np.sin(L)                        # crude declination
    # Hour angle of the sun at the site.
    ra = np.arctan2(np.cos(g) * np.sin(L), np.cos(L))            # crude right ascension
    gmst = (280.46061837 + 360.98564736629 * d) % 360.0
    lst = np.deg2rad(gmst + gs_lon_deg)
    ha = lst - ra
    sin_el = (np.sin(np.deg2rad(gs_lat_deg)) * np.sin(decl)
              + np.cos(np.deg2rad(gs_lat_deg)) * np.cos(decl) * np.cos(ha))
    return float(np.rad2deg(np.arcsin(np.clip(sin_el, -1, 1))))


def atmospheric_attenuation_db(elevation_deg: float, zenith_od: float = 0.35,
                               elev_floor: float = 1.0) -> float:
    """Atmospheric absorption in dB vs elevation.

    Uses the classic `zenith / sin(elevation)` plane-parallel slab model with a
    minimum elevation floor so the model stays finite near the horizon.
    """
    el = max(abs(float(elevation_deg)), elev_floor)
    return float(zenith_od * (1.0 / np.sin(np.deg2rad(el))))


def background_photon_rate(mjd: float, gs_lat_deg: float, gs_lon_deg: float,
                           night_rate: float = 2.0e3,
                           day_rate: float = 9.0e4) -> float:
    """Optical background photon rate (photons/s) vs day/night.

    A smooth sigmoid in solar elevation keeps the transition credible rather
    than a hard binary cut at the terminator.
    """
    sun_el = solar_elevation_deg(mjd, gs_lat_deg, gs_lon_deg)
    frac = 1.0 / (1.0 + np.exp(-(sun_el - 0.0) / 4.0))   # 0 at night, 1 at day
    return float(night_rate + frac * (day_rate - night_rate))


class LinkModel:
    """Compute link quality and noise parameters along a satellite pass.

    Parameters
    ----------
    dark_count_rate : per-detector dark count rate [Hz] (tunable QKD parameter)
    detector_eta    : detection efficiency in [0, 1]
    """

    def __init__(self, dark_count_rate: float = 100.0, detector_eta: float = 0.5,
                 zenith_od: float = 0.35):
        self.dark_count_rate = float(dark_count_rate)
        self.detector_eta = float(detector_eta)
        self.zenith_od = float(zenith_od)

    def link_quality(self, elevation_deg: float, mjd: float,
                     gs_lat_deg: float, gs_lon_deg: float) -> float:
        """Scalar link quality in [0, 1] (1 = perfect, 0 = unusable).

        Combines (a) elevation: higher elevation -> less atmosphere & shorter
        path; (b) daylight: night links are much cleaner.
        """
        el = np.clip(elevation_deg, 0.0, 90.0)
        el_factor = np.clip(el / 45.0, 0.0, 1.0) ** 0.6
        bg = background_photon_rate(mjd, gs_lat_deg, gs_lon_deg)
        # Normalised background in [0,1]; scale daylight penalty (milder than a
        # hard day/night cut so the classifier must learn the difference).
        bg_factor = 1.0 - 0.5 * np.clip((bg - 2e3) / 9e4, 0.0, 1.0)
        q = float(np.clip(0.08 + 0.92 * el_factor * bg_factor, 0.0, 1.0))
        return q

    def channel_noise(self, elevation_deg: float, mjd: float,
                      gs_lat_deg: float, gs_lon_deg: float,
                      rng: np.random.Generator | None = None) -> dict[str, float]:
        """Return a dict of channel-noise parameters for one QKD round.

        Returns
        -------
        link_quality   : scalar in [0,1]
        depol_p        : depolarising-channel probability per qubit
        flip_p         : extra bit-flip probability (from transmission errors)
        dark_frac      : probability a detector fires due to dark count
        detect_eta     : probability a transmitted photon is actually detected
        bg_rate        : background photon rate [Hz]
        """
        q = self.link_quality(elevation_deg, mjd, gs_lat_deg, gs_lon_deg)
        atten_db = atmospheric_attenuation_db(elevation_deg, self.zenith_od)
        # Depolarising noise grows as link quality drops, but a *good* link
        # keeps QBER in the 2-6% band (realistic for QKD systems).
        depol_p = float(np.clip(0.02 + 0.42 * (1.0 - q), 0.02, 0.5))
        flip_p = float(np.clip(0.005 + 0.12 * (1.0 - q), 0.003, 0.15))
        # Detection efficiency collapses with attenuation (log-normal survival).
        survival = 10.0 ** (-atten_db / 10.0)
        detect_eta = float(np.clip(self.detector_eta * survival, 0.0, 1.0))
        # Dark-count probability per detection window (~1 us).
        dark_frac = float(np.clip(self.dark_count_rate * 1e-6, 0.0, 0.5))
        bg_rate = background_photon_rate(mjd, gs_lat_deg, gs_lon_deg)
        return {
            "link_quality": q,
            "depol_p": depol_p,
            "flip_p": flip_p,
            "dark_frac": dark_frac,
            "detect_eta": detect_eta,
            "bg_rate": bg_rate,
        }

    def pass_summary(self, df: pd.DataFrame, gs_lat_deg: float,
                     gs_lon_deg: float) -> pd.DataFrame:
        """Augment a pass DataFrame with per-sample link features."""
        out = df.copy()
        qs, deps, fps, etas = [], [], [], []
        for _, row in df.iterrows():
            nz = self.channel_noise(row["elevation_deg"], row["mjd"],
                                    gs_lat_deg, gs_lon_deg)
            qs.append(nz["link_quality"])
            deps.append(nz["depol_p"])
            fps.append(nz["flip_p"])
            etas.append(nz["detect_eta"])
        out["link_quality"] = qs
        out["depol_p"] = deps
        out["flip_p"] = fps
        out["detect_eta"] = etas
        return out
