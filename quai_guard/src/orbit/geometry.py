"""Satellite orbit propagation (two-body + J2) and ground-station geometry.

All quantities are computed in a spherical, non-rotating Earth frame then
rotated into an Earth-centred, Earth-fixed (ECEF) frame using Greenwich Mean
Sidereal Time so that a fixed ground station stays fixed.

References
----------
* Vallado, D.  "Fundamentals of Astrodynamics and Applications" (J2 secular rates)
* Montenbruck & Gill, "Satellite Orbits" (Kepler solver, ECI<->ECEF rotations)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MU_EARTH = 3.986004418e14       # m^3 / s^2
R_EARTH = 6371.0e3              # mean equatorial radius [m]
J2 = 1.08262668e-3              # Earth's oblateness term
OMEGA_EARTH = 7.2921159e-5      # Earth rotation rate [rad / s]


def gmst_from_mjd(mjd: float) -> float:
    """Greenwich Mean Sidereal Time (rad) for a modified Julian date."""
    days = mjd - 51544.5
    gmst = (280.46061837 + 360.98564736629 * days) % 360.0
    return np.deg2rad(gmst)


class GroundStation:
    """A fixed ground station defined by geodetic latitude, longitude, altitude."""

    def __init__(self, lat_deg: float, lon_deg: float, alt_m: float = 0.0,
                 name: str = "GS-1"):
        self.lat_deg = float(lat_deg)
        self.lon_deg = float(lon_deg)
        self.alt_m = float(alt_m)
        self.name = name

    def ecef(self) -> np.ndarray:
        """ECEF (Earth-fixed) position vector in metres."""
        lat = np.deg2rad(self.lat_deg)
        lon = np.deg2rad(self.lon_deg)
        a = R_EARTH
        f = 1.0 / 298.257223563
        e2 = f * (2.0 - f)
        n = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
        x = (n + self.alt_m) * np.cos(lat) * np.cos(lon)
        y = (n + self.alt_m) * np.cos(lat) * np.sin(lon)
        z = (n * (1.0 - e2) + self.alt_m) * np.sin(lat)
        return np.array([x, y, z])

    def __repr__(self) -> str:
        return f"GroundStation({self.name}, {self.lat_deg:.2f}N, {self.lon_deg:.2f}E)"


class KeplerOrbit:
    """Satellite orbit from classical orbital elements with J2 secular drift.

    Parameters
    ----------
    a        : semi-major axis [m]
    e        : eccentricity [0, 1)
    i_deg    : inclination [deg]
    raan_deg : right ascension of ascending node [deg]
    argp_deg : argument of perigee [deg]
    m0_deg   : mean anomaly at epoch [deg]
    epoch_mjd: epoch as modified Julian date
    """

    def __init__(self, a: float, e: float, i_deg: float, raan_deg: float,
                 argp_deg: float, m0_deg: float, epoch_mjd: float,
                 name: str = "SAT-1"):
        self.a = float(a)
        self.e = float(e)
        self.i = np.deg2rad(float(i_deg))
        self.raan = np.deg2rad(float(raan_deg))
        self.argp = np.deg2rad(float(argp_deg))
        self.m0 = np.deg2rad(float(m0_deg))
        self.epoch_mjd = float(epoch_mjd)
        self.name = name
        self.n = np.sqrt(MU_EARTH / self.a ** 3)  # mean motion [rad / s]
        self.period = 2.0 * np.pi / self.n

        # J2 secular drift rates (rad / s)  -- Vallado eq. 9-31/9-32
        fac = 1.5 * J2 * (R_EARTH / self.a) ** 2 * self.n / (1 - self.e ** 2) ** 2
        self.raan_dot = -fac * np.cos(self.i)
        self.argp_dot = fac * (2 - 2.5 * np.sin(self.i) ** 2)

    def elements_at(self, mjd):
        """Return (a, e, i, raan, argp, M) at a modified Julian date (or array)."""
        dt = (np.asarray(mjd, dtype=float) - self.epoch_mjd) * 86400.0
        raan = self.raan + self.raan_dot * dt
        argp = self.argp + self.argp_dot * dt
        m = self.m0 + self.n * dt
        return self.a, self.e, self.i, raan, argp, m % (2 * np.pi)

    def eci_position(self, mjd):
        """Satellite position [m] in the ECI frame.

        Accepts a scalar or an array of modified Julian dates (vectorised).
        Returns shape (3,) for a scalar input and (..., 3) for arrays.
        """
        a, e, i, raan, argp, m = self.elements_at(mjd)
        scalar = np.ndim(mjd) == 0

        # Solve Kepler's equation E - e sin E = M (vectorised Newton iteration).
        e0 = m.copy()
        for _ in range(30):
            f = e0 - e * np.sin(e0) - m
            fp = 1.0 - e * np.cos(e0)
            de = f / fp
            e0 = e0 - de
            if np.all(np.abs(de) < 1e-11):
                break

        # Perifocal-frame position.
        nu = np.arctan2(np.sqrt(1 - e ** 2) * np.sin(e0), np.cos(e0) - e)
        r = a * (1 - e * np.cos(e0))

        # Perifocal -> ECI rotation: Rz(-raan) * Rx(-i) * Rz(-argp) * r_perifocal.
        cO, sO = np.cos(raan), np.sin(raan)
        cw, sw = np.cos(argp), np.sin(argp)
        ci, si = np.cos(i), np.sin(i)
        cosnu, sinnu = np.cos(nu), np.sin(nu)
        x1 = r * (cw * cosnu - sw * sinnu)
        y1 = r * (sw * cosnu + cw * sinnu)
        x2 = x1
        y2 = y1 * ci
        z2 = y1 * si
        x = cO * x2 - sO * y2
        y = sO * x2 + cO * y2
        z = z2
        out = np.stack([x, y, z], axis=-1)
        return out[0] if scalar else out

    def __repr__(self) -> str:
        return (f"KeplerOrbit({self.name}, a={self.a/1e3:.0f} km, "
                f"e={self.e:.3f}, i={np.rad2deg(self.i):.1f} deg)")


def eci_to_ecef(r_eci: np.ndarray, mjd) -> np.ndarray:
    """Rotate ECI vector(s) into the Earth-fixed frame (rotation about z by GMST)."""
    theta = gmst_from_mjd(mjd)
    c, s = np.cos(theta), np.sin(theta)
    if np.ndim(r_eci) == 1:
        return np.array([c * r_eci[0] + s * r_eci[1],
                         -s * r_eci[0] + c * r_eci[1],
                         r_eci[2]])
    return np.stack([
        c * r_eci[..., 0] + s * r_eci[..., 1],
        -s * r_eci[..., 0] + c * r_eci[..., 1],
        r_eci[..., 2],
    ], axis=-1)


def ecef_to_topocentric(r_ecef: np.ndarray, gs: GroundStation):
    """Elevation (deg), azimuth (deg), slant range (m) of point(s) seen from a station.

    Azimuth is measured clockwise from local north; elevation is above the local
    horizontal plane (0 = horizon, 90 = zenith).  Accepts a single (3,) vector or
    an array of shape (..., 3).
    """
    gs_pos = gs.ecef()
    rel = np.asarray(r_ecef) - gs_pos
    rng = np.linalg.norm(rel, axis=-1)

    lat = np.deg2rad(gs.lat_deg)
    lon = np.deg2rad(gs.lon_deg)
    # Local ENU axes (unit vectors) in ECEF frame.
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    east = np.array([-sin_lon, cos_lon, 0.0])
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])

    e = np.tensordot(rel, east, axes=(-1, 0)) / rng
    n = np.tensordot(rel, north, axes=(-1, 0)) / rng
    u = np.tensordot(rel, up, axes=(-1, 0)) / rng
    elevation = np.rad2deg(np.arcsin(np.clip(u, -1.0, 1.0)))
    azimuth = np.rad2deg(np.arctan2(e, n)) % 360.0
    return elevation, azimuth, rng


def compute_pass(orbit: KeplerOrbit, gs: GroundStation, mjd_start: float,
                 mjd_stop: float, step_s: float = 10.0,
                 min_elevation: float = 15.0) -> pd.DataFrame:
    """Propagate an orbit over a time window and extract the ground-station pass.

    Returns a DataFrame with columns:
        mjd, t_sec (seconds since window start), sat_eci_x/y/z, sat_ecef_x/y/z,
        elevation_deg, azimuth_deg, range_m, altitude_m, visible
    where `visible` marks elevation >= min_elevation (a valid QKD window).
    Vectorised, so the whole window is propagated in one shot.
    """
    mjds = np.arange(mjd_start, mjd_stop, step_s / 86400.0)
    eci = orbit.eci_position(mjds)
    ecef = eci_to_ecef(eci, mjds)
    elev, az, rng = ecef_to_topocentric(ecef, gs)
    alt = np.linalg.norm(eci, axis=-1) - R_EARTH
    return pd.DataFrame({
        "mjd": mjds,
        "t_sec": (mjds - mjd_start) * 86400.0,
        "sat_eci_x": eci[:, 0], "sat_eci_y": eci[:, 1], "sat_eci_z": eci[:, 2],
        "sat_ecef_x": ecef[:, 0], "sat_ecef_y": ecef[:, 1], "sat_ecef_z": ecef[:, 2],
        "elevation_deg": elev, "azimuth_deg": az, "range_m": rng,
        "altitude_m": alt, "visible": elev >= min_elevation,
    })


def extract_pass_segments(df: pd.DataFrame, min_elevation: float = 15.0,
                          min_samples: int = 5) -> list[pd.DataFrame]:
    """Split a pass table into contiguous `visible` segments (each a usable pass)."""
    vis = df["elevation_deg"].to_numpy() >= min_elevation
    segments, cur = [], []
    for i, ok in enumerate(vis):
        if ok:
            cur.append(i)
        else:
            if len(cur) >= min_samples:
                segments.append(df.iloc[cur].reset_index(drop=True))
            cur = []
    if len(cur) >= min_samples:
        segments.append(df.iloc[cur].reset_index(drop=True))
    return segments
