# physics.py - orbital period, eclipse modeling, tumble dynamics
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class ObjectState:
    """State of a space debris object in orbit."""
    semi_major_axis_km: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    arg_of_perigee_deg: float
    true_anomaly_deg: float
    tumble_period_s: float
    precession_period_s: float
    spin_axis: np.ndarray
    tumble_phase: float
    aspect_ratio: float
    reflectivity: float


def compute_orbital_period(semi_major_axis_km: float,
                           earth_mass_kg: float = 5.972e24) -> float:
    """Keplerian orbital period for circular orbit approximation."""
    G = 6.67430e-20
    mu = G * earth_mass_kg
    n = np.sqrt(mu / semi_major_axis_km ** 3)
    T = 2.0 * np.pi / n
    return T


def compute_eclipse_flag(state: ObjectState,
                         observer_pos_km: np.ndarray,
                         sun_dir_km: np.ndarray,
                         earth_radius_km: float = 6371.0,
                         shadow_cone_angle: float = None) -> float:
    """
    Determine if the object is in Earth's shadow (eclipse).
    Returns 1.0 if illuminated, 0.0 if in eclipse.
    """
    from .config import EARTH_SHADOW_CONE_ANGLE as _SCA
    angle = _SCA if shadow_cone_angle is None else shadow_cone_angle
    
    r = np.linalg.norm(observer_pos_km)
    obj_dir = observer_pos_km / r if r > 0 else np.zeros(3)

    shadow_check = np.dot(obj_dir, -sun_dir_km / np.linalg.norm(sun_dir_km))

    in_shadow = bool(shadow_check > 0 and r < earth_radius_km / np.sin(angle))

    return 0.0 if in_shadow else 1.0


def compute_tumble_orientation(t: float,
                               tumble_period: float,
                               precession_period: float,
                               spin_axis: np.ndarray,
                               phase: float = 0.0) -> np.ndarray:
    """
    Compute the tumble orientation at time t.
    Returns a 3x3 rotation matrix representing the object's orientation.
    """
    angle_tumble = 2.0 * np.pi * t / tumble_period
    angle_precess = 2.0 * np.pi * t / precession_period

    k = spin_axis / np.linalg.norm(spin_axis) if np.linalg.norm(spin_axis) > 0 else np.array([0.0, 0.0, 1.0])

    c = np.cos(angle_tumble)
    s = np.sin(angle_tumble)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])

    R_tumble = np.eye(3) + c * K + (1.0 - c) * (K @ K)

    world_z = np.array([0.0, 0.0, 1.0])
    if np.linalg.norm(k - world_z) < 1e-6:
        world_z = np.array([1.0, 0.0, 0.0])

    p_axis = np.cross(k, world_z)
    p_axis = p_axis / np.linalg.norm(p_axis)

    c2 = np.cos(angle_precess)
    s2 = np.sin(angle_precess)
    K2 = np.array([[0, -p_axis[2], p_axis[1]],
                   [-p_axis[2], 0, p_axis[0]],
                   [-p_axis[1], p_axis[0], 0]])

    R_precess = np.eye(3) + c2 * K2 + (1.0 - c2) * (K2 @ K2)

    R = R_precess @ R_tumble
    return R


def specular_reflectance(incident_dir: np.ndarray,
                         surface_normal: np.ndarray,
                         reflectivity: float) -> float:
    """
    Compute specular (mirror-like) reflectance for a flat panel.
    """
    dot = np.dot(incident_dir, surface_normal)
    if dot <= 0:
        return 0.0
    reflected_dir = incident_dir - 2.0 * dot * surface_normal
    return max(0.0, reflectivity)


def diffuse_reflectance(surface_normal: np.ndarray,
                        sun_dir: np.ndarray,
                        reflectivity: float) -> float:
    """
    Compute diffuse (Lambertian) reflectance.
    """
    dot = np.dot(surface_normal, sun_dir)
    if dot <= 0:
        return 0.0
    return reflectivity * dot / np.pi


def compute_light_curve_from_faces(times: np.ndarray,
                                   state: ObjectState,
                                   sun_positions: np.ndarray,
                                   observer_positions: np.ndarray,
                                   n_faces: int = 4) -> np.ndarray:
    """
    Generate a light curve given object state and geometry over time.
    """
    N = len(times)
    intensities = np.zeros(N)

    aspect = state.aspect_ratio
    reflect = state.reflectivity

    for i in range(N):
        t = times[i]
        sun_dir = sun_positions[i]
        obs_dir = observer_positions[i]

        R = compute_tumble_orientation(
            t, state.tumble_period_s, state.precession_period_s,
            state.spin_axis, state.tumble_phase
        )

        face_normals = []
        for j in range(n_faces):
            phi = 2.0 * np.pi * j / n_faces
            z = (j - n_faces / 2) / (n_faces / 2) * 0.5
            x = np.cos(phi) * np.sqrt(1 - z ** 2) * aspect
            y = np.sin(phi) * np.sqrt(1 - z ** 2) * aspect
            normal = np.array([x, y, z])
            normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else normal
            normal = R @ normal
            face_normals.append(normal)
        face_normals = np.array(face_normals)

        total = 0.0
        for normal in face_normals:
            s_term = specular_reflectance(sun_dir, normal, reflect)
            d_term = diffuse_reflectance(normal, sun_dir, reflect)

            o_dot = np.dot(normal, obs_dir)
            if o_dot > 0:
                boost = max(0.0, o_dot) * 0.5
                s_term = min(1.0, s_term + boost)

            total += s_term + d_term

        intensities[i] = total / n_faces

        eclipsed = compute_eclipse_flag(state, observer_positions[i], sun_dir)
        if eclipsed == 0.0:
            intensities[i] = 0.0

    if intensities.max() > intensities.min():
        intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())

    return intensities