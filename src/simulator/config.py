# config.py - all tunable parameters for the capstone project
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

SEED = 123
RNG = __import__("numpy").random.default_rng(SEED)

# --- Light curve generation ---
# Observation window: 720-second (12-minute) window, physically motivated as LEO pass.
# RAW SAMPLING: 5-second cadence → 144 raw samples (720/5)
# RESAMPLED MODEL INPUT: 256 observations interpolated from raw, per-curve min-max normalized to [0,1].
# Every document/UI/report must use this definition: "720-second observation window represented by 256 resampled observations."
N_SAMPLES = 256  # resampled model input length
SAMPLE_INTERVAL_SEC = 5.0  # raw cadence
TOTAL_DURATION_SEC = 720.0  # observation window (12 min), NOT 2 hours
N_TIME_STEPS = int(TOTAL_DURATION_SEC / SAMPLE_INTERVAL_SEC)  # 144 raw steps

# --- Orbital / tumble physics ---
DEFAULT_ORBIT_ALT_KM = 400.0
EARTH_RADIUS_KM = 6371.0
G = 6.67430e-20

# Tumble period ranges per class (seconds)
TUMBLE_PERIOD_RANGE = {
    0: (45, 120),
    1: (120, 300),
    2: (300, 600),
    3: (20, 60),
    4: (300, 600),
}

# Aspect ratio per class
ASPECT_RATIO_RANGE = {
    0: (1.0, 3.0),
    1: (1.5, 5.0),
    2: (3.0, 10.0),
    3: (0.5, 2.0),
    4: (2.0, 6.0),
}

# Reflectivity per class
REFLECTIVITY_RANGE = {
    0: (0.2, 0.5),
    1: (0.1, 0.3),
    2: (0.1, 0.25),
    3: (0.05, 0.15),
    4: (0.05, 0.12),
}

EARTH_SHADOW_CONE_ANGLE = 0.27

# --- Feature encoding ---
N_QUBITS = 8
N_LAYERS = 2
N_CLASSICAL_HIDDEN = 16

# --- Training ---
BATCH_SIZE = 32
MAX_EPOCHS = 20
LEARNING_RATE = 0.01
TEST_SIZE = 0.2
STRATIFY = True

# --- Noise / difficulty ---
# 25k regime: per-sample realistic variation (photon noise U[0.01,0.05],
# dropout U[0.03,0.10], exposure jitter sigma 0.05). Class 2/4 share
# 350-500s tumble band with ~12% boundary-overlap sampling.
NOISE_STD_RANGE = (0.01, 0.05)
DROPOUT_RATE = 0.05
DROPOUT_RANGE = (0.03, 0.10)
EXPOSURE_VARIANCE = 0.05