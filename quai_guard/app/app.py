"""QU-AI-GUARD — live satellite QKD demo dashboard.

Streamlit app that animates a satellite pass over the Chennai ground station,
runs *real* BB84 rounds (Qiskit-Aer backend with atmospheric noise), auto-tunes
the QKD parameters with the trained LinkPredictor, and watches the QBER stream
for eavesdropping with the trained EveDetector — all through one GuardEngine.

Run from the project root:
    streamlit run app/app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

from src.ai.eve_detector import EveDetector
from src.ai.fusion import GuardEngine
from src.ai.link_predictor import LinkPredictor
from src.orbit.geometry import GroundStation
from src.orbit.link import LinkModel
from src.quantum.attacks import EveAttack, run_attacked_sift
from src.quantum.bb84 import run_bb84_round
from src.quantum.params import QKDParams

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GS = GroundStation(lat_deg=13.0827, lon_deg=80.2707, alt_m=10.0, name="Chennai-ISRO")
N_PHOTONS = 128


@st.cache_data(show_spinner=False)
def load_orbits() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "orbits" / "passes_meta.csv")


@st.cache_data(show_spinner=False)
def load_pass(pass_id: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "orbits" / f"{pass_id}.csv")


@st.cache_resource(show_spinner=False)
def load_models():
    lp = LinkPredictor.load(ROOT / "models" / "link_predictor.joblib")
    det = EveDetector.load(ROOT / "models" / "eve_detector.joblib")
    return lp, det


def ecef_to_geodetic(x, y, z, a=6378137.0, e2=6.69437999014e-3):
    """Simple ECEF -> geodetic (lat_deg, lon_deg, alt_m)."""
    x, y, z = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float)
    p = np.hypot(x, y)
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arctan2(z, p * (1 - e2)))
    alt = np.hypot(p, z) - a
    if alt.ndim == 0:
        alt = float(alt)
    return lat, lon, alt


def draw_scene(pass_df: pd.DataFrame):
    """Static pass view over the ground track (used as the map backdrop)."""
    sat = pass_df.iloc[-1]
    sat_lat, sat_lon, _ = ecef_to_geodetic(sat["sat_ecef_x"], sat["sat_ecef_y"],
                                           sat["sat_ecef_z"])
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=110)
    lat, lon, _ = ecef_to_geodetic(pass_df["sat_ecef_x"].to_numpy(),
                                   pass_df["sat_ecef_y"].to_numpy(),
                                   pass_df["sat_ecef_z"].to_numpy())
    ax.plot(lon, lat, "-", color="#3949ab", lw=2.2, alpha=0.85,
            label="ground track")
    ax.scatter(lon, lat, c=pass_df["elevation_deg"], cmap="viridis", s=34,
               zorder=5, edgecolors="k", linewidths=0.4)
    ax.scatter([GS.lon_deg], [GS.lat_deg], s=320, marker="*", color="#e53935",
               edgecolors="k", linewidths=0.8, zorder=6, label="Ground station")
    ax.scatter([sat_lon], [sat_lat], s=260, marker="^", color="#f4511e",
               edgecolors="k", linewidths=0.8, zorder=7, label="Satellite")
    ax.set_xlabel("Longitude [deg]"); ax.set_ylabel("Latitude [deg]")
    ax.set_title("Satellite ground track", fontsize=10)
    ax.legend(loc="best", fontsize=7, framealpha=0.7)
    ax.grid(alpha=0.25)
    return fig


def build_fig(pass_df: pd.DataFrame, verdicts: list) -> plt.Figure:
    sat = pass_df.iloc[-1]
    sat_lat, sat_lon, _ = ecef_to_geodetic(sat["sat_ecef_x"], sat["sat_ecef_y"],
                                           sat["sat_ecef_z"])
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=110)
    lat, lon, _ = ecef_to_geodetic(pass_df["sat_ecef_x"].to_numpy(),
                                   pass_df["sat_ecef_y"].to_numpy(),
                                   pass_df["sat_ecef_z"].to_numpy())
    alpha = np.clip(np.linspace(0.15, 0.9, len(lat)), 0.15, 0.9)
    ax.plot(lon, lat, "-", color="#3949ab", lw=2.2, alpha=0.85)
    ax.scatter(lon, lat, c=alpha, cmap="Blues", s=26, zorder=4)
    ax.scatter([GS.lon_deg], [GS.lat_deg], s=320, marker="*", color="#e53935",
               edgecolors="k", linewidths=0.8, zorder=6)
    ax.scatter([sat_lon], [sat_lat], s=280, marker="^",
               color="#5e35b1" if not (verdicts and verdicts[-1].alarm) else "#d32f2f",
               edgecolors="k", linewidths=1.0, zorder=7,
               label="Satellite (red = alarm)")
    ax.set_xlabel("Longitude [deg]"); ax.set_ylabel("Latitude [deg]")
    ax.set_title("Live satellite pass", fontsize=10)
    ax.legend(loc="best", fontsize=7, framealpha=0.7)
    ax.grid(alpha=0.25)
    return fig


@dataclass
class LiveFrame:
    t_sec: float
    elevation: float
    link_q: float
    qber: float
    secure_bps: float
    raw_bps: float
    alarm: bool
    conf: float
    threat: str
    secure_bits: int
    secure_key_hex: str


def run_pass(pass_df: pd.DataFrame, engine: GuardEngine, eve: EveAttack,
             n_photons: int = N_PHOTONS, noise_seed: int = 42,
             n_samples: int | None = None):
    """Yield per-sample LiveFrames for a pass (already-started GuardEngine)."""
    if n_samples:
        idx = np.linspace(0, len(pass_df) - 1, n_samples).astype(int)
    else:
        idx = np.arange(len(pass_df))
    n = len(idx)
    for k, i in enumerate(idx):
        row = pass_df.iloc[i]
        link = LinkModel()
        channel = link.channel_noise(row["elevation_deg"], row["mjd"],
                                     GS.lat_deg, GS.lon_deg,
                                     np.random.default_rng(noise_seed + i))
        rng = np.random.default_rng(noise_seed + i + 100000)
        round_ = run_bb84_round(n_photons, channel, engine.current_config,
                                rng, seed_sim=noise_seed + i)
        frac_t = k / max(n - 1, 1)
        run_attacked_sift(round_, eve, rng, frac_t=frac_t)
        v = engine.update(
            qber=float(round_.qber), secure_bps=float(round_.secure_key_bps),
            raw_bps=float(round_.raw_key_bps),
            secure_key_hex=round_.secure_key_hex,
            secure_bits=round_.secure_key_bits,
            time_s=float(row["t_sec"]),
            mean_link_q=float(channel["link_quality"]),
        )
        yield LiveFrame(
            t_sec=float(row["t_sec"]), elevation=float(row["elevation_deg"]),
            link_q=float(channel["link_quality"]), qber=v.qber,
            secure_bps=v.secure_bps, raw_bps=v.raw_bps, alarm=v.alarm,
            conf=v.confidence, threat=v.threat, secure_bits=v.secure_bits,
            secure_key_hex=v.secure_key_hex,
        )


def make_main_chart(frames: list[LiveFrame]):
    fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=110)
    t = [f.t_sec for f in frames]
    ax.plot(t, [f.qber for f in frames], color="#1e88e5", lw=1.8,
            label="QBER")
    ax.axhline(0.11, color="#888", ls="--", lw=1.0, alpha=0.8, label="11% threshold")
    ax2 = ax.twinx()
    ax2.plot(t, [f.secure_bps / 1000 for f in frames], color="#43a047", lw=1.6,
             alpha=0.95, label="secure key rate")
    alarm_t = [f.t_sec for f in frames if f.alarm]
    if alarm_t:
        ax.scatter(alarm_t, [1.0] * len(alarm_t), marker="v", color="#d32f2f",
                   s=40, zorder=5, label="alarm")
    ax.set_xlabel("Time since pass start [s]")
    ax.set_ylabel("QBER", color="#1e88e5")
    ax2.set_ylabel("Secure key rate [kbps]", color="#43a047")
    ax.grid(alpha=0.25)
    ax.set_ylim(0, 0.4)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, framealpha=0.8)
    return fig


def choose_pass(meta: pd.DataFrame) -> str:
    st.sidebar.markdown("## Satellite pass")
    sats = sorted(meta["satellite"].unique())
    default_sat = "QKD-LEO" if "QKD-LEO" in sats else sats[0]
    sat = st.sidebar.selectbox("Satellite", sats, index=sats.index(default_sat))
    sel = meta[meta["satellite"] == sat]
    sel = sel.sort_values("mean_link_q", ascending=False)
    opts = {f"{r.pass_id}  (link {r.mean_link_q:.2f}, max ele {r.max_elev:.0f}°)":
            r.pass_id for _, r in sel.iterrows()}
    key = st.sidebar.selectbox("Pass", list(opts.keys()))
    return opts[key]


def main() -> None:
    st.set_page_config(page_title="QU-AI-GUARD", page_icon="🛰️",
                       layout="wide")
    st.title("QU-AI-GUARD")
    st.caption("AI-Guarded Quantum Key Distribution for Secure Satellite-to-Ground Communication")

    meta = load_orbits()
    pass_id = choose_pass(meta)

    colA, colB = st.columns([1.0, 2.2], gap="large")

    with colB:
        with st.spinner("Preparing pass ..."):
            pass_df = load_pass(pass_id)
        st.session_state.setdefault("pass_df", pass_df)
        st.session_state.setdefault("frames", [])

        # Controls
        c1, c2, c3 = st.columns(3)
        auto = c1.toggle("AI auto-tune", value=True,
                         help="LinkPredictor picks the best QKD parameters")
        eve_mode = c2.selectbox("Adversary", ["none", "intercept_resend",
                                              "pns", "intermittent"])
        strength = c3.slider("Attack strength", 0.0, 1.0, 0.5, 0.05)
        b1, b2 = st.columns(2)
        if b1.button("▶ Run live pass", type="primary"):
            st.session_state.pop("frames", None)
        if b2.button("Reset", type="secondary"):
            st.session_state.pop("frames", None)

        lp, det = load_models()
        engine = GuardEngine(link_predictor=lp, eve_detector=det)
        config = engine.start_pass(pass_df, sat_idx=int(
            meta[meta["pass_id"] == pass_id]["sat_idx"].iloc[0]),
            auto_tune=auto)
        if auto:
            mu, dr, eta = config.laser_intensity, config.dark_count_rate, config.detector_eta
            st.sidebar.success(
                f"AI-tuned: μ={mu:.2f}  dark={dr:.0f} Hz  η={eta:.2f}")
        else:
            st.sidebar.info(f"Fixed config: `{QKDParams()}`")

        if "frames" not in st.session_state:
            st.session_state["frames"] = []
        n_already = len(st.session_state["frames"])

        n_samples = 26
        with st.spinner(f"Running {n_samples} BB84 rounds ..."):
            frames = list(run_pass(pass_df, engine, EveAttack(eve_mode, strength),
                                   n_samples=n_samples))
        st.session_state["frames"] = frames

        frames = st.session_state["frames"]
        st.subheader("Live telemetry")
        fig = make_main_chart(frames)
        st.pyplot(fig)

        # Verdict strip
        last = frames[-1]
        verdict_color = {"SECURE": "🟢", "WATCH": "🟡", "INTRUSION": "🔴"}[last.threat]
        st.markdown(
            f"### {verdict_color} {last.threat} — Eve confidence "
            f"**{last.conf:.0%}** "
            f"| QBER **{last.qber*100:.1f}%** "
            f"| secure key **{last.secure_bps/1000:.0f} kbps**")
        if last.secure_key_hex:
            st.code(f"decoded key (first 8 hex): {last.secure_key_hex[:8]}",
                    language=None)

    with colA:
        st.subheader("Ground track")
        fig = build_fig(pass_df, st.session_state.get("frames", []))
        st.pyplot(fig)
        sat_name = meta[meta["pass_id"] == pass_id]["satellite"].iloc[0]
        st.caption(f"{sat_name} · samples: {len(pass_df)} · "
                   f"mean link quality: {pass_df['link_quality'].mean():.2f}")


if __name__ == "__main__":
    main()