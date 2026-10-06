# pages/1_Quantum_Lab.py — Q-ORBIT Quantum Lab
# Restyled to match the clean dark space theme.
# All backend logic (Bloch sphere, gates, Aer simulator, state evolution,
# quantum noise, real pipeline demo) fully preserved.
from __future__ import annotations
import os, sys, base64, io, time, json, traceback, math, cmath
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from app.space_theme import css, stars_html, sidebar_logo, apply_layout, CLASS_NAMES

from src.quantum.bloch import compute_bloch_vectors, bloch_sphere_figure, purity_from_bloch
from src.quantum.live_circuit import CircuitSpec
from src.quantum.backends.pennylane_backend import PennylaneBackend

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.join(BASE_DIR, "..", "..")
PHOTOS_DIR = os.path.join(ROOT, "assets", "images", "web", "photos")
REPORTS    = os.path.join(ROOT, "results", "reports")

st.set_page_config(
    page_title="Q-ORBIT — Quantum Lab",
    layout="wide", page_icon="⚛️",
    initial_sidebar_state="collapsed",
)

# ── Gate catalogue ────────────────────────────────────────────────────────────
GATE_INFO = {
    "H":    {"math": "H|0⟩ = (|0⟩+|1⟩)/√2",    "desc": "Superposition — equator."},
    "X":    {"math": "X|0⟩ = |1⟩",              "desc": "Bit-flip, 180° about X."},
    "Y":    {"math": "Y|0⟩ = i|1⟩",             "desc": "180° about Y, adds phase."},
    "Z":    {"math": "Z|+⟩ = |−⟩",              "desc": "Phase-flip, 180° about Z."},
    "S":    {"math": "S|1⟩ = i|1⟩",             "desc": "90° Z rotation, π/2 phase."},
    "T":    {"math": "T|1⟩ = e^{iπ/4}|1⟩",      "desc": "45° Z rotation, non-Clifford."},
    "RX":   {"math": "RX(θ) = cos(θ/2)I−i sin(θ/2)X", "desc": "Rotation about X."},
    "RY":   {"math": "RY(θ) = cos(θ/2)I−i sin(θ/2)Y", "desc": "Rotation about Y."},
    "RZ":   {"math": "RZ(φ) = diag(e^{−iφ/2}, e^{iφ/2})", "desc": "Rotation about Z."},
    "CNOT": {"math": "CNOT|c,t⟩ = |c,t⊕c⟩",    "desc": "Entangling, Bell creation."},
}

CRYO_STAGES = [
    {"temp": "300 K",   "name": "Room Temperature",   "top":  7, "w": 440},
    {"temp": "50 K",    "name": "Pulse-Tube 1st Stage","top": 19, "w": 380},
    {"temp": "4 K",     "name": "Cold Stage",          "top": 31, "w": 326},
    {"temp": "800 mK",  "name": "Still Plate",         "top": 44, "w": 270},
    {"temp": "100 mK",  "name": "Cold Plate",          "top": 57, "w": 214},
    {"temp": "10 mK",   "name": "Mixing Chamber",      "top": 71, "w": 162},
]


# ── helpers ───────────────────────────────────────────────────────────────────

def section(num: str, title: str, sub: str = ""):
    sub_html = (
        f'<p style="color:#6B84A8;font-size:0.88rem;line-height:1.55;margin:0.2rem 0 0 0">{sub}</p>'
        if sub else ""
    )
    st.markdown(f"""
<div style="margin:2rem 0 1rem 0;padding-bottom:0.7rem;border-bottom:1px solid #1E2E4A">
  <div style="font-family:'Space Mono',monospace;font-size:0.58rem;letter-spacing:2.5px;
              color:#A69FD6;margin-bottom:0.25rem">SECTION {num}</div>
  <h2 style="font-size:clamp(1.3rem,2.5vw,2rem);font-weight:600;color:#F0F6FF;margin:0">
    {title}
  </h2>
  {sub_html}
</div>""", unsafe_allow_html=True)


def bloch_fig_dark(theta: float, phi: float, qubit_idx: int = 0):
    """Dark-themed Bloch sphere."""
    x_s = math.sin(theta) * math.cos(phi)
    y_s = math.sin(theta) * math.sin(phi)
    z_s = math.cos(theta)
    a   = math.cos(theta / 2)
    b_c = cmath.exp(1j * phi) * math.sin(theta / 2)
    r   = math.sqrt(x_s**2 + y_s**2 + z_s**2)

    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 18)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs, opacity=0.07,
        colorscale=[[0, "#0D1628"], [1, "#1E3060"]],
        showscale=False, hoverinfo="skip",
    ))
    # Equator + meridians
    t = np.linspace(0, 2 * np.pi, 60)
    for x_c, y_c, z_c, color in [
        (np.cos(t), np.sin(t),         np.zeros_like(t), "rgba(56,189,248,0.20)"),
        (np.zeros_like(t), np.cos(t),  np.sin(t),        "rgba(166,159,214,0.15)"),
        (np.cos(t), np.zeros_like(t),  np.sin(t),        "rgba(107,132,168,0.12)"),
    ]:
        fig.add_trace(go.Scatter3d(
            x=x_c, y=y_c, z=z_c, mode="lines",
            line=dict(color=color, width=1.5),
            hoverinfo="skip", showlegend=False,
        ))
    # State vector arrow
    fig.add_trace(go.Scatter3d(
        x=[0, x_s], y=[0, y_s], z=[0, z_s],
        mode="lines", line=dict(color="#38BDF8", width=7),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=[x_s], y=[y_s], z=[z_s],
        mode="markers",
        marker=dict(size=5, color="#A69FD6",
                    line=dict(width=1.5, color="#38BDF8")),
        hoverinfo="skip", showlegend=False,
    ))
    # Axis labels
    fig.add_trace(go.Scatter3d(
        x=[0, 0, 1.22, -1.22, 0, 0],
        y=[0, 0, 0, 0, 1.22, -1.22],
        z=[1.22, -1.22, 0, 0, 0, 0],
        mode="text",
        text=["|0⟩", "|1⟩", "+X", "−X", "+Y", "−Y"],
        textfont=dict(color="#6B84A8", size=9),
        hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1.3, 1.3], showgrid=False, showticklabels=False,
                       title="", backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(range=[-1.3, 1.3], showgrid=False, showticklabels=False,
                       title="", backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(range=[-1.3, 1.3], showgrid=False, showticklabels=False,
                       title="", backgroundcolor="rgba(0,0,0,0)"),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.5, y=1.4, z=1.0)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        height=380,
        title=dict(
            text=f"q{qubit_idx} — θ {math.degrees(theta):.1f}° · φ {math.degrees(phi):.1f}° · |r| {r:.2f}",
            font=dict(size=11, color="#C8D8F0"), x=0.02,
        ),
        showlegend=False,
    )
    return fig, a, b_c, x_s, y_s, z_s


def run_aer_code(code_str: str, shots: int = 1024) -> dict:
    """Execute user Qiskit code on AerSimulator. Returns result dict."""
    start = time.time()
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        g = {"QuantumCircuit": QuantumCircuit, "AerSimulator": AerSimulator,
             "np": np, "numpy": np}
        l: dict = {}
        exec(code_str, g, l)
        ns = {**g, **l}
        qc = ns.get("qc")
        if qc is None:
            for v in ns.values():
                if isinstance(v, QuantumCircuit):
                    qc = v
                    break
        if qc is None:
            return {"status": "error",
                    "error": "No QuantumCircuit 'qc' found. Define: qc = QuantumCircuit(n)",
                    "elapsed": time.time() - start}
        has_measure = any(
            instr.operation.name == "measure" for instr in qc.data
        ) if hasattr(qc, "data") else False
        sim = AerSimulator()
        if not has_measure:
            qc2 = qc.copy()
            qc2.save_statevector()
            res = sim.run(qc2).result()
            sv = np.asarray(res.get_statevector(qc2), dtype=complex)
            probs = np.abs(sv) ** 2
            probs /= probs.sum() if probs.sum() > 0 else 1
            counts = {format(i, f"0{qc.num_qubits}b"): float(probs[i])
                      for i in range(len(probs))}
            qc_m = qc.copy()
            qc_m.measure_all()
            counts_shots = sim.run(qc_m, shots=shots).result().get_counts(qc_m)
            return {
                "status": "success", "backend": "AerSimulator (statevector)",
                "shots": shots, "counts": counts_shots, "probs": counts,
                "statevector": sv, "circuit": qc,
                "elapsed": time.time() - start,
                "depth": qc.depth(), "num_qubits": qc.num_qubits,
                "num_gates": len(qc.data),
            }
        else:
            res = sim.run(qc, shots=shots).result()
            counts = res.get_counts(qc)
            probs = {k: v / sum(counts.values()) for k, v in counts.items()}
            return {
                "status": "success", "backend": "AerSimulator",
                "shots": shots, "counts": counts, "probs": probs,
                "statevector": None, "circuit": qc,
                "elapsed": time.time() - start,
                "depth": qc.depth(), "num_qubits": qc.num_qubits,
                "num_gates": len(qc.data),
            }
    except Exception as e:
        return {"status": "error", "error": str(e) + "\n" + traceback.format_exc(limit=3),
                "elapsed": time.time() - start}


def circuit_to_image_b64(qc) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = qc.draw(output="mpl", style="clifford")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                    facecolor="#0D1628")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def mono_badge(text: str, color: str = "#38BDF8") -> str:
    return (
        f'<span style="background:rgba({int(color[1:3],16)},'
        f'{int(color[3:5],16)},{int(color[5:7],16)},0.08);'
        f'border:1px solid rgba({int(color[1:3],16)},'
        f'{int(color[3:5],16)},{int(color[5:7],16)},0.25);'
        f'color:{color};border-radius:4px;font-family:\'Space Mono\',monospace;'
        f'font-size:0.62rem;letter-spacing:1px;padding:0.22rem 0.55rem">{text}</span>'
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    st.markdown(css(), unsafe_allow_html=True)
    st.markdown(stars_html(seed=77), unsafe_allow_html=True)
    sidebar_logo(st)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
<div style="padding:1.6rem 0 1.2rem 0;border-bottom:1px solid #1E2E4A;margin-bottom:1.6rem">
  <div style="font-family:'Space Mono',monospace;font-size:0.62rem;letter-spacing:2.5px;
              color:#A69FD6;margin-bottom:0.4rem">INTERACTIVE QUANTUM LABORATORY</div>
  <h1 style="font-size:clamp(1.8rem,3.5vw,2.6rem);font-weight:700;color:#F0F6FF;
              margin:0 0 0.4rem 0">Quantum Lab</h1>
  <p style="color:#6B84A8;font-size:0.92rem;max-width:72ch;margin:0 0 0.7rem 0">
    From qubits to orbits — explore quantum states, circuits and hardware concepts. 
    Understand how these technologies connect to the Q-ORBIT hybrid classification system.
  </p>
  <div style="display:flex;gap:0.5rem;flex-wrap:wrap;align-items:center">
    <span style="display:inline-flex;align-items:center;gap:0.4rem;
                 background:rgba(100,216,168,0.07);border:1px solid rgba(100,216,168,0.2);
                 color:#64D8A8;border-radius:999px;font-family:'Space Mono',monospace;
                 font-size:0.62rem;letter-spacing:1.2px;padding:0.3rem 0.7rem">
      <span style="width:6px;height:6px;border-radius:50%;background:#64D8A8;
                   display:inline-block;animation:qo-pulse 2s infinite"></span>
      SIMULATOR READY — PennyLane default.qubit + Qiskit Aer
    </span>
    <span style="font-family:'Space Mono',monospace;font-size:0.62rem;
                 color:#6B84A8;padding:0.3rem 0.7rem;border:1px solid #1E2E4A;
                 border-radius:999px">
      Real hardware: disabled (no IBM credentials)
    </span>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Section 01 — The Machine ──────────────────────────────────────────────
    section("01", "The Quantum Computer — Dilution Refrigerator",
            "Metallic cooling stages from 300 K to 10 mK. "
            "The processor sits at the bottom plate. "
            "Select a stage to see its role.")

    sel = st.selectbox(
        "Highlight cooling stage",
        options=[f"{s['temp']} — {s['name']}" for s in CRYO_STAGES],
        index=5, key="cryo_sel",
    )
    sel_temp = sel.split(" — ")[0].strip()
    sel_info = next(s for s in CRYO_STAGES if s["temp"] == sel_temp)

    stage_explain = {
        "300 K":   "Control electronics and readout — waveform generators, amplifiers, digitizers.",
        "50 K":    "Pulse-tube first stage — removes the bulk of incoming heat, radiation shielding.",
        "4 K":     "Additional pre-cooling — superconducting coax and attenuators thermalized.",
        "800 mK":  "Still plate — pumps He-3 to drive continuous He-3/He-4 circulation.",
        "100 mK":  "Cold plate — further suppresses thermal energy, cables anchored.",
        "10 mK":   "Mixing chamber — ultra-low temperature environment for superconducting processors.",
    }

    # Build plate HTML
    plates_html = ""
    for s in CRYO_STAGES:
        active = s["temp"] == sel_temp
        bg     = "rgba(56,189,248,0.08)" if active else "#111D32"
        border = "rgba(56,189,248,0.3)" if active else "#1E2E4A"
        color  = "#38BDF8" if active else "#C8D8F0"
        plates_html += (
            f'<div style="position:absolute;left:50%;top:{s["top"]}%;'
            f'width:{s["w"]}px;height:24px;transform:translateX(-50%);'
            f'border-radius:4px;background:{bg};border:1px solid {border};'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-family:\'Space Mono\',monospace;font-size:0.6rem;'
            f'letter-spacing:1.2px;color:{color}">{s["temp"]}</div>\n'
        )

    st.markdown(f"""
<div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:12px;
            padding:1rem;margin-bottom:1rem;position:relative;height:400px;
            overflow:hidden">
  <div style="position:absolute;left:50.2%;top:8%;width:2px;height:68%;
              transform:translateX(-50%);
              background:linear-gradient(180deg,#38BDF8,#A69FD6);
              animation:qo-pulse 2.4s ease-in-out infinite;opacity:0.85"></div>
  <div style="position:absolute;left:50%;top:8%;width:1px;height:68%;
              transform:translateX(-50%);
              background:repeating-linear-gradient(180deg,rgba(107,132,168,0.25)
              0 2px,transparent 2px 9px)"></div>
  {plates_html}
  <div style="position:absolute;left:50%;top:77%;transform:translateX(-50%);
              width:88px;height:88px;border-radius:12px;background:#0D1628;
              border:1.5px solid rgba(56,189,248,0.2);
              box-shadow:0 0 24px rgba(56,189,248,0.12);
              display:grid;place-items:center">
    <div style="width:62px;height:62px;border-radius:8px;
                background:repeating-linear-gradient(0deg,rgba(56,189,248,0.1)
                0 1px,transparent 1px 10px),
                repeating-linear-gradient(90deg,rgba(166,159,214,0.08)
                0 1px,transparent 1px 12px);
                border:1px solid rgba(56,189,248,0.15)"></div>
  </div>
  <div style="position:absolute;bottom:8px;left:12px;
              font-family:'Space Mono',monospace;font-size:0.56rem;
              color:#6B84A8;background:#060B18;border:1px solid #1E2E4A;
              padding:0.22rem 0.5rem;border-radius:999px">
    He-3/He-4 dilution · 300K → 10mK
  </div>
</div>""", unsafe_allow_html=True)

    ci1, ci2 = st.columns([1.3, 1])
    with ci1:
        st.markdown(f"""
<div style="background:#0D1628;border:1px solid rgba(56,189,248,0.2);
            border-radius:10px;padding:0.9rem">
  <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
              letter-spacing:2px;color:#38BDF8;margin-bottom:0.35rem">
    {sel_info["temp"]} — {sel_info["name"]}
  </div>
  <div style="font-size:0.88rem;color:#C8D8F0;line-height:1.6">
    {stage_explain.get(sel_info["temp"], "")}
  </div>
</div>""", unsafe_allow_html=True)
    with ci2:
        st.markdown('<div style="display:grid;gap:0.3rem">', unsafe_allow_html=True)
        for s in CRYO_STAGES:
            hl = s["temp"] == sel_temp
            bg     = "rgba(56,189,248,0.08)" if hl else "#0D1628"
            border = "rgba(56,189,248,0.25)" if hl else "#1E2E4A"
            color  = "#38BDF8" if hl else "#C8D8F0"
            st.markdown(
                f'<div style="background:{bg};border:1px solid {border};'
                f'border-radius:6px;padding:0.38rem 0.7rem;'
                f'display:flex;justify-content:space-between;'
                f'font-family:\'Space Mono\',monospace;font-size:0.62rem;color:{color}">'
                f'<span>{s["name"]}</span><strong>{s["temp"]}</strong></div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 02 — The Qubit / Bloch Sphere ────────────────────────────────
    section("02", "The Qubit — Bloch Sphere Visualisation",
            "The state vector |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩ is computed live. "
            "Adjust θ (polar) and φ (azimuth) to move the arrow.")

    if "bloch_theta" not in st.session_state:
        st.session_state.bloch_theta = 0.0
    if "bloch_phi" not in st.session_state:
        st.session_state.bloch_phi = 0.0

    # Preset buttons
    pcols = st.columns(6)
    for col, (label, th, ph) in zip(pcols, [
        ("|0⟩", 0.0,             0.0),
        ("|1⟩", math.pi,         0.0),
        ("|+⟩", math.pi / 2,    0.0),
        ("|−⟩", math.pi / 2,    math.pi),
        ("|i⟩", math.pi / 2,    math.pi / 2),
        ("||−i⟩", math.pi / 2, -math.pi / 2),
    ]):
        if col.button(label, use_container_width=True, key=f"preset_{label}"):
            st.session_state.bloch_theta = th
            st.session_state.bloch_phi   = ph

    col_bloch, col_eq = st.columns([1.3, 1])
    with col_bloch:
        theta = st.slider(
            "θ — polar angle from |0⟩ [0, π]", 0.0, float(math.pi),
            float(st.session_state.bloch_theta), step=0.02, key="th_dark",
        )
        phi = st.slider(
            "φ — azimuth around Z [−π, π]", -float(math.pi), float(math.pi),
            float(st.session_state.bloch_phi), step=0.02, key="ph_dark",
        )
        st.session_state.bloch_theta = theta
        st.session_state.bloch_phi   = phi

        fig_bloch, a, b, x_b, y_b, z_b = bloch_fig_dark(theta, phi)
        st.plotly_chart(fig_bloch, use_container_width=True,
                        config={"displayModeBar": False})

        ec1, ec2, ec3 = st.columns(3)
        if ec1.button("▶ Animate φ", use_container_width=True, key="play_evo"):
            st.session_state.evolve = True
        if ec2.button("⏸ Pause",    use_container_width=True, key="pause_evo"):
            st.session_state.evolve = False
        if ec3.button("↺ Reset |0⟩", use_container_width=True, key="reset_evo"):
            st.session_state.bloch_theta = 0.0
            st.session_state.bloch_phi   = 0.0
            st.session_state.evolve      = False
            st.rerun()
        if st.session_state.get("evolve", False):
            new_phi = (phi + 0.12) % (2 * math.pi)
            if new_phi > math.pi:
                new_phi -= 2 * math.pi
            st.session_state.bloch_phi = new_phi
            time.sleep(0.12)
            st.rerun()

    with col_eq:
        P0 = abs(a) ** 2
        P1 = abs(b) ** 2
        st.markdown(f"""
<div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:10px;
            padding:1rem;font-family:'Space Mono',monospace;font-size:0.8rem;
            line-height:1.9;color:#C8D8F0">
  <div style="font-size:0.6rem;letter-spacing:2px;color:#A69FD6;
              margin-bottom:0.5rem">LIVE STATE EQUATION</div>
  |ψ⟩ = cos(θ/2)|0⟩ + e<sup>iφ</sup> sin(θ/2)|1⟩<br/>
  <span style="color:#6B84A8">θ = {theta:.3f} rad ({math.degrees(theta):.1f}°)</span><br/>
  <span style="color:#6B84A8">φ = {phi:.3f} rad ({math.degrees(phi):.1f}°)</span><br/>
  α = {a.real:+.4f}<br/>
  β = {b.real:+.4f}{b.imag:+.4f}i<br/>
  ⟨X⟩={x_b:+.3f} ⟨Y⟩={y_b:+.3f} ⟨Z⟩={z_b:+.3f}<br/>
  <span style="color:#38BDF8">P(|0⟩) = {P0*100:.1f}%</span> · 
  <span style="color:#A69FD6">P(|1⟩) = {P1*100:.1f}%</span>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div style="margin-top:0.6rem;font-family:'Space Mono',monospace;font-size:0.58rem;
            letter-spacing:1.5px;color:#6B84A8;margin-bottom:0.3rem">
  MEASUREMENT PROBABILITIES
</div>""", unsafe_allow_html=True)
        for label, pval, color in [("|0⟩", P0, "#38BDF8"), ("|1⟩", P1, "#A69FD6")]:
            st.markdown(
                f'<div style="background:#111D32;border:1px solid #1E2E4A;'
                f'border-radius:4px;height:28px;overflow:hidden;margin:4px 0">'
                f'<div style="background:{color};width:{pval*100:.1f}%;height:100%;'
                f'display:flex;align-items:center;padding-left:8px;'
                f'font-family:\'Space Mono\',monospace;font-size:0.7rem;color:#F0F6FF">'
                f'{label} {pval*100:.1f}%</div></div>',
                unsafe_allow_html=True,
            )

    # ── Section 03 — Circuit Builder ─────────────────────────────────────────
    section("03", "Quantum Gates — Circuit Builder",
            "Add gates, watch the Bloch spheres update from real PennyLane simulation. "
            "Export as Qiskit or PennyLane code.")

    if "qspec" not in st.session_state:
        st.session_state.qspec = CircuitSpec(n_qubits=2)
    nqb = st.slider("Number of qubits", 1, 8,
                    int(st.session_state.qspec.n_qubits), key="nqb")
    st.session_state.qspec.n_qubits = nqb
    qspec = st.session_state.qspec

    g1, g2, g3 = st.columns([1, 1.3, 1])

    with g1:
        st.markdown("""
<div style="font-family:'Space Mono',monospace;font-size:0.6rem;letter-spacing:2px;
            color:#38BDF8;margin-bottom:0.5rem">GATE TOOLBAR</div>""",
            unsafe_allow_html=True)
        gate = st.selectbox("Gate", list(GATE_INFO.keys()), key="gate_sel")
        info = GATE_INFO[gate]
        st.markdown(
            f'<div style="background:#111D32;border:1px solid #1E2E4A;border-radius:8px;'
            f'padding:0.6rem;font-family:\'Space Mono\',monospace;font-size:0.7rem;'
            f'color:#A69FD6;line-height:1.6;margin-bottom:0.5rem">'
            f'{info["math"]}<br/>'
            f'<span style="color:#6B84A8">{info["desc"]}</span></div>',
            unsafe_allow_html=True,
        )
        w1   = st.number_input(f"Qubit 0–{nqb-1}", 0, nqb-1, 0, key="w1")
        need2 = gate in ("CNOT", "CZ")
        w2   = st.number_input(f"Target 0–{nqb-1}", 0, nqb-1,
                               min(1, nqb-1), key="w2") if need2 else None
        ang  = (st.slider("Angle (rad)", -math.pi, math.pi, math.pi/4, key="ang")
                if gate in ("RX", "RY", "RZ") else None)

        ca, cb = st.columns(2)
        if ca.button("Add Gate", use_container_width=True, type="primary", key="add_g"):
            if gate in ("RX", "RY", "RZ"):
                qspec.add_gate(gate, [int(w1)], [float(ang)])
            elif need2:
                if int(w1) == int(w2):
                    st.toast("Control ≠ target")
                else:
                    qspec.add_gate(gate, [int(w1), int(w2)])
            else:
                qspec.add_gate(gate, [int(w1)])
            st.rerun()
        if cb.button("Undo", use_container_width=True, key="undo_g"):
            if qspec.ops:
                qspec.ops.pop()
            st.rerun()
        if st.button("Clear All", use_container_width=True, key="clear_g"):
            qspec.clear()
            st.rerun()

        pc1, pc2, pc3 = st.columns(3)
        if pc1.button("|+⟩", use_container_width=True, key="p_plus"):
            qspec.clear(); qspec.add_gate("H", [0]); st.rerun()
        if pc2.button("Bell", use_container_width=True, key="p_bell"):
            qspec.clear(); qspec.add_gate("H", [0])
            qspec.add_gate("CNOT", [0, 1] if nqb >= 2 else [0, 0]); st.rerun()
        if pc3.button("GHZ", use_container_width=True, key="p_ghz"):
            qspec.clear(); qspec.add_gate("H", [0])
            for i in range(nqb - 1):
                qspec.add_gate("CNOT", [i, i + 1])
            st.rerun()

        # wire view
        if qspec.ops:
            st.markdown("""
<div style="font-family:'Space Mono',monospace;font-size:0.58rem;letter-spacing:1.5px;
            color:#6B84A8;margin-top:0.6rem;margin-bottom:0.3rem">CIRCUIT WIRES</div>""",
                unsafe_allow_html=True)
            for qi in range(min(4, nqb)):
                chips = "".join(
                    f'<span style="background:#111D32;border:1px solid #1E2E4A;'
                    f'border-radius:4px;padding:0.1rem 0.4rem;'
                    f'font-size:0.62rem;color:#38BDF8">{op["gate"]}</span>'
                    for op in qspec.ops if qi in op["wires"]
                )
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:0.2rem;'
                    f'margin:3px 0;font-family:\'Space Mono\',monospace;'
                    f'font-size:0.6rem;color:#6B84A8">'
                    f'q{qi} ── {chips or "─ (idle) ─"}</div>',
                    unsafe_allow_html=True,
                )

    with g2:
        st.markdown("""
<div style="font-family:'Space Mono',monospace;font-size:0.6rem;letter-spacing:2px;
            color:#38BDF8;margin-bottom:0.5rem">LIVE SIMULATION — BLOCH SPHERES</div>""",
            unsafe_allow_html=True)
        try:
            res = qspec.simulate("pennylane")
            bloch_res = res.bloch_vectors
            bc1, bc2 = st.columns(2)
            for i in range(min(4, nqb)):
                bth = np.arccos(np.clip(bloch_res[i][2], -1, 1))
                bph = np.arctan2(bloch_res[i][1], bloch_res[i][0])
                fig_b, _, _, _, _, _ = bloch_fig_dark(bth, bph, qubit_idx=i)
                with (bc1 if i % 2 == 0 else bc2):
                    st.plotly_chart(fig_b, use_container_width=True,
                                    config={"displayModeBar": False},
                                    key=f"bloch_builder_{i}")
            # Probabilities
            probs = np.abs(res.statevector) ** 2
            probs /= probs.sum() if probs.sum() > 0 else 1
            top3  = np.argsort(probs)[::-1][:4]
            st.markdown(
                '<div style="font-family:\'Space Mono\',monospace;font-size:0.58rem;'
                'letter-spacing:1.5px;color:#6B84A8;margin:0.4rem 0 0.25rem 0">'
                'MEASUREMENT PROBABILITIES</div>',
                unsafe_allow_html=True,
            )
            for i in top3:
                if probs[i] < 0.001:
                    continue
                bits = format(i, f"0{nqb}b")[::-1]
                st.markdown(
                    f'<div style="background:#111D32;border:1px solid #1E2E4A;'
                    f'border-radius:4px;padding:0.3rem 0.6rem;margin:2px 0;'
                    f'display:flex;justify-content:space-between;'
                    f'font-family:\'Space Mono\',monospace;font-size:0.68rem;color:#C8D8F0">'
                    f'<span>|{bits}⟩</span><span style="color:#38BDF8">'
                    f'{probs[i]*100:.2f}%</span></div>',
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.error(f"Simulation error: {e}")

    with g3:
        st.markdown("""
<div style="font-family:'Space Mono',monospace;font-size:0.6rem;letter-spacing:2px;
            color:#38BDF8;margin-bottom:0.5rem">EXPORT CODE</div>""",
            unsafe_allow_html=True)
        ct1, ct2 = st.tabs(["Qiskit", "PennyLane"])
        with ct1:
            st.code(qspec.to_qiskit_code(), language="python")
            st.download_button(
                "⬇ Qiskit .py", data=qspec.to_qiskit_code(),
                file_name="qorbit_qiskit.py", mime="text/x-python",
                key="dl_qiskit", use_container_width=True,
            )
        with ct2:
            st.code(qspec.to_pennylane_code(), language="python")
            st.download_button(
                "⬇ PennyLane .py", data=qspec.to_pennylane_code(),
                file_name="qorbit_pennylane.py", mime="text/x-python",
                key="dl_penny", use_container_width=True,
            )

    # ── Section 04 — Qiskit Experiment ───────────────────────────────────────
    section("04", "Live Qiskit Execution — Aer Simulator",
            "Write any Qiskit circuit (name it `qc`), run on AerSimulator. "
            "Hardware path clearly disabled — no real hardware claims.")

    default_code = (
        "from qiskit import QuantumCircuit\n"
        "qc = QuantumCircuit(2)\n"
        "qc.h(0)\n"
        "qc.cx(0, 1)\n"
        "qc.measure_all()\n"
        "# Try: qc.ry(0.7, 0); qc.rz(1.2, 1)\n"
    )
    if "play_code" not in st.session_state:
        st.session_state.play_code = default_code
    st.session_state.play_code = st.text_area(
        "Qiskit circuit editor", value=st.session_state.play_code,
        height=160, key="editor",
    )
    rc1, rc2, rc3, rc4 = st.columns([1, 1, 0.8, 0.8])
    shots    = rc1.number_input("Shots", 100, 10000, 1024, step=100, key="shots")
    run_sim  = rc2.button("▶ Run Simulator", type="primary", key="run_sim",
                          use_container_width=True)
    rc3.button("⬢ Real Hardware", disabled=True,
               help="Set QISKIT_IBM_TOKEN to enable", key="run_hw",
               use_container_width=True)
    if rc4.button("↺ Reset", key="reset_code", use_container_width=True):
        st.session_state.play_code = default_code
        st.rerun()
    st.markdown("""
<div style="font-family:'Space Mono',monospace;font-size:0.58rem;color:#6B84A8;
            margin-bottom:0.5rem">
  ○ Real hardware — unavailable. Simulator mode active. 
  Results are never claimed as real hardware.
</div>""", unsafe_allow_html=True)

    if run_sim:
        with st.spinner("AerSimulator running…"):
            result = run_aer_code(st.session_state.play_code, shots=int(shots))
            st.session_state.play_result = result

    if "play_result" in st.session_state:
        result = st.session_state.play_result
        if result["status"] == "error":
            st.error(f"Circuit error:\n{result['error']}")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Backend", result["backend"].split(" ")[0])
            m2.metric("Shots",   result["shots"])
            m3.metric("Qubits",  result["num_qubits"])
            m4.metric("Depth",   result["depth"])
            st.caption(f"Elapsed {result['elapsed']*1000:.1f} ms — simulator only")

            b64 = circuit_to_image_b64(result["circuit"])
            if b64:
                st.markdown(
                    f'<div style="background:#0D1628;border:1px solid #1E2E4A;'
                    f'border-radius:8px;padding:0.5rem;margin-bottom:0.5rem">'
                    f'<img src="data:image/png;base64,{b64}" '
                    f'style="width:100%;border-radius:6px"/></div>',
                    unsafe_allow_html=True,
                )

            view = st.radio(
                "View results as",
                ["Counts", "Probabilities", "Statevector"],
                horizontal=True, key="play_view",
            )
            if view == "Counts":
                c_data = result["counts"]
                fig_c = go.Figure(go.Bar(
                    x=list(c_data.keys()), y=list(c_data.values()),
                    marker_color="#38BDF8", opacity=0.85,
                    text=list(c_data.values()), textposition="auto",
                ))
                apply_layout(fig_c, height=260, title="Measurement Counts")
                fig_c.update_layout(
                    yaxis=dict(title="Counts", gridcolor="rgba(30,46,74,0.6)"),
                )
                st.plotly_chart(fig_c, use_container_width=True,
                                config={"displayModeBar": False})
                st.json(c_data)
            elif view == "Probabilities":
                p_data = result["probs"]
                vals = [v * 100 if max(p_data.values()) <= 1 else v
                        for v in p_data.values()]
                fig_p = go.Figure(go.Bar(
                    x=list(p_data.keys()), y=vals,
                    marker_color="#A69FD6", opacity=0.85,
                    text=[f"{v:.1f}%" for v in vals], textposition="auto",
                ))
                apply_layout(fig_p, height=260, title="Measurement Probabilities (%)")
                st.plotly_chart(fig_p, use_container_width=True,
                                config={"displayModeBar": False})
            else:
                sv = result.get("statevector")
                if sv is not None:
                    st.code(
                        f"statevector =\n{np.array2string(sv, precision=4)}",
                        language="python",
                    )
                    probs_sv = np.abs(sv) ** 2
                    st.caption(
                        f"|amp|² = {np.array2string(probs_sv, precision=4)}"
                    )
                else:
                    st.info("Measurement collapses the statevector — see Counts.")

    with st.expander("Hardware credentials"):
        st.markdown("""
Set `QISKIT_IBM_TOKEN` as an environment variable, or configure 
`~/.qiskit/qiskit-ibm.json`. Install `qiskit-ibm-runtime`. 
Never paste tokens in this editor. Current status: **Simulator only**.
""")

    # ── Section 05 — State Evolution ─────────────────────────────────────────
    section("05", "Quantum State Evolution — Gate-by-Gate",
            "Step through a circuit one gate at a time. "
            "Bloch spheres update from the real PennyLane statevector.")

    if "evo_qspec" not in st.session_state:
        eq = CircuitSpec(n_qubits=2)
        eq.add_gate("H",    [0])
        eq.add_gate("CNOT", [0, 1])
        st.session_state.evo_qspec = eq
    if "evo_step" not in st.session_state:
        st.session_state.evo_step = 0

    eq     = st.session_state.evo_qspec
    eq.n_qubits = st.slider(
        "Evolution qubits", 2, 4, int(eq.n_qubits), key="evo_nq"
    )
    n_steps = len(eq.ops) + 1
    step    = int(np.clip(st.session_state.evo_step, 0, n_steps - 1))

    ep1, ep2, ep3, ep4 = st.columns(4)
    if ep1.button("◀ Prev",  use_container_width=True, key="evo_prev"):
        st.session_state.evo_step = max(0, step - 1); st.rerun()
    if ep2.button("Next ▶",  use_container_width=True, key="evo_next"):
        st.session_state.evo_step = min(n_steps - 1, step + 1); st.rerun()
    if ep3.button("↺ Reset", use_container_width=True, key="evo_reset"):
        st.session_state.evo_step = 0; st.rerun()
    ep4.progress(step / max(n_steps - 1, 1))

    labels = ["Step 0 — |00…0⟩"] + [
        f'Step {i+1} — {op["gate"]} {op["wires"]}'
        for i, op in enumerate(eq.ops)
    ]
    st.markdown(
        f'<div style="font-family:\'Space Mono\',monospace;font-size:0.68rem;'
        f'color:#38BDF8;text-align:center;margin:0.3rem 0">'
        f'{labels[min(step, len(labels)-1)]}</div>',
        unsafe_allow_html=True,
    )

    try:
        prefix = eq.ops[:step]
        if not prefix:
            sv   = np.zeros(1 << eq.n_qubits, dtype=complex)
            sv[0] = 1
            bloch_v = compute_bloch_vectors(sv, eq.n_qubits)
        else:
            r      = PennylaneBackend(eq.n_qubits).simulate_circuit_ops(prefix, eq.n_qubits)
            sv     = r.statevector
            bloch_v = r.bloch_vectors

        ecA, ecB = st.columns([1.2, 0.9])
        with ecA:
            eb1, eb2 = st.columns(2)
            for i in range(min(4, eq.n_qubits)):
                bth = np.arccos(np.clip(bloch_v[i][2], -1, 1))
                bph = np.arctan2(bloch_v[i][1], bloch_v[i][0])
                fig_evo, _, _, _, _, _ = bloch_fig_dark(bth, bph, qubit_idx=i)
                with (eb1 if i % 2 == 0 else eb2):
                    st.plotly_chart(fig_evo, use_container_width=True,
                                    config={"displayModeBar": False},
                                    key=f"evo_{step}_{i}")
        with ecB:
            probs = np.abs(sv) ** 2
            probs /= probs.sum() if probs.sum() > 0 else 1
            st.markdown("""
<div style="font-family:'Space Mono',monospace;font-size:0.58rem;letter-spacing:1.5px;
            color:#6B84A8;margin-bottom:0.3rem">STATE PROBABILITIES</div>""",
                unsafe_allow_html=True)
            for idx in np.argsort(probs)[::-1][:6]:
                if probs[idx] < 0.001:
                    continue
                bits = format(idx, f"0{eq.n_qubits}b")[::-1]
                st.markdown(
                    f'<div style="background:#111D32;border:1px solid #1E2E4A;'
                    f'border-radius:4px;padding:0.3rem 0.6rem;margin:2px 0;'
                    f'display:flex;justify-content:space-between;'
                    f'font-family:\'Space Mono\',monospace;font-size:0.68rem;color:#C8D8F0">'
                    f'<span>|{bits}⟩</span>'
                    f'<span style="color:#38BDF8">{probs[idx]*100:.2f}%</span></div>',
                    unsafe_allow_html=True,
                )
            purity_vals = [purity_from_bloch(bloch_v[i]) for i in range(eq.n_qubits)]
            st.markdown(
                f'<div style="margin-top:0.5rem;font-family:\'Space Mono\',monospace;'
                f'font-size:0.6rem;color:#6B84A8">'
                f'Purity per qubit: '
                f'{", ".join(f"q{i}:{purity_vals[i]:.2f}" for i in range(eq.n_qubits))}'
                f'</div>',
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.error(f"Evolution error: {e}")

    # ── Section 06 — Noise Demo ───────────────────────────────────────────────
    section("06", "Quantum Noise — Depolarizing Channel Demo",
            "Bell circuit (H + CNOT) under depolarizing noise. "
            "Ideal: only |00⟩ and |11⟩. Noise adds off-diagonal counts.")

    level = st.select_slider(
        "Noise level",
        options=["None", "Low (0.5%)", "Medium (2%)", "High (8%)"],
        value="None", key="noise_level",
    )
    noise_rates = {
        "None": 0.0, "Low (0.5%)": 0.005,
        "Medium (2%)": 0.02, "High (8%)": 0.08,
    }
    p = noise_rates[level]

    if st.button("Run Bell Noise Experiment", key="run_noise",
                 use_container_width=False):
        with st.spinner("Running AerSimulator…"):
            try:
                from qiskit import QuantumCircuit
                from qiskit_aer import AerSimulator
                qc_bell = QuantumCircuit(2)
                qc_bell.h(0)
                qc_bell.cx(0, 1)
                qc_bell.measure_all()
                if p == 0:
                    counts = AerSimulator().run(qc_bell, shots=2048).result().get_counts(qc_bell)
                else:
                    from qiskit_aer.noise import NoiseModel, depolarizing_error
                    nm = NoiseModel()
                    nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), ["h"])
                    nm.add_all_qubit_quantum_error(depolarizing_error(p * 1.8, 2), ["cx"])
                    counts = (
                        AerSimulator(noise_model=nm)
                        .run(qc_bell, shots=2048)
                        .result()
                        .get_counts(qc_bell)
                    )
                st.session_state.noise_counts = counts
                st.session_state.noise_level_used = level
            except Exception as e:
                st.error(f"Noise experiment error: {e}")

    if "noise_counts" in st.session_state:
        counts = st.session_state.noise_counts
        ideal  = {"00", "11"}
        colors = [
            "#38BDF8" if k.replace(" ", "") in ideal else "#E06060"
            for k in counts
        ]
        fig_n = go.Figure(go.Bar(
            x=list(counts.keys()),
            y=list(counts.values()),
            marker_color=colors,
            text=list(counts.values()),
            textposition="auto",
            hovertemplate="%{x}: %{y} shots<extra></extra>",
        ))
        apply_layout(fig_n, height=280,
                     title=f"Bell Counts — {st.session_state.noise_level_used} "
                           "(blue = ideal, red = noise-induced)")
        fig_n.update_layout(
            yaxis=dict(title="Counts", gridcolor="rgba(30,46,74,0.6)"),
        )
        st.plotly_chart(fig_n, use_container_width=True, config={"displayModeBar": False})
        ideal_total = sum(v for k, v in counts.items() if k.replace(" ", "") in ideal)
        total = sum(counts.values())
        st.markdown(
            f'<div style="font-family:\'Space Mono\',monospace;font-size:0.7rem;'
            f'color:#8A9FBF;margin-top:0.3rem">'
            f'Ideal-state fraction: {ideal_total/total*100:.1f}% · '
            f'Noise fraction: {(1-ideal_total/total)*100:.1f}%</div>',
            unsafe_allow_html=True,
        )

    # ── Section 07 — Q-ORBIT Connection ──────────────────────────────────────
    section("07", "From Qubits to Orbits — Q-ORBIT Connection",
            "How the quantum concepts demonstrated here connect to the actual classification pipeline.")

    st.markdown("""
<div style="background:#0D1628;border:1px solid rgba(56,189,248,0.2);border-radius:12px;
            padding:1.2rem;margin-bottom:1rem;overflow-x:auto">
  <div style="display:flex;align-items:center;flex-wrap:wrap;gap:0;min-width:700px">
    <div style="padding:0.5rem 0.85rem;border-radius:6px;background:#111D32;
                border:1px solid #1E2E4A;font-family:'Space Mono',monospace;
                font-size:0.65rem;color:#C8D8F0;text-align:center;white-space:nowrap">
      Light Curve<br/><span style="color:#6B84A8">256 obs</span>
    </div>
    <span style="color:#2A4060;padding:0 0.4rem">→</span>
    <div style="padding:0.5rem 0.85rem;border-radius:6px;background:rgba(56,189,248,0.06);
                border:1px solid rgba(56,189,248,0.2);font-family:'Space Mono',monospace;
                font-size:0.65rem;color:#38BDF8;text-align:center;white-space:nowrap">
      21 features
    </div>
    <span style="color:#2A4060;padding:0 0.4rem">→</span>
    <div style="padding:0.5rem 0.85rem;border-radius:6px;background:rgba(166,159,214,0.08);
                border:1px solid rgba(166,159,214,0.25);font-family:'Space Mono',monospace;
                font-size:0.65rem;color:#A69FD6;text-align:center;white-space:nowrap">
      Quantum Layer<br/><span style="color:#6B84A8">8Q RY embedding</span>
    </div>
    <span style="color:#2A4060;padding:0 0.4rem">→</span>
    <div style="padding:0.5rem 0.85rem;border-radius:6px;background:rgba(166,159,214,0.08);
                border:1px solid rgba(166,159,214,0.2);font-family:'Space Mono',monospace;
                font-size:0.65rem;color:#A69FD6;text-align:center;white-space:nowrap">
      Rot+CNOT ansatz<br/><span style="color:#6B84A8">entanglement</span>
    </div>
    <span style="color:#2A4060;padding:0 0.4rem">→</span>
    <div style="padding:0.5rem 0.85rem;border-radius:6px;background:rgba(245,166,35,0.06);
                border:1px solid rgba(245,166,35,0.2);font-family:'Space Mono',monospace;
                font-size:0.65rem;color:#F5A623;text-align:center;white-space:nowrap">
      ⟨Z⟩ × 8<br/><span style="color:#6B84A8">Pauli-Z</span>
    </div>
    <span style="color:#2A4060;padding:0 0.4rem">→</span>
    <div style="padding:0.5rem 0.85rem;border-radius:6px;background:rgba(56,189,248,0.08);
                border:1px solid rgba(56,189,248,0.25);font-family:'Space Mono',monospace;
                font-size:0.65rem;color:#38BDF8;text-align:center;white-space:nowrap">
      Classical head<br/><span style="color:#6B84A8">→ 5 classes</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # Live pipeline demo
    st.markdown("""
<div style="font-family:'Space Mono',monospace;font-size:0.6rem;letter-spacing:2px;
            color:#38BDF8;margin-bottom:0.4rem">LIVE PIPELINE DEMO</div>""",
        unsafe_allow_html=True)

    dc1, dc2, dc3 = st.columns([1, 1, 1])
    with dc1:
        cls_demo = st.selectbox(
            "Object class",
            [0, 1, 2, 3, 4],
            format_func=lambda x: CLASS_NAMES[x],
            key="cls_demo",
        )
    with dc2:
        noise_demo = st.selectbox(
            "Noise", [0.0, 0.02, 0.05], key="noise_demo"
        )
    with dc3:
        st.write("")
        run_demo = st.button("▶ Run Real Pipeline", type="primary",
                             key="run_demo", use_container_width=True)

    if run_demo:
        with st.spinner("Generating curve → 21 features → 8Q → ⟨Z⟩ → prediction…"):
            try:
                from src.simulator.lightcurve_generator import generate_single_light_curve
                from src.classical.feature_engineering import extract_all_features

                curve, _ = generate_single_light_curve(
                    int(cls_demo), noise_std=float(noise_demo), n_samples=256
                )
                feats = extract_all_features(curve, sampling_interval=720 / 255)
                feats = np.nan_to_num(feats, nan=0.0)

                # Plot curve
                fig_lc = go.Figure(go.Scatter(
                    x=np.linspace(0, 720, len(curve)).tolist(),
                    y=curve.tolist(),
                    mode="lines",
                    line=dict(color="#38BDF8", width=1.8),
                    fill="tozeroy",
                    fillcolor="rgba(56,189,248,0.07)",
                ))
                apply_layout(fig_lc, height=180,
                             title=f"Light Curve — {CLASS_NAMES[cls_demo]}")
                fig_lc.update_layout(
                    xaxis=dict(title="Time (s)", gridcolor="rgba(30,46,74,0.6)"),
                    yaxis=dict(title="Brightness", range=[0, 1.1],
                               gridcolor="rgba(30,46,74,0.6)"),
                )
                st.plotly_chart(fig_lc, use_container_width=True,
                                config={"displayModeBar": False})

                st.markdown(
                    f'<div style="font-family:\'Space Mono\',monospace;font-size:0.68rem;'
                    f'color:#A69FD6;margin:0.3rem 0">'
                    f'Features (first 6): {np.array2string(feats[:6], precision=3)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Try hybrid model
                try:
                    import torch
                    import joblib
                    from src.quantum.hybrid_model import HybridQuantumClassifier
                    mp = os.path.join(ROOT, "models", "hybrid_quantum_model.pt")
                    if os.path.exists(mp):
                        ckpt = torch.load(mp, weights_only=False, map_location="cpu")
                        m = HybridQuantumClassifier(
                            n_features=ckpt.get("n_features", 21),
                            n_qubits=ckpt.get("n_qubits", 8),
                            n_layers=ckpt.get("n_layers", 2),
                            n_classical_hidden=ckpt.get("n_classical_hidden", 16),
                            n_classes=5,
                        )
                        m.load_state_dict(ckpt["model_state_dict"])
                        m.eval()

                        ft = torch.tensor(feats[None, :], dtype=torch.float32)
                        with torch.no_grad():
                            q_feats = m.quantum_layer(ft).numpy()[0]
                            _, probs = m(ft)
                            probs = probs.numpy()[0]

                        st.markdown(
                            f'<div style="font-family:\'Space Mono\',monospace;'
                            f'font-size:0.68rem;color:#F5A623;margin:0.3rem 0">'
                            f'⟨Z⟩ × 8: {np.array2string(q_feats, precision=3)}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        pred = int(np.argmax(probs))
                        st.markdown(
                            f'<div style="background:rgba(56,189,248,0.07);'
                            f'border:1px solid rgba(56,189,248,0.25);border-radius:8px;'
                            f'padding:0.7rem;font-size:0.88rem;color:#C8D8F0">'
                            f'Prediction: <strong style="color:#38BDF8">'
                            f'{CLASS_NAMES[pred]}</strong> '
                            f'({probs[pred]*100:.1f}% confidence)</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info("Hybrid model not found. Run training first.")
                except Exception as e:
                    st.warning(f"Model inference skipped: {e}")

            except Exception as e:
                st.error(f"Pipeline error: {e}\n{traceback.format_exc(limit=2)}")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
<div style="margin-top:2rem;padding:1rem;background:#0D1628;border:1px solid #1E2E4A;
            border-radius:8px;font-size:0.75rem;color:#6B84A8;line-height:1.7">
  <strong style="color:#C8D8F0">Q-ORBIT Quantum Lab</strong> — 
  All visualisations use real PennyLane / Qiskit simulation. 
  Bloch vectors from actual statevectors. 
  No results are claimed from real quantum hardware unless 
  QISKIT_IBM_TOKEN is set and hardware is explicitly selected.<br/>
  Simulator: PennyLane default.qubit · Qiskit AerSimulator
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
else:
    main()
