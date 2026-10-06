# pages/4_Quantum.py — Q-ORBIT Quantum Architecture Page
# Shows real quantum circuit details from the trained models.
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from app.space_theme import css, stars_html, sidebar_logo, apply_layout, CLASS_COLORS, CLASS_NAMES

st.set_page_config(
    page_title="Q-ORBIT — Quantum",
    layout="wide", page_icon="⚛️",
    initial_sidebar_state="collapsed",
)

ROOT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
REPORTS = os.path.join(ROOT, "results", "reports")
MODELS  = os.path.join(ROOT, "models")

@st.cache_data
def load_fair():
    p = os.path.join(REPORTS, "fair_comparison.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}

@st.cache_data
def load_qr_meta():
    p = os.path.join(ROOT, "data", "quantum_ready", "README.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}

@st.cache_data
def load_pure_search():
    p = os.path.join(REPORTS, "pure_search.json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_ablations():
    p = os.path.join(REPORTS, "ablations.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def section(label, title, sub=""):
    sub_html = f'<p style="color:#6B84A8;font-size:0.85rem;margin:0 0 0.5rem 0">{sub}</p>' if sub else ""
    st.markdown(f"""
<div style="margin:2rem 0 0.8rem 0;padding-bottom:0.7rem;border-bottom:1px solid #1E2E4A">
  <div style="font-family:'Space Mono',monospace;font-size:0.6rem;letter-spacing:2.5px;
              color:#A69FD6;margin-bottom:0.3rem">{label}</div>
  <h2 style="font-size:1.2rem;font-weight:600;color:#F0F6FF;margin:0 0 0.15rem 0">{title}</h2>
  {sub_html}
</div>""", unsafe_allow_html=True)


def render_circuit_viz(n_qubits=8, n_layers=2):
    """Visual representation of the VQC circuit structure."""
    st.markdown(f"""
<div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:10px;
            padding:1rem 1.2rem;overflow-x:auto">
  <div style="font-family:'Space Mono',monospace;font-size:0.6rem;letter-spacing:2px;
              color:#A69FD6;margin-bottom:0.6rem">
    CIRCUIT DIAGRAM — {n_qubits} QUBITS · {n_layers} LAYERS
  </div>
  <div style="display:grid;grid-template-columns:auto 1fr;gap:0;align-items:center">""",
        unsafe_allow_html=True)

    for q in range(n_qubits):
        # qubit wire row
        wire_html = f"""
    <div style="font-family:'Space Mono',monospace;font-size:0.68rem;
                color:#6B84A8;padding:0 0.6rem 0 0;white-space:nowrap">
      |q{q}⟩
    </div>
    <div style="display:flex;align-items:center;gap:0;height:34px">
      <div style="flex:1;height:1px;background:#1E2E4A"></div>
      <div style="padding:0.3rem 0.5rem;border-radius:4px;background:#111D32;
                  border:1px solid rgba(56,189,248,0.2);
                  font-family:'Space Mono',monospace;font-size:0.6rem;
                  color:#38BDF8;white-space:nowrap;margin:0 2px">
        RY(θ{q})
      </div>
      <div style="flex:0.2;height:1px;background:#1E2E4A"></div>"""
        for L in range(n_layers):
            wire_html += f"""
      <div style="padding:0.3rem 0.5rem;border-radius:4px;background:#111D32;
                  border:1px solid rgba(166,159,214,0.25);
                  font-family:'Space Mono',monospace;font-size:0.6rem;
                  color:#A69FD6;white-space:nowrap;margin:0 2px">
        Rot
      </div>
      <div style="flex:0.2;height:1px;background:#1E2E4A"></div>"""
            if q < n_qubits - 1:
                wire_html += f"""
      <div style="padding:0.3rem 0.4rem;border-radius:4px;background:rgba(100,216,168,0.08);
                  border:1px solid rgba(100,216,168,0.2);
                  font-family:'Space Mono',monospace;font-size:0.6rem;
                  color:#64D8A8;white-space:nowrap;margin:0 2px">
        CNOT↓
      </div>
      <div style="flex:0.2;height:1px;background:#1E2E4A"></div>"""
        wire_html += f"""
      <div style="padding:0.3rem 0.5rem;border-radius:4px;background:rgba(245,166,35,0.08);
                  border:1px solid rgba(245,166,35,0.2);
                  font-family:'Space Mono',monospace;font-size:0.6rem;
                  color:#F5A623;white-space:nowrap;margin:0 2px">
        ⟨Z{q}⟩
      </div>
      <div style="flex:1;height:1px;background:#1E2E4A"></div>
    </div>"""
        st.markdown(wire_html, unsafe_allow_html=True)

    st.markdown("""
  </div>
  <div style="margin-top:0.7rem;display:flex;gap:0.6rem;flex-wrap:wrap">
    <span style="background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.2);
                 color:#38BDF8;border-radius:4px;font-family:'Space Mono',monospace;
                 font-size:0.6rem;padding:0.2rem 0.5rem">RY — angle embedding</span>
    <span style="background:rgba(166,159,214,0.08);border:1px solid rgba(166,159,214,0.2);
                 color:#A69FD6;border-radius:4px;font-family:'Space Mono',monospace;
                 font-size:0.6rem;padding:0.2rem 0.5rem">Rot(φ,θ,ω) — trainable 3-angle gate</span>
    <span style="background:rgba(100,216,168,0.08);border:1px solid rgba(100,216,168,0.2);
                 color:#64D8A8;border-radius:4px;font-family:'Space Mono',monospace;
                 font-size:0.6rem;padding:0.2rem 0.5rem">CNOT chain — entanglement</span>
    <span style="background:rgba(245,166,35,0.08);border:1px solid rgba(245,166,35,0.2);
                 color:#F5A623;border-radius:4px;font-family:'Space Mono',monospace;
                 font-size:0.6rem;padding:0.2rem 0.5rem">⟨Z⟩ — Pauli-Z measurement</span>
  </div>
</div>""", unsafe_allow_html=True)


def render_hilbert_viz(n_qubits=8):
    """Bloch-sphere projection showing 8 qubit states."""
    import math
    thetas = np.linspace(0.2, np.pi - 0.2, n_qubits)
    phis   = np.linspace(0, 2 * np.pi, n_qubits, endpoint=False)

    fig = go.Figure()
    # sphere wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        opacity=0.06,
        colorscale=[[0, "#0D1628"], [1, "#1E3060"]],
        showscale=False, hoverinfo="skip",
    ))
    # equator
    t = np.linspace(0, 2 * np.pi, 60)
    fig.add_trace(go.Scatter3d(
        x=np.cos(t), y=np.sin(t), z=np.zeros_like(t),
        mode="lines", line=dict(color="rgba(56,189,248,0.2)", width=1.5),
        hoverinfo="skip", showlegend=False,
    ))
    # qubit state vectors
    for i, (th, ph) in enumerate(zip(thetas, phis)):
        x_s = math.sin(th) * math.cos(ph)
        y_s = math.sin(th) * math.sin(ph)
        z_s = math.cos(th)
        color = CLASS_COLORS[i % len(CLASS_COLORS)]
        fig.add_trace(go.Scatter3d(
            x=[0, x_s], y=[0, y_s], z=[0, z_s],
            mode="lines+markers",
            line=dict(color=color, width=4),
            marker=dict(size=[0, 5], color=color),
            name=f"q{i}",
            hovertemplate=f"q{i}: θ={math.degrees(th):.1f}° φ={math.degrees(ph):.1f}°<extra></extra>",
        ))
    # axis labels
    fig.add_trace(go.Scatter3d(
        x=[0, 0, 1.3, -1.3], y=[0, 0, 0, 0], z=[1.3, -1.3, 0, 0],
        mode="text", text=["|0⟩", "|1⟩", "+X", "−X"],
        textfont=dict(color="#6B84A8", size=10),
        hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1.3, 1.3], showgrid=False,
                       showticklabels=False, title="",
                       backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(range=[-1.3, 1.3], showgrid=False,
                       showticklabels=False, title="",
                       backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(range=[-1.3, 1.3], showgrid=False,
                       showticklabels=False, title="",
                       backgroundcolor="rgba(0,0,0,0)"),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.5, y=1.3, z=1.0)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        height=380,
        title=dict(
            text="8 Qubits on Bloch Sphere (illustrative states)",
            font=dict(color="#C8D8F0", size=12), x=0.02,
        ),
        legend=dict(
            font=dict(color="#C8D8F0", size=9),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_param_counts():
    """Show parameter counts across qubit/layer combinations."""
    # Pure VQC: n_layers * n_qubits * 3 (Rot) + 1 (angle_scale) + n_qubits*n_classes + n_classes
    configs = []
    for nq in [4, 6, 8]:
        for nl in [1, 2, 3]:
            circuit_params = nl * nq * 3 + 1  # Rot weights + angle_scale
            readout_params = nq * 5 + 5        # Linear(nq→5)
            total = circuit_params + readout_params
            configs.append((f"{nq}q {nl}L", nq, nl, circuit_params, readout_params, total))

    fig = go.Figure()
    labels = [c[0] for c in configs]
    circuit_p = [c[3] for c in configs]
    readout_p = [c[4] for c in configs]

    fig.add_trace(go.Bar(
        name="Circuit (Rot weights)", x=labels, y=circuit_p,
        marker_color="#A69FD6", opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name="Readout (Linear 8→5)", x=labels, y=readout_p,
        marker_color="#38BDF8", opacity=0.7,
    ))
    apply_layout(fig, height=300, title="VQC Parameter Count by Configuration")
    fig.update_layout(
        barmode="stack",
        yaxis=dict(title="Parameters", gridcolor="rgba(30,46,74,0.6)"),
        legend=dict(orientation="h", y=1.02),
        annotations=[dict(
            x="8q 2L", y=configs[7][5] + 3,
            text="<b>Selected</b>", showarrow=True, arrowhead=2,
            arrowcolor="#38BDF8", font=dict(color="#38BDF8", size=10),
        )] if len(configs) > 7 else [],
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def main():
    st.markdown(css(), unsafe_allow_html=True)
    st.markdown(stars_html(seed=31), unsafe_allow_html=True)
    sidebar_logo(st)

    st.markdown("""
<div style="padding:1.6rem 0 1.2rem 0;border-bottom:1px solid #1E2E4A;margin-bottom:1.6rem">
  <div style="font-family:'Space Mono',monospace;font-size:0.62rem;letter-spacing:2.5px;
              color:#A69FD6;margin-bottom:0.4rem">QUANTUM MACHINE LEARNING</div>
  <h1 style="font-size:clamp(1.8rem,3.5vw,2.6rem);font-weight:700;color:#F0F6FF;
              margin:0 0 0.4rem 0">Quantum Architecture</h1>
  <p style="color:#6B84A8;font-size:0.92rem;max-width:70ch;margin:0 0 0.6rem 0">
    Variational Quantum Classifier (VQC) with 8 qubits, 2 layers, angle RY embedding 
    and Rot+CNOT ansatz. All representation power lives in the quantum circuit.
    Simulator: PennyLane <code>default.qubit</code>.
  </p>
  <span style="display:inline-flex;align-items:center;gap:0.4rem;
               background:rgba(100,216,168,0.07);border:1px solid rgba(100,216,168,0.2);
               color:#64D8A8;border-radius:999px;font-family:'Space Mono',monospace;
               font-size:0.62rem;letter-spacing:1.2px;padding:0.3rem 0.7rem">
    <span style="width:6px;height:6px;border-radius:50%;background:#64D8A8;
                 animation:qo-pulse 2s infinite;display:inline-block"></span>
    SIMULATOR MODE — PennyLane default.qubit — No real hardware required
  </span>
</div>""", unsafe_allow_html=True)

    fc      = load_fair()
    qr_meta = load_qr_meta()

    # ── Core specs ────────────────────────────────────────────────────────────
    section("ARCHITECTURE", "Pure Quantum VQC — Core Specification")

    c1, c2 = st.columns([1, 1.4])
    with c1:
        pure = fc.get("Pure_Quantum_VQC", {})
        acc  = pure.get("accuracy", 0)
        f1   = pure.get("f1_macro", 0)
        auc  = pure.get("roc_auc_ovr_macro", 0)

        st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;margin-bottom:0.8rem">
  <div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:8px;
              padding:0.8rem;text-align:center">
    <div style="font-size:1.6rem;font-weight:700;color:#A69FD6">8</div>
    <div style="font-family:'Space Mono',monospace;font-size:0.55rem;
                letter-spacing:1px;color:#6B84A8;margin-top:2px">QUBITS</div>
    <div style="font-size:0.7rem;color:#4A5F7A">256-D Hilbert space</div>
  </div>
  <div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:8px;
              padding:0.8rem;text-align:center">
    <div style="font-size:1.6rem;font-weight:700;color:#A69FD6">5</div>
    <div style="font-family:'Space Mono',monospace;font-size:0.55rem;
                letter-spacing:1px;color:#6B84A8;margin-top:2px">DEPTH</div>
    <div style="font-size:0.7rem;color:#4A5F7A">embed + 2×(Rot+CNOT)</div>
  </div>
  <div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:8px;
              padding:0.8rem;text-align:center">
    <div style="font-size:1.6rem;font-weight:700;color:#A69FD6">94</div>
    <div style="font-family:'Space Mono',monospace;font-size:0.55rem;
                letter-spacing:1px;color:#6B84A8;margin-top:2px">PARAMS</div>
    <div style="font-size:0.7rem;color:#4A5F7A">circuit + readout</div>
  </div>
  <div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:8px;
              padding:0.8rem;text-align:center">
    <div style="font-size:1.6rem;font-weight:700;color:#A69FD6">
      {acc*100:.1f}%
    </div>
    <div style="font-family:'Space Mono',monospace;font-size:0.55rem;
                letter-spacing:1px;color:#6B84A8;margin-top:2px">TEST ACC</div>
    <div style="font-size:0.7rem;color:#4A5F7A">F1 {f1:.3f}</div>
  </div>
</div>
<div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:8px;
            padding:0.9rem;font-size:0.82rem;color:#C8D8F0;line-height:1.8">
  <strong style="color:#A69FD6">Input:</strong> 8-D PCA-reduced features 
  (from 21 classical → z-score → PCA-8)<br/>
  <strong style="color:#A69FD6">Encoding:</strong> θ = tanh(z)·π → 
  AngleEmbedding(RY)<br/>
  <strong style="color:#A69FD6">Ansatz:</strong> Rot(φ,θ,ω) × 8 qubits × 2 layers 
  + chain CNOT entanglement<br/>
  <strong style="color:#A69FD6">Measurement:</strong> Pauli-Z expectation 
  ⟨Z⟩ per qubit → 8 values<br/>
  <strong style="color:#A69FD6">Output:</strong> Linear(8→5) → softmax → 5 classes<br/>
  <strong style="color:#A69FD6">Simulator:</strong> PennyLane default.qubit, 
  backprop differentiation<br/>
  <strong style="color:#A69FD6">PCA variance retained:</strong> 
  {sum(qr_meta.get("explained_variance",{}).get("explained_variance_ratio",[]))*100:.1f}%
</div>""", unsafe_allow_html=True)

    with c2:
        render_circuit_viz(n_qubits=8, n_layers=2)

    # ── Circuit diagram ───────────────────────────────────────────────────────
    section("CIRCUIT", "Circuit Layer Structure")
    render_circuit_viz(n_qubits=8, n_layers=2)

    # ── Encoding pipeline ─────────────────────────────────────────────────────
    section("ENCODING", "Feature → Qubit Encoding Pipeline",
            "21 classical features → dimensionality reduction → quantum angle embedding")

    ev = qr_meta.get("explained_variance", {}).get("explained_variance_ratio", [])
    total_var = sum(ev)

    st.markdown(f"""
<div style="display:flex;align-items:center;flex-wrap:wrap;gap:0;
            background:#0D1628;border:1px solid #1E2E4A;border-radius:10px;
            padding:0.9rem 1.1rem;overflow-x:auto;margin-bottom:1rem">
  <div style="padding:0.5rem 0.9rem;border-radius:6px;background:#111D32;
              border:1px solid #1E2E4A;font-family:'Space Mono',monospace;
              font-size:0.65rem;color:#C8D8F0;text-align:center;white-space:nowrap">
    21 features<br/><span style="color:#6B84A8">from light curve</span>
  </div>
  <span style="color:#2A4060;padding:0 0.5rem">→</span>
  <div style="padding:0.5rem 0.9rem;border-radius:6px;background:rgba(56,189,248,0.06);
              border:1px solid rgba(56,189,248,0.2);font-family:'Space Mono',monospace;
              font-size:0.65rem;color:#38BDF8;text-align:center;white-space:nowrap">
    z-score scaler<br/><span style="color:#6B84A8">train-fit only</span>
  </div>
  <span style="color:#2A4060;padding:0 0.5rem">→</span>
  <div style="padding:0.5rem 0.9rem;border-radius:6px;background:rgba(166,159,214,0.08);
              border:1px solid rgba(166,159,214,0.2);font-family:'Space Mono',monospace;
              font-size:0.65rem;color:#A69FD6;text-align:center;white-space:nowrap">
    PCA → 8 dims<br/><span style="color:#6B84A8">{total_var*100:.1f}% variance</span>
  </div>
  <span style="color:#2A4060;padding:0 0.5rem">→</span>
  <div style="padding:0.5rem 0.9rem;border-radius:6px;background:rgba(166,159,214,0.08);
              border:1px solid rgba(166,159,214,0.2);font-family:'Space Mono',monospace;
              font-size:0.65rem;color:#A69FD6;text-align:center;white-space:nowrap">
    θ = tanh(z)·π<br/><span style="color:#6B84A8">scale to [−π, π]</span>
  </div>
  <span style="color:#2A4060;padding:0 0.5rem">→</span>
  <div style="padding:0.5rem 0.9rem;border-radius:6px;
              background:rgba(100,216,168,0.07);border:1px solid rgba(100,216,168,0.2);
              font-family:'Space Mono',monospace;font-size:0.65rem;
              color:#64D8A8;text-align:center;white-space:nowrap">
    RY(θᵢ)|0⟩<br/><span style="color:#6B84A8">angle embedding</span>
  </div>
  <span style="color:#2A4060;padding:0 0.5rem">→</span>
  <div style="padding:0.5rem 0.9rem;border-radius:6px;background:rgba(245,166,35,0.07);
              border:1px solid rgba(245,166,35,0.2);font-family:'Space Mono',monospace;
              font-size:0.65rem;color:#F5A623;text-align:center;white-space:nowrap">
    ⟨Z⟩ × 8<br/><span style="color:#6B84A8">Pauli-Z measurement</span>
  </div>
  <span style="color:#2A4060;padding:0 0.5rem">→</span>
  <div style="padding:0.5rem 0.9rem;border-radius:6px;background:#111D32;
              border:1px solid #1E2E4A;font-family:'Space Mono',monospace;
              font-size:0.65rem;color:#C8D8F0;text-align:center;white-space:nowrap">
    Linear(8→5)<br/><span style="color:#6B84A8">class logits</span>
  </div>
</div>""", unsafe_allow_html=True)

    # PCA variance bar
    if ev:
        col_ev, col_b = st.columns([1.3, 1])
        with col_ev:
            cumev = np.cumsum(ev)
            fig_ev = go.Figure()
            fig_ev.add_trace(go.Bar(
                x=[f"PC{i+1}" for i in range(len(ev))],
                y=ev, name="Individual",
                marker_color="#A69FD6", opacity=0.85,
            ))
            fig_ev.add_trace(go.Scatter(
                x=[f"PC{i+1}" for i in range(len(ev))],
                y=cumev.tolist(), name="Cumulative",
                line=dict(color="#38BDF8", width=2),
                mode="lines+markers", yaxis="y2",
            ))
            apply_layout(fig_ev, height=260, title="PCA-8 Explained Variance")
            fig_ev.update_layout(
                yaxis=dict(title="Individual", gridcolor="rgba(30,46,74,0.6)"),
                yaxis2=dict(title="Cumulative", range=[0, 1.05],
                            overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h", y=1.02),
            )
            st.plotly_chart(fig_ev, use_container_width=True, config={"displayModeBar": False})

        with col_b:
            render_hilbert_viz(n_qubits=8)

    # ── Ablation results ──────────────────────────────────────────────────────
    section("ABLATION", "Qubit / Layer Configuration Search",
            "Grid search over 4/6/8 qubits × 1/2/3 layers on val split. "
            "8q 2L selected as best (val F1 0.4807).")

    render_param_counts()

    # Show pure search results if available
    pure_search = load_pure_search()
    if pure_search:
        st.markdown("""
<div style="font-family:'Space Mono',monospace;font-size:0.6rem;letter-spacing:2px;
            color:#6B84A8;margin-bottom:0.4rem">SEARCH RESULTS</div>""",
            unsafe_allow_html=True)
        rows = []
        if isinstance(pure_search, list):
            for item in pure_search[:15]:
                cfg = item.get("config", item)
                vm = item.get("val_metrics", {})
                tm = item.get("test_metrics", {})
                rows.append({
                    "Config": f"{cfg.get('n_qubits','?')}q {cfg.get('n_layers','?')}L",
                    "Val Acc": f"{vm.get('accuracy',0)*100:.2f}%",
                    "Val F1":  f"{vm.get('f1_macro',0):.4f}",
                    "Test Acc": f"{tm.get('accuracy',0)*100:.2f}%" if tm else "—",
                    "Params":  str(cfg.get("params", "—")),
                })
        if rows:
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── VQC results detail ────────────────────────────────────────────────────
    section("RESULTS", "Pure VQC — Test Set Results",
            "Same frozen 1,500-sample test set, seed 42.")

    if not fc:
        st.info("No results found. Run experiments first.")
        return

    pure = fc.get("Pure_Quantum_VQC", {})
    col_a, col_b = st.columns(2)

    with col_a:
        per = pure.get("per_class", {})
        fig_bar = go.Figure()
        for metric_name, key in [("F1",        "f1"),
                                   ("Precision", "precision"),
                                   ("Recall",    "recall")]:
            vals = [per.get(str(i), {}).get(key, 0) for i in range(5)]
            fig_bar.add_trace(go.Bar(name=metric_name, x=CLASS_NAMES, y=vals))
        apply_layout(fig_bar, height=320, title="Pure VQC — Per-Class Metrics")
        fig_bar.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 1.05], gridcolor="rgba(30,46,74,0.6)"),
            xaxis=dict(tickangle=-25, gridcolor="rgba(30,46,74,0.6)"),
            legend=dict(orientation="h", y=1.02),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        cm = pure.get("confusion_matrix")
        if cm:
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=[cn[:10] for cn in CLASS_NAMES],
                y=[cn[:10] for cn in CLASS_NAMES],
                colorscale=[[0, "#060B18"], [0.4, "#2A1060"], [1, "#A69FD6"]],
                text=cm, texttemplate="%{text}",
                textfont=dict(size=10, color="#F0F6FF"),
                showscale=False,
                hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
            ))
            apply_layout(fig_cm, height=320, title="Pure VQC — Confusion Matrix (test)")
            fig_cm.update_layout(
                xaxis=dict(title="Predicted", tickangle=-25, tickfont=dict(size=9),
                           gridcolor="rgba(0,0,0,0)"),
                yaxis=dict(title="True", autorange="reversed", tickfont=dict(size=9),
                           gridcolor="rgba(0,0,0,0)"),
                margin=dict(l=80, r=10, t=40, b=60),
            )
            st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

    # Summary note
    acc  = pure.get("accuracy", 0)
    f1   = pure.get("f1_macro", 0)
    st.markdown(f"""
<div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:8px;
            padding:0.9rem 1rem;font-size:0.83rem;color:#8A9FBF;line-height:1.65">
  <strong style="color:#C8D8F0">Interpretation</strong><br/>
  Pure VQC achieves {acc*100:.1f}% accuracy (F1 {f1:.3f}). This is below the classical 
  CNN (85.2%) — expected given that the VQC has only 94 trainable parameters vs ~280k 
  for the CNN, and the 8-D quantum representation carries less raw information than 
  the 256-point CNN input.<br/><br/>
  The key question is whether a <strong>hybrid</strong> combination outperforms either 
  approach alone. See the Hybrid page for results.
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
else:
    main()
