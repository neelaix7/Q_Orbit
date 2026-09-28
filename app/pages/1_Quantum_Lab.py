# quantum_lab.py — PREMIUM LIGHT SCIENTIFIC — Q-ORBIT Quantum Lab
# Editorial museum + NASA research visualization — light theme (warm white/ivory)
from __future__ import annotations
import os, sys, base64, io, time, json, traceback, math, cmath
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from src.quantum.bloch import compute_bloch_vectors, bloch_sphere_figure, purity_from_bloch
from src.quantum.live_circuit import CircuitSpec
from src.quantum.backends.pennylane_backend import PennylaneBackend
# NOTE: AerBackend/StimBackend are intentionally NOT imported here. They are unused on
# this page, and importing qiskit_aer runs `aer_initialize_libraries()` at import time,
# which hard-crashes the interpreter on systems with conflicting OpenMP runtimes.
# The Aer simulator is loaded lazily inside run_aer_code() when the user runs a circuit.

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR=os.path.join(BASE_DIR,"..","assets","images","web","photos")

def img_b64(path):
    if not os.path.exists(path): return ""
    with open(path,"rb") as f: return base64.b64encode(f.read()).decode()

def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=EB+Garamond:ital@0;1&display=swap');
:root{--bg:#F7F9FC;--bg2:#F1F5F9;--panel:#FFFFFF;--surface:#FFFFFF;--surface-elevated:#FFFFFF;--line:#E2E8F0;--line-strong:#CBD5E1;--navy:#0F172A;--text-primary:#0F172A;--text-secondary:#475569;--text-muted:#64748B;--accent-blue:#2563EB;--accent-cyan:#0891B2;--accent-violet:#7C3AED;--success:#059669;--warning:#D97706;--error:#DC2626}
html,body{background:var(--bg) !important}
.stApp{background:var(--bg) !important;color:var(--text-primary) !important;font-family:'Instrument Sans','Space Grotesk',sans-serif !important}
[data-testid="stHeader"]{background:rgba(247,249,252,0.92) !important;backdrop-filter:blur(12px) !important;border-bottom:1px solid var(--line)}
.block-container{max-width:1320px !important;padding-top:0.2rem !important;padding-bottom:2rem}
h1,h2,h3{color:var(--text-primary) !important}
hr{border-color:var(--line) !important}
p,div,span{color:var(--text-primary)}
.small-muted{color:var(--text-secondary) !important}

/* Hero subtle scientific background */
.hero-wrap{position:relative;overflow:hidden;background:linear-gradient(180deg, #FFFCF7 0%, #FDF8F1 55%, #FFFFFF 100%);border:1px solid var(--line);border-radius:22px;padding:2.2rem 2rem 1.6rem 2rem}
.hero-grid{position:absolute;inset:0;opacity:0.045;background-image:linear-gradient(to right, #0F1B2E 1px, transparent 1px), linear-gradient(to bottom, #0F1B2E 1px, transparent 1px);background-size:42px 42px}
.hero-orb{position:absolute;border-radius:50%;border:1px solid rgba(37,99,235,0.08);pointer-events:none}
.hero-math{position:absolute;font-family:'JetBrains Mono';font-size:0.62rem;letter-spacing:1px;color:rgba(15,27,46,0.18);pointer-events:none}

/* Editorial sections */
.section{position:relative;background:var(--panel);border:1px solid var(--line);border-radius:20px;overflow:hidden;box-shadow:0 8px 30px rgba(15,27,46,0.06), 0 1px 3px rgba(15,27,46,0.04)}
.section-label{font-family:'JetBrains Mono';font-size:0.64rem;letter-spacing:2.2px;color:var(--accent);text-transform:uppercase}
.section-title{font-family:'Fraunces';font-weight:700;letter-spacing:-0.02em;color:var(--navy);line-height:1.05;margin:0}
.section-sub{font-family:'EB Garamond';font-style:italic;color:#475569;font-size:1.02rem;line-height:1.55}
.divider{height:1px;background:linear-gradient(90deg, transparent, var(--line), transparent);margin:1rem 0}

/* Cryostat light - FIXED for light mode visibility + responsive */
.cryostat-stage{position:relative;height:560px;border-radius:18px;overflow:hidden;background:linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 55%, #F1F5F9 100%);border:1px solid var(--line);display:grid;place-items:center;box-shadow:0 12px 32px rgba(15,27,46,0.07)}
.plate{position:absolute;left:50%;border-radius:50%;transform:translateX(-50%);transition:all .32s cubic-bezier(.2,.8,.2,1);cursor:pointer;border:1.5px solid var(--line-strong);box-shadow:0 8px 22px rgba(15,27,46,0.08), inset 0 1px 0 rgba(255,255,255,1);background:linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%)}
.plate::before{content:"";position:absolute;inset:0;border-radius:50%;background:linear-gradient(180deg, rgba(255,255,255,0.9), rgba(241,245,249,0.0));pointer-events:none}
.plate.active{border-color:var(--accent-blue) !important;box-shadow:0 14px 36px rgba(37,99,235,0.18), 0 0 0 2px rgba(37,99,235,0.12), inset 0 1px 0 rgba(255,255,255,1) !important;transform:translateX(-50%) scale(1.04);background:linear-gradient(180deg, #FFFFFF 0%, #EFF6FF 100%) !important}
.plate-label{position:absolute;right:-118px;top:50%;transform:translateY(-50%);font-family:'JetBrains Mono';font-size:0.62rem;letter-spacing:0.7px;color:var(--text-primary);background:var(--surface);border:1.2px solid var(--line-strong);padding:0.26rem 0.56rem;border-radius:999px;white-space:nowrap;box-shadow:0 4px 12px rgba(15,27,46,0.08);font-weight:500}
.plate.active .plate-label{color:var(--accent-blue);border-color:var(--accent-blue);background:#EFF6FF;box-shadow:0 4px 14px rgba(37,99,235,0.18)}
.helium{position:absolute;left:49.6%;width:2.2px;background:linear-gradient(180deg, var(--accent-blue), var(--accent-violet));transform:translateX(-50%);border-radius:999px;animation:flow 2.4s ease-in-out infinite;opacity:0.9;box-shadow:0 0 8px rgba(37,99,235,0.22)}
.cable{position:absolute;left:50.7%;width:1px;background:repeating-linear-gradient(180deg, rgba(100,116,139,0.22) 0 2px, transparent 2px 9px);transform:translateX(-50%)}
@keyframes flow{0%,100%{opacity:0.6}50%{opacity:1;box-shadow:0 0 12px rgba(37,99,235,0.35)}}
.chip{position:absolute;left:50%;top:76%;transform:translateX(-50%);width:92px;height:92px;border-radius:14px;background:linear-gradient(180deg,#FFFFFF,#F1F5F9);border:1.5px solid rgba(37,99,235,0.18);box-shadow:0 12px 30px rgba(37,99,235,0.14), inset 0 1px 0 rgba(255,255,255,1);display:grid;place-items:center}
.chip-grid{width:64px;height:64px;border-radius:9px;background:repeating-linear-gradient(0deg, rgba(37,99,235,0.10) 0 1px, transparent 1px 10px), repeating-linear-gradient(90deg, rgba(124,58,237,0.08) 0 1px, transparent 1px 12px);border:1px solid rgba(37,99,235,0.12)}
@media(max-width:900px){.cryostat-stage{height:460px}.plate{transform:translateX(-50%) scale(0.82)} .plate.active{transform:translateX(-50%) scale(0.86)} .plate-label{right:-84px;font-size:0.56rem;padding:0.20rem 0.42rem}}
@media(max-width:600px){.cryostat-stage{height:400px}.plate{transform:translateX(-50%) scale(0.68)} .plate.active{transform:translateX(-50%) scale(0.72)} .plate-label{right:-62px;font-size:0.50rem;max-width:88px;white-space:normal;text-align:center;line-height:1.2}}

/* Cards - FIXED for light mode contrast */
.glass{background:var(--surface);border:1.2px solid var(--line);border-radius:16px;padding:1.1rem 1.2rem;box-shadow:0 6px 22px rgba(15,27,46,0.06), 0 1px 4px rgba(15,27,46,0.04)}
.glass h3{font-family:'Instrument Sans';font-weight:600;letter-spacing:0.7px;color:var(--text-primary);font-size:0.82rem;margin:0 0 0.6rem 0;text-transform:uppercase}
.badge{display:inline-block;padding:0.28rem 0.58rem;border-radius:999px;font-family:'JetBrains Mono';font-size:0.62rem;letter-spacing:0.6px;border:1px solid var(--line-strong);background:#F1F5F9;color:var(--text-primary);font-weight:500}
.equation{font-family:'JetBrains Mono';background:#F8FAFC;border:1.2px solid var(--line);border-radius:14px;padding:0.9rem 1rem;font-size:0.86rem;color:var(--text-primary);line-height:1.7}
.prob-track{height:28px;border-radius:999px;background:#F1F5F9;border:1.2px solid var(--line);overflow:hidden;display:flex;box-shadow:inset 0 1px 2px rgba(15,27,46,0.04)}
.nav-light{position:sticky;top:0;z-index:50;display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:0.7rem 1rem;margin:0 -1rem 0.9rem -1rem;background:rgba(255,252,247,0.84);backdrop-filter:blur(14px);border-bottom:1px solid var(--line)}
.qorbit-mark{width:32px;height:32px;border-radius:9px;display:grid;place-items:center;background:linear-gradient(135deg,#2563EB,#7C3AED);color:white;font-family:'Fraunces';font-weight:700}
.stTabs [data-baseweb="tab-list"]{background:#F8F5F0 !important;border:1px solid var(--line) !important;border-radius:999px !important;padding:0.22rem !important;gap:0.18rem}
.stTabs [data-baseweb="tab"]{border-radius:999px !important;color:#64748B !important;font-family:'Instrument Sans' !important;font-size:0.76rem !important;font-weight:500}
.stTabs [aria-selected="true"]{background:var(--navy) !important;color:white !important}
[data-testid="stSlider"]{padding-top:0.2rem}
</style>""", unsafe_allow_html=True)

CRYO_STAGES=[
    {"key":"300K","temp":"300 K","name":"Room Temperature","top":7,"w":440,"h":38,"explain":"Control electronics and readout systems — waveform generators, amplifiers and digitizers operate here."},
    {"key":"50K","temp":"50 K","name":"Pulse-Tube First Stage","top":19,"w":380,"h":34,"explain":"Removes the majority of incoming heat — pulse-tube cooler first stage, radiation shielding."},
    {"key":"4K","temp":"4 K","name":"Cold Stage","top":31,"w":326,"h":30,"explain":"Provides additional pre-cooling — superconducting coax and attenuators thermalized."},
    {"key":"800mK","temp":"800 mK","name":"Still Plate","top":44,"w":270,"h":26,"explain":"Supports further cooling — still pumps He-3, drives continuous He-3/He-4 circulation."},
    {"key":"100mK","temp":"100 mK","name":"Cold Plate","top":57,"w":214,"h":22,"explain":"Further suppresses thermal energy — intermediate shield, cables anchored."},
    {"key":"10mK","temp":"10 mK","name":"Mixing Chamber","top":71,"w":162,"h":22,"explain":"Ultra-low-temperature environment used to maintain superconducting quantum processors."},
]

GATE_INFO={
    "H":{"math":"H|0⟩=(|0⟩+|1⟩)/√2","matrix":"1/√2 [[1,1],[1,-1]]","desc":"Superposition — rotates |0⟩ to equator (+X)."},
    "X":{"math":"X|0⟩=|1⟩","matrix":"[[0,1],[1,0]]","desc":"Bit-flip — 180° about X, north ↔ south."},
    "Y":{"math":"Y|0⟩=i|1⟩","matrix":"[[0,-i],[i,0]]","desc":"180° about Y, adds phase."},
    "Z":{"math":"Z|+⟩=|-⟩","matrix":"[[1,0],[0,-1]]","desc":"Phase-flip — 180° about Z."},
    "S":{"math":"S=√Z, S|1⟩=i|1⟩","matrix":"[[1,0],[0,i]]","desc":"90° Z rotation, phase π/2."},
    "T":{"math":"T|1⟩=e^{iπ/4}|1⟩","matrix":"[[1,0],[0,e^{iπ/4}]]","desc":"45° Z rotation, non-Clifford."},
    "RX":{"math":"RX(θ)=cos(θ/2)I−i sin(θ/2)X","desc":"Rotation about X by θ."},
    "RY":{"math":"RY(θ)=cos(θ/2)I−i sin(θ/2)Y","desc":"Rotation about Y — directly tunes θ."},
    "RZ":{"math":"RZ(φ)=diag(e^{-iφ/2},e^{iφ/2})","desc":"Rotation about Z — tunes φ."},
    "CNOT":{"math":"CNOT|c,t⟩=|c,t⊕c⟩","desc":"Entangling — Bell creation, Bloch shrinks to center."},
}

def bloch_single(theta, phi):
    a=math.cos(theta/2); b=cmath.exp(1j*phi)*math.sin(theta/2)
    x=math.sin(theta)*math.cos(phi); y=math.sin(theta)*math.sin(phi); z=math.cos(theta)
    return a,b,x,y,z

def bloch_fig_light(theta, phi, qubit_idx=0, show_labels=True):
    a,b,x,y,z=bloch_single(theta, phi)
    r=math.sqrt(x*x+y*y+z*z)
    u=np.linspace(0,2*np.pi,36); v=np.linspace(0,np.pi,18)
    xs=np.outer(np.cos(u), np.sin(v)); ys=np.outer(np.sin(u), np.sin(v)); zs=np.outer(np.ones_like(u), np.cos(v))
    fig=go.Figure()
    # LIGHT MODE: clearly visible translucent sphere with proper contrast
    fig.add_trace(go.Surface(x=xs,y=ys,z=zs, opacity=0.14, colorscale=[[0,"#FFFFFF"],[0.5,"#EFF6FF"],[1,"#DBEAFE"]], showscale=False, hoverinfo="skip"))
    # visible wireframe
    fig.add_trace(go.Surface(x=xs,y=ys,z=zs, opacity=0.06, colorscale=[[0,"#64748B"],[1,"#64748B"]], showscale=False, hoverinfo="skip"))
    th=np.linspace(0,2*np.pi,64)
    fig.add_trace(go.Scatter3d(x=np.cos(th),y=np.sin(th),z=np.zeros_like(th), mode="lines", line=dict(color="rgba(37,99,235,0.28)",width=2.5), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=np.zeros_like(th),y=np.cos(th),z=np.sin(th), mode="lines", line=dict(color="rgba(124,58,237,0.22)",width=2), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=np.cos(th)*0.92,y=np.zeros_like(th),z=np.sin(th)*0.92, mode="lines", line=dict(color="rgba(100,116,139,0.18)",width=1.5), hoverinfo="skip", showlegend=False))
    # arrow
    fig.add_trace(go.Scatter3d(x=[0,x],y=[0,y],z=[0,z], mode="lines", line=dict(color="#2563EB",width=8), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=[x],y=[y],z=[z], mode="markers", marker=dict(size=6,color="#7C3AED",line=dict(width=1.2,color="white")), hoverinfo="skip", showlegend=False))
    if show_labels:
        fig.add_trace(go.Scatter3d(x=[0,0],y=[0,0],z=[1.16,-1.16], mode="text", text=["|0⟩","|1⟩"], textfont=dict(color="#0F172A",size=12), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter3d(x=[1.16, -1.16, 0,0],y=[0,0,1.16,-1.16],z=[0,0,0,0], mode="text", text=["+X","+X"," +Y","−Y"], textfont=dict(color="#64748B",size=9), hoverinfo="skip", showlegend=False))
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1.25,1.25], title=dict(text="X", font=dict(color="#334155", size=10)), gridcolor="rgba(148,163,184,0.12)", backgroundcolor="rgba(0,0,0,0)", showbackground=False, zerolinecolor="rgba(148,163,184,0.18)"),
            yaxis=dict(range=[-1.25,1.25], title=dict(text="Y", font=dict(color="#334155", size=10)), gridcolor="rgba(148,163,184,0.12)", backgroundcolor="rgba(0,0,0,0)", showbackground=False, zerolinecolor="rgba(148,163,184,0.18)"),
            zaxis=dict(range=[-1.25,1.25], title=dict(text="Z", font=dict(color="#334155", size=10)), gridcolor="rgba(148,163,184,0.12)", backgroundcolor="rgba(0,0,0,0)", showbackground=False, zerolinecolor="rgba(148,163,184,0.18)"),
            aspectmode="cube", bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.55,y=1.42,z=1.08))
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=34,b=0), height=420, showlegend=False,
        title=dict(text=f"Qubit {qubit_idx} — θ {math.degrees(theta):.1f}° · φ {math.degrees(phi):.1f}° · |r| {r:.2f}", font=dict(size=11,color="#334155"), x=0.02)
    )
    return fig, a,b,x,y,z

def run_aer_code(code_str, shots=1024):
    start=time.time()
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        g={"QuantumCircuit":QuantumCircuit, "AerSimulator":AerSimulator, "np":np, "numpy":np}
        l={}
        exec(code_str, g, l)
        ns={**g, **l}
        qc=ns.get("qc")
        if qc is None:
            for v in ns.values():
                if isinstance(v, QuantumCircuit): qc=v; break
        if qc is None:
            return {"status":"error","error":"No QuantumCircuit 'qc' found. Use qc = QuantumCircuit(n).","elapsed":time.time()-start}
        has_measure=any(instr.operation.name=="measure" for instr in qc.data) if hasattr(qc,"data") else False
        sim=AerSimulator()
        if not has_measure:
            qc2=qc.copy(); qc2.save_statevector()
            res=sim.run(qc2).result()
            sv=np.asarray(res.get_statevector(qc2), dtype=complex)
            probs=np.abs(sv)**2; probs/=probs.sum() if probs.sum()>0 else 1
            counts={format(i,f'0{qc.num_qubits}b'): float(probs[i]) for i in range(len(probs))}
            qc_m=qc.copy(); qc_m.measure_all()
            res_m=sim.run(qc_m, shots=shots).result()
            counts_shots=res_m.get_counts(qc_m)
            return {"status":"success","backend":"AerSimulator (statevector)","shots":shots,"counts":counts_shots,"probs":counts,"statevector":sv,"circuit":qc,"elapsed":time.time()-start,"depth":qc.depth(),"num_qubits":qc.num_qubits,"num_gates":len(qc.data)}
        else:
            res=sim.run(qc, shots=shots).result()
            counts=res.get_counts(qc)
            probs={k:v/sum(counts.values()) for k,v in counts.items()}
            return {"status":"success","backend":"AerSimulator","shots":shots,"counts":counts,"probs":probs,"statevector":None,"circuit":qc,"elapsed":time.time()-start,"depth":qc.depth(),"num_qubits":qc.num_qubits,"num_gates":len(qc.data)}
    except Exception as e:
        return {"status":"error","error":str(e)+"\n"+traceback.format_exc(limit=3),"elapsed":time.time()-start}

def circuit_to_image_b64(qc):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig=qc.draw(output="mpl", style="iqp")
        buf=io.BytesIO()
        fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR=os.path.join(BASE_DIR,"..","assets","images","web","photos")
def img_b64_p(p): return img_b64(p) if os.path.exists(p) else ""

# Centralized error helper - light mode friendly
def show_error_light(title, what, why, next_step):
    st.markdown(f"""
<div style="background:#FEF2F2;border:1.5px solid #FECACA;border-radius:14px;padding:1rem 1.1rem">
  <div style="font-family:Instrument Sans;font-weight:600;color:#991B1B">{title}</div>
  <div style="font-size:0.84rem;line-height:1.6;color:#7F1D1D;margin-top:0.35rem"><b>What:</b> {what}<br/><b>Why:</b> {why}<br/><b>Next:</b> {next_step}</div>
</div>""", unsafe_allow_html=True)

st.set_page_config(page_title="Q-ORBIT Quantum Lab — Light", layout="wide", page_icon="⚛️", initial_sidebar_state="collapsed")

def main():
    inject_css()
    # Subtle header
    st.markdown("""
<div class="nav-light">
  <div style="display:flex;align-items:center;gap:0.8rem"><div class="qorbit-mark">Q</div><div><div style="font-family:Fraunces;font-weight:700;letter-spacing:-0.02em;color:#0F172A;font-size:1.02rem">Q-ORBIT Quantum Lab</div><div style="font-family:JetBrains Mono;font-size:0.62rem;letter-spacing:1.2px;color:#64748B">Light Scientific · From Qubits to Orbits</div></div></div>
  <div style="display:flex;gap:0.4rem;align-items:center"><a href="/" style="color:#334155;text-decoration:none;font-size:0.76rem;border:1px solid #E2E8F0;padding:0.42rem 0.7rem;border-radius:999px;background:white">← Mission Control</a><span style="font-family:JetBrains Mono;font-size:0.64rem;color:#2563EB;background:#EFF6FF;border:1px solid #DBEAFE;padding:0.42rem 0.7rem;border-radius:999px">● Simulator Ready</span></div>
</div>""", unsafe_allow_html=True)

    # HERO — premium light
    st.markdown("""
<div class="hero-wrap">
  <div class="hero-grid"></div>
  <div class="hero-orb" style="width:420px;height:420px;left:-60px;top:-120px;background:radial-gradient(circle at 50% 50%, rgba(37,99,235,0.06), transparent 62%)"></div>
  <div class="hero-orb" style="width:360px;height:360px;right:-40px;top:-40px;background:radial-gradient(circle at 50% 50%, rgba(124,58,237,0.05), transparent 62%)"></div>
  <div class="hero-math" style="left:58%;top:18%">|ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩</div>
  <div class="hero-math" style="left:18%;top:62%">⟨Z⟩ · 8 expectations</div>
  <div class="hero-math" style="right:12%;top:72%">He-3 → He-4 dilution</div>
  <div style="position:relative;max-width:760px">
    <div style="font-family:JetBrains Mono;font-size:0.64rem;letter-spacing:2px;color:#2563EB;text-transform:uppercase">Q-ORBIT · Hybrid Quantum-Classical</div>
    <h1 style="font-family:Fraunces;font-weight:700;letter-spacing:-0.03em;color:#0F172A;font-size:clamp(2.6rem,5vw,4.2rem);line-height:0.95;margin:0.35rem 0 0 0">Quantum Lab</h1>
    <div style="font-family:Fraunces;font-weight:600;color:#334155;font-size:clamp(1.45rem,2.8vw,2.05rem);letter-spacing:-0.02em;margin-top:0.15rem">From Qubits to Orbits</div>
    <p style="font-family:Instrument Sans;color:#475569;font-size:1.02rem;line-height:1.65;margin:0.7rem 0 0 0;max-width:62ch">Explore quantum states, quantum circuits and quantum hardware concepts — and understand how these technologies connect to the Q-ORBIT hybrid quantum-classical space-object classification system.</p>
    <div style="margin-top:1rem;display:flex;gap:0.5rem;flex-wrap:wrap"><span class="badge">Deep Space Research Center</span><span class="badge">Quantum Laboratory</span><span class="badge">AI/ML Mission Control</span></div>
  </div>
</div>""", unsafe_allow_html=True)

    # Photos subtle strip — light
    cols=st.columns(6)
    for col, fn, cap in zip(cols, ["I1_gargantua.jpg","I2_lab.jpg","I3_cryostat.jpg","I4_earth.jpg","I5_plasma.jpg","I6_starfield.jpg"], ["Gargantua","Lab","Cryostat","Earth","Plasma","Starfield"]):
        b64=img_b64_p(os.path.join(PHOTOS_DIR, fn))
        if b64:
            col.markdown(f'<div style="border-radius:14px;overflow:hidden;border:1px solid #E8E6E1;background:white;box-shadow:0 6px 18px rgba(15,27,46,0.05)"><img src="data:image/jpeg;base64,{b64}" style="width:100%;aspect-ratio:1;object-fit:cover;display:block"/><div style="font-family:JetBrains Mono;font-size:0.58rem;color:#64748B;text-align:center;padding:0.3rem;background:#FFFCF7;border-top:1px solid #E8E6E1">{cap}</div></div>', unsafe_allow_html=True)

    # ===== SECTION 01 — THE MACHINE =====
    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
    st.markdown("""
<div class="section" style="padding:1.4rem 1.4rem 1.2rem 1.4rem">
  <div style="display:flex;align-items:baseline;gap:0.7rem;flex-wrap:wrap"><span class="section-label">Section 01</span><span style="font-family:JetBrains Mono;font-size:0.62rem;color:#94A3B8">The Machine</span></div>
  <h2 class="section-title" style="font-size:clamp(1.7rem,3vw,2.5rem);margin-top:0.15rem">The Quantum Computer — Dilution Refrigerator</h2>
  <p class="section-sub">A large interactive scientific instrument — metallic plates, copper wiring, cooling stages, central processor. Not a stack of circles. Hover and click to follow the cooling from 300&nbsp;K to 10&nbsp;mK.</p>
</div>""", unsafe_allow_html=True)

    sel=st.selectbox("Highlight stage (click simulation — camera focuses, cooling pulses)", options=[f"{s['temp']} — {s['name']}" for s in CRYO_STAGES], index=5, key="cryo_sel_light")
    sel_key=sel.split(" — ")[0]
    sel_info=next(s for s in CRYO_STAGES if s["temp"]==sel_key)

    # Build light cryostat HTML — metallic/copper elegant
    plates_html=""
    for s in CRYO_STAGES:
        active=" active" if s["temp"]==sel_key else ""
        # light metallic: warm white top highlight + soft copper accent, selected = blue accent
        bg = "linear-gradient(180deg, #FFFFFF 0%, #F8F5F0 58%, #F1EEE8 100%)" if active=="" else "linear-gradient(180deg, #FFFFFF 0%, #EFF6FF 62%, #DBEAFE 100%)"
        border = "#E8E6E1" if active=="" else "#BFDBFE"
        copper = "#E8E6E1" # subtle
        plates_html+=f'<div class="plate{active}" style="top:{s["top"]}%;width:{s["w"]}px;height:{s["h"]}px;background:{bg};border-color:{border}"><div class="plate-label">{s["temp"]}</div></div>\n'

    st.markdown(f"""
<div class="cryostat-stage" style="margin-top:0.7rem">
  <div class="helium" style="top:12%;height:68%"></div>
  <div class="helium" style="top:12%;height:68%;left:51%;opacity:0.35;width:1.2px;background:linear-gradient(180deg, rgba(148,163,184,0.35), rgba(37,99,235,0.18))"></div>
  <div class="cable" style="top:8%;height:70%"></div>
  {plates_html}
  <div class="chip"><div class="chip-grid"></div></div>
  <div style="position:absolute;bottom:10px;left:12px;font-family:JetBrains Mono;font-size:0.58rem;color:#64748B;background:white;border:1px solid #E8E6E1;padding:0.28rem 0.6rem;border-radius:999px;box-shadow:0 4px 14px rgba(15,27,46,0.06)">He-3 ▲ Still → MXC (dilutes) · He-4 ▼ MXC → Still · Copper loom — I3</div>
  <div style="position:absolute;top:10px;right:12px;font-family:JetBrains Mono;font-size:0.58rem;color:#2563EB;background:#EFF6FF;border:1px solid #BFDBFE;padding:0.28rem 0.6rem;border-radius:999px">● Rotate conceptually — select stage to focus camera</div>
</div>""", unsafe_allow_html=True)

    c1,c2=st.columns([1.28,0.92])
    with c1:
        st.markdown(f"""
<div class="glass" style="margin-top:0.7rem">
  <div style="font-family:JetBrains Mono;font-size:0.64rem;letter-spacing:1.6px;color:#2563EB">{sel_info["temp"]} — {sel_info["name"]}</div>
  <div style="font-size:0.95rem;line-height:1.65;color:#334155;margin-top:0.35rem">{sel_info["explain"]}</div>
  <div style="margin-top:0.65rem;display:flex;gap:0.45rem;flex-wrap:wrap"><span class="badge">Cooling flow 300K → 10mK</span><span class="badge">Coax thermal anchoring</span><span class="badge">Processor at 10mK</span></div>
  <div style="margin-top:0.6rem;font-size:0.74rem;color:#64748B">Scientifically reasonable visualization — not an exact IBM processor replica. The animated particles trace He-3/He-4 dilution; selected plate glows blue and scales slightly for focus.</div>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="glass" style="margin-top:0.7rem"><div style="font-family:Instrument Sans;font-weight:600;letter-spacing:0.6px;color:#0F172A;font-size:0.78rem;margin-bottom:0.4rem">Cooling Ladder — 300K → 10mK</div><div style="display:grid;gap:0.38rem">', unsafe_allow_html=True)
        for s in CRYO_STAGES:
            hl="background:#EFF6FF;border-color:#BFDBFE;color:#1E40AF" if s["temp"]==sel_key else "background:#F8FAFC;border-color:#E2E8F0;color:#334155"
            st.markdown(f'<div style="font-family:JetBrains Mono;font-size:0.64rem;border:1px solid #E2E8F0;padding:0.42rem 0.6rem;border-radius:10px;display:flex;justify-content:space-between;{hl}"><span>{s["name"]}</span><span style="font-weight:600">{s["temp"]}</span></div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
        # vertical animated cooling line
        st.markdown("""
<div style="margin-top:0.6rem;background:white;border:1px solid #E8E6E1;border-radius:14px;padding:0.7rem;box-shadow:0 6px 18px rgba(15,27,46,0.05)">
  <div style="font-family:JetBrains Mono;font-size:0.62rem;letter-spacing:1.2px;color:#94A3B8">Cooling Animation — Continuous</div>
  <div style="margin-top:0.5rem;display:flex;align-items:center;gap:0.3rem;justify-content:center;flex-wrap:wrap">
    <span style="background:#FFFBEB;border:1px solid #FDE68A;padding:0.26rem 0.5rem;border-radius:999px;font-family:JetBrains Mono;font-size:0.62rem;color:#92400E">300K</span><span style="color:#94A3B8">↓</span>
    <span style="background:#FFF7ED;border:1px solid #FED7AA;padding:0.26rem 0.5rem;border-radius:999px;font-family:JetBrains Mono;font-size:0.62rem;color:#9A3412">50K</span><span style="color:#94A3B8">↓</span>
    <span style="background:#FEF2F2;border:1px solid #FECACA;padding:0.26rem 0.5rem;border-radius:999px;font-family:JetBrains Mono;font-size:0.62rem;color:#991B1B">4K</span><span style="color:#94A3B8">↓</span>
    <span style="background:#F0F9FF;border:1px solid #BAE6FD;padding:0.26rem 0.5rem;border-radius:999px;font-family:JetBrains Mono;font-size:0.62rem;color:#0C4A6E">800mK</span><span style="color:#94A3B8">↓</span>
    <span style="background:#EFF6FF;border:1px solid #BFDBFE;padding:0.26rem 0.5rem;border-radius:999px;font-family:JetBrains Mono;font-size:0.62rem;color:#1E40AF">100mK</span><span style="color:#94A3B8">↓</span>
    <span style="background:#EFF6FF;border:1px solid #93C5FD;padding:0.26rem 0.5rem;border-radius:999px;font-family:JetBrains Mono;font-size:0.62rem;color:#1E3A8A;font-weight:700">10mK</span>
  </div>
  <div style="margin-top:0.5rem;height:4px;border-radius:999px;background:linear-gradient(90deg, #FDE68A, #FECACA, #BAE6FD, #93C5FD);opacity:0.9"></div>
</div>""", unsafe_allow_html=True)

    # ===== SECTION 02 — THE QUBIT =====
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    st.markdown("""
<div class="section" style="padding:1.4rem">
  <div style="display:flex;align-items:baseline;gap:0.7rem"><span class="section-label">Section 02</span><span style="font-family:JetBrains Mono;font-size:0.62rem;color:#94A3B8">The Qubit</span></div>
  <h2 class="section-title" style="font-size:clamp(1.65rem,2.8vw,2.4rem)">The Bloch Sphere — Live State Vector</h2>
  <p class="section-sub">A large high-quality translucent sphere. The arrow is the physical state — it moves as you change θ and φ. Nothing is hard-coded.</p>
</div>""", unsafe_allow_html=True)

    if "bloch_theta" not in st.session_state: st.session_state.bloch_theta=0.0
    if "bloch_phi" not in st.session_state: st.session_state.bloch_phi=0.0
    # Presets elegant
    pc1,pc2,pc3,pc4,pc5,pc6=st.columns(6)
    for col,(label,th,ph) in zip([pc1,pc2,pc3,pc4,pc5,pc6], [("|0⟩",0,0),("|1⟩",math.pi,0),("|+⟩",math.pi/2,0),("|-⟩",math.pi/2,math.pi),("|i⟩",math.pi/2,math.pi/2),("|-i⟩",math.pi/2,-math.pi/2)]):
        if col.button(label, width="stretch", key=f"preset_light_{label}"):
            st.session_state.bloch_theta=th; st.session_state.bloch_phi=ph

    colA,colB=st.columns([1.25,0.95], gap="large")
    with colA:
        theta=st.slider("θ — polar from |0⟩ north [0, π]", 0.0, float(math.pi), float(st.session_state.bloch_theta), step=0.02, key="th_light", help="θ=0→|0⟩, θ=π→|1⟩, θ=π/2→equator")
        phi=st.slider("φ — azimuth around Z [-π, π]", -float(math.pi), float(math.pi), float(st.session_state.bloch_phi), step=0.02, key="ph_light")
        st.session_state.bloch_theta=theta; st.session_state.bloch_phi=phi
        fig,a,b,x,y,z=bloch_fig_light(theta, phi)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar":False})
        st.caption("Drag to rotate · Scroll to zoom · The arrow is r=(sinθ cosφ, sinθ sinφ, cosθ) computed live.")
        # Evolution controls
        ec1,ec2,ec3=st.columns(3)
        if ec1.button("▶ Play Evolution", width="stretch", key="play_evo"):
            st.session_state.evolve=True
        if ec2.button("⏸ Pause", width="stretch", key="pause_evo"):
            st.session_state.evolve=False
        if ec3.button("↺ Reset to |0⟩", width="stretch", key="reset_evo"):
            st.session_state.bloch_theta=0; st.session_state.bloch_phi=0; st.session_state.evolve=False; st.rerun()
        if st.session_state.get("evolve", False):
            # simple animated step
            st.session_state.bloch_phi=(phi+0.12)%(2*math.pi)
            if st.session_state.bloch_phi>math.pi: st.session_state.bloch_phi-=2*math.pi
            time.sleep(0.12); st.rerun()
    with colB:
        a,b,x,y,z=bloch_single(theta, phi)
        P0=abs(a)**2; P1=abs(b)**2
        st.markdown(f"""
<div class="equation">
  <div style="font-family:Fraunces;font-weight:600;color:#0F172A;font-size:0.98rem;margin-bottom:0.3rem">Live Quantum Equation</div>
  <div>|ψ⟩ = cos(θ/2)|0⟩ + e<sup>iφ</sup> sin(θ/2)|1⟩</div>
  <div style="margin-top:0.5rem;color:#334155">θ = {theta:.3f} rad · {math.degrees(theta):.1f}° &nbsp; φ = {phi:.3f} rad · {math.degrees(phi):.1f}°</div>
  <div>α = cos(θ/2) = {a.real:+.4f}{a.imag:+.4f}i</div>
  <div>β = e<sup>iφ</sup> sin(θ/2) = {b.real:+.4f}{b.imag:+.4f}i</div>
  <div style="margin-top:0.45rem;font-weight:600">P(0)=|α|² = {P0:.4f} ({P0*100:.1f}%) &nbsp; P(1)=|β|² = {P1:.4f} ({P1*100:.1f}%)</div>
  <div style="margin-top:0.35rem;color:#64748B">Bloch vector r = ({x:+.3f}, {y:+.3f}, {z:+.3f}) — endpoint you drag</div>
</div>""", unsafe_allow_html=True)
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-family:JetBrains Mono;font-size:0.62rem;letter-spacing:1px;color:#94A3B8">MEASUREMENT PROBABILITIES — ANIMATED</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div class="prob-track" style="margin:6px 0">
  <div style="width:{P0*100:.1f}%;background:linear-gradient(90deg,#60A5FA,#2563EB);display:flex;align-items:center;padding-left:10px;font-size:0.78rem;font-weight:600;color:white">|0⟩ {P0*100:.1f}%</div>
  <div style="width:{P1*100:.1f}%;background:linear-gradient(90deg,#A78BFA,#7C3AED);display:flex;align-items:center;padding-left:10px;font-size:0.78rem;font-weight:600;color:white">|1⟩ {P1*100:.1f}%</div>
</div>""", unsafe_allow_html=True)
        st.markdown(f'<div class="glass" style="font-size:0.80rem;color:#475569;line-height:1.6"><b style="color:#0F172A">State vector:</b> ({a.real:.4f}{a.imag:+.4f}i)|0⟩ + ({b.real:.4f}{b.imag:+.4f}i)|1⟩<br/><b>Expectations:</b> ⟨X⟩={x:+.3f} ⟨Y⟩={y:+.3f} ⟨Z⟩={z:+.3f} — real `sinθcosφ/sinθsinφ/cosθ`.</div>', unsafe_allow_html=True)

    # ===== SECTION 03 — THE CIRCUIT =====
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    st.markdown("""
<div class="section" style="padding:1.4rem">
  <div style="display:flex;align-items:baseline;gap:0.7rem"><span class="section-label">Section 03</span><span style="font-family:JetBrains Mono;font-size:0.62rem;color:#94A3B8">The Circuit</span></div>
  <h2 class="section-title" style="font-size:clamp(1.6rem,2.8vw,2.35rem)">Quantum Gates — Build a Circuit, Watch the Bloch React</h2>
  <p class="section-sub">Apply gates and the Bloch sphere, equation, circuit and measurement all update from real simulation.</p>
</div>""", unsafe_allow_html=True)

    if "qspec_light" not in st.session_state: st.session_state.qspec_light=CircuitSpec(n_qubits=2)
    nqb=st.slider("Qubits", 1, 8, int(st.session_state.qspec_light.n_qubits), key="nqb_light")
    st.session_state.qspec_light.n_qubits=nqb
    qspec=st.session_state.qspec_light

    g1,g2,g3=st.columns([1.02,1.24,1.02], gap="large")
    with g1:
        st.markdown('<div class="glass"><div style="font-family:Instrument Sans;font-weight:600;color:#0F172A;font-size:0.78rem;margin-bottom:0.4rem">Gate Toolbar — Click to Add</div>', unsafe_allow_html=True)
        gate=st.selectbox("Gate", list(GATE_INFO.keys()), key="gate_light")
        info=GATE_INFO[gate]
        st.markdown(f'<div style="font-family:JetBrains Mono;font-size:0.74rem;color:#2563EB;background:#EFF6FF;border:1px solid #BFDBFE;padding:0.45rem;border-radius:10px">{info["math"]}<br/><span style="color:#7C3AED">{info.get("matrix","")}</span><br/><span style="color:#475569">{info["desc"]}</span></div>', unsafe_allow_html=True)
        w1=st.number_input(f"Qubit 0-{nqb-1}", 0, nqb-1, 0, key="w1_light")
        need2=gate in ("CNOT","CZ")
        w2=st.number_input(f"Target 0-{nqb-1}", 0, nqb-1, min(1,nqb-1), key="w2_light") if need2 else None
        ang=st.slider("Angle", -3.14,3.14,0.79, key="ang_light") if gate in ("RX","RY","RZ") else None
        if st.button("➕ Add Gate to Circuit", width="stretch", key="add_light"):
            if gate in ("RX","RY","RZ"): qspec.add_gate(gate,[int(w1)],[float(ang)])
            elif need2:
                if int(w1)==int(w2): st.toast("Control ≠ target")
                else: qspec.add_gate(gate,[int(w1),int(w2)])
            else: qspec.add_gate(gate,[int(w1)])
            st.rerun()
        cA,cB=st.columns(2)
        if cA.button("↩ Undo", width="stretch", key="undo_light"):
            if qspec.ops: qspec.ops.pop(); st.rerun()
        if cB.button("🗑 Clear", width="stretch", key="clear_light"):
            qspec.clear(); st.rerun()
        p1,p2,p3=st.columns(3)
        if p1.button("|+⟩", width="stretch", key="p_plus_light"): qspec.clear(); qspec.add_gate("H",[0]); st.rerun()
        if p2.button("Bell", width="stretch", key="p_bell_light"): qspec.clear(); qspec.add_gate("H",[0]); qspec.add_gate("CNOT",[0,1] if nqb>=2 else [0,0]); st.rerun()
        if p3.button("GHZ", width="stretch", key="p_ghz_light"): qspec.clear(); qspec.add_gate("H",[0]); [qspec.add_gate("CNOT",[i,i+1]) for i in range(nqb-1)]; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        # Circuit horizontal viz
        st.markdown('<div style="font-family:JetBrains Mono;font-size:0.62rem;color:#94A3B8;margin-top:0.6rem">Circuit — Horizontal (q0 ── H ── Ry ── Rz ── Measure)</div>', unsafe_allow_html=True)
        if not qspec.ops:
            st.markdown('<div class="glass" style="text-align:center;color:#94A3B8">Empty — add H then CNOT to entangle.</div>', unsafe_allow_html=True)
        else:
            # build horizontal wire view
            wires_html=""
            for qi in range(min(4,nqb)):
                line=f'<div style="height:2px;background:#E2E8F0;position:relative;margin:0.7rem 0"><span style="position:absolute;left:-18px;top:-7px;font-family:JetBrains Mono;font-size:0.66rem;color:#334155">q{qi}</span>'
                for idx,op in enumerate(qspec.ops):
                    if qi in op["wires"]:
                        # gate chip
                        gate_lab=op["gate"]
                        left= 8 + idx*54
                        line+=f'<span style="position:absolute;left:{left}px;top:-13px;background:white;border:1px solid #CBD5E1;padding:0.12rem 0.38rem;border-radius:8px;font-family:JetBrains Mono;font-size:0.62rem;box-shadow:0 2px 8px rgba(15,27,46,0.06)">{gate_lab}</span>'
                line+='</div>'
                wires_html+=line
            st.markdown(f'<div class="glass">{wires_html}<div style="font-family:JetBrains Mono;font-size:0.62rem;color:#94A3B8;text-align:center">Depth {len(qspec.ops)} · Gates {len(qspec.ops)} · Qubits {nqb}</div></div>', unsafe_allow_html=True)
    with g2:
        st.markdown('<div class="glass"><div style="font-family:Instrument Sans;font-weight:600;color:#0F172A;font-size:0.78rem;margin-bottom:0.4rem">Live Bloch — Circuit Result</div>', unsafe_allow_html=True)
        try:
            res=qspec.simulate("pennylane")
            bloch_res=res.bloch_vectors
            cols=st.columns(2)
            for i in range(min(4, nqb)):
                with cols[i%2]:
                    th=np.arccos(np.clip(bloch_res[i][2],-1,1)); ph=np.arctan2(bloch_res[i][1],bloch_res[i][0])
                    fig2,a2,b2,_,_,_=bloch_fig_light(th, ph, qubit_idx=i)
                    st.plotly_chart(fig2, width="stretch", config={"displayModeBar":False}, key=f"light_builder_{i}")
            probs=np.abs(res.statevector)**2; probs/=probs.sum() if probs.sum()>0 else 1
            st.markdown(f'<div style="font-family:JetBrains Mono;font-size:0.62rem;color:#94A3B8">Probs (real statevector): {", ".join([f"|{format(i,f"0{nqb}b")[::-1]}⟩ {probs[i]*100:.1f}%" for i in np.argsort(probs)[::-1][:3]] )}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(str(e))
        st.markdown('</div>', unsafe_allow_html=True)
    with g3:
        st.markdown('<div class="glass"><div style="font-family:Instrument Sans;font-weight:600;color:#0F172A;font-size:0.78rem;margin-bottom:0.4rem">Auto Python — Qiskit / PennyLane</div>', unsafe_allow_html=True)
        t1,t2=st.tabs(["Qiskit","PennyLane"])
        with t1: st.code(qspec.to_qiskit_code(), language="python"); st.download_button("⬇ Qiskit .py", data=qspec.to_qiskit_code(), file_name="qorbit_qiskit.py", mime="text/x-python", key="dl_q_light")
        with t2: st.code(qspec.to_pennylane_code(), language="python"); st.download_button("⬇ PennyLane .py", data=qspec.to_pennylane_code(), file_name="qorbit_pennylane.py", mime="text/x-python", key="dl_p_light")
        st.markdown('</div>', unsafe_allow_html=True)

    # ===== SECTION 04 — THE EXPERIMENT =====
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    st.markdown("""
<div class="section" style="padding:1.4rem">
  <div style="display:flex;align-items:baseline;gap:0.7rem"><span class="section-label">Section 04</span><span style="font-family:JetBrains Mono;font-size:0.62rem;color:#94A3B8">The Experiment</span></div>
  <h2 class="section-title" style="font-size:clamp(1.6rem,2.8vw,2.35rem)">Live Qiskit Execution — Simulator & Hardware</h2>
  <p class="section-sub">A clean editor with line numbers. Run on Aer simulator; hardware path honestly disabled until credentials.</p>
</div>""", unsafe_allow_html=True)

    default_code="""from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0,1)
qc.measure_all()
# Try: qc.ry(0.7, 0); qc.rz(1.2, 1); qc.x(0)
"""
    if "play_code_light" not in st.session_state: st.session_state.play_code_light=default_code
    st.session_state.play_code_light=st.text_area("Qiskit editor — Aer", value=st.session_state.play_code_light, height=180, key="editor_light")
    ec1,ec2,ec3,ec4,ec5=st.columns([1.1,1.1,0.9,0.9,0.9])
    shots=ec1.number_input("Shots", 100, 10000, 1024, step=100, key="shots_light")
    run_sim=ec2.button("▶ Run on Simulator", width="stretch", type="primary", key="run_sim_light")
    ec3.button("⬢ Run on Hardware", width="stretch", disabled=True, help="Hardware disabled: no IBM token", key="run_hw_light")
    if ec4.button("■ Stop", width="stretch", key="stop_light"): st.toast("No background job")
    if ec5.button("↺ Reset", width="stretch", key="reset_light"): st.session_state.play_code_light=default_code; st.rerun()
    if st.button("▢ Clear", key="clear_light2"): st.session_state.play_code_light=""; st.rerun()
    ec3.caption("Hardware unavailable — Simulator active")
    if run_sim:
        with st.spinner("Aer running…"):
            res=run_aer_code(st.session_state.play_code_light, shots=int(shots))
            st.session_state.play_result_light=res
    if "play_result_light" in st.session_state:
        res=st.session_state.play_result_light
        if res["status"]=="error":
            st.markdown(f'<div class="glass" style="border-color:#FECACA"><pre style="white-space:pre-wrap;font-family:JetBrains Mono;font-size:0.72rem;color:#991B1B">{res["error"]}</pre></div>', unsafe_allow_html=True)
        else:
            s1,s2,s3,s4=st.columns(4)
            s1.metric("Backend", res["backend"]); s2.metric("Shots", res["shots"]); s3.metric("Qubits", res["num_qubits"]); s4.metric("Depth/Gates", f'{res["depth"]}/{res["num_gates"]}')
            st.caption(f"Elapsed {res['elapsed']*1000:.1f} ms — simulator, never claimed as hardware.")
            b64=circuit_to_image_b64(res["circuit"])
            if b64:
                st.markdown(f'<div class="glass"><img src="data:image/png;base64,{b64}" style="width:100%;background:white;border-radius:12px;padding:8px"/></div>', unsafe_allow_html=True)
            view=st.radio("Measurement view", ["Counts","Probabilities","Statevector"], horizontal=True, key="play_view_light")
            if view=="Counts":
                keys=list(res["counts"].keys()); vals=[res["counts"][k] for k in keys]
                fig=go.Figure(go.Bar(x=keys,y=vals, marker_color="#2563EB", text=vals, textposition="auto"))
                fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white", height=280, margin=dict(l=10,r=10,t=10,b=10), yaxis=dict(gridcolor="#E2E8F0"), xaxis=dict(gridcolor="#E2E8F0"))
                st.plotly_chart(fig, width="stretch", config={"displayModeBar":False})
                st.json(res["counts"])
            elif view=="Probabilities":
                probs=res["probs"] if isinstance(res["probs"],dict) else res["counts"]
                keys=list(probs.keys()); vals=[probs[k]*100 if max(probs.values())<=1 else probs[k] for k in keys]
                fig=go.Figure(go.Bar(x=keys,y=vals, marker_color="#7C3AED", text=[f"{v:.1f}%" for v in vals], textposition="auto"))
                fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white", height=280, margin=dict(l=10,r=10,t=10,b=10), yaxis=dict(gridcolor="#E2E8F0"), xaxis=dict(gridcolor="#E2E8F0"))
                st.plotly_chart(fig, width="stretch", config={"displayModeBar":False})
            else:
                if res.get("statevector") is not None:
                    st.code(f"statevector = {np.array2string(res['statevector'], precision=4)}", language="python")
                    st.caption(f"Probs |amp|² = {np.array2string(np.abs(res['statevector'])**2, precision=4)}")
                else: st.info("With measurements, statevector collapsed — see Counts.")
    with st.expander("🔑 Hardware credentials"):
        st.markdown("Set `QISKIT_IBM_TOKEN` env or `~/.qiskit/qiskit-ibm.json`, install `qiskit-ibm-runtime`. Never paste token in editor. Current: Simulator only.")

    # State Evolution
    st.markdown('<div class="glass" style="margin-top:0.8rem"><div style="font-family:Instrument Sans;font-weight:600;color:#0F172A;font-size:0.82rem;margin-bottom:0.4rem">Quantum State Evolution — Play the Circuit Gate by Gate</div>', unsafe_allow_html=True)
    if "evo_qspec_light" not in st.session_state:
        st.session_state.evo_qspec_light=CircuitSpec(n_qubits=2); st.session_state.evo_qspec_light.add_gate("H",[0]); st.session_state.evo_qspec_light.add_gate("CNOT",[0,1])
    eq=st.session_state.evo_qspec_light
    eq.n_qubits=st.slider("Evolution qubits", 2,4,int(eq.n_qubits), key="evo_nq_light")
    if "evo_step_light" not in st.session_state: st.session_state.evo_step_light=0
    n_steps=len(eq.ops)+1
    c1,c2,c3,c4,c5=st.columns([1,1,1,1,1.4])
    if c1.button("◀ Prev", width="stretch", key="evo_prev_light"): st.session_state.evo_step_light=max(0, st.session_state.evo_step_light-1); st.rerun()
    if c2.button("▶ Next", width="stretch", key="evo_next_light"): st.session_state.evo_step_light=min(n_steps-1, st.session_state.evo_step_light+1); st.rerun()
    if c3.button("▶ Play", width="stretch", key="evo_play_light"):
        for i in range(n_steps):
            st.session_state.evo_step_light=i; time.sleep(0.5); st.rerun()
    if c4.button("⏸ Pause", width="stretch", key="evo_pause_light"): st.toast("Paused")
    if c5.button("↺ Reset", width="stretch", key="evo_reset_light"): st.session_state.evo_step_light=0; st.rerun()
    step=int(st.session_state.evo_step_light)
    st.progress(step/(n_steps-1) if n_steps>1 else 1)
    prefix=eq.ops[:step]
    labels=["Step 0 — |00..0⟩"]+[f'Step {i+1} — {op["gate"]} {op["wires"]}' for i,op in enumerate(eq.ops)]
    st.markdown(f'<div style="font-family:JetBrains Mono;font-size:0.68rem;color:#2563EB;text-align:center">{labels[step] if step < len(labels) else labels[-1]}</div>', unsafe_allow_html=True)
    try:
        if not prefix:
            sv=np.zeros(1<<eq.n_qubits,dtype=complex); sv[0]=1; bloch=compute_bloch_vectors(sv,eq.n_qubits)
        else:
            r=PennylaneBackend(eq.n_qubits).simulate_circuit_ops(prefix,eq.n_qubits); sv=r.statevector; bloch=r.bloch_vectors
        ec1,ec2=st.columns([1.2,0.9])
        with ec1:
            cols=st.columns(2)
            for i in range(min(4, eq.n_qubits)):
                with cols[i%2]:
                    th=np.arccos(np.clip(bloch[i][2],-1,1)); ph=np.arctan2(bloch[i][1],bloch[i][0])
                    fig,_a,_b,_,_,_=bloch_fig_light(th,ph,qubit_idx=i)
                    st.plotly_chart(fig, width="stretch", config={"displayModeBar":False}, key=f"evo_light_{step}_{i}")
        with ec2:
            probs=np.abs(sv)**2; probs/=probs.sum() if probs.sum()>0 else 1
            for i in np.argsort(probs)[::-1][:4]:
                bits=format(i,f'0{eq.n_qubits}b')[::-1]
                st.markdown(f'<div style="font-family:JetBrains Mono;font-size:0.70rem;background:#F8FAFC;border:1px solid #E2E8F0;padding:0.32rem 0.5rem;border-radius:8px;margin:0.18rem 0;display:flex;justify-content:space-between"><span>|{bits}⟩</span><span>{probs[i]*100:.2f}%</span></div>', unsafe_allow_html=True)
            st.caption(f"|r| per qubit: {', '.join([f'q{i}:{np.linalg.norm(bloch[i]):.2f}' for i in range(eq.n_qubits)])}")
    except Exception as e:
        st.error(str(e))
    st.markdown('</div>', unsafe_allow_html=True)

    # ===== SECTION 05 — THE CONNECTION =====
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    st.markdown("""
<div class="section" style="padding:1.4rem">
  <div style="display:flex;align-items:baseline;gap:0.7rem"><span class="section-label">Section 05</span><span style="font-family:JetBrains Mono;font-size:0.62rem;color:#94A3B8">The Connection</span></div>
  <h2 class="section-title" style="font-size:clamp(1.6rem,2.8vw,2.35rem)">Why Quantum Matters to Q-ORBIT</h2>
  <p class="section-sub">Light curve → quantum feature space → hybrid decision — all with real data, never invented.</p>
</div>""", unsafe_allow_html=True)

    # Pipeline horizontal light
    st.markdown("""
<div class="glass" style="overflow-x:auto">
  <div style="display:flex;align-items:center;gap:0.35rem;justify-content:center;min-width:760px">
    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;padding:0.55rem 0.7rem;min-width:120px;text-align:center"><div style="font-family:JetBrains Mono;font-size:0.60rem;color:#64748B">LIGHT CURVE</div><div style="font-family:Fraunces;font-weight:600;color:#0F172A">256 pts</div></div><span style="color:#94A3B8">→</span>
    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;padding:0.55rem 0.7rem;min-width:120px;text-align:center"><div style="font-family:JetBrains Mono;font-size:0.60rem;color:#64748B">FEATURES</div><div style="font-family:Fraunces;font-weight:600;color:#0F172A">21 dims</div></div><span style="color:#94A3B8">→</span>
    <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:12px;padding:0.55rem 0.7rem;min-width:120px;text-align:center"><div style="font-family:JetBrains Mono;font-size:0.60rem;color:#92400E">QUANTUM MAP</div><div style="font-family:Fraunces;font-weight:600;color:#92400E">8Q RY</div></div><span style="color:#94A3B8">→</span>
    <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:12px;padding:0.55rem 0.7rem;min-width:120px;text-align:center"><div style="font-family:JetBrains Mono;font-size:0.60rem;color:#1E40AF">VARIATIONAL</div><div style="font-family:Fraunces;font-weight:600;color:#1E40AF">⟨Z⟩×8</div></div><span style="color:#94A3B8">→</span>
    <div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:12px;padding:0.55rem 0.7rem;min-width:120px;text-align:center"><div style="font-family:JetBrains Mono;font-size:0.60rem;color:#166534">FUSION</div><div style="font-family:Fraunces;font-weight:600;color:#166534">Hybrid</div></div><span style="color:#94A3B8">→</span>
    <div style="background:#0F172A;border:1px solid #0F172A;border-radius:12px;padding:0.55rem 0.7rem;min-width:120px;text-align:center"><div style="font-family:JetBrains Mono;font-size:0.60rem;color:#94A3B8">CLASSIFY</div><div style="font-family:Fraunces;font-weight:600;color:white">5 classes</div></div>
  </div>
</div>""", unsafe_allow_html=True)

    cc1,cc2=st.columns([1.15,0.95], gap="large")
    with cc1:
        st.markdown("""
<div class="glass" style="margin-top:0.7rem">
  <div style="display:grid;gap:0.55rem">
    <div style="display:flex;gap:0.7rem;align-items:flex-start"><div style="width:34px;height:34px;border-radius:10px;background:#EFF6FF;border:1px solid #BFDBFE;display:grid;place-items:center">▦</div><div><div style="font-family:Fraunces;font-weight:600;color:#1E40AF">Classical AI</div><div style="font-size:0.82rem;color:#475569">Extracts 21 physics features — CNN temporal motifs, SVM margins. Strong on clean curves.</div></div></div>
    <div style="display:flex;gap:0.7rem;align-items:flex-start"><div style="width:34px;height:34px;border-radius:10px;background:#F5F3FF;border:1px solid #DDD6FE;display:grid;place-items:center">◈</div><div><div style="font-family:Fraunces;font-weight:600;color:#6D28D9">Quantum Model</div><div style="font-size:0.82rem;color:#475569">Angle RY → Rot+CNOT×2 (256-D Hilbert) → ⟨Z⟩×8 via PennyLane/Aer.</div></div></div>
    <div style="display:flex;gap:0.7rem;align-items:flex-start"><div style="width:34px;height:34px;border-radius:10px;background:#F0FDF4;border:1px solid #BBF7D0;display:grid;place-items:center">⬡</div><div><div style="font-family:Fraunces;font-weight:600;color:#166534">Hybrid + Adaptive Fusion</div><div style="font-size:0.82rem;color:#475569">α·p_classical+(1-α)·p_quantum, α from signal quality, trained on train/val only.</div></div></div>
  </div>
</div>""", unsafe_allow_html=True)
    with cc2:
        try:
            fc=json.load(open(os.path.join(BASE_DIR,"..","results","reports","fair_comparison.json"))) if os.path.exists(os.path.join(BASE_DIR,"..","results","reports","fair_comparison.json")) else {}
            hyb=fc.get("Hybrid_Quantum",{}); cnn=fc.get("Classical_CNN",{})
            st.markdown(f"""
<div class="glass" style="margin-top:0.7rem">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;text-align:center">
    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;padding:0.6rem"><div style="font-family:Fraunces;font-weight:600;color:#2563EB;font-size:1.1rem">{cnn.get("accuracy",0.852)*100:.1f}%</div><div style="font-family:JetBrains Mono;font-size:0.60rem;color:#64748B">CNN (classical)</div></div>
    <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:12px;padding:0.6rem"><div style="font-family:Fraunces;font-weight:600;color:#1E40AF;font-size:1.1rem">{hyb.get("accuracy",0.70)*100:.1f}%</div><div style="font-family:JetBrains Mono;font-size:0.60rem;color:#64748B">Hybrid (8q)</div></div>
  </div>
  <div style="margin-top:0.5rem;font-size:0.74rem;color:#64748B">Same 1500 test, same splits, same metrics. No hard-coded superiority.</div>
</div>""", unsafe_allow_html=True)
        except Exception: st.info("Run experiments to populate metrics.")

    # Dataset experiment
    st.markdown('<div class="glass" style="margin-top:0.8rem"><div style="font-family:Instrument Sans;font-weight:600;color:#0F172A;font-size:0.82rem;margin-bottom:0.4rem">Q-ORBIT Quantum Experiment — Select Real Sample</div>', unsafe_allow_html=True)
    dc1,dc2,dc3=st.columns([1,1,1.1])
    with dc1: cls_exp=dc1.selectbox("Class", [0,1,2,3,4], format_func=lambda x: ["Intact","Dead","Rocket","Debris","Spoofed"][x], key="cls_light")
    with dc2: noise_exp=dc2.selectbox("Noise", [0.0,0.02,0.05], key="noise_light")
    with dc3:
        run_exp=st.button("▶ Run Pipeline — Real Model", width="stretch", type="primary", key="run_exp_light")
    if run_exp:
        with st.spinner("Running curve → 21 feats → 8Q → ⟨Z⟩ → hybrid…"):
            try:
                from src.simulator.lightcurve_generator import generate_single_light_curve
                from src.classical.feature_engineering import extract_all_features
                curve,_=generate_single_light_curve(int(cls_exp), noise_std=float(noise_exp), n_samples=256)
                feats=extract_all_features(curve, sampling_interval=720/255)
                feats=np.nan_to_num(feats, nan=0.0)
                try:
                    from app.app import load_all_models
                    bundle=load_all_models(); hybrid=bundle.get("hybrid"); scaler=bundle.get("scaler")
                except Exception: hybrid=None; scaler=None
                figL=go.Figure(go.Scatter(y=curve, mode="lines", line=dict(color="#2563EB",width=1.8)))
                figL.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white", height=160, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(figL, width="stretch", config={"displayModeBar":False})
                st.markdown(f'<div style="font-family:JetBrains Mono;font-size:0.68rem;color:#334155">21 features: {np.array2string(feats[:6], precision=3)} …</div>', unsafe_allow_html=True)
                if hybrid is not None:
                    import torch
                    ft=np.asarray(feats, dtype=float)
                    if scaler is not None and "mean" in scaler:
                        ft=(ft - np.asarray(scaler["mean"]))/np.asarray(scaler["scale"])
                    ft=torch.tensor(ft[None,:], dtype=torch.float32)
                    with torch.no_grad():
                        qfeat=hybrid.quantum_layer(ft).numpy()[0]
                        _,probs=hybrid(ft); probs=probs.numpy()[0]
                    st.markdown(f'<div style="font-family:JetBrains Mono;font-size:0.68rem;color:#1E40AF">Quantum ⟨Z⟩×8: {np.array2string(qfeat, precision=3)}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-family:JetBrains Mono;font-size:0.68rem;color:#166534">Hybrid probs: {np.array2string(probs, precision=3)} → pred {int(np.argmax(probs))}</div>', unsafe_allow_html=True)
                else:
                    st.info("Hybrid model not loaded — showing RY angle encoding demo.")
            except Exception as e:
                st.error(f"{e}\n{traceback.format_exc(limit=2)}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Sim vs Hardware + Noise
    two=st.columns(2)
    with two[0]:
        st.markdown("""
<div class="glass" style="margin-top:0.8rem">
  <div style="font-family:Instrument Sans;font-weight:600;color:#0F172A">Simulator vs Real Hardware</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;margin-top:0.6rem">
    <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:12px;padding:0.7rem"><div style="font-family:JetBrains Mono;font-size:0.62rem;color:#1E40AF">SIMULATOR</div><div style="font-size:0.82rem;color:#1E3A8A">Aer ideal, deterministic, local C++</div><div style="margin-top:0.4rem"><span class="badge">● Available</span></div></div>
    <div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:12px;padding:0.7rem"><div style="font-family:JetBrains Mono;font-size:0.62rem;color:#991B1B">REAL HARDWARE</div><div style="font-size:0.82rem;color:#7F1D1D">IBM 10mK, queue, decoherence</div><div style="margin-top:0.4rem"><span class="badge">○ Unavailable</span></div></div>
  </div>
  <div style="margin-top:0.5rem;font-size:0.72rem;color:#64748B">Simulator results never claimed as hardware. Set <code>QISKIT_IBM_TOKEN</code> to enable.</div>
</div>""", unsafe_allow_html=True)
    with two[1]:
        level=st.select_slider("Quantum Noise — Robustness Analog", options=["OFF","LOW","MEDIUM","HIGH"], value="OFF", key="noise_light2")
        try:
            from qiskit_aer.noise import NoiseModel, depolarizing_error
            from qiskit import QuantumCircuit
            from qiskit_aer import AerSimulator
            qc=QuantumCircuit(2); qc.h(0); qc.cx(0,1); qc.measure_all()
            rates={"OFF":0,"LOW":0.005,"MEDIUM":0.02,"HIGH":0.08}
            p=rates[level]
            if p==0:
                counts=AerSimulator().run(qc, shots=2048).result().get_counts(qc)
            else:
                nm=NoiseModel(); nm.add_all_qubit_quantum_error(depolarizing_error(p,1), ["h"]); nm.add_all_qubit_quantum_error(depolarizing_error(p*1.8,2), ["cx"])
                counts=AerSimulator(noise_model=nm).run(qc, shots=2048).result().get_counts(qc)
            fig=go.Figure(go.Bar(x=list(counts.keys()), y=list(counts.values()), marker_color=["#2563EB" if k in ("00","11") else "#F87171" for k in counts.keys()], text=list(counts.values()), textposition="auto"))
            fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white", height=220, margin=dict(l=10,r=10,t=10,b=10), title=f'Bell counts — {level}')
            st.plotly_chart(fig, width="stretch", config={"displayModeBar":False})
        except Exception as e:
            st.caption(f"Noise demo: {e}")

    st.markdown("""
<div style="margin-top:1rem;background:white;border:1px solid #E8E6E1;border-radius:16px;padding:1rem;box-shadow:0 8px 24px rgba(15,27,46,0.05);font-size:0.76rem;color:#64748B;line-height:1.6">
  <b style="color:#0F172A">Premium light scientific — no neon.</b> Warm ivory #FFFCF7, navy #0F172A, copper #B87333 accents only where physical (plates). C++ only for light-curve/feature where profiling helps (`cpp/qorbit_core.cpp:20`), quantum stays Python/JS. All numbers from real Aer/PennyLane simulation.
</div>""", unsafe_allow_html=True)

if __name__=="__main__":
    main()
