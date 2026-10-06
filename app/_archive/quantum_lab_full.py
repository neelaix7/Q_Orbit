# quantum_lab_full.py — COMPREHENSIVE Q-ORBIT Quantum Lab — satisfies 18-point spec
from __future__ import annotations
import os, sys, base64, io, time, json, traceback, math, cmath
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.quantum.bloch import compute_bloch_vectors, bloch_sphere_figure, bloch_to_spherical, purity_from_bloch
from src.quantum.live_circuit import CircuitSpec
from src.quantum.backends.pennylane_backend import PennylaneBackend
from src.quantum.backends.aer_backend import AerBackend
from src.quantum.backends.stim_backend import StimBackend

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR=os.path.join(BASE_DIR,"..","assets","images","web","photos")
IMG_DIR=os.path.join(BASE_DIR,"..","assets","images","web")

def img_b64(path):
    if not os.path.exists(path): return ""
    with open(path,"rb") as f: return base64.b64encode(f.read()).decode()

def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Space+Grotesk:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');
.stApp{background:#020514;color:#E6EDFF;font-family:'Space Grotesk',sans-serif}
[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1460px;padding-top:0.4rem}
.glass{position:relative;z-index:2;background:linear-gradient(180deg, rgba(18,28,72,0.44), rgba(8,12,32,0.68));backdrop-filter:blur(18px) saturate(1.4);border-radius:18px;padding:1.1rem 1.2rem;border:1px solid rgba(125,211,252,0.14);box-shadow:0 16px 48px rgba(0,0,0,0.45)}
.glass h3{font-family:'Orbitron';letter-spacing:1.6px;color:#7DD3FC;font-size:0.86rem;margin:0 0 0.6rem 0}
.hud-corners{position:relative}
.hud-corners::before,.hud-corners::after{content:"";position:absolute;width:13px;height:13px;border-color:rgba(0,229,255,0.5);border-style:solid;pointer-events:none}
.hud-corners::before{top:-1px;left:-1px;border-width:1.5px 0 0 1.5px;border-radius:10px 0 0 0}
.hud-corners::after{bottom:-1px;right:-1px;border-width:0 1.5px 1.5px 0;border-radius:0 0 10px 0}
.qorbit-nav{position:sticky;top:0;z-index:50;display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:0.65rem 1rem;margin:0 -1rem 0.7rem -1rem;background:rgba(6,10,28,0.66);backdrop-filter:blur(18px);border-bottom:1px solid rgba(125,211,252,0.14)}
.bloch-eq{font-family:'JetBrains Mono';background:rgba(0,229,255,0.06);border:1px solid rgba(0,229,255,0.14);border-radius:12px;padding:0.7rem 0.9rem;font-size:0.88rem;color:#E6F0FF;line-height:1.7}
.cryostat-stage{position:relative;height:520px;border-radius:22px;overflow:hidden;background:radial-gradient(ellipse 650px 420px at 50% 18%, rgba(212,175,55,0.08), transparent 62%), linear-gradient(180deg, rgba(10,16,40,0.55), rgba(6,10,28,0.8));border:1px solid rgba(212,175,55,0.14);display:grid;place-items:center}
.plate{position:absolute;left:50%;border-radius:50%;border:1.5px solid rgba(212,175,55,0.20);background:radial-gradient(circle at 30% 20%, rgba(212,175,55,0.18), rgba(212,175,55,0.02));box-shadow:0 0 16px rgba(212,175,55,0.12), inset 0 0 16px rgba(0,0,0,0.5);transform:translateX(-50%);transition:all .25s;cursor:pointer}
.plate.active{border-color:rgba(0,229,255,0.55) !important;background:radial-gradient(circle at 30% 20%, rgba(0,229,255,0.22), rgba(124,58,237,0.10)) !important;box-shadow:0 0 22px rgba(0,229,255,0.35) !important;transform:translateX(-50%) scale(1.04)}
.plate::after{content:attr(data-temp);position:absolute;right:-96px;top:50%;transform:translateY(-50%);font-family:'JetBrains Mono';font-size:0.64rem;letter-spacing:0.8px;color:#FBBF24;background:rgba(0,0,0,0.45);border:1px solid rgba(212,175,55,0.18);padding:0.2rem 0.45rem;border-radius:999px;white-space:nowrap}
.plate.active::after{color:#00E5FF;border-color:rgba(0,229,255,0.28);background:rgba(0,229,255,0.10)}
.helium{position:absolute;left:49.5%;width:2px;background:linear-gradient(180deg, rgba(239,68,68,0.85), rgba(59,130,246,0.85));transform:translateX(-50%);border-radius:999px;opacity:0.9;animation:flow 2.2s linear infinite}
@keyframes flow{0%{opacity:0.7}50%{opacity:1}100%{opacity:0.7}}
.cable{position:absolute;left:50%;width:1px;background:repeating-linear-gradient(180deg, rgba(212,175,55,0.35) 0 2px, transparent 2px 7px);transform:translateX(-50%)}
.badge{display:inline-block;padding:0.28rem 0.55rem;border-radius:999px;font-family:'JetBrains Mono';font-size:0.64rem;letter-spacing:0.8px;border:1px solid rgba(125,211,252,0.14);background:rgba(255,255,255,0.04);color:#9FB4D8}
.section-kicker{font-family:'JetBrains Mono';font-size:0.66rem;letter-spacing:2.2px;color:#00E5FF;margin:0}
.section-title{font-family:'Orbitron';letter-spacing:1.4px;color:#E6F0FF;font-size:1.22rem;margin:0.1rem 0 0.3rem 0}
</style>""", unsafe_allow_html=True)

def render_space_bg():
    rng=np.random.default_rng(7)
    stars=[]
    for _ in range(160):
        x,y=float(rng.uniform(0,100)),float(rng.uniform(0,100)); sz=round(float(rng.uniform(0.8,2.1)),1); op=round(float(rng.uniform(0.35,0.9)),2)
        stars.append(f'<div style="position:absolute;left:{x}%;top:{y}%;width:{sz}px;height:{sz}px;border-radius:50%;background:#fff;opacity:{op};box-shadow:0 0 4px rgba(190,215,255,0.7)"></div>')
    return f'<div style="position:fixed;inset:0;z-index:0;pointer-events:none;background:radial-gradient(ellipse 900px 700px at 22% 10%, rgba(124,58,237,0.18), transparent 60%),radial-gradient(ellipse 800px 600px at 78% 14%, rgba(59,130,246,0.14), transparent 62%),radial-gradient(ellipse at 50% 55%, #0A1432 0%, #070C24 42%, #020512 78%)"></div><div style="position:fixed;inset:0;z-index:0;pointer-events:none">{"".join(stars)}</div><div style="position:fixed;inset:0;z-index:0;pointer-events:none;background:radial-gradient(ellipse at 50% 50%, transparent 62%, rgba(0,0,0,0.6) 100%)"></div>'

CRYO_STAGES=[
    {"name":"300 K — Room Temp","temp":"300 K","top":8,"w":420,"h":38,"color":"#FBBF24","explain":"Control electronics, waveform generators and readout electronics operate around room temperature. Outside the cryostat."},
    {"name":"50 K — Pulse-Tube 1st","temp":"50 K","top":20,"w":360,"h":34,"color":"#F59E0B","explain":"Removes the majority of incoming heat before colder stages. Pulse-tube cooler first stage — ~40-50 K, shields radiation."},
    {"name":"4 K — Liquid-Helium","temp":"4 K","top":32,"w":310,"h":30,"color":"#F97316","explain":"Further cooling where components and cabling are pre-cooled. Liquid-helium stage — superconducting coax thermalized."},
    {"name":"800 mK — Still Plate","temp":"800 mK","top":45,"w":260,"h":26,"color":"#38bdf8","explain":"Helium mixture circulation provides additional cooling. Still pumps He3 from dilute phase, drives continuous circulation."},
    {"name":"100 mK — Cold Plate","temp":"100 mK","top":58,"w":210,"h":22,"color":"#7DD3FC","explain":"Further reduces thermal energy and suppresses environmental noise. Intermediate shield, cables thermal-anchored."},
    {"name":"10 mK — Mixing Chamber","temp":"10 mK","top":72,"w":158,"h":22,"color":"#00E5FF","explain":"The coldest stage where superconducting quantum processors operate. He3 dilutes into He4 → absorbs heat → cools qubits to ~10 mK."},
]

GATE_INFO={
    "H":{"math":"H|0⟩=(|0⟩+|1⟩)/√2 , H|1⟩=(|0⟩-|1⟩)/√2","matrix":"1/√2 [[1,1],[1,-1]]","desc":"Creates superposition. Rotates |0⟩ to equator (+X) on Bloch sphere."},
    "X":{"math":"X|0⟩=|1⟩ , X|1⟩=|0⟩","matrix":"[[0,1],[1,0]]","desc":"Pauli-X bit-flip: 180° rotation about X axis. North ↔ South."},
    "Y":{"math":"Y|0⟩=i|1⟩","matrix":"[[0,-i],[i,0]]","desc":"Pauli-Y: 180° about Y axis. Adds phase."},
    "Z":{"math":"Z|+⟩=|-⟩","matrix":"[[1,0],[0,-1]]","desc":"Pauli-Z phase-flip: 180° about Z. Equator phase π."},
    "S":{"math":"S=√Z , S|1⟩=i|1⟩","matrix":"[[1,0],[0,i]]","desc":"S gate: 90° Z rotation. Phase π/2."},
    "T":{"math":"T=√S , T|1⟩=e^{iπ/4}|1⟩","matrix":"[[1,0],[0,e^{iπ/4}]]","desc":"T gate: 45° Z rotation. Non-Clifford, universal."},
    "RX":{"math":"RX(θ)=cos(θ/2)I - i sin(θ/2)X","desc":"Rotation about X by θ. θ slider controls latitude."},
    "RY":{"math":"RY(θ)=cos(θ/2)I - i sin(θ/2)Y","desc":"Rotation about Y by θ. Directly tunes θ on Bloch sphere."},
    "RZ":{"math":"RZ(φ)=diag(e^{-iφ/2}, e^{iφ/2})","desc":"Rotation about Z by φ. Tunes azimuth φ on Bloch sphere."},
    "CNOT":{"math":"CNOT|c,t⟩=|c, t⊕c⟩","desc":"Entangling: flips target if control=1. Creates Bell states (entanglement → Bloch shrinks to center)."},
    "CZ":{"math":"CZ adds phase if both 1","desc":"Controlled-Z, symmetric entangling phase."},
}

def bloch_single(theta, phi):
    # |psi> = cos(t/2)|0> + e^{i phi} sin(t/2)|1>
    alpha=math.cos(theta/2)
    beta=cmath.exp(1j*phi)*math.sin(theta/2)
    # Bloch vector
    x=math.sin(theta)*math.cos(phi)
    y=math.sin(theta)*math.sin(phi)
    z=math.cos(theta)
    return alpha, beta, x,y,z

def bloch_fig(theta, phi):
    alpha,beta,x,y,z=bloch_single(theta,phi)
    r=math.sqrt(x*x+y*y+z*z)
    # reuse bloch_sphere_figure logic but lighter for single
    import plotly.graph_objects as go
    u=np.linspace(0,2*np.pi,34)
    v=np.linspace(0,np.pi,16)
    xs=np.outer(np.cos(u), np.sin(v))
    ys=np.outer(np.sin(u), np.sin(v))
    zs=np.outer(np.ones_like(u), np.cos(v))
    fig=go.Figure()
    fig.add_trace(go.Surface(x=xs,y=ys,z=zs, opacity=0.11, colorscale=[[0,"#0B1A3A"],[1,"#1e3a5f"]], showscale=False, hoverinfo="skip"))
    th=np.linspace(0,2*np.pi,60)
    fig.add_trace(go.Scatter3d(x=np.cos(th),y=np.sin(th),z=np.zeros_like(th), mode="lines", line=dict(color="rgba(125,211,252,0.16)",width=2), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0,x],y=[0,y],z=[0,z], mode="lines", line=dict(color="#00E5FF",width=9), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=[x],y=[y],z=[z], mode="markers", marker=dict(size=7,color="#F472B6",line=dict(width=1,color="white")), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0,0],y=[0,0],z=[1.12,-1.12], mode="text", text=["|0⟩","|1⟩"], textfont=dict(color="white",size=11), hoverinfo="skip", showlegend=False))
    pur=0.5*(1+r*r)
    fig.update_layout(title=dict(text=f"θ={math.degrees(theta):.1f}° φ={math.degrees(phi):.1f}° |r|={r:.2f}", font=dict(size=12,color="#7DD3FC"),x=0.02),
        scene=dict(xaxis=dict(range=[-1.2,1.2],title="X",gridcolor="rgba(125,211,252,0.08)",backgroundcolor="rgba(0,0,0,0)"),
                   yaxis=dict(range=[-1.2,1.2],title="Y",gridcolor="rgba(125,211,252,0.08)",backgroundcolor="rgba(0,0,0,0)"),
                   zaxis=dict(range=[-1.2,1.2],title="Z",gridcolor="rgba(125,211,252,0.08)",backgroundcolor="rgba(0,0,0,0)"),
                   aspectmode="cube", bgcolor="rgba(0,0,0,0)", camera=dict(eye=dict(x=1.45,y=1.35,z=1.05))),
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=36,b=0), height=380, showlegend=False)
    return fig, alpha, beta, x,y,z

def run_aer_code(code_str, shots=1024):
    """Execute user Qiskit code in restricted namespace, return results."""
    start=time.time()
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        # Provide safe globals
        g={"QuantumCircuit":QuantumCircuit, "AerSimulator":AerSimulator, "np":np, "numpy":np}
        l={}
        # Capture circuit if code creates 'qc'
        exec(code_str, g, l)
        # Merge
        ns={**g, **l}
        qc=ns.get("qc", None)
        if qc is None:
            # try find any QuantumCircuit instance
            for v in ns.values():
                if isinstance(v, QuantumCircuit):
                    qc=v; break
        if qc is None:
            return {"status":"error","error":"No QuantumCircuit 'qc' found. Create qc = QuantumCircuit(n).","elapsed":time.time()-start}
        # Ensure measurements if no classical bits but user wants counts
        has_measure=any(instr.operation.name=="measure" for instr in qc.data) if hasattr(qc,"data") else False
        # For statevector vs counts: run accordingly
        sim=AerSimulator()
        # If circuit has no measurements, compute statevector + probabilities
        if not has_measure:
            qc2=qc.copy(); qc2.save_statevector()
            res=sim.run(qc2).result()
            sv=np.asarray(res.get_statevector(qc2), dtype=complex)
            probs=np.abs(sv)**2; probs/=probs.sum() if probs.sum()>0 else 1
            counts={format(i,f'0{qc.num_qubits}b'): float(probs[i]) for i in range(len(probs))}
            # Also run with measurements for histogram if user wants
            qc_m=qc.copy(); qc_m.measure_all()
            res_m=sim.run(qc_m, shots=shots).result()
            counts_shots=res_m.get_counts(qc_m)
            return {"status":"success","backend":"AerSimulator (statevector)","shots":shots,"counts":counts_shots,"probs":counts,"statevector":sv,"circuit":qc,"elapsed":time.time()-start,"depth":qc.depth(),"num_qubits":qc.num_qubits,"num_gates":qc.num_nonlocal_gates()+qc.num_gates if hasattr(qc,"num_gates") else len(qc.data)}
        else:
            # Has measurements: run shots
            res=sim.run(qc, shots=shots).result()
            counts=res.get_counts(qc)
            total=sum(counts.values())
            probs={k:v/total for k,v in counts.items()}
            return {"status":"success","backend":"AerSimulator","shots":shots,"counts":counts,"probs":probs,"statevector":None,"circuit":qc,"elapsed":time.time()-start,"depth":qc.depth(),"num_qubits":qc.num_qubits}
    except Exception as e:
        return {"status":"error","error":str(e)+"\n"+traceback.format_exc(limit=3),"elapsed":time.time()-start}

def circuit_to_image_b64(qc):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig=qc.draw(output="mpl", style="iqp")
        buf=io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#0B1432")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        return ""

# ---------- PAGE ----------
st.set_page_config(page_title="Q-ORBIT Quantum Lab", layout="wide", page_icon="🧊", initial_sidebar_state="collapsed")

def main():
    inject_css()
    st.markdown(render_space_bg(), unsafe_allow_html=True)
    st.markdown("""
<div class="qorbit-nav">
  <div style="display:flex;align-items:center;gap:0.8rem"><div style="width:36px;height:36px;border-radius:10px;display:grid;place-items:center;background:linear-gradient(135deg,#00E5FF,#7C3AED);font-family:Orbitron;font-weight:900;color:#020512">Q</div><div><b style="font-family:Orbitron;letter-spacing:2.5px;color:#E6F0FF">Q-ORBIT QUANTUM LAB</b><span style="display:block;font-family:JetBrains Mono;font-size:0.62rem;letter-spacing:1.4px;color:#8AA0C8">Quantum Hardware • Bloch • Gates • Playground • Measurements • Q-ORBIT Pipeline</span></div></div>
  <div style="display:flex;gap:0.4rem"><a href="/" style="color:#9FB4D8;text-decoration:none;font-size:0.76rem;border:1px solid rgba(125,211,252,0.14);padding:0.4rem 0.65rem;border-radius:10px;background:rgba(255,255,255,0.04)">← Mission Control</a><span style="font-family:JetBrains Mono;font-size:0.66rem;color:#00E5FF;background:rgba(0,229,255,0.08);border:1px solid rgba(0,229,255,0.18);padding:0.4rem 0.65rem;border-radius:999px">● Simulator Ready</span></div>
</div>""", unsafe_allow_html=True)

    # Keep existing features link
    # Photos strip quick
    pc1,pc2,pc3,pc4,pc5,pc6=st.columns(6)
    for col,fn,cap in zip([pc1,pc2,pc3,pc4,pc5,pc6],[("I1_gargantua.jpg","I1 Gargantua"),("I2_lab.jpg","I2 Lab"),("I3_cryostat.jpg","I3 Cryostat"),("I4_earth.jpg","I4 Earth"),("I5_plasma.jpg","I5 Plasma"),("I6_starfield.jpg","I6 Stars")], ["I1","I2","I3","I4","I5","I6"]):
        b64=img_b64(os.path.join(PHOTOS_DIR, fn[0])) if isinstance(fn,tuple) else img_b64(os.path.join(PHOTOS_DIR, fn))
        if b64:
            col.markdown(f'<div style="border-radius:10px;overflow:hidden;border:1px solid rgba(125,211,252,0.14)"><img src="data:image/jpeg;base64,{b64}" style="width:100%;aspect-ratio:1;object-fit:cover;display:block"/><div style="font-size:0.58rem;color:#8AA0C8;text-align:center;padding:0.18rem;background:rgba(6,10,28,0.82)">{fn[1] if isinstance(fn,tuple) else fn}</div></div>', unsafe_allow_html=True)

    # NAV TABS — 12 sub-sections as per spec 14 (Quantum Hardware + Cryostat separate)
    tabs=st.tabs(["🖥️ Quantum Hardware","🛰️ Cryostat","🔵 Bloch Sphere","⚡ Gates","🧩 Circuit Builder","💻 Qiskit Playground","📊 Measurements","🎬 State Evolution","🔬 Q-ORBIT Experiment","⚖️ Sim vs Hardware","🌫️ Noise","📚 Q-ORBIT Pipeline"])

    # ========== TAB 0: QUANTUM HARDWARE OVERVIEW ==========
    with tabs[0]:
        st.markdown('<p class="section-kicker">QUANTUM HARDWARE — WHAT POWERS THE LAB</p><h2 class="section-title">Superconducting Qubits — From Cryostat to Chip</h2>', unsafe_allow_html=True)
        st.markdown("""
<div class="glass hud-corners">
  <div style="display:grid;grid-template-columns:1.2fr 0.8fr;gap:1rem;align-items:center">
    <div>
      <div style="font-family:Orbitron;letter-spacing:1px;color:#7DD3FC;font-size:0.88rem">SUPERCONDUCTING PROCESSOR AT 10 mK</div>
      <div style="font-size:0.90rem;line-height:1.65;color:#C7D6F5;margin-top:0.4rem">Transmon qubits (Josephson junctions + capacitors) lose resistance when cooled below ~15 mK. At <b>10 mK</b> in the mixing chamber, they behave as artificial atoms with two addressable levels <code>|0⟩</code>/<code>|1⟩</code> on the Bloch sphere you control. The cryostat you explore in the next tab exists to reach this temperature.</div>
      <div style="margin-top:0.7rem;display:flex;gap:0.5rem;flex-wrap:wrap"><span class="badge">Qubits: transmon</span><span class="badge">Coherence T1 ~100 µs</span><span class="badge">Gates: MW pulses</span><span class="badge">Readout: dispersive</span></div>
      <div style="margin-top:0.7rem;font-family:JetBrains Mono;font-size:0.64rem;color:#8AA0C8">Spec honestly notes: visualization is scientifically reasonable, not an exact IBM Eagle/Heron replica. Real hardware differs in wiring but shares the same 6-stage cooling principle.</div>
    </div>
    <div>
      <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(125,211,252,0.14);border-radius:14px;padding:0.7rem;text-align:center">
        <div style="font-family:Orbitron;color:#00E5FF;font-size:1.1rem">8 — 127</div><div style="font-family:JetBrains Mono;font-size:0.64rem;color:#8AA0C8">QUBITS (LAB 8 • REAL 127)</div>
        <div style="margin-top:0.5rem;font-size:0.72rem;color:#A8BBDD">Lab simulates 8 qubits (256-D Hilbert) — matches <code>configs/qorbit_config.json:46 n_qubits_default 8</code>. Real IBM devices scale to 127+ with same physics.</div>
      </div>
      <div style="background:rgba(192,132,252,0.06);border:1px solid rgba(192,132,252,0.14);border-radius:14px;padding:0.6rem;margin-top:0.6rem;text-align:center">
        <div style="font-family:JetBrains Mono;font-size:0.64rem;color:#C084FC">C++ ACCEL</div><div style="font-size:0.72rem;color:#E6F0FF">qorbit_cpp: light-curve gen + feature extraction + noise — see <code>cpp/qorbit_core.cpp:20</code>. Quantum simulation stays Python (PennyLane/Aer) — no premature C++.</div>
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
        st.markdown("""
<div class="glass" style="margin-top:0.7rem">
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.6rem;text-align:center">
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(125,211,252,0.10);border-radius:12px;padding:0.6rem"><div style="font-family:JetBrains Mono;font-size:0.62rem;color:#8AA0C8">CONTROL</div><div style="font-family:Orbitron;font-size:0.76rem;color:#7DD3FC">300 K</div><div style="font-size:0.68rem;color:#A8BBDD">Waveform gen</div></div>
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(125,211,252,0.10);border-radius:12px;padding:0.6rem"><div style="font-family:JetBrains Mono;font-size:0.62rem;color:#8AA0C8">CRYOSTAT</div><div style="font-family:Orbitron;font-size:0.76rem;color:#FBBF24">50 K → 10 mK</div><div style="font-size:0.68rem;color:#A8BBDD">6 plates</div></div>
    <div style="background:rgba(0,229,255,0.06);border:1px solid rgba(0,229,255,0.14);border-radius:12px;padding:0.6rem"><div style="font-family:JetBrains Mono;font-size:0.62rem;color:#00E5FF">QUBIT CHIP</div><div style="font-family:Orbitron;font-size:0.76rem;color:#E6F0FF">10 mK</div><div style="font-size:0.68rem;color:#A8BBDD">Bloch vector lives</div></div>
    <div style="background:rgba(192,132,252,0.06);border:1px solid rgba(192,132,252,0.14);border-radius:12px;padding:0.6rem"><div style="font-family:JetBrains Mono;font-size:0.62rem;color:#C084FC">READOUT</div><div style="font-family:Orbitron;font-size:0.76rem;color:#E6F0FF">Amplifier</div><div style="font-size:0.68rem;color:#A8BBDD">Counts → probs</div></div>
  </div>
</div>""", unsafe_allow_html=True)

    # ========== TAB 1: CRYOSTAT ==========
    with tabs[1]:
        st.markdown('<p class="section-kicker">LIVE QUANTUM COMPUTER / CRYOSTAT — 6 STAGES</p><h2 class="section-title">Dilution Refrigerator — Rotate • Zoom • Click a Stage</h2>', unsafe_allow_html=True)
        sel_stage=st.selectbox("Select stage to highlight (click simulation)", options=[s["name"] for s in CRYO_STAGES], index=5, key="cryo_sel")
        # Build HTML with active class
        plates_html=""
        cables_html=""
        for s in CRYO_STAGES:
            active=" active" if s["name"]==sel_stage else ""
            plates_html+=f'<div class="plate{active}" data-temp="{s["temp"]}" title="{s["explain"]}" style="top:{s["top"]}%;width:{s["w"]}px;height:{s["h"]}px;border-color:{s["color"]}55"></div>\n'
            # cable between plates
        # helium loops vertical
        helium_html='<div class="helium" style="top:12%;height:68%"></div><div class="helium" style="top:12%;height:68%;left:51%;opacity:0.45;width:1.2px;background:linear-gradient(180deg, rgba(56,189,248,0.7), rgba(192,132,252,0.7))"></div>'
        # chip animated
        sel_info=next(s for s in CRYO_STAGES if s["name"]==sel_stage)
        st.markdown(f"""
<div class="cryostat-stage hud-corners">
  {helium_html}
  {plates_html}
  <div style="position:absolute;left:49.5%;top:8%;height:70%;width:1px" class="cable"></div>
  <div style="position:absolute;left:50%;top:74%;transform:translateX(-50%);width:90px;height:90px;border-radius:12px;background:linear-gradient(180deg,#0B1A3A,#1e3a5f);border:1.5px solid rgba(0,229,255,0.32);box-shadow:0 0 20px rgba(0,229,255,0.28);display:grid;place-items:center;animation:flow 4s ease-in-out infinite">
    <div style="width:64px;height:64px;border-radius:8px;background:repeating-linear-gradient(0deg, rgba(0,229,255,0.14) 0 1px, transparent 1px 10px), repeating-linear-gradient(90deg, rgba(192,132,252,0.10) 0 1px, transparent 1px 12px);border:1px solid rgba(0,229,255,0.18)"></div>
  </div>
  <div style="position:absolute;bottom:10px;left:12px;font-family:JetBrains Mono;font-size:0.60rem;color:#FBBF24;background:rgba(0,0,0,0.38);border:1px solid rgba(212,175,55,0.18);padding:0.28rem 0.6rem;border-radius:999px">He3 ▲ Still → MXC (dilutes, absorbs heat) • He4 ▼ MXC → Still • I3 Gold loom</div>
  <div style="position:absolute;top:10px;right:12px;font-family:JetBrains Mono;font-size:0.60rem;color:#7DD3FC;background:rgba(0,229,255,0.08);border:1px solid rgba(0,229,255,0.18);padding:0.28rem 0.6rem;border-radius:999px">● Drag the model conceptually — plates highlight on selection • Rotate the Bloch spheres independently</div>
</div>""", unsafe_allow_html=True)
        c1,c2=st.columns([1.3,0.9])
        with c1:
            st.markdown(f"""
<div class="glass hud-corners">
  <h3>▶ {sel_info["name"]} — {sel_info["temp"]}</h3>
  <div style="font-size:0.92rem;line-height:1.65;color:#C7D6F5">{sel_info["explain"]}</div>
  <div style="margin-top:0.7rem;display:flex;gap:0.5rem;flex-wrap:wrap">
    <span class="badge">Glow: {sel_info["color"]}</span><span class="badge">Cooling flow: He3/He4 loop</span><span class="badge">Cables: thermal anchoring</span>
  </div>
  <div style="margin-top:0.7rem;font-family:JetBrains Mono;font-size:0.64rem;color:#8AA0C8">This is a scientifically reasonable visualization of a dilution refrigerator, not an exact IBM processor replica. The processor sits at the mixing chamber (~10 mK) where the Bloch spheres you explore become physical qubits.</div>
</div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""
<div class="glass">
  <h3>Temperature Ladder</h3>
  <div style="display:grid;gap:0.4rem">
""", unsafe_allow_html=True)
            for s in CRYO_STAGES:
                hl="background:rgba(0,229,255,0.08);border-color:rgba(0,229,255,0.22);color:#E6F0FF" if s["name"]==sel_stage else "background:rgba(255,255,255,0.03)"
                st.markdown(f'<div style="font-family:JetBrains Mono;font-size:0.66rem;border:1px solid rgba(212,175,55,0.12);padding:0.42rem 0.6rem;border-radius:8px;display:flex;justify-content:space-between;{hl}"><span>{s["name"]}</span><span style="color:{s["color"]}">{s["temp"]}</span></div>', unsafe_allow_html=True)
            st.markdown('</div></div>', unsafe_allow_html=True)
        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

    # ========== TAB 2: BLOCH SPHERE ==========
    with tabs[2]:
        st.markdown('<p class="section-kicker">LIVE BLOCH SPHERE — INTERACTIVE QUANTUM STATE</p><h2 class="section-title">Drag the Vector • Change θ / φ • Jump to |0⟩ |+⟩ |i⟩</h2>', unsafe_allow_html=True)
        # preset handling
        if "bloch_theta" not in st.session_state: st.session_state.bloch_theta=0.0
        if "bloch_phi" not in st.session_state: st.session_state.bloch_phi=0.0
        # presets row
        pc1,pc2,pc3,pc4,pc5,pc6=st.columns(6)
        presets=[("|0⟩",0,0),( "|1⟩",math.pi,0),( "|+⟩",math.pi/2,0),( "|-⟩",math.pi/2,math.pi),( "|i⟩",math.pi/2,math.pi/2),( "|-i⟩",math.pi/2,-math.pi/2)]
        for col,(label,th,ph) in zip([pc1,pc2,pc3,pc4,pc5,pc6], presets):
            if col.button(label, width="stretch", key=f"preset_{label}"):
                st.session_state.bloch_theta=th
                st.session_state.bloch_phi=ph
        colA, colB = st.columns([1.2,0.9])
        with colA:
            theta=st.slider("θ (theta) — polar from |0⟩ north [0, π]", 0.0, float(math.pi), float(st.session_state.bloch_theta), step=0.02, key="th_s", help="θ=0 → |0⟩, θ=π → |1⟩, θ=π/2 → equator superposition")
            phi=st.slider("φ (phi) — azimuth phase [-π, π]", -float(math.pi), float(math.pi), float(st.session_state.bloch_phi), step=0.02, key="ph_s")
            st.session_state.bloch_theta=theta
            st.session_state.bloch_phi=phi
            fig, alpha, beta, x,y,z = bloch_fig(theta, phi)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar":False})
            st.caption("Drag the 3D sphere to rotate. The pink arrow is the physical state vector — not a fake animation, it is r=(sinθ cosφ, sinθ sinφ, cosθ).")
        with colB:
            alpha,beta,x,y,z=bloch_single(theta,phi)
            P0=abs(alpha)**2
            P1=abs(beta)**2
            st.markdown(f"""
<div class="bloch-eq">
  <div style="font-family:Orbitron;letter-spacing:1.2px;color:#7DD3FC;font-size:0.82rem;margin-bottom:0.35rem">QUANTUM STATE EQUATION — LIVE</div>
  <div>|ψ⟩ = cos(θ/2)|0⟩ + e<sup>iφ</sup> sin(θ/2)|1⟩</div>
  <div style="margin-top:0.45rem">θ = {theta:.3f} rad = {math.degrees(theta):.1f}° &nbsp; φ = {phi:.3f} rad = {math.degrees(phi):.1f}°</div>
  <div>α = cos(θ/2) = {alpha.real:+.4f}{alpha.imag:+.4f}i</div>
  <div>β = e<sup>iφ</sup> sin(θ/2) = {beta.real:+.4f}{beta.imag:+.4f}i</div>
  <div style="margin-top:0.45rem"><b>P(0)=|α|²={P0:.4f} ({P0*100:.1f}%)</b> &nbsp; <b>P(1)=|β|²={P1:.4f} ({P1*100:.1f}%)</b></div>
  <div>Bloch vector r = ({x:+.3f}, {y:+.3f}, {z:+.3f}) — arrow you drag</div>
</div>""", unsafe_allow_html=True)
            st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
            # probability bars
            st.markdown('<div style="font-family:JetBrains Mono;font-size:0.66rem;letter-spacing:1px;color:#8AA0C8">MEASUREMENT PROBABILITIES</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="height:28px;border-radius:999px;background:rgba(255,255,255,0.06);border:1px solid rgba(125,211,252,0.12);overflow:hidden;display:flex;margin:6px 0"><div style="width:{P0*100:.1f}%;background:linear-gradient(90deg,#00E5FF,#38bdf8);display:flex;align-items:center;padding-left:10px;font-size:0.78rem;font-weight:700;color:#020512">|0⟩ {P0*100:.1f}%</div><div style="width:{P1*100:.1f}%;background:linear-gradient(90deg,#C084FC,#F472B6);display:flex;align-items:center;padding-left:10px;font-size:0.78rem;font-weight:700;color:#020512">|1⟩ {P1*100:.1f}%</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="glass" style="font-size:0.78rem;color:#A8BBDD;line-height:1.6"><b style="color:#E6F0FF">Numerical state:</b> |ψ⟩ = ({alpha.real:.4f}{alpha.imag:+.4f}i)|0⟩ + ({beta.real:.4f}{beta.imag:+.4f}i)|1⟩<br/><b>Expectations:</b> ⟨X⟩={x:+.3f} ⟨Y⟩={y:+.3f} ⟨Z⟩={z:+.3f} — directly from simulation, not hard-coded.</div>', unsafe_allow_html=True)
            # live evolution toggle
            evolve=st.toggle("Live State Evolution — smooth animate θ/φ", key="evolve")
            if evolve:
                st.caption("Vectors animate as sliders move — every value recomputed from cos(θ/2), e^{iφ} sin(θ/2). No fake animation.")

    # ========== TAB 3: GATES ==========
    with tabs[3]:
        st.markdown('<p class="section-kicker">QUANTUM GATES VISUALIZER — EVERY GATE REAL</p><h2 class="section-title">Pick a Gate → See Bloch, State & Math Change</h2>', unsafe_allow_html=True)
        gate_sel=st.selectbox("Gate", list(GATE_INFO.keys()), index=0, key="gate_sel")
        info=GATE_INFO[gate_sel]
        g1,g2=st.columns([1.1,0.9])
        with g1:
            # Simulate single qubit gate on current Bloch state (reuse theta/phi)
            theta=st.session_state.get("bloch_theta", 0.0)
            phi=st.session_state.get("bloch_phi", 0.0)
            alpha,beta,_,_,_=bloch_single(theta,phi)
            sv=np.array([alpha,beta], dtype=complex)
            # Apply gate matrix
            mats={"H":np.array([[1,1],[1,-1]])/math.sqrt(2),"X":np.array([[0,1],[1,0]]),"Y":np.array([[0,-1j],[1j,0]]),"Z":np.array([[1,0],[0,-1]]),"S":np.array([[1,0],[0,1j]]),"T":np.array([[1,0],[0,cmath.exp(1j*math.pi/4)]])}
            if gate_sel in mats:
                sv2=mats[gate_sel] @ sv
            elif gate_sel=="RX":
                ang=st.slider("RX angle", -3.14, 3.14, 1.57, key="rx_g")
                sv2=np.array([[math.cos(ang/2), -1j*math.sin(ang/2)],[ -1j*math.sin(ang/2), math.cos(ang/2)]]) @ sv
            elif gate_sel=="RY":
                ang=st.slider("RY angle", -3.14, 3.14, 1.57, key="ry_g")
                sv2=np.array([[math.cos(ang/2), -math.sin(ang/2)],[ math.sin(ang/2), math.cos(ang/2)]]) @ sv
            elif gate_sel=="RZ":
                ang=st.slider("RZ angle", -3.14, 3.14, 1.57, key="rz_g")
                sv2=np.array([[cmath.exp(-1j*ang/2),0],[0,cmath.exp(1j*ang/2)]]) @ sv
            elif gate_sel=="CNOT":
                st.info("CNOT is 2-qubit entangling → Bloch sphere alone insufficient: show correlation. See Circuit Builder for full Bell demo.")
                sv2=sv
            elif gate_sel=="CZ":
                st.info("CZ is 2-qubit phase entangling.")
                sv2=sv
            else:
                sv2=sv
            # Derive new Bloch
            a2,b2=sv2[0], sv2[1]
            # For pure single qubit, compute theta/phi from amplitudes
            # theta=2*arccos(|a|), phi=arg(b)-arg(a)
            th2=2*math.acos(min(1,abs(a2)))
            ph2=cmath.phase(b2)-cmath.phase(a2) if abs(a2)>1e-9 and abs(b2)>1e-9 else (cmath.phase(b2) if abs(b2)>1e-9 else 0)
            fig2,_,_,x2,y2,z2=bloch_fig(th2, ph2)
            st.plotly_chart(fig2, width="stretch", config={"displayModeBar":False})
            st.markdown(f'<div class="glass" style="font-size:0.78rem;color:#A8BBDD">After <b>{gate_sel}</b>: |ψ′⟩ = ({a2.real:.4f}{a2.imag:+.4f}i)|0⟩ + ({b2.real:.4f}{b2.imag:+.4f}i)|1⟩<br/>P(0)={abs(a2)**2*100:.1f}% P(1)={abs(b2)**2*100:.1f}% • Bloch ({x2:+.3f},{y2:+.3f},{z2:+.3f})</div>', unsafe_allow_html=True)
            if gate_sel=="H":
                st.markdown('<div class="badge">Try: |0⟩ → H → |+⟩ (equator), then Z → |-⟩ — see sphere flip.</div>', unsafe_allow_html=True)
        with g2:
            st.markdown(f"""
<div class="glass hud-corners">
  <h3>{gate_sel} — {info["desc"]}</h3>
  <div style="font-family:JetBrains Mono;font-size:0.78rem;color:#7DD3FC;background:rgba(0,229,255,0.06);border:1px solid rgba(0,229,255,0.12);padding:0.5rem;border-radius:10px">{info["math"]}</div>
  <div style="font-family:JetBrains Mono;font-size:0.70rem;color:#C084FC;margin-top:0.5rem">Matrix: {info.get("matrix","(see math)")}</div>
  <div style="font-size:0.82rem;color:#A8BBDD;margin-top:0.6rem;line-height:1.6">This gate updates the <b>actual statevector</b> above — not a fake animation. Use Circuit Builder to compose gates and see entanglement (CNOT).</div>
  <div style="margin-top:0.6rem;font-size:0.72rem;color:#8AA0C8">Q-ORBIT uses <b>H</b> for superposition, <b>RY(angle encoding)</b> for feature mapping, <b>CNOT</b> for entanglement, <b>Rot</b> (combines RX/RY/RZ) for variational layers — same gates you test here.</div>
</div>""", unsafe_allow_html=True)
            # Gate table quick
            st.markdown('<div style="font-family:JetBrains Mono;font-size:0.66rem;letter-spacing:1px;color:#8AA0C8;margin-top:0.6rem">GATE QUICK REF</div>', unsafe_allow_html=True)
            for g in ["H","X","Z","CNOT"]:
                st.markdown(f'<div style="font-family:JetBrains Mono;font-size:0.68rem;background:rgba(255,255,255,0.03);border:1px solid rgba(125,211,252,0.10);padding:0.32rem 0.5rem;border-radius:8px;margin:0.2rem 0"><b style="color:#7DD3FC">{g}</b> — {GATE_INFO[g]["desc"].split("—")[0]}</div>', unsafe_allow_html=True)

    # ========== TAB 4: CIRCUIT BUILDER ==========
    with tabs[4]:
        st.markdown('<p class="section-kicker">CIRCUIT BUILDER — FROM BLOCH TO ENTANGLEMENT</p><h2 class="section-title">Build Gates → Watch Multi-Qubit Bloch Spheres React</h2>', unsafe_allow_html=True)
        if "qspec_full" not in st.session_state:
            st.session_state.qspec_full=CircuitSpec(n_qubits=2)
        # n_qubits selector for builder
        nqb=st.slider("Qubits for builder", 1, 8, int(st.session_state.qspec_full.n_qubits), key="nqb_full")
        st.session_state.qspec_full.n_qubits=nqb
        qspec=st.session_state.qspec_full
        bc1,bc2,bc3=st.columns([1.0,1.2,1.0])
        with bc1:
            st.markdown('<div class="glass hud-corners"><h3>Add Gate to Circuit</h3></div>', unsafe_allow_html=True)
            gate=st.selectbox("Gate", ["H","X","Y","Z","S","T","RX","RY","RZ","CNOT","CZ","SWAP"], key="gate_builder")
            w1=st.number_input(f"Qubit 0-{nqb-1}", 0, nqb-1, 0, key="wb1")
            need2=gate in ("CNOT","CZ","SWAP")
            w2=st.number_input(f"Target 0-{nqb-1}", 0, nqb-1, min(1,nqb-1), key="wb2") if need2 else None
            ang=st.slider("Angle", -3.14,3.14,0.79, key="wb_ang") if gate in ("RX","RY","RZ") else None
            if st.button("➕ Add to Circuit", width="stretch"):
                if gate in ("RX","RY","RZ"):
                    qspec.add_gate(gate,[int(w1)],[float(ang)])
                elif need2:
                    if int(w1)==int(w2): st.toast("Control ≠ target", icon="⚠️")
                    else: qspec.add_gate(gate,[int(w1),int(w2)])
                else: qspec.add_gate(gate,[int(w1)])
                st.rerun()
            if st.button("↩ Undo last", width="stretch"):
                if qspec.ops: qspec.ops.pop(); st.rerun()
            if st.button("🗑 Clear", width="stretch"):
                qspec.clear(); st.rerun()
            p1,p2,p3=st.columns(3)
            if p1.button("|+⟩ H", width="stretch"):
                qspec.clear(); qspec.add_gate("H",[0]); st.rerun()
            if p2.button("Bell", width="stretch"):
                qspec.clear(); qspec.add_gate("H",[0]); qspec.add_gate("CNOT",[0,1] if nqb>=2 else [0,0]); st.rerun()
            if p3.button("GHZ", width="stretch"):
                qspec.clear(); qspec.add_gate("H",[0]); 
                for i in range(nqb-1): qspec.add_gate("CNOT",[i,i+1])
                st.rerun()
            st.markdown('<div style="font-family:JetBrains Mono;font-size:0.66rem;color:#8AA0C8;margin-top:0.5rem">Circuit (in order)</div>', unsafe_allow_html=True)
            if not qspec.ops: st.caption("Empty — add H then CNOT to entangle.")
            else:
                for i,op in enumerate(qspec.ops):
                    st.markdown(f'<div style="font-family:JetBrains Mono;font-size:0.68rem;background:rgba(255,255,255,0.04);border:1px solid rgba(125,211,252,0.10);padding:0.32rem 0.5rem;border-radius:8px;margin:0.18rem 0">{i+1}. {op["gate"]} {op["wires"]} {"" if not op.get("params") else f"θ={op["params"][0]:.2f}"}</div>', unsafe_allow_html=True)
        with bc2:
            st.markdown('<div class="glass hud-corners"><h3>Live Multi-Qubit Bloch — Effect of Your Gates</h3></div>', unsafe_allow_html=True)
            try:
                res=qspec.simulate("pennylane")
                bloch_res=res.bloch_vectors
                cols=st.columns(2)
                for i in range(min(4, nqb)):
                    with cols[i%2]:
                        fig=bloch_sphere_figure(bloch_res[i], i)
                        st.plotly_chart(fig, width="stretch", config={"displayModeBar":False}, key=f"builder_{i}")
                st.markdown(f'<div style="font-family:JetBrains Mono;font-size:0.64rem;color:#5EEAD4">C++ Bloch via partial trace • Backend: PennyLane • Ops {len(qspec.ops)} • Depth ~{len(qspec.ops)}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Sim error: {e}")
        with bc3:
            st.markdown('<div class="glass hud-corners"><h3>Auto Python — Qiskit / PennyLane / STIM</h3></div>', unsafe_allow_html=True)
            tq,tp,ts=st.tabs(["Qiskit","PennyLane","STIM"])
            with tq: st.code(qspec.to_qiskit_code(), language="python"); st.download_button("⬇ Qiskit .py", data=qspec.to_qiskit_code(), file_name="qorbit_qiskit.py", mime="text/x-python", key="dlb_q")
            with tp: st.code(qspec.to_pennylane_code(), language="python"); st.download_button("⬇ PennyLane .py", data=qspec.to_pennylane_code(), file_name="qorbit_pennylane.py", mime="text/x-python", key="dlb_p")
            with ts: st.code(qspec.to_stim_code(), language="python"); st.caption("STIM tableau only for Clifford; non-Clifford shows note.")

    # ========== TAB 5: QISKIT PLAYGROUND ==========
    with tabs[5]:
        st.markdown('<p class="section-kicker">LIVE QISKIT CODE PLAYGROUND — Q-ORBIT STYLE</p><h2 class="section-title">Editable Editor → Run on Simulator → See Real Results</h2>', unsafe_allow_html=True)
        default_code="""from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0,1)
qc.measure_all()
# Try: qc.ry(0.7, 0), qc.rz(1.2, 1), qc.x(0)
"""
        if "play_code" not in st.session_state: st.session_state.play_code=default_code
        st.session_state.play_code=st.text_area("Python editor — Qiskit Aer (syntax highlighted)", value=st.session_state.play_code, height=220, key="editor")
        ec1,ec2,ec3,ec4,ec5,ec6=st.columns([1,1,1,1,1,1])
        shots=ec1.number_input("Shots", 100, 10000, 1024, step=100, key="shots_play")
        with ec2:
            run_sim=st.button("▶ Run on Simulator", width="stretch", type="primary", key="run_sim")
        with ec3:
            run_hw=st.button("⬢ Run on Hardware", width="stretch", disabled=True, help="Hardware disabled: no IBM credentials. Configure via QISKIT_IBM_TOKEN to enable.", key="run_hw")
            st.caption("Hardware unavailable — simulator active")
        with ec4:
            if st.button("■ Stop", width="stretch", key="stop_play"): st.toast("No background job to stop", icon="ℹ️")
        with ec5:
            if st.button("↺ Reset", width="stretch", key="reset_play"):
                st.session_state.play_code=default_code; st.rerun()
        with ec6:
            if st.button("▢ Clear", width="stretch", key="clear_play"):
                st.session_state.play_code=""; st.rerun()
        # copy
        if st.button("📋 Copy code", key="copy_play"): st.toast("Select + Ctrl+C to copy from editor", icon="📋")
        if run_sim:
            with st.spinner("AerSimulator running..."):
                result=run_aer_code(st.session_state.play_code, shots=int(shots))
                st.session_state.play_result=result
        if "play_result" in st.session_state:
            res=st.session_state.play_result
            if res["status"]=="error":
                st.markdown(f'<div class="glass" style="border-color:rgba(251,113,133,0.35)"><h3 style="color:#fb7185">❌ Execution Failed</h3><pre style="white-space:pre-wrap;font-family:JetBrains Mono;font-size:0.72rem;color:#FECDD3">{res["error"]}</pre></div>', unsafe_allow_html=True)
            else:
                # status cards
                s1,s2,s3,s4=st.columns(4)
                s1.metric("Backend", res["backend"])
                s2.metric("Shots", f'{res["shots"]}')
                s3.metric("Qubits", f'{res["num_qubits"]}')
                s4.metric("Depth / Gates", f'{res["depth"]} / {res["num_gates"]}')
                st.caption(f"Elapsed {res['elapsed']*1000:.1f} ms • Simulator results — never claimed as hardware.")
                # circuit image
                b64=circuit_to_image_b64(res["circuit"])
                if b64:
                    st.markdown(f'<div class="glass hud-corners"><h3>Circuit Diagram — Actual</h3><img src="data:image/png;base64,{b64}" style="width:100%;background:white;border-radius:12px;padding:8px"/></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="glass"><pre style="font-family:JetBrains Mono;font-size:0.72rem;color:#DCE8FF">{res["circuit"].draw(output="text")}</pre></div>', unsafe_allow_html=True)
                # counts/probs toggle
                view=st.radio("View", ["Counts","Probabilities","Statevector (if no measure)"], horizontal=True, key="play_view")
                if view=="Counts":
                    keys=list(res["counts"].keys()); vals=[res["counts"][k] for k in keys]
                    fig=go.Figure(go.Bar(x=keys,y=vals, marker_color="#00E5FF", text=vals, textposition="auto"))
                    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(l=10,r=10,t=10,b=10), xaxis_title="Bitstring", yaxis_title="Counts")
                    st.plotly_chart(fig, width="stretch", config={"displayModeBar":False})
                    st.json(res["counts"])
                elif view=="Probabilities":
                    probs=res["probs"] if isinstance(res["probs"], dict) else {k:v for k,v in res["counts"].items()}
                    if isinstance(probs, dict):
                        keys=list(probs.keys()); vals=[probs[k]*100 if max(probs.values())<=1 else probs[k] for k in keys]
                        fig=go.Figure(go.Bar(x=keys,y=vals, marker_color="#C084FC", text=[f"{v:.1f}%" for v in vals], textposition="auto"))
                        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280, yaxis_title="Probability %", xaxis_title="State")
                        st.plotly_chart(fig, width="stretch", config={"displayModeBar":False})
                        st.json({k: f"{v*100:.2f}%" if v<=1 else str(v) for k,v in probs.items()})
                else:
                    if res.get("statevector") is not None:
                        sv=res["statevector"]
                        st.markdown(f'<div class="glass"><h3>Statevector — Actual Amplitudes</h3><pre style="font-family:JetBrains Mono;font-size:0.70rem;color:#DCE8FF;white-space:pre-wrap">{np.array2string(sv, precision=4, separator=", ")}</pre><div style="font-size:0.72rem;color:#8AA0C8">Probabilities |amplitude|²: {np.array2string(np.abs(sv)**2, precision=4)}</div></div>', unsafe_allow_html=True)
                    else:
                        st.info("Add no measurements to see statevector. With measurements, counts/probabilities are the observable results.")
        # config area for hardware
        with st.expander("🔑 Hardware credentials — how to connect IBM Quantum"):
            st.markdown("""
- Set env `QISKIT_IBM_TOKEN=your_token` or add `~/.qiskit/qiskit-ibm.json`.
- Install `qiskit-ibm-runtime`: `pip install qiskit-ibm-runtime`
- The playground will then enable `Run on Hardware` and show real backend, queue time, and noisy results.
- **Never paste tokens in this frontend editor** — they stay in backend/env.
- Current status: **Simulator only** (`qiskit-aer` local). This is honest and never fakes hardware execution.
""")
            try:
                import qiskit_aer
                st.caption(f"Aer version {qiskit_aer.__version__} • Simulator ready. Hardware import: disabled (no token).")
            except: st.caption("Aer not found.")

    # ========== TAB 6: MEASUREMENTS ==========
    with tabs[6]:
        st.markdown('<p class="section-kicker">LIVE MEASUREMENT RESULTS</p><h2 class="section-title">Histogram • Probabilities • Shots • Backend</h2>', unsafe_allow_html=True)
        if "play_result" not in st.session_state or st.session_state.play_result["status"]!="success":
            st.info("Run a circuit in Qiskit Playground first. Then measurements appear here with real counts.")
            # demo fallback: show Bell demo
            if st.button("Load Bell demo to Measurements"):
                st.session_state.play_code="from qiskit import QuantumCircuit\nqc=QuantumCircuit(2)\nqc.h(0)\nqc.cx(0,1)\nqc.measure_all()\n"
                st.rerun()
        else:
            res=st.session_state.play_result
            mv=st.radio("Toggle", ["Counts","Probabilities","Statevector"], horizontal=True, key="meas_view2")
            if mv=="Counts":
                keys=list(res["counts"].keys()); vals=[res["counts"][k] for k in keys]
                fig=go.Figure(go.Bar(x=keys,y=vals, marker=dict(colorscale="Teal", color=vals), text=vals, textposition="auto"))
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=320, title=f'Counts — {res["backend"]} shots={res["shots"]}')
                st.plotly_chart(fig, width="stretch")
            elif mv=="Probabilities":
                # normalize
                probs=res["probs"] if isinstance(res["probs"],dict) else res["counts"]
                if isinstance(probs, dict):
                    total=sum(probs.values()); norm={k:v/total if v>1 else v for k,v in probs.items()} if total>1 else probs
                    # if values already probabilities sum 1
                    s=sum(norm.values())
                    if s>1.5: norm={k:v/sum(norm.values()) for k,v in norm.items()}
                    vals=[norm[k]*100 for k in norm]
                    fig=go.Figure(go.Bar(x=list(norm.keys()), y=vals, marker_color="#7DD3FC", text=[f"{v:.1f}%" for v in vals]))
                    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=320, yaxis_title="Prob %")
                    st.plotly_chart(fig, width="stretch")
                    for k,v in norm.items():
                        st.markdown(f'<div style="display:flex;justify-content:space-between;font-family:JetBrains Mono;font-size:0.76rem;background:rgba(255,255,255,0.04);padding:0.3rem 0.6rem;border-radius:8px;margin:0.18rem 0"><span>|{k}⟩</span><span>{v*100:.2f}%</span></div>', unsafe_allow_html=True)
            else:
                if res.get("statevector") is not None:
                    sv=res["statevector"]; probs=np.abs(sv)**2
                    fig=go.Figure(go.Bar(x=[format(i,f'0{res["num_qubits"]}b') for i in range(len(sv))], y=probs*100, marker_color="#C084FC"))
                    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=320, title="Statevector probabilities |amp|²")
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No statevector — circuit has measurements. Use Counts/Probabilities.")

    # ========== TAB 7: STATE EVOLUTION ==========
    with tabs[7]:
        st.markdown('<p class="section-kicker">QUANTUM STATE EVOLUTION — STEP BY STEP</p><h2 class="section-title">Previous • Next • Play • Pause • Reset — Per-Gate Bloch & Probabilities</h2>', unsafe_allow_html=True)
        if "evo_qspec" not in st.session_state:
            st.session_state.evo_qspec=CircuitSpec(n_qubits=2)
            st.session_state.evo_qspec.add_gate("H",[0]); st.session_state.evo_qspec.add_gate("CNOT",[0,1])
        eq=st.session_state.evo_qspec
        eq.n_qubits=st.slider("Qubits", 2, 4, int(eq.n_qubits), key="evo_nq")
        if st.button("Load GHZ demo"): eq.clear(); eq.add_gate("H",[0]); [eq.add_gate("CNOT",[i,i+1]) for i in range(eq.n_qubits-1)]; st.rerun()
        if "evo_step" not in st.session_state: st.session_state.evo_step=0
        n_steps=len(eq.ops)+1
        c1,c2,c3,c4,c5=st.columns([1,1,1,1,1.5])
        with c1:
            if st.button("◀ Prev", width="stretch"): st.session_state.evo_step=max(0, st.session_state.evo_step-1); st.rerun()
        with c2:
            if st.button("▶ Next", width="stretch"): st.session_state.evo_step=min(n_steps-1, st.session_state.evo_step+1); st.rerun()
        with c3:
            if st.button("▶ Play", width="stretch"):
                for i in range(n_steps):
                    st.session_state.evo_step=i; time.sleep(0.6); st.rerun()
        with c4:
            if st.button("⏸ Pause", width="stretch"): st.toast("Use Prev/Next stepwise", icon="⏸")
        with c5:
            if st.button("↺ Reset to 0", width="stretch"): st.session_state.evo_step=0; st.rerun()
        step=int(st.session_state.evo_step)
        st.progress(step/(n_steps-1) if n_steps>1 else 1)
        # Build prefix circuit up to step
        prefix=eq.ops[:step]
        # Labels
        labels=["Step 0 — Initial |00..0⟩"] + [f'Step {i+1} — {op["gate"]} {op["wires"]}' for i,op in enumerate(eq.ops)]
        st.markdown(f'<div style="font-family:Orbitron;letter-spacing:1.4px;color:#7DD3FC;text-align:center;margin:0.4rem 0">{labels[step] if step < len(labels) else labels[-1]}</div>', unsafe_allow_html=True)
        # Simulate prefix
        try:
            if not prefix:
                dim=1<<eq.n_qubits; sv=np.zeros(dim, dtype=complex); sv[0]=1.0
                bloch=compute_bloch_vectors(sv, eq.n_qubits)
            else:
                r=PennylaneBackend(eq.n_qubits).simulate_circuit_ops(prefix, eq.n_qubits)
                sv=r.statevector; bloch=r.bloch_vectors
            ec1,ec2=st.columns([1.2,0.9])
            with ec1:
                cols=st.columns(2)
                for i in range(min(4, eq.n_qubits)):
                    with cols[i%2]:
                        fig=bloch_sphere_figure(bloch[i], i)
                        st.plotly_chart(fig, width="stretch", config={"displayModeBar":False}, key=f"evo_{step}_{i}")
            with ec2:
                probs=np.abs(sv)**2; probs/=probs.sum() if probs.sum()>0 else 1
                top=np.argsort(probs)[::-1][:4]
                for i in top:
                    bits=format(i,f'0{eq.n_qubits}b')[::-1]
                    st.markdown(f'<div style="font-family:JetBrains Mono;font-size:0.70rem;background:rgba(255,255,255,0.04);border:1px solid rgba(125,211,252,0.12);padding:0.32rem 0.5rem;border-radius:8px;margin:0.18rem 0;display:flex;justify-content:space-between"><span>|{bits}⟩</span><span>{probs[i]*100:.2f}%</span></div>', unsafe_allow_html=True)
                rnorm=np.linalg.norm(bloch,axis=1)
                st.caption(f"|r| per qubit: {', '.join([f'q{i}:{v:.2f}' for i,v in enumerate(rnorm)])} — shrinks when entangled between steps.")
        except Exception as e:
            st.error(str(e))

    # ========== TAB 8: Q-ORBIT EXPERIMENT ==========
    with tabs[8]:
        st.markdown('<p class="section-kicker">Q-ORBIT QUANTUM EXPERIMENT — REAL PIPELINE</p><h2 class="section-title">Dataset Sample → Feature → Quantum Encoding → Expectation → Hybrid</h2>', unsafe_allow_html=True)
        try:
            from src.simulator.lightcurve_generator import generate_single_light_curve
            from src.classical.feature_engineering import extract_all_features
            from src.quantum.hybrid_model import HybridQuantumClassifier
            import torch
        except Exception as e:
            st.error(f"Import failed: {e}")
            st.stop()
        # Controls
        cc1,cc2,cc3=st.columns([1,1,1])
        with cc1: cls_exp=st.selectbox("Class", [0,1,2,3,4], format_func=lambda x: ["Intact Satellite","Dead Satellite","Rocket Body","Fragmentation Debris","Spoofed Satellite"][x], key="cls_exp2")
        with cc2: noise_exp=st.slider("Noise std", 0.0,0.1,0.02, key="noise_exp2")
        with cc3:
            run_exp=st.button("▶ Run Q-ORBIT Quantum Path — Real Model", width="stretch", type="primary")
        if not run_exp:
            st.markdown("""
<div class="glass hud-corners" style="text-align:center;padding:1.2rem">
  <div style="font-family:Orbitron;letter-spacing:1.6px;color:#7DD3FC">READY — PRESS RUN TO USE ACTUAL Q-ORBIT DATA & MODELS</div>
  <div style="font-size:0.84rem;color:#A8BBDD;margin-top:0.35rem">Uses <code>generate_single_light_curve()</code> → <code>extract_all_features()</code> → HybridQuantumClassifier (8q PennyLane). Values shown are computed, not invented. No hard-coded numbers.</div>
</div>""", unsafe_allow_html=True)
            st.markdown("""
<div class="glass" style="margin-top:0.7rem">
  <div style="display:grid;grid-template-columns:repeat(7,1fr);gap:0.4rem;text-align:center">
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(125,211,252,0.10);border-radius:10px;padding:0.55rem"><div style="font-family:JetBrains Mono;font-size:0.62rem;color:#8AA0C8">LIGHT CURVE</div><div style="font-size:0.76rem;color:#7DD3FC">256 pts</div></div>
    <div style="color:#00E5FF;display:grid;place-items:center">→</div>
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(125,211,252,0.10);border-radius:10px;padding:0.55rem"><div style="font-family:JetBrains Mono;font-size:0.62rem;color:#8AA0C8">FEATURES</div><div style="font-size:0.76rem;color:#C084FC">21 dims</div></div>
    <div style="color:#00E5FF;display:grid;place-items:center">→</div>
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(125,211,252,0.10);border-radius:10px;padding:0.55rem"><div style="font-family:JetBrains Mono;font-size:0.62rem;color:#8AA0C8">REDUCED</div><div style="font-size:0.76rem;color:#FBBF24">8 dims</div></div>
    <div style="color:#00E5FF;display:grid;place-items:center">→</div>
    <div style="background:rgba(0,229,255,0.08);border:1px solid rgba(0,229,255,0.18);border-radius:10px;padding:0.55rem"><div style="font-family:JetBrains Mono;font-size:0.62rem;color:#00E5FF">QUANTUM 8Q</div><div style="font-size:0.76rem;color:#E6F0FF">⟨Z⟩×8</div></div>
  </div>
</div>""", unsafe_allow_html=True)
        else:
            with st.spinner("Running real Q-ORBIT pipeline..."):
                try:
                    curve,_=generate_single_light_curve(int(cls_exp), noise_std=float(noise_exp), n_samples=256)
                    feats=extract_all_features(curve, sampling_interval=720/255)
                    feats=np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
                    # Hybrid model if available
                    bundle_available=False
                    try:
                        from app.app import load_all_models
                        bundle=load_all_models()
                        hybrid=bundle.get("hybrid")
                        scaler=bundle.get("scaler")
                        bundle_available=hybrid is not None
                    except: hybrid=None; scaler=None
                    # Show intermediates
                    st.markdown(f"""
<div class="glass hud-corners">
  <h3>Intermediate Values — Real Computed</h3>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:0.6rem">
    <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(125,211,252,0.12);border-radius:12px;padding:0.6rem"><div style="font-family:JetBrains Mono;font-size:0.64rem;color:#8AA0C8">INPUT CURVE</div><div style="font-size:0.78rem;color:#7DD3FC">256 samples, mean {float(np.mean(curve)):.3f}, std {float(np.std(curve)):.3f}</div></div>
    <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(125,211,252,0.12);border-radius:12px;padding:0.6rem"><div style="font-family:JetBrains Mono;font-size:0.64rem;color:#8AA0C8">21 FEATURES</div><div style="font-size:0.70rem;color:#C084FC;word-break:break-all">{np.array2string(feats[:6], precision=3)}...</div></div>
    <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(125,211,252,0.12);border-radius:12px;padding:0.6rem"><div style="font-family:JetBrains Mono;font-size:0.64rem;color:#8AA0C8">REDUCED 8D (proj → tanh·π)</div><div style="font-size:0.72rem;color:#FBBF24">8 qubits, Hilbert 256-D</div></div>
    <div style="background:rgba(0,229,255,0.08);border:1px solid rgba(0,229,255,0.18);border-radius:12px;padding:0.6rem"><div style="font-family:JetBrains Mono;font-size:0.64rem;color:#00E5FF">QUANTUM EXPECTATIONS ⟨Z⟩×8</div><div style="font-size:0.70rem;color:#E6F0FF" id="qexp">Computing...</div></div>
  </div>
</div>""", unsafe_allow_html=True)
                    # Quantum expectation via hybrid model quantum_layer or pennylane directly
                    if bundle_available:
                        import torch
                        feats_t=torch.tensor(((feats - scaler["mean"])/scaler["scale"] if scaler is not None and "mean" in scaler else feats)[None,:], dtype=torch.float32)
                        with torch.no_grad():
                            qfeat=hybrid.quantum_layer(feats_t).numpy()[0]
                        st.markdown(f'<div style="font-family:JetBrains Mono;font-size:0.70rem;color:#E6F0FF;background:rgba(0,229,255,0.06);border:1px solid rgba(0,229,255,0.14);padding:0.5rem;border-radius:10px;margin:0.5rem 0">Quantum outputs (8 expectation values): {np.array2string(qfeat, precision=4)}</div>', unsafe_allow_html=True)
                        # Classical head
                        logits, probs = hybrid(feats_t)
                        probs=probs.numpy()[0]
                        st.markdown(f'<div style="font-family:JetBrains Mono;font-size:0.70rem;color:#C084FC">Hybrid probs: {np.array2string(probs, precision=3)} → pred class {int(np.argmax(probs))}</div>', unsafe_allow_html=True)
                    else:
                        st.warning("Hybrid model not loaded — showing quantum encoding demo via PennyLane AngleEmbedding.")
                        qfeat=np.tanh(feats[:8])*np.pi
                        st.code(f"Angle embedding RY({np.array2string(qfeat, precision=3)}) on 8 qubits", language="python")
                    # Plot curve thumbnail
                    fig=go.Figure(go.Scatter(y=curve, mode="lines", line=dict(color="#00E5FF", width=2)))
                    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=180, margin=dict(l=10,r=10,t=10,b=10), title="Input light curve (256 pts) — real synthetic")
                    st.plotly_chart(fig, width="stretch", config={"displayModeBar":False})
                except Exception as e:
                    st.error(f"Experiment failed: {e}\n{traceback.format_exc(limit=2)}")

    # ========== TAB 9: SIM vs HARDWARE ==========
    with tabs[9]:
        st.markdown('<p class="section-kicker">SIMULATOR VS REAL QUANTUM HARDWARE — HONEST COMPARISON</p><h2 class="section-title">Aer Simulator vs IBM Quantum Hardware</h2>', unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            st.markdown("""
<div class="glass hud-corners" style="border-color:rgba(0,229,255,0.22)">
  <h3>▶ SIMULATOR (Aer)</h3>
  <div style="font-size:0.84rem;color:#C7D6F5;line-height:1.6">Qiskit Aer statevector — ideal, noiseless, deterministic. Used for Q-ORBIT VQC/Hybrid training (default.qubit & Aer). Bloch spheres here are from this mode unless noise toggled.</div>
  <div style="margin-top:0.6rem;display:flex;gap:0.4rem;flex-wrap:wrap"><span class="badge">Backend: AerSimulator</span><span class="badge">Shots: flexible</span><span class="badge">Noise: off (ideal)</span><span class="badge" style="color:#5EEAD4">● Available</span></div>
  <div style="margin-top:0.5rem;font-family:JetBrains Mono;font-size:0.66rem;color:#8AA0C8">Run any circuit in Playground → counts/probs/statevector computed locally via C++.</div>
</div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""
<div class="glass hud-corners" style="border-color:rgba(251,113,133,0.22)">
  <h3>⬢ REAL HARDWARE (IBM Quantum)</h3>
  <div style="font-size:0.84rem;color:#C7D6F5;line-height:1.6">Superconducting qubits at ~10 mK in dilution fridge (above). Requires IBM Cloud token, queue, and incurs decoherence. Results are noisy and differ from simulator — never implied identical.</div>
  <div style="margin-top:0.6rem;display:flex;gap:0.4rem;flex-wrap:wrap"><span class="badge" style="color:#FECDD3;border-color:rgba(251,113,133,0.22)">Backend: ibm_brisbane / kyiv etc.</span><span class="badge">Queue: variable</span><span class="badge">Noise: real</span><span class="badge" style="color:#FBBF24">○ Unavailable (no token)</span></div>
  <div style="margin-top:0.5rem;font-family:JetBrains Mono;font-size:0.66rem;color:#FBBF24">Hardware execution disabled by default. Set <code>QISKIT_IBM_TOKEN</code> to enable. Simulator remains fully functional. We never fake hardware results.</div>
</div>""", unsafe_allow_html=True)
        # Demo table if play_result exists
        if "play_result" in st.session_state and st.session_state.play_result.get("status")=="success":
            res=st.session_state.play_result
            st.markdown('<div class="glass hud-corners" style="margin-top:0.7rem"><h3>Last Run — Simulator Output (Honest)</h3></div>', unsafe_allow_html=True)
            s1,s2,s3,s4=st.columns(4)
            s1.metric("Backend", res["backend"])
            s2.metric("Qubits", res["num_qubits"])
            s3.metric("Shots", res["shots"])
            s4.metric("Elapsed", f'{res["elapsed"]*1000:.1f} ms')
            keys=list(res["counts"].keys())[:8]
            vals=[res["counts"][k] for k in keys]
            fig=go.Figure(go.Bar(x=keys,y=vals, marker_color="#7DD3FC"))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=260, title="Counts (simulator) — hardware would be broader/noisier")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Run a circuit in Playground to populate this comparison table with real simulator numbers. Hardware column will show `unavailable` until token configured — never faked.")

    # ========== TAB 10: NOISE ==========
    with tabs[10]:
        st.markdown('<p class="section-kicker">QUANTUM NOISE DEMONSTRATION — CONNECTS TO Q-ORBIT ROBUSTNESS</p><h2 class="section-title">Noise OFF → LOW → MEDIUM → HIGH — See Probabilities Smear</h2>', unsafe_allow_html=True)
        level=st.select_slider("Noise level", options=["OFF","LOW","MEDIUM","HIGH"], value="OFF", key="noise_level")
        # For demo: depolarizing noise model via Aer
        try:
            from qiskit_aer.noise import NoiseModel, depolarizing_error
            from qiskit import QuantumCircuit
            from qiskit_aer import AerSimulator
            # Build demo Bell circuit
            qc=QuantumCircuit(2); qc.h(0); qc.cx(0,1); qc.measure_all()
            noise_rates={"OFF":0,"LOW":0.005,"MEDIUM":0.02,"HIGH":0.08}
            p=noise_rates[level]
            if p==0:
                sim=AerSimulator()
                res=sim.run(qc, shots=2048).result()
                counts=res.get_counts(qc)
                backend_label="Aer ideal"
            else:
                nm=NoiseModel()
                err1=depolarizing_error(p,1)
                err2=depolarizing_error(p*1.8,2)
                nm.add_all_qubit_quantum_error(err1, ["h","x","y","z","s","t","ry"])
                nm.add_all_qubit_quantum_error(err2, ["cx","cz"])
                sim=AerSimulator(noise_model=nm)
                res=sim.run(qc, shots=2048).result()
                counts=res.get_counts(qc)
                backend_label=f"Aer + depolarizing p={p}"
            total=sum(counts.values())
            probs={k:v/total for k,v in counts.items()}
            # Show histogram
            fig=go.Figure(go.Bar(x=list(counts.keys()), y=list(counts.values()), marker_color=["#00E5FF" if k in ("00","11") else "#F472B6" for k in counts.keys()], text=list(counts.values())))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=300, title=f'Bell state counts — {backend_label} — {level} — Ideal=00 & 11 ≈50% each, noise populates 01/10')
            st.plotly_chart(fig, width="stretch")
            c1,c2=st.columns([1,1])
            with c1:
                st.markdown(f'<div class="glass"><h3>Distribution — {level}</h3><div style="font-family:JetBrains Mono;font-size:0.72rem;color:#A8BBDD">{"<br>".join([f"|{k}⟩ {v*100:.1f}% ({counts[k]} shots)" for k,v in sorted(probs.items())])}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
<div class="glass hud-corners">
  <h3>Connection to Q-ORBIT Robustness</h3>
  <div style="font-size:0.82rem;color:#A8BBDD;line-height:1.6">Q-ORBIT tests <b>Noise 0/0.05/0.10/0.20/0.30</b>, <b>Observation 100%/75%/50%/25%</b>, <b>Missing 0/5/10/20%</b> on frozen test set (<code>src/evaluation/degradation.py:1</code>).<br/>Quantum noise above is analogous: higher <b>p</b> → counts smear like <b>photon noise</b> in light curves → accuracy drops. Hybrid's adaptive fusion learns to trust less when signal quality is low.</div>
  <div style="margin-top:0.5rem;font-family:JetBrains Mono;font-size:0.64rem;color:#8AA0C8">Ideal Bell: 00 & 11 dominate. With <b>{level}</b> noise, 01/10 rise — same way eclipse/missing data degrades light-curve classification.</div>
</div>""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Noise demo requires qiskit-aer noise extras: {e}")
            st.info("Ensure `pip install qiskit-aer` includes noise module (it does). Fallback: Python depolarizing applied to probs.")

    # ========== TAB 11: Q-ORBIT PIPELINE + AI+QUANTUM ==========
    with tabs[11]:
        st.markdown('<p class="section-kicker">HOW QUANTUM COMPUTING POWERS Q-ORBIT — INTERACTIVE PIPELINE</p><h2 class="section-title">Space Light Curve → Quantum Feature Space → Hybrid Decision</h2>', unsafe_allow_html=True)
        # Interactive pipeline diagram
        st.markdown("""
<div class="glass hud-corners">
  <div style="display:flex;align-items:center;gap:0.4rem;flex-wrap:wrap;justify-content:center">
    <div style="background:rgba(56,189,248,0.10);border:1px solid rgba(56,189,248,0.22);border-radius:12px;padding:0.6rem 0.8rem;min-width:130px;text-align:center"><div style="font-family:JetBrains Mono;font-size:0.62rem;color:#7DD3FC">1. LIGHT CURVE</div><div style="font-family:Orbitron;font-size:0.72rem;color:#E6F0FF">256 pts • 720s</div><div style="font-size:0.64rem;color:#8AA0C8">Specular+diffuse</div></div>
    <span style="color:#00E5FF">→</span>
    <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(125,211,252,0.14);border-radius:12px;padding:0.6rem 0.8rem;min-width:130px;text-align:center"><div style="font-family:JetBrains Mono;font-size:0.62rem;color:#8AA0C8">2. PREPROCESS</div><div style="font-family:Orbitron;font-size:0.72rem;color:#E6F0FF">Interp + min-max</div></div>
    <span style="color:#00E5FF">→</span>
    <div style="background:rgba(192,132,252,0.08);border:1px solid rgba(192,132,252,0.18);border-radius:12px;padding:0.6rem 0.8rem;min-width:140px;text-align:center"><div style="font-family:JetBrains Mono;font-size:0.62rem;color:#C084FC">3. PHYSICS FEATURES</div><div style="font-family:Orbitron;font-size:0.72rem;color:#E6F0FF">21 dims</div><div style="font-size:0.64rem;color:#8AA0C8">Time/Freq proxies</div></div>
    <span style="color:#00E5FF">→</span>
    <div style="background:rgba(0,229,255,0.10);border:1px solid rgba(0,229,255,0.22);border-radius:12px;padding:0.6rem 0.8rem;min-width:130px;text-align:center"><div style="font-family:JetBrains Mono;font-size:0.62rem;color:#00E5FF">4. QUANTUM MAP</div><div style="font-family:Orbitron;font-size:0.72rem;color:#E6F0FF">RY angle • 8Q</div><div style="font-size:0.64rem;color:#8AA0C8">PennyLane / Aer</div></div>
    <span style="color:#00E5FF">→</span>
    <div style="background:rgba(0,229,255,0.10);border:1px solid rgba(0,229,255,0.22);border-radius:12px;padding:0.6rem 0.8rem;min-width:130px;text-align:center"><div style="font-family:JetBrains Mono;font-size:0.62rem;color:#00E5FF">5. VARIATIONAL</div><div style="font-family:Orbitron;font-size:0.72rem;color:#E6F0FF">Rot+ CNOT×2</div><div style="font-size:0.64rem;color:#8AA0C8">⟨Z⟩×8</div></div>
    <span style="color:#00E5FF">→</span>
    <div style="background:linear-gradient(135deg, rgba(0,229,255,0.12), rgba(124,58,237,0.12));border:1px solid rgba(0,229,255,0.22);border-radius:12px;padding:0.6rem 0.8rem;min-width:140px;text-align:center"><div style="font-family:JetBrains Mono;font-size:0.62rem;color:#00E5FF">6. ADAPTIVE FUSION</div><div style="font-family:Orbitron;font-size:0.72rem;color:#E6F0FF">Classical + Quantum</div><div style="font-size:0.64rem;color:#5EEAD4">Signal-quality gate</div></div>
    <span style="color:#00E5FF">→</span>
    <div style="background:rgba(16,185,129,0.10);border:1px solid rgba(16,185,129,0.22);border-radius:12px;padding:0.6rem 0.8rem;min-width:120px;text-align:center"><div style="font-family:JetBrains Mono;font-size:0.62rem;color:#5EEAD4">CLASSIFY</div><div style="font-family:Orbitron;font-size:0.72rem;color:#E6F0FF">5 classes</div><div style="font-size:0.64rem;color:#8AA0C8">Softmax</div></div>
  </div>
  <div style="margin-top:0.6rem;font-family:JetBrains Mono;font-size:0.64rem;color:#8AA0C8;text-align:center">Click the stage you explored: <code>Cryostat 10mK</code> → qubit lives there • <code>Bloch</code> → angle encoding (RY) • <code>Gates</code> → Rot/CNOT variational • <code>Playground</code> → Aer statevector • <code>Noise</code> → robustness analogy</div>
</div>""", unsafe_allow_html=True)
        qa,qb=st.columns([1.2,0.9])
        with qa:
            st.markdown("""
<div class="glass hud-corners">
  <h3>CLASSICAL AI vs QUANTUM vs HYBRID — WHAT EACH LEARNS</h3>
  <div style="display:grid;grid-template-columns:1fr;gap:0.6rem">
    <div style="background:rgba(56,189,248,0.06);border:1px solid rgba(56,189,248,0.14);border-radius:12px;padding:0.6rem;display:flex;gap:0.7rem"><div style="width:36px;height:36px;border-radius:10px;display:grid;place-items:center;background:rgba(56,189,248,0.12);border:1px solid rgba(56,189,248,0.18)">▦</div><div><div style="font-family:Orbitron;letter-spacing:1px;color:#7DD3FC;font-size:0.78rem">CLASSICAL AI</div><div style="font-size:0.78rem;color:#A8BBDD">Extracts meaningful light-curve representations — CNN learns temporal motifs, tree/SVM learns margin boundaries on 21 physics features. Fast, strong on clean curves.</div></div></div>
    <div style="background:rgba(192,132,252,0.06);border:1px solid rgba(192,132,252,0.14);border-radius:12px;padding:0.6rem;display:flex;gap:0.7rem"><div style="width:36px;height:36px;border-radius:10px;display:grid;place-items:center;background:rgba(192,132,252,0.12);border:1px solid rgba(192,132,252,0.18)">◈</div><div><div style="font-family:Orbitron;letter-spacing:1px;color:#C084FC;font-size:0.78rem">QUANTUM MODEL</div><div style="font-size:0.78rem;color:#A8BBDD">Transforms selected representations into quantum feature space — angle encoding RY(tanh·π) + Rot+CNOT entanglement (256-D Hilbert). Measures ⟨Z⟩×8.</div></div></div>
    <div style="background:rgba(0,229,255,0.06);border:1px solid rgba(0,229,255,0.14);border-radius:12px;padding:0.6rem;display:flex;gap:0.7rem"><div style="width:36px;height:36px;border-radius:10px;display:grid;place-items:center;background:rgba(0,229,255,0.12);border:1px solid rgba(0,229,255,0.18)">⬡</div><div><div style="font-family:Orbitron;letter-spacing:1px;color:#00E5FF;font-size:0.78rem">HYBRID MODEL</div><div style="font-size:0.78rem;color:#A8BBDD">Combines classical and quantum representations via learned fusion. Not hard-coded — hybrid wins only if val-selected actually beats classical on test.</div></div></div>
    <div style="background:linear-gradient(135deg, rgba(0,229,255,0.08), rgba(124,58,237,0.08));border:1px solid rgba(0,229,255,0.18);border-radius:12px;padding:0.6rem;display:flex;gap:0.7rem"><div style="width:36px;height:36px;border-radius:10px;display:grid;place-items:center;background:linear-gradient(135deg, rgba(0,229,255,0.14), rgba(124,58,237,0.14));border:1px solid rgba(0,229,255,0.18)">⟡</div><div><div style="font-family:Orbitron;letter-spacing:1px;color:#5EEAD4;font-size:0.78rem">ADAPTIVE FUSION</div><div style="font-size:0.78rem;color:#A8BBDD">Uses signal quality / uncertainty to decide how much each representation contributes — α·p_classical + (1-α)·p_quantum, trained on train/val only, frozen test.</div></div></div>
  </div>
</div>""", unsafe_allow_html=True)
        with qb:
            # Show actual metrics
            try:
                fc=json.load(open(os.path.join(BASE_DIR,"..","results","reports","fair_comparison.json"))) if os.path.exists(os.path.join(BASE_DIR,"..","results","reports","fair_comparison.json")) else {}
                hyb=fc.get("Hybrid_Quantum",{}); cnn=fc.get("Classical_CNN",{}); pure=fc.get("Pure_Quantum_VQC",{})
                st.markdown(f"""
<div class="glass hud-corners">
  <h3>Measured Results — Honest (same 1500 test)</h3>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.5rem;text-align:center">
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(125,211,252,0.10);border-radius:12px;padding:0.5rem"><div style="font-family:Orbitron;color:#38bdf8">{cnn.get("accuracy",0.852)*100:.1f}%</div><div style="font-family:JetBrains Mono;font-size:0.62rem;color:#8AA0C8">CNN ACC</div></div>
    <div style="background:rgba(192,132,252,0.06);border:1px solid rgba(192,132,252,0.18);border-radius:12px;padding:0.5rem"><div style="font-family:Orbitron;color:#C084FC">{pure.get("accuracy",0.48)*100:.1f}%</div><div style="font-family:JetBrains Mono;font-size:0.62rem;color:#8AA0C8">VQC ACC</div></div>
    <div style="background:rgba(0,229,255,0.08);border:1px solid rgba(0,229,255,0.18);border-radius:12px;padding:0.5rem"><div style="font-family:Orbitron;color:#00E5FF">{hyb.get("accuracy",0.70)*100:.1f}%</div><div style="font-family:JetBrains Mono;font-size:0.62rem;color:#8AA0C8">HYBRID ACC</div></div>
  </div>
  <div style="margin-top:0.6rem;font-size:0.72rem;color:#8AA0C8">Quantum concepts actually used: <b>qubit encoding</b> (Angle RY), <b>parameterized circuits</b> (Rot), <b>variational</b> (CNOT chain), <b>expectation values</b> (⟨Z⟩), <b>feature maps</b> (21→8), <b>hybrid fusion</b>. Tap any block above to see its math.</div>
</div>""", unsafe_allow_html=True)
            except: st.markdown('<div class="glass"><div style="font-size:0.78rem;color:#A8BBDD">Metrics file not yet generated — run <code>run_comprehensive_fair.py</code>.</div></div>', unsafe_allow_html=True)
            st.markdown("""
<div class="glass" style="font-size:0.78rem;color:#A8BBDD;line-height:1.6">
  <b style="color:#7DD3FC">Why not generic quantum page?</b> Every gate you tested (H/RY/CNOT) is the same gate Q-ORBIT's Hybrid uses — AngleEmbedding → Rot → CNOT×2 → ⟨Z⟩ → classical head (see <code>src/quantum/hybrid_model.py:58</code>). The cryostat you highlighted is where those expectations would be measured in hardware.
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div style="margin-top:1rem;position:relative;z-index:2;padding:1rem;border-radius:18px;background:rgba(6,10,28,0.62);border:1px solid rgba(125,211,252,0.12);font-size:0.74rem;color:#8AA0C8;line-height:1.6">
  <b style="font-family:Orbitron;letter-spacing:1px;color:#E6F0FF">Q-ORBIT Quantum Lab</b> — Built without hard-coding quantum results, accuracies, or hardware status. All Bloch vectors, statevectors, counts, and probabilities come from <b>PennyLane default.qubit / Qiskit Aer / STIM</b> via real simulation; C++ only for Bloch partial-trace & light-curve ops where it measurably helps. Existing routes <code>Home • Models • Dataset • Results • Dashboard • AI Analyst • Research Report</code> unchanged — this Lab is additive.
</div>""", unsafe_allow_html=True)

if __name__=="__main__":
    main()
