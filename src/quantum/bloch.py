from __future__ import annotations
import numpy as np
from typing import Tuple

def compute_bloch_vectors(statevector: np.ndarray, n_qubits:int)->np.ndarray:
    """Compute Bloch vectors (n_qubits,3) from statevector (2**n,) complex.
    Tries C++ accelerated path via qorbit_cpp, falls back to numpy partial trace.
    Returns x,y,z in [-1,1] each.
    """
    statevector=np.asarray(statevector, dtype=np.complex128)
    dim=1<<n_qubits
    if statevector.size!=dim:
        raise ValueError(f"statevector size {statevector.size} != 2**{n_qubits}={dim}")
    # normalize
    nrm=np.linalg.norm(statevector)
    if nrm>1e-12:
        statevector=statevector/nrm
    # Wire order: PennyLane/Qiskit use wire 0 as most-significant bit in statevector index,
    # while bit k in integer index corresponds to qubit (n-1 - wire). We compute in bit order
    # then reverse to wire order so sphere 0 ↔ wire 0 (user expectation).
    def _to_wire_order(arr):
        # reverse qubit order to map bit -> wire
        return arr[::-1].copy() if n_qubits>1 else arr
    # Try C++
    try:
        import qorbit_cpp
        if hasattr(qorbit_cpp, "bloch_vectors"):
            bloch=qorbit_cpp.bloch_vectors(statevector, n_qubits)
            bloch=np.asarray(bloch, dtype=float)
            return _to_wire_order(bloch)
    except Exception:
        pass
    # Numpy fallback
    out=np.zeros((n_qubits,3), dtype=float)
    # Build density matrix implicitly via loop similar to C++ but vectorized per qubit
    for k in range(n_qubits):
        rho00=0j; rho01=0j; rho11=0j
        for i in range(dim):
            for j in range(dim):
                mask=~(1<<k)
                if (i & mask) != (j & mask):
                    continue
                contrib=statevector[i]*np.conj(statevector[j])
                a=(i>>k)&1; b=(j>>k)&1
                if a==0 and b==0: rho00+=contrib
                elif a==0 and b==1: rho01+=contrib
                elif a==1 and b==1: rho11+=contrib
        rho10=np.conj(rho01)
        x=(rho01+rho10).real
        y=(1j*(rho10 - rho01)).real
        z=(rho00 - rho11).real
        out[k,0]=float(np.clip(x,-1,1))
        out[k,1]=float(np.clip(y,-1,1))
        out[k,2]=float(np.clip(z,-1,1))
    return _to_wire_order(out)

def bloch_to_spherical(bloch: np.ndarray)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """(n,3) -> theta, phi, r. theta in [0,pi] from |0> north pole, phi in [-pi,pi], r in [0,1]."""
    bloch=np.asarray(bloch, dtype=float)
    x,y,z=bloch[:,0], bloch[:,1], bloch[:,2]
    r=np.sqrt(x*x+y*y+z*z)
    r=np.clip(r,0,1)
    theta=np.arccos(np.clip(z/np.maximum(r,1e-12), -1,1))
    # for pure mixed recall r~0 -> theta meaningless; set 0
    theta=np.where(r<1e-9, 0.0, theta)
    phi=np.arctan2(y,x)
    return theta, phi, r

def state_probabilities(statevector: np.ndarray)->np.ndarray:
    sv=np.asarray(statevector, dtype=np.complex128)
    probs=np.abs(sv)**2
    s=probs.sum()
    if s>1e-12: probs/=s
    return probs

def purity_from_bloch(bloch: np.ndarray)->np.ndarray:
    """purity = (1+|r|^2)/2 per qubit. Accepts a single (3,) vector or a batch (n,3)."""
    b = np.asarray(bloch, dtype=float)
    r = np.linalg.norm(b, axis=-1)
    return 0.5 * (1 + r * r)

def bloch_sphere_figure(bloch_vec, qubit_idx=0, title_suffix=""):
    """Create Plotly 3D Bloch sphere figure with arrow for single qubit."""
    import plotly.graph_objects as go
    import numpy as np
    x,y,z=float(bloch_vec[0]), float(bloch_vec[1]), float(bloch_vec[2])
    r=np.sqrt(x*x+y*y+z*z)
    # sphere mesh
    u=np.linspace(0,2*np.pi,36)
    v=np.linspace(0,np.pi,18)
    xs=np.outer(np.cos(u), np.sin(v))
    ys=np.outer(np.sin(u), np.sin(v))
    zs=np.outer(np.ones_like(u), np.cos(v))
    fig=go.Figure()
    fig.add_trace(go.Surface(x=xs,y=ys,z=zs, opacity=0.12, colorscale=[[0,"#0B1A3A"],[1,"#1e3a5f"]], showscale=False, hoverinfo="skip"))
    # equator / meridians
    theta=np.linspace(0,2*np.pi,64)
    fig.add_trace(go.Scatter3d(x=np.cos(theta),y=np.sin(theta),z=np.zeros_like(theta), mode="lines", line=dict(color="rgba(125,211,252,0.22)",width=2), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=np.zeros_like(theta),y=np.cos(theta),z=np.sin(theta), mode="lines", line=dict(color="rgba(192,132,252,0.18)",width=2), hoverinfo="skip", showlegend=False))
    # axes
    for ax,col in [(np.array([[0,0],[0,0],[-1.15,1.15]]),"#7DD3FC")]:
        pass
    # arrow shaft
    fig.add_trace(go.Scatter3d(x=[0,x],y=[0,y],z=[0,z], mode="lines", line=dict(color="#00E5FF", width=8), hoverinfo="skip", showlegend=False))
    # arrow tip
    fig.add_trace(go.Scatter3d(x=[x],y=[y],z=[z], mode="markers", marker=dict(size=7, color="#F472B6", line=dict(width=1,color="white")), hoverinfo="skip", showlegend=False))
    # pole labels
    fig.add_trace(go.Scatter3d(x=[0,0],y=[0,0],z=[1.12,-1.12], mode="text", text=["|0⟩","|1⟩"], textposition="middle center", textfont=dict(color="white", size=11), hoverinfo="skip", showlegend=False))
    pur=0.5*(1+r*r)
    ent = 1.0 - r  # 0=unentangled (pure), 1=max entangled (mixed)
    title=f"Qubit {qubit_idx} — r=({x:.2f},{y:.2f},{z:.2f}) | |r|={r:.2f} | purity={pur:.2f}"
    if title_suffix: title+=f" {title_suffix}"
    fig.update_layout(title=dict(text=title, font=dict(size=12,color="#7DD3FC"), x=0.02),
                      scene=dict(xaxis=dict(range=[-1.2,1.2], title="X", gridcolor="rgba(125,211,252,0.1)", backgroundcolor="rgba(0,0,0,0)"),
                                 yaxis=dict(range=[-1.2,1.2], title="Y", gridcolor="rgba(125,211,252,0.1)", backgroundcolor="rgba(0,0,0,0)"),
                                 zaxis=dict(range=[-1.2,1.2], title="Z", gridcolor="rgba(125,211,252,0.1)", backgroundcolor="rgba(0,0,0,0)"),
                                 aspectmode="cube", bgcolor="rgba(0,0,0,0)", camera=dict(eye=dict(x=1.45,y=1.35,z=1.05))),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=38,b=0), height=360, showlegend=False)
    return fig

def build_before_after_snapshots(n_qubits=8, features=None, weights=None):
    """Utility to produce Bloch before/after snapshots for docs/tests.
    Returns dict with keys 'zero', 'superposition', 'entangled' each BackendResult.
    """
    from .backends.pennylane_backend import PennylaneBackend
    pb=PennylaneBackend(n_qubits=n_qubits)
    # zero state
    zero=np.zeros(1<<n_qubits, dtype=complex); zero[0]=1.0
    from .bloch import compute_bloch_vectors
    b_zero=compute_bloch_vectors(zero, n_qubits)
    # Need backend results; construct manually via Aer/Pennylane ops
    # superposition: H on q0
    sup=pb.simulate_circuit_ops([{"gate":"H","wires":[0]}], n_qubits=n_qubits)
    ent_ops=[{"gate":"H","wires":[0]}, {"gate":"CNOT","wires":[0,1]}]
    if n_qubits>2:
        # extend GHZ-ish chain
        for i in range(1, n_qubits-1):
            ent_ops.append({"gate":"CNOT","wires":[i,i+1]})
    ent=pb.simulate_circuit_ops(ent_ops, n_qubits=n_qubits)
    return {"zero": zero, "superposition": sup.statevector, "entangled": ent.statevector}
