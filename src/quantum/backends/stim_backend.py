from __future__ import annotations
import numpy as np
try:
    import stim
    HAS_STIM=True
except Exception:
    HAS_STIM=False
from .base import BackendResult

CLIFFORD_GATES={"H","X","Y","Z","S","CNOT","CZ","SWAP"}

class StimBackend:
    def __init__(self, n_qubits:int=8):
        self.n_qubits=n_qubits

    def is_clifford(self, ops)->bool:
        for op in ops:
            g=op.get("gate","")
            # RX/RY/RZ with arbitrary angles are not Clifford unless angle is pi/2 multiples -> treat as non-Clifford for simplicity
            if g not in CLIFFORD_GATES:
                if g in ("RX","RY","RZ","Rot","T","Toffoli"):
                    return False
        return True

    def simulate_circuit_ops(self, ops, n_qubits:int=None):
        n=n_qubits or self.n_qubits
        if not HAS_STIM or not self.is_clifford(ops):
            # fallback to pennylane for non-Clifford
            from .pennylane_backend import PennylaneBackend
            return PennylaneBackend(n).simulate_circuit_ops(ops, n)
        # Stim tableau simulation -> convert to statevector via stim's to_state_vector? stim 1.16 has state_vector
        try:
            sim=stim.TableauSimulator()
            for op in ops:
                g=op.get("gate","")
                w=op.get("wires",[0])
                if g=="H": sim.h(w[0])
                elif g=="X": sim.x(w[0])
                elif g=="Y": sim.y(w[0])
                elif g=="Z": sim.z(w[0])
                elif g=="S": sim.s(w[0])
                elif g=="CNOT": sim.cx(w[0],w[1])
                elif g=="CZ": sim.cz(w[0],w[1])
                elif g=="SWAP": sim.swap(w[0],w[1])
            # get statevector via to_state_vector
            if hasattr(sim, "to_state_vector"):
                sv=np.asarray(sim.to_state_vector(), dtype=complex)
            elif hasattr(sim, "state_vector"):
                sv=np.asarray(sim.state_vector(), dtype=complex)
            else:
                # fallback: use stim's Tableau.to_state_vector
                tab=sim.current_inverse_tableau().inverse()
                sv=np.asarray(tab.to_state_vector(), dtype=complex)
            # stim returns little-endian? ensure length 2**n
            expected=2**n
            if len(sv)!=expected:
                # pad/truncate
                tmp=np.zeros(expected, dtype=complex); tmp[:min(len(sv),expected)]=sv[:expected]; sv=tmp
                nrm=np.linalg.norm(sv)
                if nrm>1e-12: sv=sv/nrm
            from ..bloch import compute_bloch_vectors
            bloch=compute_bloch_vectors(sv,n)
            return BackendResult(statevector=sv, n_qubits=n, bloch_vectors=bloch, metadata={"backend":"stim","is_clifford":True})
        except Exception as e:
            from .pennylane_backend import PennylaneBackend
            rb=PennylaneBackend(n).simulate_circuit_ops(ops,n)
            rb.metadata["stim_fallback"]=str(e)
            return rb

    def tableau_text(self, ops, n_qubits:int=None)->str:
        if not HAS_STIM:
            return "STIM not installed — pip install stim"
        if not self.is_clifford(ops):
            return "Non-Clifford circuit — STIM tableau only for Clifford (H,S,CNOT,CZ). Use AER/PennyLane."
        n=n_qubits or self.n_qubits
        c=stim.Circuit()
        for op in ops:
            g=op.get("gate","")
            w=op.get("wires",[0])
            if g=="H": c.append("H", w)
            elif g=="X": c.append("X", w)
            elif g=="Y": c.append("Y", w)
            elif g=="Z": c.append("Z", w)
            elif g=="S": c.append("S", w)
            elif g=="CNOT": c.append("CX", w)
            elif g=="CZ": c.append("CZ", w)
            elif g=="SWAP": c.append("SWAP", w)
        t=stim.Tableau.from_circuit(c)
        return str(t)
