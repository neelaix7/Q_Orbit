from __future__ import annotations
import numpy as np
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    HAS_AER=True
except Exception:
    HAS_AER=False
from .base import BackendResult

class AerBackend:
    def __init__(self, n_qubits:int=8):
        self.n_qubits=n_qubits
        if HAS_AER:
            self.sim=AerSimulator(method="statevector")
        else:
            self.sim=None

    def _build_circuit_features(self, features, weights=None, feature_scale=np.pi):
        from qiskit import QuantumCircuit
        n=self.n_qubits
        qc=QuantumCircuit(n)
        f=np.asarray(features, dtype=float).reshape(-1)
        if len(f)>=n: f=f[:n]
        else:
            tmp=np.zeros(n); tmp[:len(f)]=f; f=tmp
        f=np.tanh(f)*feature_scale
        for i in range(n):
            qc.ry(float(f[i]), i)
        if weights is not None:
            w=np.asarray(weights, dtype=float)
            if w.ndim==3:
                n_layers=w.shape[0]
                for l in range(n_layers):
                    for i in range(n):
                        qc.u(float(w[l,i,0]), float(w[l,i,1]), float(w[l,i,2]), i)  # U = Rot
                    for i in range(n-1):
                        qc.cx(i,i+1)
            elif w.ndim==2:
                for i in range(min(n, w.shape[0])):
                    qc.u(float(w[i,0]), float(w[i,1]), float(w[i,2]), i)
                for i in range(n-1): qc.cx(i,i+1)
        qc.save_statevector()
        return qc

    def simulate(self, features, weights=None, feature_scale=np.pi, noise_model=None) -> BackendResult:
        if not HAS_AER:
            # fallback to pennylane
            from .pennylane_backend import PennylaneBackend
            return PennylaneBackend(self.n_qubits).simulate(features, weights, feature_scale)
        qc=self._build_circuit_features(features, weights, feature_scale)
        result=self.sim.run(qc, noise_model=noise_model).result()
        sv=np.asarray(result.get_statevector(qc), dtype=complex)
        nrm=np.linalg.norm(sv)
        if nrm>1e-12: sv=sv/nrm
        from ..bloch import compute_bloch_vectors
        bloch=compute_bloch_vectors(sv, self.n_qubits)
        return BackendResult(statevector=sv, n_qubits=self.n_qubits, bloch_vectors=bloch, metadata={"backend":"aer","feature_scale":feature_scale})

    def simulate_circuit_ops(self, ops, n_qubits:int=None) -> BackendResult:
        if not HAS_AER:
            from .pennylane_backend import PennylaneBackend
            return PennylaneBackend(n_qubits or self.n_qubits).simulate_circuit_ops(ops, n_qubits)
        from qiskit import QuantumCircuit
        n=n_qubits or self.n_qubits
        qc=QuantumCircuit(n)
        for op in ops:
            g=op.get("gate","")
            w=op.get("wires",[0])
            p=op.get("params",[])
            if g=="H": qc.h(w[0])
            elif g=="X": qc.x(w[0])
            elif g=="Y": qc.y(w[0])
            elif g=="Z": qc.z(w[0])
            elif g=="S": qc.s(w[0])
            elif g=="T": qc.t(w[0])
            elif g=="RX": qc.rx(float(p[0]) if p else 0, w[0])
            elif g=="RY": qc.ry(float(p[0]) if p else 0, w[0])
            elif g=="RZ": qc.rz(float(p[0]) if p else 0, w[0])
            elif g=="Rot":
                # Rot(phi,theta,omega)= U3(theta,phi,omega) up to phase
                phi,theta,omega = (float(p[0]),float(p[1]),float(p[2])) if len(p)>=3 else (0,0,0)
                qc.u(theta, phi, omega, w[0])
            elif g=="CNOT": qc.cx(w[0],w[1])
            elif g=="CZ": qc.cz(w[0],w[1])
            elif g=="SWAP": qc.swap(w[0],w[1])
            elif g=="Toffoli": qc.ccx(w[0],w[1],w[2])
        qc.save_statevector()
        result=self.sim.run(qc).result()
        sv=np.asarray(result.get_statevector(qc), dtype=complex)
        from ..bloch import compute_bloch_vectors
        bloch=compute_bloch_vectors(sv,n)
        return BackendResult(statevector=sv, n_qubits=n, bloch_vectors=bloch, metadata={"backend":"aer_ops"})
