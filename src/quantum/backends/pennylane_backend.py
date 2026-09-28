from __future__ import annotations
import numpy as np
import pennylane as qml
from .base import BackendResult

class PennylaneBackend:
    """PennyLane default.qubit statevector backend for Bloch extraction."""
    def __init__(self, n_qubits:int=8):
        self.n_qubits=n_qubits
        self.dev=qml.device("default.qubit", wires=n_qubits)
    
    def simulate(self, features: np.ndarray, weights: np.ndarray = None, feature_scale: float = np.pi) -> BackendResult:
        """Simulate AngleEmbedding + optional variational layer. If weights is None -> just embedding."""
        n=self.n_qubits
        # truncate/pad features to n_qubits
        f=np.asarray(features, dtype=float).reshape(-1)
        if len(f)>=n:
            f=f[:n]
        else:
            tmp=np.zeros(n); tmp[:len(f)]=f; f=tmp
        f=np.tanh(f)*feature_scale

        if weights is None:
            n_layers=0
            w=np.zeros((0,n,3))
        else:
            w=np.asarray(weights, dtype=float)
            if w.ndim==3:
                n_layers=w.shape[0]
            else:
                # assume flat -> reshape
                n_layers=2
                w=w.reshape(n_layers,n,3) if w.size==n_layers*n*3 else np.zeros((n_layers,n,3))

        @qml.qnode(self.dev)
        def circ(feat, wts):
            qml.AngleEmbedding(feat, wires=range(n), rotation="Y")
            for l in range(n_layers if weights is not None else 0):
                for i in range(n):
                    qml.Rot(wts[l,i,0], wts[l,i,1], wts[l,i,2], wires=i)
                for i in range(n-1):
                    qml.CNOT(wires=[i,i+1])
            return qml.state()
        
        # need to pass w if n_layers>0 else dummy
        if weights is not None and w.size>0:
            sv = circ(f, w)
        else:
            # define simple version without weights param
            @qml.qnode(self.dev)
            def circ2(feat):
                qml.AngleEmbedding(feat, wires=range(n), rotation="Y")
                return qml.state()
            sv = circ2(f)
        sv=np.asarray(sv, dtype=complex)
        # normalize
        nrm=np.linalg.norm(sv)
        if nrm>1e-12:
            sv=sv/nrm
        # bloch via python (fast for 8q)
        from ..bloch import compute_bloch_vectors
        bloch=compute_bloch_vectors(sv, n)
        return BackendResult(statevector=sv, n_qubits=n, bloch_vectors=bloch, metadata={"backend":"pennylane","feature_scale":feature_scale,"n_layers":n_layers if weights is not None else 0})

    def simulate_circuit_ops(self, ops, n_qubits:int=None) -> BackendResult:
        """ops: list of dict {gate: str, wires: list, params: list}. Generic circuit for live simulator."""
        n=n_qubits or self.n_qubits
        dev=qml.device("default.qubit", wires=n)
        @qml.qnode(dev)
        def circ():
            for op in ops:
                g=op.get("gate","")
                wires=op.get("wires",[0])
                params=op.get("params",[])
                if g=="H":
                    qml.Hadamard(wires=wires[0])
                elif g=="X":
                    qml.PauliX(wires=wires[0])
                elif g=="Y":
                    qml.PauliY(wires=wires[0])
                elif g=="Z":
                    qml.PauliZ(wires=wires[0])
                elif g=="S":
                    qml.S(wires=wires[0])
                elif g=="T":
                    qml.T(wires=wires[0])
                elif g in ("RX","RY","RZ"):
                    ang=params[0] if params else 0.0
                    if g=="RX": qml.RX(ang, wires=wires[0])
                    elif g=="RY": qml.RY(ang, wires=wires[0])
                    else: qml.RZ(ang, wires=wires[0])
                elif g=="Rot":
                    qml.Rot(params[0], params[1], params[2], wires=wires[0])
                elif g=="CNOT":
                    qml.CNOT(wires=wires)
                elif g=="CZ":
                    qml.CZ(wires=wires)
                elif g=="SWAP":
                    qml.SWAP(wires=wires)
                elif g=="Toffoli":
                    qml.Toffoli(wires=wires)
            return qml.state()
        sv=np.asarray(circ(), dtype=complex)
        from ..bloch import compute_bloch_vectors
        bloch=compute_bloch_vectors(sv,n)
        return BackendResult(statevector=sv, n_qubits=n, bloch_vectors=bloch, metadata={"backend":"pennylane_ops","n_ops":len(ops)})
