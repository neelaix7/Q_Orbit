from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np
import re

GATE_PALETTE=[
    ("H", "Hadamard — superposition", "H"),
    ("X", "Pauli-X — bit flip", "X"),
    ("Y", "Pauli-Y", "Y"),
    ("Z", "Pauli-Z — phase flip", "Z"),
    ("S", "S phase", "S"),
    ("T", "T π/8", "T"),
    ("RX", "RX(θ) param", "RX"),
    ("RY", "RY(θ) param", "RY"),
    ("RZ", "RZ(θ) param", "RZ"),
    ("CNOT", "CNOT control→target", "CNOT"),
    ("CZ", "CZ", "CZ"),
    ("SWAP", "SWAP", "SWAP"),
]

@dataclass
class CircuitSpec:
    n_qubits: int = 8
    ops: List[Dict[str, Any]] = field(default_factory=list)

    def add_gate(self, gate:str, wires:List[int], params:List[float]=None):
        self.ops.append({"gate": gate, "wires": list(wires), "params": list(params or [])})

    def clear(self):
        self.ops=[]

    def to_qiskit_code(self)->str:
        lines=[f"from qiskit import QuantumCircuit",
               f"qc = QuantumCircuit({self.n_qubits})"]
        for op in self.ops:
            g=op["gate"]; w=op["wires"]; p=op.get("params",[])
            if g=="H": lines.append(f"qc.h({w[0]})")
            elif g=="X": lines.append(f"qc.x({w[0]})")
            elif g=="Y": lines.append(f"qc.y({w[0]})")
            elif g=="Z": lines.append(f"qc.z({w[0]})")
            elif g=="S": lines.append(f"qc.s({w[0]})")
            elif g=="T": lines.append(f"qc.t({w[0]})")
            elif g in ("RX","RY","RZ"):
                ang=p[0] if p else 0.0
                lines.append(f"qc.{g.lower()}({ang:.4f}, {w[0]})  # θ={ang:.3f} rad")
            elif g=="Rot":
                phi,theta,omega=p[0],p[1],p[2] if len(p)>=3 else (0,0,0)
                lines.append(f"qc.u({theta:.4f}, {phi:.4f}, {omega:.4f}, {w[0]})  # Rot(φ,θ,ω)")
            elif g=="CNOT": lines.append(f"qc.cx({w[0]}, {w[1]})")
            elif g=="CZ": lines.append(f"qc.cz({w[0]}, {w[1]})")
            elif g=="SWAP": lines.append(f"qc.swap({w[0]}, {w[1]})")
            elif g=="Toffoli": lines.append(f"qc.ccx({w[0]}, {w[1]}, {w[2]})")
        lines.append("qc.save_statevector()  # AER")
        lines.append("from qiskit_aer import AerSimulator; sv = AerSimulator().run(qc).result().get_statevector(qc)")
        return "\n".join(lines)

    def to_pennylane_code(self)->str:
        lines=["import pennylane as qml",
               f"dev = qml.device('default.qubit', wires={self.n_qubits})",
               "@qml.qnode(dev)",
               "def circuit():"
               ]
        if not self.ops:
            lines.append("    return qml.state()")
        else:
            for op in self.ops:
                g=op["gate"]; w=op["wires"]; p=op.get("params",[])
                if g=="H": lines.append(f"    qml.Hadamard(wires={w[0]})")
                elif g=="X": lines.append(f"    qml.PauliX(wires={w[0]})")
                elif g=="Y": lines.append(f"    qml.PauliY(wires={w[0]})")
                elif g=="Z": lines.append(f"    qml.PauliZ(wires={w[0]})")
                elif g=="S": lines.append(f"    qml.S(wires={w[0]})")
                elif g=="T": lines.append(f"    qml.T(wires={w[0]})")
                elif g in ("RX","RY","RZ"):
                    ang=p[0] if p else 0.0
                    lines.append(f"    qml.{g}( {ang:.4f}, wires={w[0]})")
                elif g=="Rot":
                    phi,theta,omega=p[0],p[1],p[2] if len(p)>=3 else (0,0,0)
                    lines.append(f"    qml.Rot({phi:.4f}, {theta:.4f}, {omega:.4f}, wires={w[0]})")
                elif g=="CNOT": lines.append(f"    qml.CNOT(wires={w})")
                elif g=="CZ": lines.append(f"    qml.CZ(wires={w})")
                elif g=="SWAP": lines.append(f"    qml.SWAP(wires={w})")
            lines.append("    return qml.state()")
        lines.append("sv = circuit()")
        return "\n".join(lines)

    def to_stim_code(self)->str:
        lines=["import stim", f"c = stim.Circuit()"]
        for op in self.ops:
            g=op["gate"]; w=op["wires"]
            if g=="H": lines.append(f"c.append('H', {w})")
            elif g=="X": lines.append(f"c.append('X', {w})")
            elif g=="Y": lines.append(f"c.append('Y', {w})")
            elif g=="Z": lines.append(f"c.append('Z', {w})")
            elif g=="S": lines.append(f"c.append('S', {w})")
            elif g=="CNOT": lines.append(f"c.append('CX', {w})")
            elif g=="CZ": lines.append(f"c.append('CZ', {w})")
            elif g in ("RX","RY","RZ","Rot","T"): lines.append(f"# {g} non-Clifford — not in STIM tableau (use AER/PennyLane)")
        lines.append("sim = stim.TableauSimulator(); sim.do(c); print(sim.current_inverse_tableau())")
        return "\n".join(lines)

    def simulate(self, backend: str="pennylane"):
        backend=backend.lower()
        if backend=="aer":
            from .backends.aer_backend import AerBackend
            return AerBackend(self.n_qubits).simulate_circuit_ops(self.ops, self.n_qubits)
        elif backend=="stim":
            from .backends.stim_backend import StimBackend
            return StimBackend(self.n_qubits).simulate_circuit_ops(self.ops, self.n_qubits)
        else:
            from .backends.pennylane_backend import PennylaneBackend
            return PennylaneBackend(self.n_qubits).simulate_circuit_ops(self.ops, self.n_qubits)
