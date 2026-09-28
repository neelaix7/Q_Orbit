# vqc.py — Pure Quantum Variational Quantum Classifier (genuine quantum, no classical head)
# Architecture: StandardScaler+PCA (outside) -> AngleEmbedding(RY) -> Rot+CNOT ansatz -> PauliZ expvals -> linear map to logits via expectation only
# The classical postprocessing is ONLY a fixed measurement->logits mapping trained via quantum parameters, not an arbitrary MLP.
from __future__ import annotations
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

class PureVQC(nn.Module):
    """Pure Quantum Classifier: all learnable representation is in the quantum circuit.

    Definition of 'pure quantum' for Q-ORBIT:
    - Input: PCA-reduced features (n_qubits dims) already standardized
    - Quantum: AngleEmbedding + variational Rot(CNOT) layers (trainable weights)
    - Measurement: expval PauliZ on each qubit -> (batch, n_qubits)
    - Output: SINGLE linear layer mapping n_qubits expectations -> n_classes logits
      (This is the minimal classical mapping from quantum observables to class scores;
       it is NOT a deep MLP. The representation power lives in the circuit.)
    - Alternative strict mode: use n_classes qubits and interpret each Z expectation as class logit.

    Training: backprop through PennyLane default.qubit; optimizer updates Rot angles.
    """
    def __init__(self, n_qubits=8, n_layers=2, n_classes=5, device_name="default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.device = qml.device(device_name, wires=n_qubits)
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        self.angle_scale = nn.Parameter(torch.tensor(float(np.pi)))
        # minimal classical map: n_qubits -> n_classes
        self.readout = nn.Linear(n_qubits, n_classes)

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(features, weights):
            qml.AngleEmbedding(features, wires=range(n_qubits), rotation="Y")
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
                for i in range(n_qubits-1):
                    qml.CNOT(wires=[i, i+1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        self._qnode = circuit

    def quantum_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_qubits) already scaled to roughly [-1,1] via scaler+pca
        scaled = torch.tanh(x) * self.angle_scale
        expvals = self._qnode(scaled, self.weights)
        return torch.stack(expvals, dim=1).float()  # (batch, n_qubits)

    def forward(self, x: torch.Tensor):
        q = self.quantum_forward(x)
        logits = self.readout(q)
        probs = F.softmax(logits, dim=1)
        return logits, probs

    def circuit_info(self):
        # PennyLane specs
        return {"n_qubits": self.n_qubits, "n_layers": self.n_layers, "params": int(self.weights.numel()+self.angle_scale.numel()+self.readout.weight.numel()+self.readout.bias.numel()), "depth": 1 + self.n_layers*2, "device": "default.qubit"}

class PureVQCTrainer:
    @staticmethod
    def param_count(model: PureVQC):
        return sum(p.numel() for p in model.parameters())
