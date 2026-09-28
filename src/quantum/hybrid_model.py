# hybrid_model.py - PennyLane quantum layer (8 qubits) + classical head
from __future__ import annotations
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pennylane as qn
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------
# Trainable quantum layer
# -----------------------

class QuantumLayer(nn.Module):
    """
    Trainable PennyLane quantum circuit integrated with PyTorch.

    Architecture:
    1. Learned classical projection: Linear(n_features -> n_qubits)
    2. Angle embedding (RY) of the 8 projected features into 8 qubits
    3. Variational ansatz: Rot gates + CNOT entangling, repeated n_layers times
    4. Measurement: expectation of Pauli-Z on each qubit -> (batch, n_qubits)

    Gradients flow from the classical head through the measured expectation
    values into the rotation angles using PennyLane's backprop differentiation.
    """
    def __init__(
        self,
        n_features: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Learned projection from raw features into qubit space
        self.projection = nn.Linear(n_features, n_qubits)

        # Trainable scaling of the normalized projection: angles land in [0, pi]
        self.angle_scale = nn.Parameter(torch.tensor(np.pi))

        # Trainable variational weights: (n_layers, n_qubits, 3) for Rot gates
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)

        self.device = qn.device(device_name, wires=n_qubits)

    def _circuit(self):
        """Build the PennyLane circuit as a QNode on this layer's device."""
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        device = self.device

        @qn.qnode(device, interface="torch", diff_method="backprop")
        def circuit(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            qn.AngleEmbedding(features, wires=range(n_qubits), rotation="Y")
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qn.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)
                for i in range(n_qubits - 1):
                    qn.CNOT(wires=[i, i + 1])
            return [qn.expval(qn.PauliZ(i)) for i in range(n_qubits)]

        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: project features, encode into qubits, run circuit.

        Parameters
        ----------
        x: torch.Tensor of shape (batch, n_features)

        Returns
        -------
        expvals: torch.Tensor of shape (batch, n_qubits) - Pauli-Z expectations
        """
        projected = self.projection(x)
        # Normalize angles to [-1, 1] then scale with a trainable factor so the
        # embedding stays in [0, pi] and gradients flow into the scale too
        projected = torch.tanh(projected) * self.angle_scale
        circuit = self._circuit()
        expvals = circuit(projected, self.weights)
        return torch.stack(expvals, dim=1).float()


# -----------------------
# Full hybrid model: quantum layer + classical head
# -----------------------

class HybridQuantumClassifier(nn.Module):
    """
    Hybrid Quantum + Classical neural network for light curve classification.

    Architecture:
    - Quantum layer: projects + angle-encodes features, runs variational circuit
    - Classical head: Dense(ReLU) -> Dense(softmax) over 5 classes
    """
    def __init__(
        self,
        n_features: int = 21,
        n_qubits: int = 8,
        n_layers: int = 2,
        n_classical_hidden: int = 16,
        n_classes: int = 5,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classical_hidden = n_classical_hidden
        self.n_classes = n_classes

        self.quantum_layer = QuantumLayer(
            n_features=n_features,
            n_qubits=n_qubits,
            n_layers=n_layers,
            device_name=device_name,
        )

        # Classical head over the 8 measured quantum expectation values
        self.classical_head = nn.Sequential(
            nn.Linear(n_qubits, n_classical_hidden),
            nn.BatchNorm1d(n_classical_hidden),
            nn.ReLU(),
            nn.Linear(n_classical_hidden, n_classical_hidden),
            nn.ReLU(),
            nn.Linear(n_classical_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the hybrid model.

        Parameters
        ----------
        x: torch.Tensor of shape (batch, n_features)

        Returns
        -------
        logits: torch.Tensor of shape (batch, n_classes)
        probs: torch.Tensor of shape (batch, n_classes)
        """
        q_features = self.quantum_layer(x)
        logits = self.classical_head(q_features)
        probs = F.softmax(logits, dim=1)
        return logits, probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels."""
        with torch.no_grad():
            logits, probs = self.forward(x)
            return torch.argmax(probs, dim=1)


# -----------------------
# Quantum kernel SVM alternative
# -----------------------

class QuantumKernelSVM:
    """
    Quantum kernel SVM using PennyLane's kernel estimation.

    The quantum kernel computes K(x, x') = |⟨ψ(x)|ψ(x')⟩|^2 via the
    transition-amplitude trick: encode x, inverse-encode x', and read the
    probability of the all-zero state.
    """
    def __init__(
        self,
        n_qubits: int = 8,
        device_name: str = "default.qubit",
    ):
        self.n_qubits = n_qubits
        self.device = qn.device(device_name, wires=n_qubits)
        self.support_vectors: Optional[np.ndarray] = None
        self.support_labels: Optional[np.ndarray] = None

    def _kernel_circuit(self):
        n_qubits = self.n_qubits
        device = self.device

        @qn.qnode(device)
        def circuit(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
            qn.AngleEmbedding(x1, wires=range(n_qubits), rotation="Y")
            qn.adjoint(qn.AngleEmbedding)(x2, wires=range(n_qubits), rotation="Y")
            return qn.probs(wires=range(n_qubits))

        return circuit

    def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """K(x1, x2) = |⟨ψ(x1)|ψ(x2)⟩|^2 = probability of the all-zero state."""
        circuit = self._kernel_circuit()
        probs = circuit(x1, x2)
        return float(probs[0])

    def fit(self, X: np.ndarray, y: np.ndarray, sample_size: int = 300):
        """Fit a quantum-kernel SVM using scikit-learn with a precomputed kernel."""
        from sklearn.svm import SVC

        n_samples = len(X)
        # Sub-sample for tractability (O(n^2) kernel evaluations)
        idx = np.random.RandomState(0).choice(n_samples, min(sample_size, n_samples), replace=False)
        self.support_vectors = X[idx]
        self.support_labels = y[idx]

        # Compute kernel matrix on the support subset
        n_sv = len(self.support_vectors)
        K = np.zeros((n_sv, n_sv))
        circuit = self._kernel_circuit()
        for i in range(n_sv):
            for j in range(n_sv):
                probs = circuit(self.support_vectors[i], self.support_vectors[j])
                K[i, j] = probs[0]

        self.clf = SVC(kernel="precomputed", C=1.0, probability=True, random_state=0)
        self.clf.fit(K, self.support_labels)
        self._circuit = circuit

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X using the trained quantum kernel."""
        if self.support_vectors is None:
            raise RuntimeError("QuantumKernelSVM must be fitted before prediction.")
        # Compute kernel between X and support vectors
        n_test = len(X)
        n_sv = len(self.support_vectors)
        K = np.zeros((n_test, n_sv))
        circuit = self._circuit
        for i in range(n_test):
            for j in range(n_sv):
                probs = circuit(X[i], self.support_vectors[j])
                K[i, j] = probs[0]
        return self.clf.predict(K)