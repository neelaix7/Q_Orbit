# feature_map.py - angle/amplitude encoding of light curve -> qubits
from __future__ import annotations
from typing import Optional
import numpy as np
import pennylane as qn


def angle_feature_map(x: np.ndarray, n_qubits: int, name: str = "feature_map") -> qn.QNode:
    """
    Angle (Amplitude) embedding: encode each feature into the amplitude of a qubit state.
    |ψ⟩ = cos(x_i/2)|0⟩ + sin(x_i/2)|1⟩
    
    Parameters
    ----------
    x: np.ndarray of shape (n_features,) - input features, normalized to [0,1]
    n_qubits: number of qubits (typically 4-8)
    name: name for the PennyLane node
    
    Returns
    -------
    qnode: PennyLane QNode with the feature map applied
    """
    assert len(x) <= n_qubits, f"Number of features ({len(x)}) exceeds qubits ({n_qubits})"
    
    @qn.qnode(name=name)
    def circuit():
        for i in range(len(x)):
            qn.RY(x[i], wires=i)
        return qn.state()
    
    return circuit


def reuploading_feature_map(
    x: np.ndarray,
    n_qubits: int,
    n_layers: int = 2,
    name: str = "reuploading",
) -> qn.QNode:
    """
    ZZFeatureMap-style re-uploading: features are re-encoded multiple layers.
    Each layer encodes all features, followed by entangling gates.
    """
    assert len(x) <= n_qubits, f"Features ({len(x)}) exceed qubits ({n_qubits})"
    
    @qn.qnode("default.qubit")
    def circuit():
        for layer in range(n_layers):
            for i in range(len(x)):
                qn.RY(x[i] * (layer + 1), wires=i)
            for i in range(n_qubits - 1):
                qn.CNOT(wires=[i, i + 1])
        return qn.state()
    
    return circuit


def fourier_feature_map(x: np.ndarray, n_qubits: int, name: str = "fourier") -> qn.QNode:
    """
    Fourier feature map: encode (cos(x), sin(x)) pairs to capture phase information.
    Important for light curves because brightness periodicity carries orbital information.
    """
    assert len(x) <= n_qubits, f"Features ({len(x)}) exceed qubits ({n_qubits})"
    
    @qn.qnode(name=name)
    def circuit():
        for i in range(len(x)):
            qn.RY(x[2 * i], wires=i)
            qn.RY(x[2 * i + 1] if 2 * i + 1 < len(x) else 0, wires=i)
        return qn.state()
    
    return circuit


def validate_feature_map(x: np.ndarray, n_qubits: int, method: str = "angle") -> np.ndarray:
    """Validate and normalize features for quantum encoding."""
    x = np.asarray(x, dtype=float)
    
    if len(x) == n_qubits:
        return x
    
    if len(x) < n_qubits:
        padded = np.zeros(n_qubits)
        padded[:len(x)] = x
        return padded
    
    # Compress: take uniformly spaced samples
    indices = np.linspace(0, len(x) - 1, n_qubits, dtype=int)
    return x[indices]


def encode_to_qubits(
    curve_features: np.ndarray,
    n_qubits: int = 8,
    method: str = "angle",
) -> qn.QNode:
    """Encode light curve features into qubit states for quantum processing."""
    from src.simulator.config import N_QUBITS
    n_qubits = n_qubits or N_QUBITS
    
    features = validate_feature_map(curve_features, n_qubits, method)
    
    if method == "angle":
        return angle_feature_map(features, n_qubits)
    elif method == "fourier":
        return fourier_feature_map(features, n_qubits)
    elif method == "reuploading":
        return reuploading_feature_map(features, n_qubits, n_layers=2)
    else:
        return angle_feature_map(features, n_qubits)