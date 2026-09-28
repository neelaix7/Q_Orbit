# hybrid_variants.py — Hybrid A/B/C/D for Q-ORBIT
from __future__ import annotations
import torch, torch.nn as nn
from src.quantum.hybrid_model import HybridQuantumClassifier
from src.classical.cnn_baseline import LightCurveCNN

class HybridA(nn.Module):
    """A: Features → PCA → Quantum map → Classical classifier (PCA already in reducer, here just wrapper)"""
    def __init__(self, n_qubits=8, n_layers=2, hidden=16):
        super().__init__()
        # same as HybridQuantumClassifier but expects PCA input (n_features=n_qubits)
        self.model = HybridQuantumClassifier(n_features=n_qubits, n_qubits=n_qubits, n_layers=n_layers, n_classical_hidden=hidden)
    def forward(self, x): return self.model(x)

class HybridB(nn.Module):
    """B: Features → trainable quantum layer → MLP (canonical hybrid, retained)"""
    def __init__(self, n_features=21, n_qubits=8, n_layers=2, hidden=16):
        super().__init__()
        self.model = HybridQuantumClassifier(n_features=n_features, n_qubits=n_qubits, n_layers=n_layers, n_classical_hidden=hidden)
    def forward(self, x): return self.model(x)

class HybridC(nn.Module):
    """C: 1D-CNN feature extractor (frozen early) → dim reduction → quantum → classifier
    CNN extractor outputs 64*32=2048 → Linear→n_qubits → quantum """
    def __init__(self, n_qubits=8, n_layers=2, hidden=16):
        super().__init__()
        self.cnn = LightCurveCNN(n_classes=5)
        # freeze first two conv blocks optionally, but here trainable for simplicity
        self.proj = nn.Linear(64*32, n_qubits)  # 64*32 from cnn flat before FC
        self.quantum = HybridQuantumClassifier(n_features=n_qubits, n_qubits=n_qubits, n_layers=n_layers, n_classical_hidden=hidden)
        # we bypass cnn's FC and use quantum

    def forward(self, x_curves: torch.Tensor):  # (batch,1,256)
        # extract conv features
        h = self.cnn.conv1(x_curves); h = self.cnn.bn1(h); h = torch.relu(h); h = self.cnn.pool1(h)
        h = self.cnn.conv2(h); h = self.cnn.bn2(h); h = torch.relu(h); h = self.cnn.pool2(h)
        h = self.cnn.conv3(h); h = self.cnn.bn3(h); h = torch.relu(h); h = self.cnn.pool3(h)
        flat = h.view(h.size(0), -1)  # 64*32
        q_in = self.proj(flat)
        logits, probs = self.quantum(q_in)
        return logits, probs

class HybridD(nn.Module):
    """D: Dual-branch classical (21) + quantum (PCA 8) → concat → fusion → classifier"""
    def __init__(self, n_qubits=8, n_layers=2, hidden=16):
        super().__init__()
        self.quantum_branch = HybridQuantumClassifier(n_features=n_qubits, n_qubits=n_qubits, n_layers=n_layers, n_classical_hidden=hidden)
        # quantum_branch already has classical head; we use its quantum expectations as features
        self.fusion = nn.Sequential(nn.Linear(21 + n_qubits, hidden), nn.ReLU(), nn.Linear(hidden, 5))

    def forward(self, x_features: torch.Tensor, x_pca: torch.Tensor):
        # x_features (batch,21), x_pca (batch,n_qubits)
        # FIX: quantum branch must remain differentiable (no torch.no_grad) so Rot
        # weights train; fuse expectations then classify fused vector.
        q_exp = self.quantum_branch.quantum_layer(x_pca)  # (batch, n_qubits)
        fused = torch.cat([x_features, q_exp], dim=1)
        logits = self.fusion(fused)
        probs = torch.softmax(logits, dim=1)
        return logits, probs
