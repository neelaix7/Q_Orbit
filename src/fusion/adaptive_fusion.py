# adaptive_fusion.py — Q-ORBIT Adaptive Quantum-Classical Fusion Engine (learned gating)
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, json, os
from typing import Dict

class AdaptiveFusionGate(nn.Module):
    """
    Input: concat[ classical_rep (d_c), quantum_rep (d_q), signal_quality (1-7) ] → gating
    Output: weight alpha for classical, (1-alpha) for quantum, fused representation.
    Simple learned gate: Linear → tanh → Linear → sigmoid. Trainable, val-selected, frozen test.
    Fusion modes: 'prob' (weighted probs) or 'rep' (weighted reps then classifier)
    """
    def __init__(self, d_classical: int = 21, d_quantum: int = 8, d_quality: int = 7, hidden: int = 16, mode: str = "prob"):
        super().__init__()
        self.mode = mode
        self.d_c, self.d_q, self.d_quality = d_classical, d_quantum, d_quality
        self.gate = nn.Sequential(
            nn.Linear(d_classical + d_quantum + d_quality, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()  # alpha
        )
        # for rep fusion, also need classifier after fusion
        if mode == "rep":
            self.classifier = nn.Sequential(nn.Linear(d_classical + d_quantum, hidden), nn.ReLU(), nn.Linear(hidden, 5))
        else:
            self.classifier = None

    def forward(self, classical_rep: torch.Tensor, quantum_rep: torch.Tensor, quality: torch.Tensor):
        # quality: (batch, d_quality)
        x = torch.cat([classical_rep, quantum_rep, quality], dim=1)
        alpha = self.gate(x)  # (batch,1) ∈(0,1)
        if self.mode == "prob":
            # caller will provide probs: fusion = alpha * p_c + (1-alpha)*p_q
            return alpha
        else:
            # rep fusion: weighted reps then classifier on the FUSED vector
            fused = torch.cat([alpha * classical_rep, (1 - alpha) * quantum_rep], dim=1)
            logits = self.classifier(fused)
            return alpha, fused, logits

class WeightedProbFusion:
    """Non-parametric weighted prob fusion with learned scalar alpha (single param) — simplest, data-free baseline."""
    def __init__(self, alpha: float = 0.6): self.alpha = float(alpha)
    def fuse(self, p_c: np.ndarray, p_q: np.ndarray) -> np.ndarray:
        return self.alpha * p_c + (1-self.alpha) * p_q

def train_adaptive_gate(classical_probs_train, quantum_probs_train, y_train, quality_train, classical_probs_val, quantum_probs_val, y_val, quality_val, hidden=16, lr=0.01, epochs=50):
    """Trains gate to minimize val CE via Adam, returns best gate + alpha hist. Minimal to satisfy requirement."""
    import torch.optim as optim
    d_q = quantum_probs_train.shape[1] if quantum_probs_train.ndim>1 else 5
    # we use dummy reps = probs for training (prob fusion mode)
    # gate input dims: classical 5 (prob) + quantum 5 + quality 7 → we set d_c=5,d_q=5
    gate = AdaptiveFusionGate(d_classical=5, d_quantum=5, d_quality=quality_train.shape[1], hidden=hidden, mode="prob")
    opt = optim.Adam(gate.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    # build tensors
    Xc_tr = torch.tensor(classical_probs_train, dtype=torch.float32)
    Xq_tr = torch.tensor(quantum_probs_train, dtype=torch.float32)
    Q_tr = torch.tensor(quality_train, dtype=torch.float32)
    Xc_val = torch.tensor(classical_probs_val, dtype=torch.float32)
    Xq_val = torch.tensor(quantum_probs_val, dtype=torch.float32)
    Q_val = torch.tensor(quality_val, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    best_f1 = -1; best_state = None
    from src.evaluation.compare_three import compute_all_metrics
    for ep in range(epochs):
        gate.train(); opt.zero_grad()
        alpha = gate(Xc_tr, Xq_tr, Q_tr)  # (batch,1)
        fused = alpha * Xc_tr + (1-alpha) * Xq_tr
        loss = crit(torch.log(fused+1e-9), y_tr)  # fused is prob, use NLL
        loss.backward(); opt.step()
        # val
        gate.eval()
        with torch.no_grad():
            alpha_v = gate(Xc_val, Xq_val, Q_val)
            fused_v = alpha_v * Xc_val + (1-alpha_v) * Xq_val
            pred = torch.argmax(fused_v, dim=1).numpy()
            metrics = compute_all_metrics(y_val, pred, fused_v.numpy())
            if metrics["f1_macro"] > best_f1:
                best_f1 = metrics["f1_macro"]; best_state = {k:v.cpu() for k,v in gate.state_dict().items()}
    if best_state: gate.load_state_dict(best_state)
    return gate, best_f1

def infer_fused_probs(gate: AdaptiveFusionGate, p_c: np.ndarray, p_q: np.ndarray, quality: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gate.eval()
    with torch.no_grad():
        alpha = gate(torch.tensor(p_c, dtype=torch.float32), torch.tensor(p_q, dtype=torch.float32), torch.tensor(quality, dtype=torch.float32)).numpy()  # (n,1)
    fused = alpha * p_c + (1-alpha) * p_q
    return fused, alpha.squeeze()
