import sys
sys.path.insert(0, r"D:\Capstone")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.quantum.hybrid_model import HybridQuantumClassifier

model = HybridQuantumClassifier(n_features=21, n_qubits=8, n_layers=2, n_classical_hidden=64, n_classes=5)
print("Hybrid model created. Parameters:")
for name, p in model.named_parameters():
    print(f"  {name}: {tuple(p.shape)}")

batch = torch.randn(16, 21)
logits, probs = model(batch)
print(f"\nForward pass: logits={tuple(logits.shape)}, probs={tuple(probs.shape)}")
print("probs row sums:", probs.sum(dim=1).detach().numpy())

loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0]))
loss.backward()
print("\nGradients computed successfully")
for name, p in model.named_parameters():
    if p.grad is not None and p.grad.abs().mean().item() > 0:
        print(f"  {name}: grad shape={tuple(p.grad.shape)}, mean={p.grad.abs().mean().item():.6f}")

# A few optimizer steps to confirm weights move
opt = optim.Adam(model.parameters(), lr=0.01)
l = list(model.parameters())[0].clone()
for _ in range(5):
    opt.zero_grad()
    lg, pr = model(batch)
    loss = nn.CrossEntropyLoss()(lg, torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0]))
    loss.backward()
    opt.step()
print("\nOptimizer steps worked, loss now:", loss.item())

print("\nHybrid model SUCCESS")