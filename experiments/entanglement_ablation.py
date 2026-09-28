# entanglement_ablation.py — WITH vs WITHOUT entanglement, same data/hypers, val-selected.
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, torch.nn as nn
import pennylane as qml
from src.evaluation.compare_three import compute_all_metrics

CFG = json.load(open("configs/qorbit_config.json"))

class VQC(nn.Module):
    def __init__(self, nq=8, nl=2, entangle=True):
        super().__init__()
        self.nq, self.nl, self.ent = nq, nl, entangle
        self.dev = qml.device("default.qubit", wires=nq)
        self.w = nn.Parameter(torch.randn(nl, nq, 3) * 0.1)
        self.sc = nn.Parameter(torch.tensor(float(np.pi)))
        self.ro = nn.Linear(nq, 5)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circ(f, w):
            qml.AngleEmbedding(f, wires=range(nq), rotation="Y")
            for l in range(nl):
                for i in range(nq):
                    qml.Rot(w[l, i, 0], w[l, i, 1], w[l, i, 2], wires=i)
                if entangle:
                    for i in range(nq - 1):
                        qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(nq)]
        self._c = circ

    def forward(self, x):
        e = torch.stack(self._c(torch.tanh(x) * self.sc, self.w), dim=1).float()
        return self.ro(e)

def run(ent, Xtr, ytr, Xva, yva, Xte, yte, epochs=6):
    m = VQC(entangle=ent)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()
    Xt, Xv, Xe = (torch.tensor(a, dtype=torch.float32) for a in (Xtr, Xva, Xte))
    yt = torch.tensor(ytr, dtype=torch.long)
    best, bs = -1, None
    for _ in range(epochs):
        m.train(); opt.zero_grad(); crit(m(Xt), yt).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = m(Xv).argmax(1).numpy()
            f1 = compute_all_metrics(yva, p, np.eye(5)[p])["f1_macro"]
            if f1 > best: best, bs = f1, {k: v.cpu() for k, v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval()
    with torch.no_grad():
        lg = m(Xe); pr = lg.argmax(1).numpy(); pb = torch.softmax(lg, 1).numpy()
    mt = compute_all_metrics(yte, pr, pb)
    return {"val_f1_best": float(best), "test_acc": float(mt["accuracy"]), "test_f1": float(mt["f1_macro"])}

def main():
    dd = {k: np.load(f"data/quantum_ready/{k}8.npz") for k in ("train", "val", "test")}
    # subset for speed (stratified-ish random, seed 42)
    rng = np.random.RandomState(42)
    it = rng.choice(len(dd["train"]["X"]), 2000, replace=False)
    iv = rng.choice(len(dd["val"]["X"]), 800, replace=False)
    out = {"WITH_ENTANGLEMENT": run(True, dd["train"]["X"][it], dd["train"]["y"][it].astype(int), dd["val"]["X"][iv], dd["val"]["y"][iv].astype(int), dd["test"]["X"], dd["test"]["y"].astype(int)),
           "WITHOUT_ENTANGLEMENT": run(False, dd["train"]["X"][it], dd["train"]["y"][it].astype(int), dd["val"]["X"][iv], dd["val"]["y"][iv].astype(int), dd["test"]["X"], dd["test"]["y"].astype(int))}
    out["note"] = "same 8D data, same init scale, Adam 0.01, subset for speed; entanglement NOT assumed better"
    json.dump(out, open("results/reports/entanglement_ablation.json", "w", encoding="utf-8"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
