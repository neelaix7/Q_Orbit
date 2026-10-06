# optimize_hybrid_quick.py — bounded legitimate Hybrid improvement attempt.
# Protocol: train on TRAIN, select on VAL F1, ship only if val F1 beats the
# current champion (0.7443). TEST is evaluated once, for the val-winner only.
import os, sys, json, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.classical.feature_engineering import extract_all_features
from src.quantum.hybrid_model import HybridQuantumClassifier
from src.evaluation.compare_three import compute_all_metrics

SEED = 123
torch.manual_seed(SEED); np.random.seed(SEED)
CHAMPION_VAL_F1 = 0.7442565859360253

CANDIDATES = [
    {"name": "q8l3h32_do", "n_qubits": 8, "n_layers": 3, "n_hidden": 32,
     "lr": 0.005, "epochs": 15, "batch": 64, "dropout": 0.2, "class_weight": True},
    {"name": "q8l2h64_cw", "n_qubits": 8, "n_layers": 2, "n_hidden": 64,
     "lr": 0.005, "epochs": 15, "batch": 64, "dropout": 0.1, "class_weight": True},
]


class HybridDropout(HybridQuantumClassifier):
    def __init__(self, *a, dropout=0.1, **k):
        super().__init__(*a, **k)
        nq, nh = self.n_qubits, self.n_classical_hidden
        self.classical_head = nn.Sequential(
            nn.Linear(nq, nh), nn.BatchNorm1d(nh), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(nh, nh), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(nh, self.n_classes))


def feats(curves):
    return np.array([extract_all_features(c) for c in curves], dtype=np.float32)


def main():
    tr = np.load("data/splits/train.npz", allow_pickle=True)
    va = np.load("data/splits/val.npz", allow_pickle=True)
    te = np.load("data/splits/test.npz", allow_pickle=True)
    ctr, ytr = tr["curves"], tr["labels"].astype(int)
    cva, yva = va["curves"], va["labels"].astype(int)
    cte, yte = te["curves"], te["labels"].astype(int)
    print(f"train {len(ctr)} val {len(cva)} test {len(cte)}", flush=True)

    Ftr, Fva = feats(ctr), feats(cva)
    mean, std = Ftr.mean(0), Ftr.std(0) + 1e-8
    Xtr, Xva = (Ftr - mean) / std, (Fva - mean) / std
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)

    best = None
    for cfg in CANDIDATES:
        model = HybridDropout(n_features=21, n_qubits=cfg["n_qubits"],
                              n_layers=cfg["n_layers"], n_classical_hidden=cfg["n_hidden"],
                              dropout=cfg["dropout"])
        cw = None
        if cfg["class_weight"]:
            cnt = np.bincount(ytr, minlength=5).astype(float)
            cw = torch.tensor(cnt.sum() / (5 * cnt), dtype=torch.float32)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        crit = nn.CrossEntropyLoss(weight=cw)
        n = len(Xtr_t)
        best_val, best_state = -1, None
        t0 = time.time()
        for ep in range(1, cfg["epochs"] + 1):
            model.train()
            idx = torch.randperm(n)
            tot, nb = 0.0, 0
            for s in range(0, n, cfg["batch"]):
                bi = idx[s:s + cfg["batch"]]
                opt.zero_grad()
                logits, _ = model(Xtr_t[bi])
                loss = crit(logits, ytr_t[bi])
                loss.backward(); opt.step()
                tot += loss.item(); nb += 1
            model.eval()
            with torch.no_grad():
                _, p = model(Xva_t)
                m = compute_all_metrics(yva, torch.argmax(p, 1).numpy(), p.numpy())
            print(f"[{cfg['name']}] ep{ep}: loss {tot/max(nb,1):.4f} "
                  f"val acc {m['accuracy']:.4f} val F1 {m['f1_macro']:.4f}", flush=True)
            if m["f1_macro"] > best_val:
                best_val, best_state = m["f1_macro"], {k: v.cpu() for k, v in model.state_dict().items()}
        print(f"[{cfg['name']}] best val F1 {best_val:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if best is None or best_val > best[0]:
            best = (best_val, cfg, best_state)

    val_f1, cfg, state = best
    print(f"Winner: {cfg['name']} val F1 {val_f1:.4f} vs champion {CHAMPION_VAL_F1:.4f}", flush=True)
    if val_f1 <= CHAMPION_VAL_F1:
        print("No improvement — keeping shipped model. Nothing written.", flush=True)
        return
    # Ship: single TEST evaluation of the val-winner only
    Fte = feats(cte)
    Xte = (Fte - mean) / std
    model = HybridDropout(n_features=21, n_qubits=cfg["n_qubits"],
                          n_layers=cfg["n_layers"], n_classical_hidden=cfg["n_hidden"],
                          dropout=cfg["dropout"])
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        _, p = model(torch.tensor(Xte, dtype=torch.float32))
        p = p.numpy(); pred = np.argmax(p, axis=1)
    tm = compute_all_metrics(yte, pred, p)
    print(f"TEST acc {tm['accuracy']:.4f} F1 {tm['f1_macro']:.4f}", flush=True)
    import shutil
    shutil.copy("models/hybrid_quantum_model.pt", "models/hybrid_quantum_model.prev.pt")
    torch.save({"model_state_dict": state, "n_features": 21,
                "n_qubits": cfg["n_qubits"], "n_layers": cfg["n_layers"],
                "n_classical_hidden": cfg["n_hidden"],
                "accuracy": tm["accuracy"] * 100, "val_f1": val_f1,
                "config": cfg, "param_count": sum(v.numel() for v in state.values())}, 
               "models/hybrid_quantum_model.pt")
    print("Shipped new champion.", flush=True)


if __name__ == "__main__":
    main()
