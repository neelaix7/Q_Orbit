# train_adaptive_real.py — REAL adaptive fusion: RF(classical) vs PureVQC(quantum), gate on quality.
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, joblib
from src.classical.feature_engineering import extract_all_features
from src.fusion.signal_quality import analyze_signal_quality
from src.fusion.adaptive_fusion import train_adaptive_gate, infer_fused_probs
from src.evaluation.compare_three import compute_all_metrics
from src.features.reducer import QuantumReducer
from src.quantum.vqc import PureVQC

CFG = json.load(open("configs/qorbit_config.json")); SEED = CFG["seed"]

def FQ(curves):
    f = np.array([extract_all_features(c) for c in curves], dtype=np.float32)
    q = np.array([list(analyze_signal_quality(c).values())[:7] for c in curves], dtype=np.float32)
    if q.shape[1] != 7: q = np.pad(q, ((0, 0), (0, 7 - q.shape[1])))
    return f, q

@torch.no_grad()
def vqc_probs(red, wpath, Xraw_feats, m, s):
    ck = torch.load(wpath, map_location="cpu", weights_only=False)
    mdl = PureVQC(8, 2); mdl.load_state_dict(ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck); mdl.eval()
    Xq = (red.transform(Xraw_feats) - m) / s
    out = []
    for i in range(0, len(Xq), 256):
        lg, pb = mdl(torch.tensor(Xq[i:i+256], dtype=torch.float32))
        out.append(pb.numpy())
    return np.vstack(out)

def main():
    tr = np.load(CFG["dataset"]["train_path"]); va = np.load(CFG["dataset"]["val_path"]); te = np.load(CFG["dataset"]["test_path"])
    Ftr, Qtr = FQ(tr["curves"]); Fva, Qva = FQ(va["curves"]); Fte, Qte = FQ(te["curves"])
    ytr, yva, yte = tr["labels"].astype(int), va["labels"].astype(int), te["labels"].astype(int)
    sc = joblib.load("models/feature_scaler.joblib")
    if isinstance(sc, dict):  # stored as {mean, scale, ...}
        m0, s0 = sc["mean"], sc["scale"]
        Xtr, Xva, Xte = (Ftr-m0)/s0, (Fva-m0)/s0, (Fte-m0)/s0
    else:
        Xtr, Xva, Xte = sc.transform(Ftr), sc.transform(Fva), sc.transform(Fte)
    qm = np.nan_to_num(np.vstack([Qtr, Qva]).mean(0)); qs = np.nan_to_num(np.vstack([Qtr, Qva]).std(0)) + 1e-8
    Qtr, Qva, Qte = (Qtr-qm)/qs, (Qva-qm)/qs, (Qte-qm)/qs
    _rf = joblib.load("models/classical_rf_model.joblib")
    rf = _rf["model"] if isinstance(_rf, dict) and "model" in _rf else _rf
    pc_tr, pc_va, pc_te = rf.predict_proba(Xtr), rf.predict_proba(Xva), rf.predict_proba(Xte)
    red = QuantumReducer.load("models/pure_vqc_reducer.joblib")
    qd = np.load("data/quantum_ready/train8.npz") if os.path.exists("data/quantum_ready/train8.npz") else None
    # norm stats used in train_pure_quantum: recompute from train for stability
    Xq_tr_full = red.transform(Ftr)
    m, s = Xq_tr_full.mean(0), Xq_tr_full.std(0) + 1e-8
    pq_tr = vqc_probs(red, "models/pure_quantum_vqc.pt", Ftr, m, s)
    pq_va = vqc_probs(red, "models/pure_quantum_vqc.pt", Fva, m, s)
    pq_te = vqc_probs(red, "models/pure_quantum_vqc.pt", Fte, m, s)
    gate, best = train_adaptive_gate(pc_tr, pq_tr, ytr, Qtr, pc_va, pq_va, yva, Qva, hidden=16, lr=0.01, epochs=40)
    pf, alpha = infer_fused_probs(gate, pc_te, pq_te, Qte)
    mt = compute_all_metrics(yte, pf.argmax(1), pf)
    os.makedirs("models", exist_ok=True); torch.save(gate.state_dict(), "models/adaptive_gate.pt")
    json.dump({"test_acc": float(mt["accuracy"]), "test_f1": float(mt["f1_macro"]),
               "val_f1_best": float(best), "alpha_mean": float(alpha.mean()), "alpha_std": float(alpha.std()),
               "note": "REAL: RF(200) classical vs trained PureVQC-8q2L quantum + 7 quality features, val-selected"},
              open("results/reports/adaptive_fusion.json", "w", encoding="utf-8"), indent=2)
    st = 0.6 * pc_te + 0.4 * pq_te
    json.dump({"adaptive": mt, "static_0.6": compute_all_metrics(yte, st.argmax(1), st)},
              open("results/reports/adaptive_vs_static.json", "w", encoding="utf-8"), indent=2)
    print(f"Adaptive REAL acc {mt['accuracy']:.4f} f1 {mt['f1_macro']:.4f} alpha {alpha.mean():.3f}±{alpha.std():.3f}")

if __name__ == "__main__":
    main()
