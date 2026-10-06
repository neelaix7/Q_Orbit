# build_quantum_ready.py — MASTER 21D -> selection -> zscore -> PCA8 -> quantum_ready/
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, joblib
from src.features.reducer import QuantumReducer
from src.classical.feature_engineering import extract_all_features

CFG = json.load(open("configs/qorbit_config.json"))

def feats(c): return np.array([extract_all_features(x) for x in c], dtype=np.float32)

def main():
    fa = json.load(open("results/reports/feature_analysis.json")) if os.path.exists("results/reports/feature_analysis.json") else None
    tr = np.load(CFG["dataset"]["train_path"]); va = np.load(CFG["dataset"]["val_path"]); te = np.load(CFG["dataset"]["test_path"])
    Ftr, Fva, Fte = feats(tr["curves"]), feats(va["curves"]), feats(te["curves"])
    red = QuantumReducer(n_components=8); red.fit(Ftr)  # train-only fit
    Qtr, Qva, Qte = red.transform(Ftr), red.transform(Fva), red.transform(Fte)
    # second standardization for VQC stability (matches train_pure_quantum.py)
    m, s = Qtr.mean(0), Qtr.std(0) + 1e-8
    os.makedirs("data/quantum_ready", exist_ok=True)
    np.savez_compressed("data/quantum_ready/train8.npz", X=(Qtr-m)/s, y=tr["labels"])
    np.savez_compressed("data/quantum_ready/val8.npz", X=(Qva-m)/s, y=va["labels"])
    np.savez_compressed("data/quantum_ready/test8.npz", X=(Qte-m)/s, y=te["labels"])
    red.save("data/quantum_ready/reducer8.joblib")
    joblib.dump({"mean": m, "scale": s}, "data/quantum_ready/norm8.joblib")
    info = {"n_qubits": 8, "encoding": "angle_RY: theta=tanh(z)*pi, |psi(x)>=x_i Ry(x_i)|0>",
            "explained_variance": red.info(), "why_8": "matches 8-qubit VQC/hybrid ansatz (depth 5, 94/1798 params); PCA8 retains majority variance; 4/6/8 ablation in search_pure_quick.py",
            "seed": 123, "regime": "clean", "fit_on": "train only", "source": "SAME master 21D features"}
    json.dump(info, open("data/quantum_ready/README.json", "w", encoding="utf-8"), indent=2)
    print("quantum_ready saved:", Qtr.shape, Qva.shape, Qte.shape, red.info())

if __name__ == "__main__":
    main()
