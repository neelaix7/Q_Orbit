# quantum_kernel_demo.py — optional small quantum-kernel SVM (subsampled, honest cost).
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.quantum.hybrid_model import QuantumKernelSVM
from src.evaluation.compare_three import compute_all_metrics

def main(n_train=80, n_test=80):
    q = np.load("data/quantum_ready/train8.npz"); v = np.load("data/quantum_ready/test8.npz")
    rng = np.random.RandomState(42)
    it = rng.choice(len(q["X"]), n_train, replace=False); ie = rng.choice(len(v["X"]), n_test, replace=False)
    Xtr, ytr, Xte, yte = q["X"][it], q["y"][it].astype(int), v["X"][ie], v["y"][ie].astype(int)
    # tanh*pi angle normalization consistent with VQC
    Xtr = np.tanh(Xtr) * np.pi; Xte = np.tanh(Xte) * np.pi
    t0 = time.time()
    ksvm = QuantumKernelSVM(n_qubits=8); ksvm.fit(Xtr, ytr, sample_size=n_train)
    pred = ksvm.predict(Xte)
    # predict_proba unavailable -> one-hot for metrics compat
    proba = np.eye(5)[pred]
    m = compute_all_metrics(yte, pred, proba)
    m.update({"n_train": n_train, "n_test": n_test, "seconds": time.time()-t0,
              "note": "K(x,x')=|<psi(x)|psi(x')>|^2 AngleEmbedding+adjoint; subsampled O(n^2); optional comparison vs classical RBF SVM"})
    json.dump(m, open("results/reports/quantum_kernel.json", "w", encoding="utf-8"), indent=2)
    print(json.dumps({k: m[k] for k in ("accuracy","f1_macro","seconds","n_train","n_test")}, indent=2))

if __name__ == "__main__":
    main()
