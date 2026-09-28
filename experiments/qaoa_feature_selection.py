# qaoa_feature_selection.py — SMALL RESEARCH PROTOTYPE (not a classifier).
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

NAMES = ["mean","std","variance","median","skewness","kurtosis","peak_to_peak","amplitude","energy","above_median_fraction","min_value","dominant_freq","dominant_magnitude","harmonic_energy","fft_entropy","n_harmonics","flash_count","rise_time","fall_time","eclipse_fraction","period_estimate"]

def bitstring_cost(bits, Q):
    x = np.array(bits, dtype=float)
    return float(x @ Q @ x)

def qaoa_solve(Q, p=1, steps=40, lr=0.2, seed=42):
    n = Q.shape[0]
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="autograd")
    def circ(g, b):
        for i in range(n):
            qml.Hadamard(wires=i)
        for k in range(p):
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(Q[i, j]) > 1e-12:
                        qml.IsingZZ(2 * g[k] * Q[i, j], wires=[i, j])
            for i in range(n):
                if abs(Q[i, i]) > 1e-12:
                    qml.RZ(2 * g[k] * Q[i, i], wires=i)
            for i in range(n):
                qml.RX(2 * b[k], wires=i)
        return qml.probs(wires=range(n))

    def cost(g, b):
        pr = circ(g, b)
        tot = 0.0
        for s in range(2 ** n):
            bits = [(s >> (n - 1 - i)) & 1 for i in range(n)]
            tot = tot + pr[s] * bitstring_cost(bits, Q)
        return tot

    rng = np.random.RandomState(seed)
    g = pnp.array(rng.uniform(0, 1, p), requires_grad=True)
    b = pnp.array(rng.uniform(0, 1, p), requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=lr)
    for _ in range(steps):
        g, b = opt.step(cost, g, b)
    probs = np.array(circ(g, b))
    best = int(np.argmax(probs))
    bits = [(best >> (n - 1 - i)) & 1 for i in range(n)]
    return bits, probs, float(cost(g, b))

def main(n_feat=8, k_target=4, p=1):
    from src.classical.feature_engineering import extract_all_features
    CFG = json.load(open("configs/qorbit_config.json"))
    fa = json.load(open("results/reports/feature_analysis.json"))
    mi_all = np.array([fa["mutual_info"][n] for n in NAMES])
    cand = np.argsort(mi_all)[::-1][:n_feat]
    tr = np.load(CFG["dataset"]["train_path"]); va = np.load(CFG["dataset"]["val_path"])
    Ftr = np.array([extract_all_features(c) for c in tr["curves"]], dtype=float)[:, cand]
    Fva = np.array([extract_all_features(c) for c in va["curves"]], dtype=float)[:, cand]
    ytr, yva = tr["labels"].astype(int), va["labels"].astype(int)
    rel = mutual_info_classif(Ftr, ytr, random_state=42)
    C = np.abs(np.corrcoef(Ftr, rowvar=False)); C = np.nan_to_num(C, nan=0.0)
    lam, mu = 0.5, 1.0
    Q = np.zeros((n_feat, n_feat))
    for i in range(n_feat):
        Q[i, i] = -rel[i] + mu * (1 - 2 * k_target)
        for j in range(i + 1, n_feat):
            Q[i, j] = lam * C[i, j] + 2 * mu
    bits, probs, ecost = qaoa_solve(Q, p=p)
    sel = [i for i, v in enumerate(bits) if v == 1] or [int(np.argmax(rel))]
    greedy = list(np.argsort(rel)[::-1][:max(len(sel), 1)])
    def val_f1(cols):
        rf = RandomForestClassifier(200, random_state=42, n_jobs=-1).fit(Ftr[:, cols], ytr)
        return float(f1_score(yva, rf.predict(Fva[:, cols]), average="macro"))
    out = {"objective": "min xTQx: -relevance + redundancy + mu(|S|-k)^2",
           "candidates": [NAMES[int(c)] for c in cand], "QUBO_Q": Q.tolist(),
           "QAOA_bits": bits, "QAOA_expected_cost": ecost,
           "QAOA_subset": [NAMES[int(cand[i])] for i in sel], "QAOA_valF1": val_f1(sel),
           "greedy_subset": [NAMES[int(cand[i])] for i in greedy], "greedy_valF1": val_f1(greedy),
           "label": "RESEARCH PROTOTYPE: QAOA proposes feature subset; validation is classical RF val-F1; NOT a classifier"}
    json.dump(out, open("results/reports/qaoa_selection.json", "w", encoding="utf-8"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "QUBO_Q"}, indent=2))

if __name__ == "__main__":
    main()
