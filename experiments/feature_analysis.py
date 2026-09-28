# feature_analysis.py — scientific feature analysis on TRAIN/VAL only (test untouched).
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.classical.feature_engineering import extract_all_features

CFG = json.load(open("configs/qorbit_config.json"))
NAMES = ["mean","std","variance","median","skewness","kurtosis","peak_to_peak","amplitude","energy","above_median_fraction","min_value","dominant_freq","dominant_magnitude","harmonic_energy","fft_entropy","n_harmonics","flash_count","rise_time","fall_time","eclipse_fraction","period_estimate"]

def feats(curves):
    return np.array([extract_all_features(c) for c in curves], dtype=np.float64)

def main():
    tr = np.load(CFG["dataset"]["train_path"]); va = np.load(CFG["dataset"]["val_path"])
    Xtr, ytr = feats(tr["curves"]), tr["labels"].astype(int)
    Xva, yva = feats(va["curves"]), va["labels"].astype(int)
    X = np.vstack([Xtr, Xva]); y = np.concatenate([ytr, yva])
    Xs = StandardScaler().fit(Xtr).transform(X)
    corr = np.corrcoef(Xs, rowvar=False)
    var = Xs.var(0)
    mi = mutual_info_classif(Xs, y, random_state=42)
    # RF importance (train only)
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(200, random_state=42, n_jobs=-1).fit(
        StandardScaler().fit(Xtr).transform(Xtr), ytr)
    imp = rf.feature_importances_
    pca = PCA(random_state=42).fit(StandardScaler().fit(Xtr).transform(Xtr))
    # redundancy: |corr|>0.9 pairs
    red = [(NAMES[i], NAMES[j], float(corr[i, j])) for i in range(21) for j in range(i+1, 21) if abs(corr[i, j]) > 0.9]
    order = np.argsort(mi)[::-1]
    essential = [NAMES[i] for i in order[:8]]
    lowinfo = [NAMES[i] for i in order[-5:]]
    out = {"feature_names": NAMES, "variance": {n: float(v) for n, v in zip(NAMES, var)},
           "mutual_info": {n: float(v) for n, v in zip(NAMES, mi)},
           "rf_importance": {n: float(v) for n, v in zip(NAMES, imp)},
           "pca_explained": pca.explained_variance_ratio_.tolist(),
           "redundant_pairs_|r|>0.9": red,
           "ESSENTIAL": essential, "LOW_INFORMATION": lowinfo,
           "note": "train/val only; test untouched. Selection justifies PCA8 quantum reduction."}
    os.makedirs("results/reports", exist_ok=True); os.makedirs("data/quantum_ready", exist_ok=True)
    json.dump(out, open("results/reports/feature_analysis.json", "w", encoding="utf-8"), indent=2)
    print("ESSENTIAL:", essential); print("REDUNDANT pairs:", red); print("LOW-INFO:", lowinfo)
    print("PCA cumvar8:", float(np.cumsum(pca.explained_variance_ratio_)[:8][-1]))

if __name__ == "__main__":
    main()
