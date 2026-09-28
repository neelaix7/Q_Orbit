# optimize_hybrid_full.py — systematic hybrid optimization with validation selection, no test leakage
from __future__ import annotations
import os, sys, json, time, traceback
import numpy as np, torch, torch.nn as nn, joblib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.classical.feature_engineering import extract_all_features
from src.quantum.hybrid_model import HybridQuantumClassifier
from src.features.reducer import QuantumReducer
from src.evaluation.compare_three import compute_all_metrics
from sklearn.feature_selection import SelectKBest, mutual_info_classif

CFG_PATH="configs/qorbit_config.json"
with open(CFG_PATH) as f: CFG=json.load(f)
SEED=CFG["seed"]
torch.manual_seed(SEED); np.random.seed(SEED)

GRID=[
    # Baseline-like variants sweeping quantum capacity
    {"name":"q4l2h16_lr0.01", "n_qubits":4, "n_layers":2, "hidden":16, "lr":0.01, "epochs":15, "pca":False, "k_best":None},
    {"name":"q6l2h16_lr0.01", "n_qubits":6, "n_layers":2, "hidden":16, "lr":0.01, "epochs":15, "pca":False, "k_best":None},
    {"name":"q8l2h16_lr0.01", "n_qubits":8, "n_layers":2, "hidden":16, "lr":0.01, "epochs":15, "pca":False, "k_best":None},
    {"name":"q8l3h16_lr0.01", "n_qubits":8, "n_layers":3, "hidden":16, "lr":0.01, "epochs":15, "pca":False, "k_best":None},
    {"name":"q8l2h32_lr0.01", "n_qubits":8, "n_layers":2, "hidden":32, "lr":0.01, "epochs":15, "pca":False, "k_best":None},
    {"name":"q8l2h64_lr0.01", "n_qubits":8, "n_layers":2, "hidden":64, "lr":0.01, "epochs":15, "pca":False, "k_best":None},
    # Feature selection variants
    {"name":"q8l2h32_k15", "n_qubits":8, "n_layers":2, "hidden":32, "lr":0.01, "epochs":15, "pca":False, "k_best":15},
    {"name":"q8l2h32_k12", "n_qubits":8, "n_layers":2, "hidden":32, "lr":0.01, "epochs":15, "pca":False, "k_best":12},
    # PCA variants (quantum gets PCA 8, classical head still hybrid)
    {"name":"q8l2h32_pca8", "n_qubits":8, "n_layers":2, "hidden":32, "lr":0.01, "epochs":15, "pca":8, "k_best":None},
    {"name":"q8l2h32_lr0.005", "n_qubits":8, "n_layers":2, "hidden":32, "lr":0.005, "epochs":18, "pca":False, "k_best":None},
    {"name":"q8l2h16_lr0.005_pca", "n_qubits":8, "n_layers":2, "hidden":16, "lr":0.005, "epochs":18, "pca":8, "k_best":None},
    {"name":"q6l3h32", "n_qubits":6, "n_layers":3, "hidden":32, "lr":0.01, "epochs":15, "pca":False, "k_best":None},
]

BATCH=64

def prepare_features():
    tr=np.load(CFG["dataset"]["train_path"], allow_pickle=True)
    va=np.load(CFG["dataset"]["val_path"], allow_pickle=True)
    te=np.load(CFG["dataset"]["test_path"], allow_pickle=True)
    ct,yt=tr["curves"], tr["labels"].astype(int)
    cv,yv=va["curves"], va["labels"].astype(int)
    cte,yte=te["curves"], te["labels"].astype(int)
    print(f"Loaded splits train {len(ct)} val {len(cv)} test {len(cte)}")
    f_train=np.array([extract_all_features(c) for c in ct], dtype=np.float32)
    f_val=np.array([extract_all_features(c) for c in cv], dtype=np.float32)
    f_test=np.array([extract_all_features(c) for c in cte], dtype=np.float32)
    return (f_train,f_val,f_test),(yt,yv,yte),(ct,cv,cte)

def train_one(cfg, f_train,y_train,f_val,y_val,f_test,y_test):
    nq=cfg["n_qubits"]; nl=cfg["n_layers"]; hidden=cfg["hidden"]; lr=cfg["lr"]; epochs=cfg["epochs"]
    # preprocessing: fit scaler on train only
    mean=f_train.mean(axis=0); std=f_train.std(axis=0)+1e-8
    # handle k_best feature selection
    selector=None
    pca_reducer=None
    if cfg["k_best"] is not None:
        k=cfg["k_best"]
        selector=SelectKBest(mutual_info_classif, k=k)
        # fit on standardized?
        X_std_train=(f_train-mean)/std
        selector.fit(X_std_train, y_train)
        # transform all
        X_train_sel=selector.transform(X_std_train)
        X_val_sel=selector.transform((f_val-mean)/std)
        X_test_sel=selector.transform((f_test-mean)/std)
        # need to map selector output dim to n_qubits? Hybrid expects n_features = k
        n_features=k
        X_train, X_val, X_test = X_train_sel, X_val_sel, X_test_sel
        # re-standardize after selection? already std, but ensure zero mean
        # no extra scaling
    elif cfg["pca"] is not None:
        n_components=cfg["pca"]
        reducer=QuantumReducer(n_components=n_components)
        reducer.fit(f_train)
        X_train=reducer.transform(f_train)
        # standardize PCA output (reducer already standardizes internally, but output is PCA space)
        # we additionally z-score PCA output using train PCA mean/std
        pca_train_mean=X_train.mean(axis=0); pca_train_std=X_train.std(axis=0)+1e-8
        X_train=(X_train - pca_train_mean)/pca_train_std
        X_val=(reducer.transform(f_val) - pca_train_mean)/pca_train_std
        X_test=(reducer.transform(f_test) - pca_train_mean)/pca_train_std
        n_features=n_components
        pca_reducer=(reducer, pca_train_mean, pca_train_std)
    else:
        X_train=(f_train-mean)/std
        X_val=(f_val-mean)/std
        X_test=(f_test-mean)/std
        n_features=X_train.shape[1]

    # Hybrid model expects n_features = input dim, n_qubits = quantum width
    # If n_features != n_qubits, projection handles it. Keep n_qubits as cfg.
    model=HybridQuantumClassifier(n_features=n_features, n_qubits=nq, n_layers=nl, n_classical_hidden=hidden)
    opt=torch.optim.Adam(model.parameters(), lr=lr)
    crit=nn.CrossEntropyLoss()
    Xtr_t=torch.tensor(X_train, dtype=torch.float32); ytr_t=torch.tensor(y_train, dtype=torch.long)
    Xva_t=torch.tensor(X_val, dtype=torch.float32); yva_t=torch.tensor(y_val, dtype=torch.long)
    Xte_t=torch.tensor(X_test, dtype=torch.float32)
    n=len(Xtr_t)
    best_val_f1=-1; best_state=None; best_epoch=0
    t0=time.time()
    for epoch in range(1, epochs+1):
        model.train()
        idx=torch.randperm(n)
        for s in range(0,n,BATCH):
            bi=idx[s:s+BATCH]
            opt.zero_grad()
            logits,_=model(Xtr_t[bi])
            loss=crit(logits, ytr_t[bi])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            logits,proba=model(Xva_t)
            pred=torch.argmax(proba,dim=1).numpy()
            metrics=compute_all_metrics(y_val,pred,proba.numpy())
            f1=metrics["f1_macro"]
            if f1>best_val_f1:
                best_val_f1=f1; best_state={k:v.cpu() for k,v in model.state_dict().items()}; best_epoch=epoch
    train_time=time.time()-t0
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        t1=time.time()
        logits,proba=model(Xte_t); proba=proba.numpy(); pred=np.argmax(proba,axis=1)
        infer_ms=(time.time()-t1)/len(Xte_t)*1000
        test_metrics=compute_all_metrics(y_test,pred,proba)
        logits,proba=model(Xva_t); proba=proba.numpy(); pred=np.argmax(proba,axis=1)
        val_metrics=compute_all_metrics(y_val,pred,proba)
    params=sum(p.numel() for p in model.parameters())
    return {
        "config": cfg,
        "n_features": n_features,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_time": train_time,
        "infer_ms": infer_ms,
        "param_count": params,
        "circuit_depth": 1+nl*2,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "selector": selector,
        "pca_reducer": pca_reducer,
        "mean": mean, "std": std
    }, model

def main():
    (f_train,f_val,f_test),(y_train,y_val,y_test),(ct,cv,cte)=prepare_features()
    results=[]
    best=None; best_model=None; best_cfg=None
    for i,cfg in enumerate(GRID):
        print(f"\n[{i+1}/{len(GRID)}] {cfg['name']} q{cfg['n_qubits']} l{cfg['n_layers']} h{cfg['hidden']} lr{cfg['lr']} k_best={cfg['k_best']} pca={cfg['pca']}")
        try:
            res, model = train_one(cfg, f_train,y_train,f_val,y_val,f_test,y_test)
            print(f"  val F1 {res['val_metrics']['f1_macro']:.4f} acc {res['val_metrics']['accuracy']:.4f} | test acc {res['test_metrics']['accuracy']:.4f} F1 {res['test_metrics']['f1_macro']:.4f} params {res['param_count']} epoch {res['best_epoch']}")
            results.append({k:v for k,v in res.items() if k not in ["selector","pca_reducer","mean","std"]})
            # attach serializable metrics
            results[-1]["val_metrics"]=res["val_metrics"]
            results[-1]["test_metrics"]=res["test_metrics"]
            if best is None or res["best_val_f1"]>best["best_val_f1"]:
                best=res; best_model=model; best_cfg=cfg
        except Exception as e:
            traceback.print_exc()
            print("  failed",e)
            results.append({"config":cfg,"error":str(e)})
    os.makedirs("results/reports", exist_ok=True)
    with open("results/reports/hybrid_optimization.json", "w", encoding="utf-8") as f:
        json.dump(results,f,indent=2)
    with open("results/reports/hybrid_optimization.md", "w", encoding="utf-8") as f:
        f.write("# Hybrid Optimization — validation-selected, no test leakage\n\n")
        f.write(f"Seed {SEED} | Grid {len(GRID)} configs | Batch {BATCH}\n\n")
        f.write("| # | Config | Qubits | Layers | Hidden | LR | k_best | PCA | Val F1 | Val Acc | Test Acc | Test F1 | Params | Depth | Train(s) | Best Epoch |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for idx,r in enumerate(results):
            if "error" in r:
                f.write(f"| {idx+1} | {r['config']['name']} | - | - | - | - | - | - | ERR | ERR | ERR | ERR | - | - | - | - |\n")
            else:
                c=r["config"]
                f.write(f"| {idx+1} | {c['name']} | {c['n_qubits']} | {c['n_layers']} | {c['hidden']} | {c['lr']} | {c['k_best']} | {c['pca']} | {r['val_metrics']['f1_macro']:.4f} | {r['val_metrics']['accuracy']:.4f} | {r['test_metrics']['accuracy']:.4f} | {r['test_metrics']['f1_macro']:.4f} | {r['param_count']} | {r['circuit_depth']} | {r['train_time']:.1f} | {r['best_epoch']} |\n")
        if best:
            f.write(f"\n**Best on VAL:** {best['config']['name']} qubits={best['config']['n_qubits']} layers={best['config']['n_layers']} hidden={best['config']['hidden']} -> Val F1 {best['val_metrics']['f1_macro']:.4f} Test F1 {best['test_metrics']['f1_macro']:.4f} Test Acc {best['test_metrics']['accuracy']:.4f}\n")
            f.write(f"\n*Test set was frozen and evaluated once per config; selection was on VAL only.*\n")

    if best and best_model is not None:
        # Save best hybrid as production model
        os.makedirs("models", exist_ok=True)
        # Need to decide what to save: for selector/pca variants, need extra artifacts
        ckpt={
            "model_state_dict": best_model.state_dict(),
            "n_features": best["n_features"],
            "n_qubits": best["config"]["n_qubits"],
            "n_layers": best["config"]["n_layers"],
            "n_classical_hidden": best["config"]["hidden"],
            "accuracy": float(best["test_metrics"]["accuracy"]*100),
            "val_f1": float(best["val_metrics"]["f1_macro"]),
            "config": best["config"]
        }
        torch.save(ckpt, "models/hybrid_quantum_model.pt")
        print(f"\nSaved best hybrid to models/hybrid_quantum_model.pt: {best['config']['name']} test acc {best['test_metrics']['accuracy']:.4f}")
        # Save selector/pca if needed
        if best["selector"] is not None:
            joblib.dump(best["selector"], "models/hybrid_selector.joblib")
            print("Saved hybrid_selector.joblib")
            # also save mean/std for fallback
            np.savez("models/hybrid_pca_stats.npz", mean=best["mean"], std=best["std"])
        elif best["pca_reducer"] is not None:
            reducer, m, s = best["pca_reducer"]
            reducer.save("models/hybrid_reducer.joblib")
            np.savez("models/hybrid_pca_stats.npz", mean=m, std=s)
            print("Saved hybrid_reducer + pca stats")
        else:
            # save feature scaler for standard hybrid
            joblib.dump({"mean":best["mean"],"std":best["std"],"scale":best["std"]}, "models/feature_scaler.joblib")
            # cleanup old selector/reducer if best is standard
            for p in ["models/hybrid_selector.joblib","models/hybrid_reducer.joblib","models/hybrid_pca_stats.npz"]:
                if os.path.exists(p):
                    try: os.remove(p)
                    except Exception: pass
            print("Saved feature_scaler, cleaned selector/reducer")
        # also copy to val-selected marker
        with open("models/hybrid_best_val.json", "w", encoding="utf-8") as f:
            json.dump({"config":best["config"],"val_f1":best["val_metrics"]["f1_macro"],"test_acc":best["test_metrics"]["accuracy"],"test_f1":best["test_metrics"]["f1_macro"]},f,indent=2)
    print("\nDone hybrid optimization.")

if __name__=="__main__":
    main()
