# search_hybrid.py — systematic hybrid grid search, validation-based best selection, never touches test for tuning
from __future__ import annotations
import os, sys, json, time, itertools, traceback
import numpy as np, torch, torch.nn as nn, joblib
sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))
from src.classical.feature_engineering import extract_all_features
from src.quantum.hybrid_model import HybridQuantumClassifier
from src.features.reducer import QuantumReducer
from src.evaluation.compare_three import compute_all_metrics
from sklearn.feature_selection import SelectKBest, f_classif

CFG_PATH="configs/qorbit_config.json"
with open(CFG_PATH) as f: CFG=json.load(f)
SEED=CFG["seed"]
torch.manual_seed(SEED); np.random.seed(SEED)

# Grid — focus effort on hybrid as requested
GRID=[
    # qubits, layers, hidden, pca, k_best
    {"n_qubits":4, "n_layers":1, "n_hidden":16, "use_pca":False, "k_best":None},
    {"n_qubits":4, "n_layers":2, "n_hidden":16, "use_pca":False, "k_best":None},
    {"n_qubits":6, "n_layers":2, "n_hidden":16, "use_pca":False, "k_best":None},
    {"n_qubits":6, "n_layers":2, "n_hidden":32, "use_pca":False, "k_best":None},
    {"n_qubits":6, "n_layers":3, "n_hidden":16, "use_pca":False, "k_best":None},
    {"n_qubits":8, "n_layers":1, "n_hidden":16, "use_pca":False, "k_best":None},
    {"n_qubits":8, "n_layers":2, "n_hidden":16, "use_pca":False, "k_best":None}, # baseline
    {"n_qubits":8, "n_layers":2, "n_hidden":32, "use_pca":False, "k_best":None},
    {"n_qubits":8, "n_layers":2, "n_hidden":64, "use_pca":False, "k_best":None},
    {"n_qubits":8, "n_layers":3, "n_hidden":16, "use_pca":False, "k_best":None},
    {"n_qubits":8, "n_layers":3, "n_hidden":32, "use_pca":False, "k_best":None},
    {"n_qubits":8, "n_layers":2, "n_hidden":16, "use_pca":True, "k_best":None}, # PCA variant
    {"n_qubits":8, "n_layers":2, "n_hidden":16, "use_pca":False, "k_best":12}, # feature selection variant
    {"n_qubits":6, "n_layers":2, "n_hidden":16, "use_pca":False, "k_best":12},
]

EPOCHS=8
BATCH=32
LR=0.01
# quick mode: if env var QORBIT_QUICK set, reduce grid for demo speed
import os as _os
if _os.environ.get("QORBIT_QUICK")=="1":
    GRID=GRID[:6]

def train_one(curves_train, y_train, curves_val, y_val, curves_test, y_test, cfg):
    nq=cfg["n_qubits"]; nl=cfg["n_layers"]; nh=cfg["n_hidden"]; use_pca=cfg["use_pca"]; k_best=cfg["k_best"]
    # features
    f_train=np.array([extract_all_features(c) for c in curves_train],dtype=np.float32)
    f_val=np.array([extract_all_features(c) for c in curves_val],dtype=np.float32)
    f_test=np.array([extract_all_features(c) for c in curves_test],dtype=np.float32)
    # optional feature selection fit on train only
    selector=None
    if k_best is not None:
        selector=SelectKBest(f_classif, k=k_best)
        selector.fit(f_train, y_train)
        f_train=selector.transform(f_train)
        f_val=selector.transform(f_val)
        f_test=selector.transform(f_test)
        n_features=k_best
    else:
        n_features=f_train.shape[1]
    # optional PCA for hybrid? if use_pca we reduce to n_qubits then still use Hybrid with n_features=n_qubits
    reducer=None; mean=None; std=None; m_pca=None; s_pca=None
    if use_pca:
        reducer=QuantumReducer(n_components=nq)
        reducer.fit(f_train)
        pc_train=reducer.transform(f_train)
        m_pca=pc_train.mean(axis=0); s_pca=pc_train.std(axis=0)+1e-8
        X_train=(pc_train-m_pca)/s_pca
        X_val=(reducer.transform(f_val)-m_pca)/s_pca
        X_test=(reducer.transform(f_test)-m_pca)/s_pca
        n_features=nq
    else:
        mean=f_train.mean(axis=0); std=f_train.std(axis=0)+1e-8
        X_train=(f_train-mean)/std; X_val=(f_val-mean)/std; X_test=(f_test-mean)/std

    # build hybrid
    model=HybridQuantumClassifier(n_features=n_features, n_qubits=nq, n_layers=nl, n_classical_hidden=nh, n_classes=5)
    opt=torch.optim.Adam(model.parameters(), lr=LR)
    crit=nn.CrossEntropyLoss()
    Xtr_t=torch.tensor(X_train,dtype=torch.float32); ytr_t=torch.tensor(y_train,dtype=torch.long)
    Xva_t=torch.tensor(X_val,dtype=torch.float32); yva_t=torch.tensor(y_val,dtype=torch.long)
    Xte_t=torch.tensor(X_test,dtype=torch.float32); yte_t=torch.tensor(y_test,dtype=torch.long)
    n=len(Xtr_t)
    t0=time.time()
    best_val_f1=-1; best_state=None; best_val_metrics=None; history=[]
    for epoch in range(1,EPOCHS+1):
        model.train()
        idx=torch.randperm(n)
        total=0; bs=0
        for s in range(0,n,BATCH):
            bi=idx[s:s+BATCH]
            opt.zero_grad()
            logits,_=model(Xtr_t[bi])
            loss=crit(logits, ytr_t[bi])
            loss.backward(); opt.step()
            total+=loss.item(); bs+=1
        # val eval
        model.eval()
        with torch.no_grad():
            logits,proba=model(Xva_t)
            pred=torch.argmax(proba,dim=1).numpy()
            proba_n=proba.numpy()
            metrics=compute_all_metrics(y_val,pred,proba_n)
        history.append({"epoch":epoch,"loss":total/max(bs,1),"val_f1":metrics["f1_macro"],"val_acc":metrics["accuracy"]})
        if metrics["f1_macro"]>best_val_f1:
            best_val_f1=metrics["f1_macro"]; best_val_metrics=metrics
            best_state={k:v.cpu() for k,v in model.state_dict().items()}
    train_time=time.time()-t0
    # reload best
    if best_state: model.load_state_dict(best_state)
    # test eval with best
    model.eval()
    with torch.no_grad():
        t1=time.time()
        logits,proba=model(Xte_t)
        proba_n=proba.numpy(); pred=np.argmax(proba_n,axis=1)
        infer_time=(time.time()-t1)/len(Xte_t)
        test_metrics=compute_all_metrics(y_test,pred,proba_n)
        # val metrics again with best
        logits,proba=model(Xva_t)
        proba_n=proba.numpy(); pred=np.argmax(proba_n,axis=1)
        val_metrics=compute_all_metrics(y_val,pred,proba_n)
    param_count=sum(p.numel() for p in model.parameters())
    circuit_depth=1 + nl*2
    # qubit/metrics
    result={
        "config":cfg,
        "val_metrics":val_metrics,
        "test_metrics":test_metrics,
        "history":history,
        "train_time":train_time,
        "infer_time_per_sample":infer_time,
        "param_count":param_count,
        "circuit_depth":circuit_depth,
        "n_qubits":nq,
        "n_layers":nl,
    }
    # save scaler/reducer for best later
    return result, model, (mean,std,m_pca,s_pca,reducer,selector)

def main():
    tr=np.load(CFG["dataset"]["train_path"],allow_pickle=True)
    va=np.load(CFG["dataset"]["val_path"],allow_pickle=True)
    te=np.load(CFG["dataset"]["test_path"],allow_pickle=True)
    curves_train,y_train=tr["curves"],tr["labels"].astype(int)
    curves_val,y_val=va["curves"],va["labels"].astype(int)
    curves_test,y_test=te["curves"],te["labels"].astype(int)
    print(f"Hybrid search: {len(GRID)} configs, epochs {EPOCHS}, seed {SEED}")
    all_results=[]
    best=None; best_model_state=None; best_aux=None
    for i,cfg in enumerate(GRID):
        print(f"\n[{i+1}/{len(GRID)}] qubits={cfg['n_qubits']} layers={cfg['n_layers']} hidden={cfg['n_hidden']} pca={cfg['use_pca']} k={cfg['k_best']}")
        try:
            res, model, aux = train_one(curves_train,y_train,curves_val,y_val,curves_test,y_test,cfg)
            print(f"  val f1 {res['val_metrics']['f1_macro']:.4f} acc {res['val_metrics']['accuracy']:.4f} | test acc {res['test_metrics']['accuracy']:.4f} | train {res['train_time']:.1f}s params {res['param_count']}")
            all_results.append(res)
            if best is None or res["val_metrics"]["f1_macro"]>best["val_metrics"]["f1_macro"]:
                best=res; best_model_state={k:v.cpu() for k,v in model.state_dict().items()}
                best_aux=aux
        except Exception as e:
            traceback.print_exc()
            print(f"  FAILED config {cfg}: {e}")
            all_results.append({"config":cfg,"error":str(e)})
    # pick best
    os.makedirs("results/reports",exist_ok=True); os.makedirs("models",exist_ok=True)
    # clean for json
    def clean(m):
        if m is None or "error" in m: return m
        # remove confusion matrix huge? keep but ensure serializable
        out={}
        for k,v in m.items():
            if k in ("val_metrics","test_metrics"):
                out[k]={kk:vv for kk,vv in v.items() if not kk.startswith("_")}
            elif k=="history":
                out[k]=v
            else:
                out[k]=v
        return out
    with open("results/reports/hybrid_search.json", "w", encoding="utf-8") as f:
        json.dump([clean(r) for r in all_results], f, indent=2)
    # markdown report
    with open("results/reports/hybrid_search.md", "w", encoding="utf-8") as f:
        f.write("# Hybrid Search — validation-selected best (no test leakage)\n\n")
        f.write(f"Seed {SEED} | Grid {len(GRID)} configs | Epochs {EPOCHS}\n\n")
        f.write("| # | Qubits | Layers | Hidden | PCA | k_best | Val Acc | Val F1 | Test Acc | Test F1 | Params | Depth | Train(s) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for idx,r in enumerate(all_results):
            if "error" in r: f.write(f"| {idx+1} | {r['config']['n_qubits']} | {r['config']['n_layers']} | {r['config']['n_hidden']} | {r['config']['use_pca']} | {r['config']['k_best']} | ERR | ERR | ERR | ERR | - | - | - |\n")
            else: f.write(f"| {idx+1} | {r['n_qubits']} | {r['n_layers']} | {r['config']['n_hidden']} | {r['config']['use_pca']} | {r['config']['k_best']} | {r['val_metrics']['accuracy']:.4f} | {r['val_metrics']['f1_macro']:.4f} | {r['test_metrics']['accuracy']:.4f} | {r['test_metrics']['f1_macro']:.4f} | {r['param_count']} | {r['circuit_depth']} | {r['train_time']:.1f} |\n")
        if best:
            f.write(f"\n**Best on VAL:** qubits={best['n_qubits']} layers={best['n_layers']} hidden={best['config']['n_hidden']} pca={best['config']['use_pca']} k_best={best['config']['k_best']} → Val F1 {best['val_metrics']['f1_macro']:.4f} Test F1 {best['test_metrics']['f1_macro']:.4f}\n")
    if best:
        print(f"\nBEST hybrid: qubits {best['n_qubits']} layers {best['n_layers']} hidden {best['config']['n_hidden']} val f1 {best['val_metrics']['f1_macro']:.4f} test acc {best['test_metrics']['accuracy']:.4f}")
        # save best model to standard path for dashboard
        mean,std,m_pca,s_pca,reducer,selector = best_aux
        payload={
            "model_state_dict": best_model_state,
            "n_features": best["config"].get("k_best") if best["config"].get("k_best") else (best["n_qubits"] if best["config"]["use_pca"] else 21),
            "n_qubits": best["n_qubits"],
            "n_layers": best["n_layers"],
            "n_classical_hidden": best["config"]["n_hidden"],
            "accuracy": best["test_metrics"]["accuracy"]*100,
            "val_f1": best["val_metrics"]["f1_macro"],
            "config": best["config"],
            "param_count": best["param_count"],
            "circuit_depth": best["circuit_depth"],
        }
        # adjust n_features correctly for Hybrid which stores original feature dim
        if best["config"]["use_pca"]:
            payload["n_features"]=best["n_qubits"]
        elif best["config"]["k_best"]:
            payload["n_features"]=best["config"]["k_best"]
        else:
            payload["n_features"]=21
        torch.save(payload, "models/hybrid_quantum_model.pt")
        print("Saved best hybrid to models/hybrid_quantum_model.pt")
        # also save its preprocessing auxiliaries so fair_comparison can reconstruct
        if reducer is not None:
            reducer.save("models/hybrid_reducer.joblib")
        if selector is not None:
            joblib.dump(selector, "models/hybrid_selector.joblib")
        # save scaler for non-pca case
        if mean is not None:
            joblib.dump({"mean":mean,"std":std}, "models/feature_scaler.joblib")
        if m_pca is not None:
            np.savez("models/hybrid_pca_stats.npz", mean=m_pca, std=s_pca)

if __name__=="__main__":
    main()
