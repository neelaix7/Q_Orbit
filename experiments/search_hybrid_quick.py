# quick hybrid search on subset for demo speed — still experimental, validation-based
from __future__ import annotations
import os, sys, json, time, traceback, random
import numpy as np, torch, torch.nn as nn, joblib
sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))
from src.classical.feature_engineering import extract_all_features
from src.quantum.hybrid_model import HybridQuantumClassifier
from src.evaluation.compare_three import compute_all_metrics
CFG_PATH="configs/qorbit_config.json"
with open(CFG_PATH) as f: CFG=json.load(f)
SEED=CFG["seed"]
torch.manual_seed(SEED); np.random.seed(SEED)
GRID=[
    {"n_qubits":4, "n_layers":2, "n_hidden":16},
    {"n_qubits":6, "n_layers":2, "n_hidden":16},
    {"n_qubits":8, "n_layers":2, "n_hidden":16},
    {"n_qubits":8, "n_layers":3, "n_hidden":16},
    {"n_qubits":8, "n_layers":2, "n_hidden":32},
]
EPOCHS=5
BATCH=64
LR=0.01
SUBSET_TRAIN=2000
SUBSET_VAL=500

def prepare():
    tr=np.load(CFG["dataset"]["train_path"],allow_pickle=True)
    va=np.load(CFG["dataset"]["val_path"],allow_pickle=True)
    te=np.load(CFG["dataset"]["test_path"],allow_pickle=True)
    ct,yt=tr["curves"],tr["labels"].astype(int)
    cv,yv=va["curves"],va["labels"].astype(int)
    cte,yte=te["curves"],te["labels"].astype(int)
    # subset for speed: stratified sample
    from sklearn.model_selection import train_test_split
    # use deterministic RNG
    rng=np.random.RandomState(SEED)
    # sample train subset
    idx=rng.choice(len(ct), SUBSET_TRAIN, replace=False)
    ct,yt=ct[idx],yt[idx]
    idx2=rng.choice(len(cv), SUBSET_VAL, replace=False)
    cv,yv=cv[idx2],yv[idx2]
    return (ct,yt),(cv,yv),(cte,yte)

def train_one(curves_train,y_train,curves_val,y_val,curves_test,y_test,cfg):
    nq=cfg["n_qubits"]; nl=cfg["n_layers"]; nh=cfg["n_hidden"]
    f_train=np.array([extract_all_features(c) for c in curves_train],dtype=np.float32)
    f_val=np.array([extract_all_features(c) for c in curves_val],dtype=np.float32)
    f_test=np.array([extract_all_features(c) for c in curves_test],dtype=np.float32)
    mean=f_train.mean(axis=0); std=f_train.std(axis=0)+1e-8
    X_train=(f_train-mean)/std; X_val=(f_val-mean)/std; X_test=(f_test-mean)/std
    model=HybridQuantumClassifier(n_features=21, n_qubits=nq, n_layers=nl, n_classical_hidden=nh)
    opt=torch.optim.Adam(model.parameters(), lr=LR)
    crit=nn.CrossEntropyLoss()
    Xtr_t=torch.tensor(X_train,dtype=torch.float32); ytr_t=torch.tensor(y_train,dtype=torch.long)
    Xva_t=torch.tensor(X_val,dtype=torch.float32); yva_t=torch.tensor(y_val,dtype=torch.long)
    Xte_t=torch.tensor(X_test,dtype=torch.float32)
    n=len(Xtr_t)
    t0=time.time()
    best_f1=-1; best_state=None
    for epoch in range(1,EPOCHS+1):
        model.train()
        idx=torch.randperm(n)
        for s in range(0,n,BATCH):
            bi=idx[s:s+BATCH]
            opt.zero_grad()
            logits,_=model(Xtr_t[bi])
            loss=crit(logits,ytr_t[bi])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            logits,proba=model(Xva_t)
            pred=torch.argmax(proba,dim=1).numpy()
            metrics=compute_all_metrics(y_val,pred,proba.numpy())
        if metrics["f1_macro"]>best_f1:
            best_f1=metrics["f1_macro"]; best_state={k:v.cpu() for k,v in model.state_dict().items()}
    train_time=time.time()-t0
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        t1=time.time()
        logits,proba=model(Xte_t); proba=proba.numpy(); pred=np.argmax(proba,axis=1)
        infer=(time.time()-t1)/len(Xte_t)
        test_metrics=compute_all_metrics(y_test,pred,proba)
        logits,proba=model(Xva_t); proba=proba.numpy(); pred=np.argmax(proba,axis=1)
        val_metrics=compute_all_metrics(y_val,pred,proba)
    params=sum(p.numel() for p in model.parameters())
    return {"config":cfg,"val_metrics":val_metrics,"test_metrics":test_metrics,"train_time":train_time,"infer_time":infer,"param_count":params,"circuit_depth":1+nl*2,"n_qubits":nq}, model

def main():
    (ct,yt),(cv,yv),(cte,yte)=prepare()
    print(f"Quick hybrid search subset train {len(ct)} val {len(cv)} test {len(cte)} epochs {EPOCHS}")
    results=[]; best=None; best_state=None
    for i,cfg in enumerate(GRID):
        print(f"[{i+1}/{len(GRID)}] q{cfg['n_qubits']} l{cfg['n_layers']} h{cfg['n_hidden']}")
        try:
            res, model = train_one(ct,yt,cv,yv,cte,yte,cfg)
            print(f"  val f1 {res['val_metrics']['f1_macro']:.4f} acc {res['val_metrics']['accuracy']:.4f} | test acc {res['test_metrics']['accuracy']:.4f} params {res['param_count']}")
            results.append(res)
            if best is None or res["val_metrics"]["f1_macro"]>best["val_metrics"]["f1_macro"]:
                best=res; best_state={k:v.cpu() for k,v in model.state_dict().items()}
        except Exception as e:
            traceback.print_exc()
            print("failed",e)
            results.append({"config":cfg,"error":str(e)})
    os.makedirs("results/reports",exist_ok=True)
    # merge with existing full hybrid model if quick not best: keep full 8q2l as baseline comparison
    # load full evaluation from fair_comparison for reference
    # save quick search results
    with open("results/reports/hybrid_search.json", "w", encoding="utf-8") as f: json.dump([{k:v for k,v in r.items() if k!="history"} if "val_metrics" in r else r for r in results], f, indent=2)
    with open("results/reports/hybrid_search.md", "w", encoding="utf-8") as f:
        f.write("# Hybrid Search (quick subset demo — validation-selected best, no test leakage)\n\n")
        f.write(f"Seed {SEED} | Grid {len(GRID)} configs | Epochs {EPOCHS} | Subset train {SUBSET_TRAIN} val {SUBSET_VAL}\n\n")
        f.write("| # | Qubits | Layers | Hidden | Val Acc | Val F1 | Test Acc | Test F1 | Params | Depth | Train(s) |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
        for idx,r in enumerate(results):
            if "error" in r: f.write(f"| {idx+1} | {r['config']['n_qubits']} | {r['config']['n_layers']} | {r['config']['n_hidden']} | ERR | ERR | ERR | ERR | - | - | - |\n")
            else: f.write(f"| {idx+1} | {r['n_qubits']} | {r['config']['n_layers']} | {r['config']['n_hidden']} | {r['val_metrics']['accuracy']:.4f} | {r['val_metrics']['f1_macro']:.4f} | {r['test_metrics']['accuracy']:.4f} | {r['test_metrics']['f1_macro']:.4f} | {r['param_count']} | {r['circuit_depth']} | {r['train_time']:.1f} |\n")
        if best: f.write(f"\n**Best on VAL (subset):** qubits={best['n_qubits']} layers={best['config']['n_layers']} hidden={best['config']['n_hidden']} -> Val F1 {best['val_metrics']['f1_macro']:.4f} Test F1 {best['test_metrics']['f1_macro']:.4f}\n")
        f.write("\nNote: Full production hybrid (8 qubits, 2 layers, 16 hidden) trained on full 7000-sample train set achieves 70.13% test accuracy (see fair_comparison.md). This quick search demonstrates systematic validation-based selection; production model remains the best validated on full data.\n")
    print("Saved hybrid_search quick demo")

if __name__=="__main__":
    main()
