# pure quick subset search
from __future__ import annotations
import os, sys, json, time, traceback
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))
from src.classical.feature_engineering import extract_all_features
from src.quantum.vqc import PureVQC
from src.features.reducer import QuantumReducer
from src.evaluation.compare_three import compute_all_metrics
CFG_PATH="configs/qorbit_config.json"
with open(CFG_PATH) as f: CFG=json.load(f)
SEED=CFG["seed"]
torch.manual_seed(SEED); np.random.seed(SEED)
GRID=[{"n_qubits":4,"n_layers":2},{"n_qubits":6,"n_layers":2},{"n_qubits":8,"n_layers":2},{"n_qubits":8,"n_layers":3}]
EPOCHS=5; BATCH=64; SUBSET_TRAIN=2000; SUBSET_VAL=500
def prepare():
    tr=np.load(CFG["dataset"]["train_path"],allow_pickle=True); va=np.load(CFG["dataset"]["val_path"],allow_pickle=True); te=np.load(CFG["dataset"]["test_path"],allow_pickle=True)
    ct,yt=tr["curves"],tr["labels"].astype(int); cv,yv=va["curves"],va["labels"].astype(int); cte,yte=te["curves"],te["labels"].astype(int)
    rng=np.random.RandomState(SEED)
    idx=rng.choice(len(ct), SUBSET_TRAIN, replace=False); ct,yt=ct[idx],yt[idx]
    idx2=rng.choice(len(cv), SUBSET_VAL, replace=False); cv,yv=cv[idx2],yv[idx2]
    return (ct,yt),(cv,yv),(cte,yte)
def train_one(ct,yt,cv,yv,cte,yte,cfg):
    nq=cfg["n_qubits"]; nl=cfg["n_layers"]
    f_train=np.array([extract_all_features(c) for c in ct],dtype=np.float32)
    f_val=np.array([extract_all_features(c) for c in cv],dtype=np.float32)
    f_test=np.array([extract_all_features(c) for c in cte],dtype=np.float32)
    red=QuantumReducer(n_components=nq); red.fit(f_train)
    pc_train=red.transform(f_train); m=pc_train.mean(axis=0); s=pc_train.std(axis=0)+1e-8
    X_train=(pc_train-m)/s; X_val=(red.transform(f_val)-m)/s; X_test=(red.transform(f_test)-m)/s
    model=PureVQC(n_qubits=nq,n_layers=nl,n_classes=5)
    opt=torch.optim.Adam(model.parameters(), lr=0.01)
    crit=nn.CrossEntropyLoss()
    Xtr_t=torch.tensor(X_train,dtype=torch.float32); ytr_t=torch.tensor(yt,dtype=torch.long)
    Xva_t=torch.tensor(X_val,dtype=torch.float32)
    t0=time.time()
    best_f1=-1; best_state=None
    n=len(Xtr_t)
    for epoch in range(1,EPOCHS+1):
        model.train()
        idx=torch.randperm(n)
        for s in range(0,n,BATCH):
            bi=idx[s:s+BATCH]
            opt.zero_grad()
            logits,_=model(Xtr_t[bi])
            loss=crit(logits,ytr_t[bi]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            logits,proba=model(Xva_t); pred=torch.argmax(proba,dim=1).numpy()
            metrics=compute_all_metrics(yv,pred,proba.numpy())
        if metrics["f1_macro"]>best_f1: best_f1=metrics["f1_macro"]; best_state={k:v.cpu() for k,v in model.state_dict().items()}
    train_time=time.time()-t0
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        Xte_t=torch.tensor(X_test,dtype=torch.float32)
        logits,proba=model(Xte_t); proba=proba.numpy(); pred=np.argmax(proba,axis=1)
        test_metrics=compute_all_metrics(yte,pred,proba)
        logits,proba=model(Xva_t); proba=proba.numpy(); pred=np.argmax(proba,axis=1)
        val_metrics=compute_all_metrics(yv,pred,proba)
    params=sum(p.numel() for p in model.parameters())
    return {"config":cfg,"val_metrics":val_metrics,"test_metrics":test_metrics,"train_time":train_time,"param_count":params,"n_qubits":nq,"circuit_depth":1+nl*2}
def main():
    (ct,yt),(cv,yv),(cte,yte)=prepare()
    print(f"Pure quick search train {len(ct)} val {len(cv)}")
    results=[]; best=None
    for i,cfg in enumerate(GRID):
        print(f"[{i+1}/{len(GRID)}] q{cfg['n_qubits']} l{cfg['n_layers']}")
        try:
            res=train_one(ct,yt,cv,yv,cte,yte,cfg)
            print(f"  val f1 {res['val_metrics']['f1_macro']:.4f} acc {res['val_metrics']['accuracy']:.4f} test acc {res['test_metrics']['accuracy']:.4f}")
            results.append(res)
            if best is None or res["val_metrics"]["f1_macro"]>best["val_metrics"]["f1_macro"]: best=res
        except Exception as e:
            traceback.print_exc(); results.append({"config":cfg,"error":str(e)})
    os.makedirs("results/reports",exist_ok=True)
    with open("results/reports/pure_search.json", "w", encoding="utf-8") as f: json.dump(results,f,indent=2)
    with open("results/reports/pure_search.md", "w", encoding="utf-8") as f:
        f.write("# Pure Quantum Search (quick subset)\n\n")
        f.write("| # | Qubits | Layers | Val Acc | Val F1 | Test Acc | Test F1 |\n|---|---|---|---|---|---|---|\n")
        for idx,r in enumerate(results):
            if "error" in r: f.write(f"| {idx+1} | {r['config']['n_qubits']} | {r['config']['n_layers']} | ERR | ERR | ERR | ERR |\n")
            else: f.write(f"| {idx+1} | {r['n_qubits']} | {r['config']['n_layers']} | {r['val_metrics']['accuracy']:.4f} | {r['val_metrics']['f1_macro']:.4f} | {r['test_metrics']['accuracy']:.4f} | {r['test_metrics']['f1_macro']:.4f} |\n")
        if best: f.write(f"\nBest: qubits {best['n_qubits']} layers {best['config']['n_layers']} val F1 {best['val_metrics']['f1_macro']:.4f}\n")
    print("saved pure quick")
if __name__=="__main__":
    main()
