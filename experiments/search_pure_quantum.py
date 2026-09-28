# search_pure_quantum.py — Pure VQC grid search on validation, select best pure config
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

GRID=[
    {"n_qubits":4, "n_layers":2},
    {"n_qubits":6, "n_layers":2},
    {"n_qubits":6, "n_layers":3},
    {"n_qubits":8, "n_layers":2},
    {"n_qubits":8, "n_layers":3},
    {"n_qubits":4, "n_layers":3},
]

EPOCHS=15
BATCH=32
LR=0.01

def train_one(curves_train,y_train,curves_val,y_val,curves_test,y_test,cfg):
    nq=cfg["n_qubits"]; nl=cfg["n_layers"]
    f_train=np.array([extract_all_features(c) for c in curves_train],dtype=np.float32)
    f_val=np.array([extract_all_features(c) for c in curves_val],dtype=np.float32)
    f_test=np.array([extract_all_features(c) for c in curves_test],dtype=np.float32)
    reducer=QuantumReducer(n_components=nq)
    reducer.fit(f_train)
    pc_train=reducer.transform(f_train)
    m=pc_train.mean(axis=0); s=pc_train.std(axis=0)+1e-8
    X_train=(pc_train-m)/s
    X_val=(reducer.transform(f_val)-m)/s
    X_test=(reducer.transform(f_test)-m)/s
    model=PureVQC(n_qubits=nq,n_layers=nl,n_classes=5)
    opt=torch.optim.Adam(model.parameters(), lr=LR)
    crit=nn.CrossEntropyLoss()
    Xtr_t=torch.tensor(X_train,dtype=torch.float32); ytr_t=torch.tensor(y_train,dtype=torch.long)
    Xva_t=torch.tensor(X_val,dtype=torch.float32); yva_t=torch.tensor(y_val,dtype=torch.long)
    Xte_t=torch.tensor(X_test,dtype=torch.float32); yte_t=torch.tensor(y_test,dtype=torch.long)
    n=len(Xtr_t)
    t0=time.time()
    best_f1=-1; best_state=None; best_metrics=None
    for epoch in range(1,EPOCHS+1):
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
            proba_n=proba.numpy()
            metrics=compute_all_metrics(y_val,pred,proba_n)
        if metrics["f1_macro"]>best_f1:
            best_f1=metrics["f1_macro"]; best_metrics=metrics
            best_state={k:v.cpu() for k,v in model.state_dict().items()}
    train_time=time.time()-t0
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        t1=time.time()
        logits,proba=model(Xte_t)
        proba_n=proba.numpy(); pred=np.argmax(proba_n,axis=1)
        infer_time=(time.time()-t1)/len(Xte_t)
        test_metrics=compute_all_metrics(y_test,pred,proba_n)
        logits,proba=model(Xva_t)
        proba_n=proba.numpy(); pred=np.argmax(proba_n,axis=1)
        val_metrics=compute_all_metrics(y_val,pred,proba_n)
    param_count=sum(p.numel() for p in model.parameters())
    return {"config":cfg,"val_metrics":val_metrics,"test_metrics":test_metrics,"train_time":train_time,"infer_time":infer_time,"param_count":param_count,"circuit_depth":1+nl*2,"n_qubits":nq}, model, reducer, (m,s)

def main():
    tr=np.load(CFG["dataset"]["train_path"],allow_pickle=True)
    va=np.load(CFG["dataset"]["val_path"],allow_pickle=True)
    te=np.load(CFG["dataset"]["test_path"],allow_pickle=True)
    curves_train,y_train=tr["curves"],tr["labels"].astype(int)
    curves_val,y_val=va["curves"],va["labels"].astype(int)
    curves_test,y_test=te["curves"],te["labels"].astype(int)
    print(f"Pure VQC search {len(GRID)} configs epochs {EPOCHS}")
    results=[]; best=None; best_state=None; best_reducer=None; best_stats=None
    for i,cfg in enumerate(GRID):
        print(f"[{i+1}/{len(GRID)}] qubits {cfg['n_qubits']} layers {cfg['n_layers']}")
        try:
            res, model, reducer, stats = train_one(curves_train,y_train,curves_val,y_val,curves_test,y_test,cfg)
            print(f"  val f1 {res['val_metrics']['f1_macro']:.4f} acc {res['val_metrics']['accuracy']:.4f} | test acc {res['test_metrics']['accuracy']:.4f} params {res['param_count']}")
            results.append(res)
            if best is None or res["val_metrics"]["f1_macro"]>best["val_metrics"]["f1_macro"]:
                best=res; best_state={k:v.cpu() for k,v in model.state_dict().items()}; best_reducer=reducer; best_stats=stats
        except Exception as e:
            traceback.print_exc()
            print(f"  failed {e}")
            results.append({"config":cfg,"error":str(e)})
    os.makedirs("results/reports",exist_ok=True)
    with open("results/reports/pure_search.json", "w", encoding="utf-8") as f:
        json.dump(results,f,indent=2)
    with open("results/reports/pure_search.md", "w", encoding="utf-8") as f:
        f.write("# Pure Quantum Search — validation best\n\n")
        f.write("| # | Qubits | Layers | Val Acc | Val F1 | Test Acc | Test F1 | Params |\n|---|---|---|---|---|---|---|---|\n")
        for idx,r in enumerate(results):
            if "error" in r: f.write(f"| {idx+1} | {r['config']['n_qubits']} | {r['config']['n_layers']} | ERR | ERR | ERR | ERR | - |\n")
            else: f.write(f"| {idx+1} | {r['n_qubits']} | {r['config']['n_layers']} | {r['val_metrics']['accuracy']:.4f} | {r['val_metrics']['f1_macro']:.4f} | {r['test_metrics']['accuracy']:.4f} | {r['test_metrics']['f1_macro']:.4f} | {r['param_count']} |\n")
        if best: f.write(f"\n**Best:** qubits {best['n_qubits']} layers {best['config']['n_layers']} val F1 {best['val_metrics']['f1_macro']:.4f}\n")
    if best:
        print(f"\nBEST pure qubits {best['n_qubits']} layers {best['config']['n_layers']} val f1 {best['val_metrics']['f1_macro']:.4f}")
        m,s=best_stats
        torch.save({"model_state_dict":best_state,"n_qubits":best["n_qubits"],"n_layers":best["config"]["n_layers"],"accuracy":best["test_metrics"]["accuracy"]*100,"mean":m,"std":s}, "models/pure_quantum_vqc.pt")
        best_reducer.save("models/pure_vqc_reducer.joblib")
        print("Saved best pure to models/pure_quantum_vqc.pt")

if __name__=="__main__":
    main()
