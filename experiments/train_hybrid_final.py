# train_hybrid_final.py — retrain top 2 hybrid configs on FULL data, val-selected
import os, sys, json, time
import numpy as np, torch, torch.nn as nn, joblib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.classical.feature_engineering import extract_all_features
from src.quantum.hybrid_model import HybridQuantumClassifier
from src.evaluation.compare_three import compute_all_metrics
CFG=json.load(open("configs/qorbit_config.json"))
SEED=CFG["seed"]
torch.manual_seed(SEED); np.random.seed(SEED)
BATCH=64
EPOCHS=20

def load():
    tr=np.load(CFG["dataset"]["train_path"], allow_pickle=True)
    va=np.load(CFG["dataset"]["val_path"], allow_pickle=True)
    te=np.load(CFG["dataset"]["test_path"], allow_pickle=True)
    ct,yt=tr["curves"],tr["labels"].astype(int)
    cv,yv=va["curves"],va["labels"].astype(int)
    cte,yte=te["curves"],te["labels"].astype(int)
    f_tr=np.array([extract_all_features(c) for c in ct],dtype=np.float32)
    f_va=np.array([extract_all_features(c) for c in cv],dtype=np.float32)
    f_te=np.array([extract_all_features(c) for c in cte],dtype=np.float32)
    mean=f_tr.mean(axis=0); std=f_tr.std(axis=0)+1e-8
    X_tr=(f_tr-mean)/std; X_va=(f_va-mean)/std; X_te=(f_te-mean)/std
    return (X_tr,X_va,X_te),(yt,yv,yte),(mean,std)

GRID=[
    {"name":"q8l2h16_lr0.01", "n_qubits":8,"n_layers":2,"hidden":16,"lr":0.01},
    {"name":"q8l2h32_lr0.01", "n_qubits":8,"n_layers":2,"hidden":32,"lr":0.01},
    {"name":"q8l2h32_lr0.005", "n_qubits":8,"n_layers":2,"hidden":32,"lr":0.005},
    {"name":"q8l3h32_lr0.01", "n_qubits":8,"n_layers":3,"hidden":32,"lr":0.01},
]

def train_cfg(cfg, X_tr,y_tr,X_va,y_va,X_te,y_te):
    nq=cfg["n_qubits"]; nl=cfg["n_layers"]; hidden=cfg["hidden"]; lr=cfg["lr"]
    model=HybridQuantumClassifier(n_features=21, n_qubits=nq, n_layers=nl, n_classical_hidden=hidden)
    opt=torch.optim.Adam(model.parameters(), lr=lr)
    crit=nn.CrossEntropyLoss()
    Xtr_t=torch.tensor(X_tr,dtype=torch.float32); ytr_t=torch.tensor(y_tr,dtype=torch.long)
    Xva_t=torch.tensor(X_va,dtype=torch.float32); Xte_t=torch.tensor(X_te,dtype=torch.float32)
    n=len(Xtr_t)
    best_f1=-1; best_state=None
    t0=time.time()
    for epoch in range(1, EPOCHS+1):
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
            f1=compute_all_metrics(y_va,pred,proba.numpy())["f1_macro"]
            print(f"  epoch {epoch:2d} val F1 {f1:.4f}")
            if f1>best_f1:
                best_f1=f1; best_state={k:v.cpu() for k,v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits,proba=model(Xte_t); pred=np.argmax(proba.numpy(),axis=1); test_metrics=compute_all_metrics(y_te,pred,proba.numpy())
        logits,proba=model(Xva_t); pred=np.argmax(proba.numpy(),axis=1); val_metrics=compute_all_metrics(y_va,pred,proba.numpy())
    train_time=time.time()-t0
    return model, val_metrics, test_metrics, train_time

def main():
    (X_tr,X_va,X_te),(y_tr,y_va,y_te),(mean,std)=load()
    print(f"Full data X_tr {X_tr.shape} X_va {X_va.shape} X_te {X_te.shape}")
    results=[]
    best=None; best_model=None
    for cfg in GRID:
        print(f"\n=== {cfg['name']} q{cfg['n_qubits']} l{cfg['n_layers']} h{cfg['hidden']} lr{cfg['lr']} ===")
        model,val_metrics,test_metrics,train_time=train_cfg(cfg,X_tr,y_tr,X_va,y_va,X_te,y_te)
        print(f" -> val F1 {val_metrics['f1_macro']:.4f} acc {val_metrics['accuracy']:.4f} | test F1 {test_metrics['f1_macro']:.4f} acc {test_metrics['accuracy']:.4f} time {train_time:.1f}s")
        results.append({"config":cfg,"val_metrics":val_metrics,"test_metrics":test_metrics,"train_time":train_time,"param_count":sum(p.numel() for p in model.parameters())})
        if best is None or val_metrics["f1_macro"]>best["val_metrics"]["f1_macro"]:
            best=results[-1]; best_model=model
    print("\n=== FINAL BEST ===")
    print(best)
    # save best
    if best:
        cfg=best["config"]
        ckpt={"model_state_dict":best_model.state_dict(),"n_features":21,"n_qubits":cfg["n_qubits"],"n_layers":cfg["n_layers"],"n_classical_hidden":cfg["hidden"],"accuracy":float(best["test_metrics"]["accuracy"]*100),"val_f1":float(best["val_metrics"]["f1_macro"]),"config":cfg}
        torch.save(ckpt,"models/hybrid_quantum_model.pt")
        joblib.dump({"mean":mean,"std":std,"scale":std},"models/feature_scaler.joblib")
        for p in ["models/hybrid_selector.joblib","models/hybrid_reducer.joblib","models/hybrid_pca_stats.npz"]:
            if os.path.exists(p):
                try: os.remove(p)
                except Exception: pass
        print(f"Saved best {cfg['name']} test acc {best['test_metrics']['accuracy']:.4f}")
        with open("models/hybrid_best_val.json", "w", encoding="utf-8") as f: json.dump({"config":cfg,"val_f1":best["val_metrics"]["f1_macro"],"test_acc":best["test_metrics"]["accuracy"],"test_f1":best["test_metrics"]["f1_macro"]},f,indent=2)
        with open("results/reports/hybrid_final.json", "w", encoding="utf-8") as f: json.dump(results,f,indent=2)

if __name__=="__main__":
    main()
