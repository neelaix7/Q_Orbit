# optimize_hybrid_fast.py — subset grid search + full training of best
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
    {"name":"q8l2h16", "n_qubits":8, "n_layers":2, "hidden":16, "lr":0.01, "epochs":8, "pca":False, "k_best":None},
    {"name":"q8l2h32", "n_qubits":8, "n_layers":2, "hidden":32, "lr":0.01, "epochs":8, "pca":False, "k_best":None},
    {"name":"q8l3h16", "n_qubits":8, "n_layers":3, "hidden":16, "lr":0.01, "epochs":8, "pca":False, "k_best":None},
    {"name":"q6l2h32", "n_qubits":6, "n_layers":2, "hidden":32, "lr":0.01, "epochs":8, "pca":False, "k_best":None},
    {"name":"q8l2h32_k15", "n_qubits":8, "n_layers":2, "hidden":32, "lr":0.01, "epochs":8, "pca":False, "k_best":15},
    {"name":"q8l2h32_pca8", "n_qubits":8, "n_layers":2, "hidden":32, "lr":0.01, "epochs":8, "pca":8, "k_best":None},
    {"name":"q8l2h32_lr0.005", "n_qubits":8, "n_layers":2, "hidden":32, "lr":0.005, "epochs":8, "pca":False, "k_best":None},
    {"name":"q4l2h16", "n_qubits":4, "n_layers":2, "hidden":16, "lr":0.01, "epochs":8, "pca":False, "k_best":None},
]

BATCH=64
SUBSET_TRAIN=3000
SUBSET_VAL=600

def load_and_subset():
    tr=np.load(CFG["dataset"]["train_path"], allow_pickle=True)
    va=np.load(CFG["dataset"]["val_path"], allow_pickle=True)
    te=np.load(CFG["dataset"]["test_path"], allow_pickle=True)
    ct,yt=tr["curves"], tr["labels"].astype(int)
    cv,yv=va["curves"], va["labels"].astype(int)
    cte,yte=te["curves"], te["labels"].astype(int)
    # deterministic subset
    rng=np.random.RandomState(SEED)
    idx=rng.choice(len(ct), SUBSET_TRAIN, replace=False)
    ct_s, yt_s = ct[idx], yt[idx]
    idx2=rng.choice(len(cv), SUBSET_VAL, replace=False)
    cv_s, yv_s = cv[idx2], yv[idx2]
    # features
    f_train=np.array([extract_all_features(c) for c in ct_s], dtype=np.float32)
    f_val=np.array([extract_all_features(c) for c in cv_s], dtype=np.float32)
    # for final test we use full test but also need full train for final retrain
    f_train_full=np.array([extract_all_features(c) for c in ct], dtype=np.float32)
    f_val_full=np.array([extract_all_features(c) for c in cv], dtype=np.float32)
    f_test=np.array([extract_all_features(c) for c in cte], dtype=np.float32)
    return (f_train,f_val,f_train_full,f_val_full,f_test),(yt_s,yv_s,yt,yv,yte),(ct_s,cv_s,ct,cv,cte)

def train_one_subset(cfg, f_train,y_train,f_val,y_val,f_test,y_test):
    nq=cfg["n_qubits"]; nl=cfg["n_layers"]; hidden=cfg["hidden"]; lr=cfg["lr"]; epochs=cfg["epochs"]
    mean=f_train.mean(axis=0); std=f_train.std(axis=0)+1e-8
    selector=None; pca_reducer=None; n_features=None
    if cfg["k_best"] is not None:
        k=cfg["k_best"]
        selector=SelectKBest(mutual_info_classif, k=k)
        X_train=(f_train-mean)/std
        selector.fit(X_train, y_train)
        X_train=selector.transform(X_train)
        X_val=selector.transform((f_val-mean)/std)
        X_test=selector.transform((f_test-mean)/std)
        n_features=k
    elif cfg["pca"] is not None:
        n_components=cfg["pca"]
        reducer=QuantumReducer(n_components=n_components)
        reducer.fit(f_train)
        X_train=reducer.transform(f_train)
        m=X_train.mean(axis=0); s=X_train.std(axis=0)+1e-8
        X_train=(X_train-m)/s
        X_val=(reducer.transform(f_val)-m)/s
        X_test=(reducer.transform(f_test)-m)/s
        n_features=n_components
        pca_reducer=(reducer,m,s)
    else:
        X_train=(f_train-mean)/std
        X_val=(f_val-mean)/std
        X_test=(f_test-mean)/std
        n_features=X_train.shape[1]
    model=HybridQuantumClassifier(n_features=n_features, n_qubits=nq, n_layers=nl, n_classical_hidden=hidden)
    opt=torch.optim.Adam(model.parameters(), lr=lr)
    crit=nn.CrossEntropyLoss()
    Xtr_t=torch.tensor(X_train,dtype=torch.float32); ytr_t=torch.tensor(y_train,dtype=torch.long)
    Xva_t=torch.tensor(X_val,dtype=torch.float32)
    n=len(Xtr_t)
    best_f1=-1; best_state=None
    t0=time.time()
    for epoch in range(1,epochs+1):
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
            f1=compute_all_metrics(y_val,pred,proba.numpy())["f1_macro"]
            if f1>best_f1:
                best_f1=f1; best_state={k:v.cpu() for k,v in model.state_dict().items()}
    train_time=time.time()-t0
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        Xte_t=torch.tensor(X_test,dtype=torch.float32)
        logits,proba=model(Xte_t); proba=proba.numpy(); pred=np.argmax(proba,axis=1)
        test_metrics=compute_all_metrics(y_test,pred,proba)
        logits,proba=model(Xva_t); proba=proba.numpy(); pred=np.argmax(proba,axis=1)
        val_metrics=compute_all_metrics(y_val,pred,proba)
    params=sum(p.numel() for p in model.parameters())
    return {"config":cfg,"n_features":n_features,"val_metrics":val_metrics,"test_metrics":test_metrics,"train_time":train_time,"param_count":params,"best_val_f1":best_f1}, model, (selector,pca_reducer,mean,std)

def main():
    (f_train,f_val,f_train_full,f_val_full,f_test),(y_train_s,y_val_s,y_train_full,y_val_full,y_test),_ = load_and_subset()
    print(f"Subset grid train {len(f_train)} val {len(f_val)} | Full train {len(f_train_full)} val {len(f_val_full)} test {len(f_test)}")
    results=[]
    best=None; best_model=None; best_aux=None
    for i,cfg in enumerate(GRID):
        print(f"\n[{i+1}/{len(GRID)}] {cfg['name']} q{cfg['n_qubits']} l{cfg['n_layers']} h{cfg['hidden']} lr{cfg['lr']} k={cfg['k_best']} pca={cfg['pca']}")
        try:
            res, model, aux = train_one_subset(cfg, f_train,y_train_s,f_val,y_val_s,f_test,y_test)
            print(f"  val F1 {res['val_metrics']['f1_macro']:.4f} acc {res['val_metrics']['accuracy']:.4f} | test F1 {res['test_metrics']['f1_macro']:.4f} acc {res['test_metrics']['accuracy']:.4f}")
            results.append({k:v for k,v in res.items() if k not in ["val_metrics","test_metrics"]})
            results[-1]["val_metrics"]=res["val_metrics"]
            results[-1]["test_metrics"]=res["test_metrics"]
            if best is None or res["best_val_f1"]>best["best_val_f1"]:
                best=res; best_model=model; best_aux=aux
        except Exception as e:
            traceback.print_exc()
            results.append({"config":cfg,"error":str(e)})
    os.makedirs("results/reports",exist_ok=True)
    with open("results/reports/hybrid_optimization.json", "w", encoding="utf-8") as f: json.dump(results,f,indent=2)
    with open("results/reports/hybrid_optimization.md", "w", encoding="utf-8") as f:
        f.write("# Hybrid Optimization — subset grid search, val-selected\n\n")
        f.write(f"| # | Config | Q | L | H | LR | k_best | PCA | Val F1 | Val Acc | Test F1 | Test Acc | Params | Train(s) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for idx,r in enumerate(results):
            if "error" in r: f.write(f"| {idx+1} | {r['config']['name']} | ERR |\n")
            else: f.write(f"| {idx+1} | {r['config']['name']} | {r['config']['n_qubits']} | {r['config']['n_layers']} | {r['config']['hidden']} | {r['config']['lr']} | {r['config']['k_best']} | {r['config']['pca']} | {r['val_metrics']['f1_macro']:.4f} | {r['val_metrics']['accuracy']:.4f} | {r['test_metrics']['f1_macro']:.4f} | {r['test_metrics']['accuracy']:.4f} | {r['param_count']} | {r['train_time']:.1f} |\n")
        if best: f.write(f"\n**Best subset on VAL:** {best['config']['name']} Val F1 {best['val_metrics']['f1_macro']:.4f}\n")
    if best is None:
        print("No best found"); return
    print(f"\nBest subset config: {best['config']['name']} — retraining on FULL data for 20 epochs...")
    # retrain best on full data for 20 epochs with same preprocessing pipeline but fit on full train
    cfg=best["config"].copy()
    cfg["epochs"]=20
    # use full train/val for final
    selector,pca_reducer,mean,std = best_aux
    # Need to refit selector/reducer on full train if needed
    # We'll redo train_one but with full data and longer epochs
    # Helper: reuse train_one_subset but with full arrays and 20 epochs
    # create a dedicated full training
    # For simplicity, call train_one_subset with full arrays but modify to fit on full
    # Let's implement inline full training
    nq=cfg["n_qubits"]; nl=cfg["n_layers"]; hidden=cfg["hidden"]; lr=cfg["lr"]; epochs=20
    # Prepare features: f_train_full etc already available in outer scope? We'll reload
    # Reload to ensure correct
    tr=np.load(CFG["dataset"]["train_path"],allow_pickle=True); va=np.load(CFG["dataset"]["val_path"],allow_pickle=True); te=np.load(CFG["dataset"]["test_path"],allow_pickle=True)
    ct,yt=tr["curves"],tr["labels"].astype(int); cv,yv=va["curves"],va["labels"].astype(int); cte,yte=te["curves"],te["labels"].astype(int)
    f_train_full=np.array([extract_all_features(c) for c in ct],dtype=np.float32)
    f_val_full_np=np.array([extract_all_features(c) for c in cv],dtype=np.float32)
    f_test_full=np.array([extract_all_features(c) for c in cte],dtype=np.float32)
    mean_full=f_train_full.mean(axis=0); std_full=f_train_full.std(axis=0)+1e-8
    if cfg["k_best"] is not None:
        k=cfg["k_best"]
        sel=SelectKBest(mutual_info_classif,k=k)
        X_train_full=sel.fit_transform((f_train_full-mean_full)/std_full, yt)
        X_val_full=sel.transform((f_val_full_np-mean_full)/std_full)
        X_test_full=sel.transform((f_test_full-mean_full)/std_full)
        n_features=k
        selector_final=sel
        pca_final=None
    elif cfg["pca"] is not None:
        nc=cfg["pca"]
        red=QuantumReducer(n_components=nc); red.fit(f_train_full)
        Xp=red.transform(f_train_full); m=Xp.mean(axis=0); s=Xp.std(axis=0)+1e-8
        X_train_full=(Xp-m)/s; X_val_full=(red.transform(f_val_full_np)-m)/s; X_test_full=(red.transform(f_test_full)-m)/s
        n_features=nc
        selector_final=None; pca_final=(red,m,s)
    else:
        X_train_full=(f_train_full-mean_full)/std_full
        X_val_full=(f_val_full_np-mean_full)/std_full
        X_test_full=(f_test_full-mean_full)/std_full
        n_features=X_train_full.shape[1]
        selector_final=None; pca_final=None
    model=HybridQuantumClassifier(n_features=n_features, n_qubits=nq, n_layers=nl, n_classical_hidden=hidden)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    crit=nn.CrossEntropyLoss()
    Xtr_t=torch.tensor(X_train_full,dtype=torch.float32); ytr_t=torch.tensor(yt,dtype=torch.long)
    Xva_t=torch.tensor(X_val_full,dtype=torch.float32)
    Xte_t=torch.tensor(X_test_full,dtype=torch.float32)
    n=len(Xtr_t)
    best_f1=-1; best_state=None
    t0=time.time()
    for epoch in range(1,epochs+1):
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
            f1=compute_all_metrics(yv,pred,proba.numpy())["f1_macro"]
            acc=compute_all_metrics(yv,pred,proba.numpy())["accuracy"]
            print(f"  epoch {epoch} val F1 {f1:.4f} acc {acc:.4f}")
            if f1>best_f1:
                best_f1=f1; best_state={k:v.cpu() for k,v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits,proba=model(Xte_t); proba=proba.numpy(); pred=np.argmax(proba,axis=1)
        test_metrics=compute_all_metrics(yte,pred,proba)
        logits,proba=model(Xva_t); proba=proba.numpy(); pred=np.argmax(proba,axis=1)
        val_metrics=compute_all_metrics(yv,pred,proba)
    print(f"FINAL full train best val F1 {val_metrics['f1_macro']:.4f} acc {val_metrics['accuracy']:.4f} | test F1 {test_metrics['f1_macro']:.4f} acc {test_metrics['accuracy']:.4f}")
    os.makedirs("models",exist_ok=True)
    ckpt={"model_state_dict":model.state_dict(),"n_features":n_features,"n_qubits":nq,"n_layers":nl,"n_classical_hidden":hidden,"accuracy":float(test_metrics["accuracy"]*100),"val_f1":float(val_metrics["f1_macro"]),"config":cfg}
    torch.save(ckpt,"models/hybrid_quantum_model.pt")
    # save aux
    if selector_final is not None:
        joblib.dump(selector_final,"models/hybrid_selector.joblib")
        np.savez("models/hybrid_pca_stats.npz", mean=mean_full, std=std_full)
        for p in ["models/hybrid_reducer.joblib"]:
            if os.path.exists(p): os.remove(p)
        print("Saved selector")
    elif pca_final is not None:
        red,m,s=pca_final
        red.save("models/hybrid_reducer.joblib")
        np.savez("models/hybrid_pca_stats.npz", mean=m, std=s)
        joblib.dump({"mean":mean_full,"std":std_full}, "models/feature_scaler.joblib")
        if os.path.exists("models/hybrid_selector.joblib"): os.remove("models/hybrid_selector.joblib")
        print("Saved reducer")
    else:
        joblib.dump({"mean":mean_full,"std":std_full,"scale":std_full},"models/feature_scaler.joblib")
        for p in ["models/hybrid_selector.joblib","models/hybrid_reducer.joblib","models/hybrid_pca_stats.npz"]:
            if os.path.exists(p):
                try: os.remove(p)
                except Exception: pass
        print("Saved feature_scaler")
    with open("models/hybrid_best_val.json", "w", encoding="utf-8") as f:
        json.dump({"config":cfg,"val_f1":val_metrics["f1_macro"],"test_acc":test_metrics["accuracy"],"test_f1":test_metrics["f1_macro"]},f,indent=2)
    # also update hybrid_optimization.json with final
    results.append({"config":cfg,"note":"FULL_RETRAIN_20epochs","val_metrics":val_metrics,"test_metrics":test_metrics})
    with open("results/reports/hybrid_optimization.json", "w", encoding="utf-8") as f: json.dump(results,f,indent=2)
    print("Done.")

if __name__=="__main__":
    main()
