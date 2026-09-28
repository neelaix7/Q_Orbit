# run_fair_comparison.py — Fair 3-way comparison on frozen test set + robustness sweeps
# Uses same dataset, same splits, same preprocessing, same metrics. No tuning on test.
from __future__ import annotations
import os, sys, json, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))
from src.classical.feature_engineering import extract_all_features
from src.classical.cnn_baseline import LightCurveCNN
from src.quantum.hybrid_model import HybridQuantumClassifier
from src.quantum.vqc import PureVQC
from src.features.reducer import QuantumReducer
from src.evaluation.compare_three import compute_all_metrics
from src.evaluation.degradation import add_gaussian_noise, truncate_observation, add_missing
import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIG_PATH="configs/qorbit_config.json"
with open(CONFIG_PATH) as f: CFG=json.load(f)
SEED=CFG["seed"]

def load_frozen():
    tr=np.load(CFG["dataset"]["train_path"],allow_pickle=True)
    va=np.load(CFG["dataset"]["val_path"],allow_pickle=True)
    te=np.load(CFG["dataset"]["test_path"],allow_pickle=True)
    return (tr["curves"],tr["labels"].astype(int)), (va["curves"],va["labels"].astype(int)), (te["curves"],te["labels"].astype(int))

def curves_to_features(curves, mean=None, std=None, reducer=None):
    feats=np.array([extract_all_features(c) for c in curves],dtype=np.float32)
    if reducer is not None:
        feats=reducer.transform(feats)
        # reducer already includes scaler; still standardize after PCA for circuit stability using train stats
        if mean is not None:
            feats=(feats-mean)/std
        return feats
    if mean is not None:
        feats=(feats-mean)/std
    return feats

def predict_torch(model, X, batch=256):
    model.eval()
    preds=[]; probas=[]
    with torch.no_grad():
        for i in range(0,len(X),batch):
            xb=torch.tensor(X[i:i+batch],dtype=torch.float32)
            # CNN expects (batch,1,256) ; feature models expect (batch,n_features)
            if isinstance(model, LightCurveCNN):
                xb=xb.unsqueeze(1) if xb.dim()==2 else xb  # but for CNN we pass raw curves, not features — caller handles
            out=model(xb) if not isinstance(model, LightCurveCNN) else model(xb)
            if isinstance(out, tuple):
                logits, proba = out
            else:
                logits=out; proba=torch.softmax(logits,dim=1)
            probas.append(proba.cpu().numpy()); preds.append(torch.argmax(proba,dim=1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probas)

def main():
    os.makedirs("results/reports",exist_ok=True); os.makedirs("results/figures",exist_ok=True)
    (curves_train,y_train),(curves_val,y_val),(curves_test,y_test)=load_frozen()
    print(f"Frozen splits: train {len(curves_train)} val {len(curves_val)} test {len(curves_test)} seed {SEED}")

    # Feature preprocessing: fit scaler on TRAIN features only
    feats_train_raw=np.array([extract_all_features(c) for c in curves_train],dtype=np.float32)
    mean=feats_train_raw.mean(axis=0); std=feats_train_raw.std(axis=0)+1e-8
    # save scaler
    joblib.dump({"mean":mean,"std":std}, "models/feature_scaler.joblib")

    X_train = (feats_train_raw-mean)/std
    X_test  = curves_to_features(curves_test, mean, std)
    X_val   = curves_to_features(curves_val, mean, std)

    # PCA reducers for quantum (fit on train only) — for n_qubits variants
    reducers={}
    for nq in CFG["quantum"]["n_qubits_options"]:
        r=QuantumReducer(n_components=nq)
        r.fit(feats_train_raw)
        # standardize PCA output using train PCA stats
        pc_train=r.transform(feats_train_raw)
        m2=pc_train.mean(axis=0); s2=pc_train.std(axis=0)+1e-8
        reducers[nq]=(r,m2,s2)
        r.save(f"models/reducer_{nq}q.joblib")
        print(f"Reducer {nq}q explained var {np.round(r.pca.explained_variance_ratio_,3)} sum {r.pca.explained_variance_ratio_.sum():.3f}")

    # default 8q for all quantum models
    r8,m8,s8=reducers[8]
    Xq_train=(r8.transform(feats_train_raw)-m8)/s8
    Xq_test=(r8.transform(np.array([extract_all_features(c) for c in curves_test]))-m8)/s8
    Xq_val=(r8.transform(np.array([extract_all_features(c) for c in curves_val]))-m8)/s8

    results={}

    # 1. Classical CNN (raw curves)
    cnn_path="models/classical_cnn_model.pt"
    cnn=LightCurveCNN(n_classes=5)
    if os.path.exists(cnn_path):
        ckpt=torch.load(cnn_path,weights_only=False,map_location="cpu")
        cnn.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded CNN from {cnn_path}")
    else:
        print("CNN checkpoint not found — will evaluate randomly initialized (run train_classical first)")
    cnn.eval()
    with torch.no_grad():
        X_test_cnn=torch.tensor(curves_test,dtype=torch.float32).unsqueeze(1)
        logits=cnn(X_test_cnn)
        proba=torch.softmax(logits,dim=1).numpy()
        pred=np.argmax(proba,axis=1)
    results["Classical_CNN"]=compute_all_metrics(y_test,pred,proba)
    results["Classical_CNN"]["_pred"]=pred  # for confusion later

    # 2. Classical SVM (21 features)
    svm_path="models/classical_svm_model.joblib"
    svm_bundle=joblib.load(svm_path) if os.path.exists(svm_path) else None
    if svm_bundle is not None:
        from src.classical.cnn_baseline import LightCurveSVM
        # bundle contains sklearn SVC under 'model'
        clf=svm_bundle["model"]
        # features already standardized with same mean/std
        from sklearn.preprocessing import StandardScaler
        # predict_proba if available else decision
        try:
            proba_svm=clf.predict_proba(X_test)
        except Exception:
            # fallback: one-hot
            pred_svm=clf.predict(X_test)
            proba_svm=np.eye(5)[pred_svm]
            proba_svm=proba_svm/np.clip(proba_svm.sum(axis=1,keepdims=True),1e-9,None)
            pred_svm=np.argmax(proba_svm,axis=1)
        else:
            pred_svm=np.argmax(proba_svm,axis=1)
        results["Classical_SVM"]=compute_all_metrics(y_test,pred_svm,proba_svm)
    else:
        # train quickly if missing
        from sklearn.svm import SVC
        clf=SVC(kernel="rbf",C=1.0,gamma="scale",probability=True,random_state=SEED)
        clf.fit(X_train, y_train)
        proba_svm=clf.predict_proba(X_test); pred_svm=np.argmax(proba_svm,axis=1)
        results["Classical_SVM"]=compute_all_metrics(y_test,pred_svm,proba_svm)

    # 3. Pure Quantum VQC (8 qubits, PCA)
    pure_path="models/pure_quantum_vqc.pt"
    pure=PureVQC(n_qubits=8,n_layers=2,n_classes=5)
    if os.path.exists(pure_path):
        ckpt=torch.load(pure_path,weights_only=False,map_location="cpu")
        pure.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded Pure VQC from {pure_path}")
    else:
        print("Pure VQC checkpoint not found — untrained (run train_pure_quantum)")
    pure.eval()
    with torch.no_grad():
        # use Xq_test (PCA 8 dims)
        logits,proba=pure(torch.tensor(Xq_test,dtype=torch.float32))
        proba=proba.numpy(); pred=np.argmax(proba,axis=1)
    results["Pure_Quantum_VQC"]=compute_all_metrics(y_test,pred,proba)

    # 4. Hybrid (21 features -> quantum layer)
    hyb_path="models/hybrid_quantum_model.pt"
    hyb=HybridQuantumClassifier(n_features=21,n_qubits=8,n_layers=2,n_classical_hidden=16)
    if os.path.exists(hyb_path):
        ckpt=torch.load(hyb_path,weights_only=False,map_location="cpu")
        hyb.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded Hybrid from {hyb_path}")
    else:
        print("Hybrid checkpoint not found")
    hyb.eval()
    with torch.no_grad():
        logits,proba=hyb(torch.tensor(X_test,dtype=torch.float32))
        proba=proba.numpy(); pred=np.argmax(proba,axis=1)
    results["Hybrid_Quantum"]=compute_all_metrics(y_test,pred,proba)

    # Save main table
    table=[{"model":k,"accuracy":v["accuracy"],"precision_macro":v["precision_macro"],"recall_macro":v["recall_macro"],"f1_macro":v["f1_macro"],"f1_weighted":v["f1_weighted"],"roc_auc":v.get("roc_auc_ovr_macro")} for k,v in results.items()]
    with open("results/reports/fair_comparison.json", "w", encoding="utf-8") as f:
        json.dump({k:{kk:vv for kk,vv in v.items() if not kk.startswith("_")} for k,v in results.items()}, f, indent=2)
    # markdown
    with open("results/reports/fair_comparison.md", "w", encoding="utf-8") as f:
        f.write("# Q-ORBIT Fair Three-Way Comparison (70/15/15 frozen, scaler fit on train only)\n\n")
        f.write(f"Seed {SEED} | Test {len(y_test)} samples\n\n")
        f.write("| Model | Accuracy | Precision_macro | Recall_macro | F1_macro | F1_weighted | ROC-AUC |\n|---|---|---|---|---|---|---|\n")
        for r in table:
            f.write(f"| {r['model']} | {r['accuracy']:.4f} | {r['precision_macro']:.4f} | {r['recall_macro']:.4f} | {r['f1_macro']:.4f} | {r['f1_weighted']:.4f} | {str(round(r['roc_auc'],4) if r['roc_auc'] else '-') } |\n")
        f.write("\n## Per-class F1\n\n| Class | Classical_CNN | Classical_SVM | Pure_Quantum | Hybrid |\n|---|---|---|---|---|\n")
        for i,name in enumerate(CFG["dataset"]["class_names"]):
            f.write(f"| {name} | {results['Classical_CNN']['per_class'][str(i)]['f1'] if str(i) in results['Classical_CNN']['per_class'] else results['Classical_CNN']['per_class'][i]['f1']:.4f} | {results['Classical_SVM']['per_class'][i]['f1']:.4f} | {results['Pure_Quantum_VQC']['per_class'][i]['f1']:.4f} | {results['Hybrid_Quantum']['per_class'][i]['f1']:.4f} |\n")

    print("Saved fair_comparison.json/md")
    for r in table:
        print(f"{r['model']:18s} acc {r['accuracy']:.4f} f1_macro {r['f1_macro']:.4f}")

    # Robustness sweeps on TEST ONLY
    noise_levels=CFG["robustness"]["noise_levels"]
    obs_fracs=CFG["robustness"]["observation_fractions"]
    miss_fracs=CFG["robustness"]["missing_fractions"]
    # helper to evaluate given degraded curves_test
    def eval_all_on_curves(curves_deg):
        # recompute features with same scaler/reducer
        X_deg = np.array([extract_all_features(c) for c in curves_deg],dtype=np.float32)
        X_deg_s = (X_deg-mean)/std
        Xq_deg = (r8.transform(X_deg)-m8)/s8
        # CNN
        with torch.no_grad():
            logits=cnn(torch.tensor(curves_deg,dtype=torch.float32).unsqueeze(1))
            proba=torch.softmax(logits,dim=1).numpy(); pred=np.argmax(proba,axis=1)
            acc_c=float((pred==y_test).mean())
            # SVM
            proba_s=clf.predict_proba(X_deg_s) if 'clf' in locals() else np.eye(5)[pred]
            pred_s=np.argmax(proba_s,axis=1); acc_s=float((pred_s==y_test).mean())
            # pure
            _,proba_p=pure(torch.tensor(Xq_deg,dtype=torch.float32)); proba_p=proba_p.numpy(); pred_p=np.argmax(proba_p,axis=1); acc_p=float((pred_p==y_test).mean())
            # hybrid
            _,proba_h=hyb(torch.tensor(X_deg_s,dtype=torch.float32)); proba_h=proba_h.numpy(); pred_h=np.argmax(proba_h,axis=1); acc_h=float((pred_h==y_test).mean())
        return {"Classical_CNN":acc_c,"Classical_SVM":acc_s,"Pure_Quantum":acc_p,"Hybrid":acc_h}

    # noise sweep
    rows=[]
    for ns in noise_levels:
        deg=add_gaussian_noise(curves_test, ns, seed=SEED) if ns>0 else curves_test
        rows.append((ns, eval_all_on_curves(deg)))
    with open("results/reports/robustness_noise.json", "w", encoding="utf-8") as f:
        json.dump([{"noise":ns, **r} for ns,r in rows], f, indent=2)
    # plot
    fig,ax=plt.subplots(figsize=(8,5))
    for k in ["Classical_CNN","Classical_SVM","Pure_Quantum","Hybrid"]:
        ax.plot([r[0] for r in rows],[r[1][k] for r in rows],marker="o",label=k)
    ax.set_xlabel("Gaussian noise std"); ax.set_ylabel("Accuracy"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("results/figures/robustness_noise.png",dpi=150); plt.close()

    # observation length
    rows2=[]
    for f in obs_fracs:
        deg=truncate_observation(curves_test, f, seed=SEED) if f<1.0 else curves_test
        rows2.append((f, eval_all_on_curves(deg)))
    with open("results/reports/robustness_observation.json", "w", encoding="utf-8") as f:
        json.dump([{"observation_fraction":fr, **r} for fr,r in rows2], f, indent=2)
    fig,ax=plt.subplots(figsize=(8,5))
    for k in ["Classical_CNN","Classical_SVM","Pure_Quantum","Hybrid"]:
        ax.plot([r[0] for r in rows2],[r[1][k] for r in rows2],marker="o",label=k)
    ax.set_xlabel("Observation fraction"); ax.set_ylabel("Accuracy"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("results/figures/robustness_observation.png",dpi=150); plt.close()

    # missing
    rows3=[]
    for mf in miss_fracs:
        deg=add_missing(curves_test, mf, seed=SEED) if mf>0 else curves_test
        rows3.append((mf, eval_all_on_curves(deg)))
    with open("results/reports/robustness_missing.json", "w", encoding="utf-8") as f:
        json.dump([{"missing_fraction":mf, **r} for mf,r in rows3], f, indent=2)
    fig,ax=plt.subplots(figsize=(8,5))
    for k in ["Classical_CNN","Classical_SVM","Pure_Quantum","Hybrid"]:
        ax.plot([r[0] for r in rows3],[r[1][k] for r in rows3],marker="o",label=k)
    ax.set_xlabel("Missing fraction"); ax.set_ylabel("Accuracy"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("results/figures/robustness_missing.png",dpi=150); plt.close()

    # comparison bar plot
    fig,ax=plt.subplots(figsize=(10,6))
    models=list(results.keys()); accs=[results[m]["accuracy"] for m in models]; f1s=[results[m]["f1_macro"] for m in models]
    x=np.arange(len(models)); ax.bar(x-0.2, accs, width=0.4, label="Accuracy"); ax.bar(x+0.2, f1s, width=0.4, label="F1-macro")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=12, ha="right"); ax.legend(); ax.grid(axis="y",alpha=0.3); ax.set_ylim(0,1)
    plt.tight_layout(); plt.savefig("results/figures/fair_comparison.png",dpi=150); plt.close()
    print("Robustness plots saved.")

if __name__=="__main__":
    main()
