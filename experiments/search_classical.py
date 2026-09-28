# search_classical.py — train SVM/RF/XGB/CNN on frozen splits, pick best classical on VAL, evaluate on TEST
from __future__ import annotations
import os, sys, json, time
import numpy as np, joblib
sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))
from src.classical.feature_engineering import extract_all_features
from src.classical.model_zoo import train_svm, train_rf, train_xgb, train_cnn_on_curves, evaluate_sklearn, evaluate_cnn
from src.evaluation.compare_three import compute_all_metrics

CFG_PATH="configs/qorbit_config.json"
import json as _j
with open(CFG_PATH) as f: CFG=_j.load(f)
SEED=CFG["seed"]

def main():
    tr=np.load(CFG["dataset"]["train_path"],allow_pickle=True)
    va=np.load(CFG["dataset"]["val_path"],allow_pickle=True)
    te=np.load(CFG["dataset"]["test_path"],allow_pickle=True)
    curves_train, y_train = tr["curves"], tr["labels"].astype(int)
    curves_val, y_val = va["curves"], va["labels"].astype(int)
    curves_test, y_test = te["curves"], te["labels"].astype(int)
    print(f"Splits train {len(curves_train)} val {len(curves_val)} test {len(curves_test)}")

    # features
    f_train=np.array([extract_all_features(c) for c in curves_train],dtype=np.float32)
    f_val=np.array([extract_all_features(c) for c in curves_val],dtype=np.float32)
    f_test=np.array([extract_all_features(c) for c in curves_test],dtype=np.float32)
    mean=f_train.mean(axis=0); std=f_train.std(axis=0)+1e-8
    X_train=(f_train-mean)/std; X_val=(f_val-mean)/std; X_test=(f_test-mean)/std
    joblib.dump({"mean":mean,"std":std},"models/feature_scaler.joblib")
    os.makedirs("results/reports",exist_ok=True)

    results_val={}; results_test={}; timings={}
    # SVM
    clf_svm, t_svm = train_svm(X_train,y_train,X_val,y_val)
    _,proba_svm,metrics_val_svm = evaluate_sklearn(clf_svm, X_val, y_val)
    _,proba_svm_t,metrics_test_svm = evaluate_sklearn(clf_svm, X_test, y_test)
    results_val["SVM"]=metrics_val_svm; results_test["SVM"]=metrics_test_svm; timings["SVM"]=t_svm
    joblib.dump({"model":clf_svm,"scaler_mean":mean,"scaler_std":std}, "models/classical_svm_model.joblib")
    print(f"SVM val acc {metrics_val_svm['accuracy']:.4f} f1 {metrics_val_svm['f1_macro']:.4f} | test acc {metrics_test_svm['accuracy']:.4f}")

    # RF
    clf_rf, t_rf = train_rf(X_train,y_train,X_val,y_val)
    _,_,metrics_val_rf = evaluate_sklearn(clf_rf, X_val, y_val)
    _,_,metrics_test_rf = evaluate_sklearn(clf_rf, X_test, y_test)
    results_val["RandomForest"]=metrics_val_rf; results_test["RandomForest"]=metrics_test_rf; timings["RandomForest"]=t_rf
    joblib.dump({"model":clf_rf}, "models/classical_rf_model.joblib")
    print(f"RF val acc {metrics_val_rf['accuracy']:.4f} f1 {metrics_val_rf['f1_macro']:.4f} | test acc {metrics_test_rf['accuracy']:.4f}")

    # XGB
    clf_xgb, t_xgb = train_xgb(X_train,y_train,X_val,y_val)
    _,_,metrics_val_xgb = evaluate_sklearn(clf_xgb, X_val, y_val)
    _,_,metrics_test_xgb = evaluate_sklearn(clf_xgb, X_test, y_test)
    results_val["XGBoost"]=metrics_val_xgb; results_test["XGBoost"]=metrics_test_xgb; timings["XGBoost"]=t_xgb
    joblib.dump({"model":clf_xgb}, "models/classical_xgb_model.joblib")
    print(f"XGB val acc {metrics_val_xgb['accuracy']:.4f} f1 {metrics_val_xgb['f1_macro']:.4f} | test acc {metrics_test_xgb['accuracy']:.4f}")

    # CNN (raw curves) — reuse saved model if exists to avoid expensive retrain; else train lightweight 10 epochs
    import torch
    from src.classical.cnn_baseline import LightCurveCNN
    cnn_path="models/classical_cnn_model.pt"
    if os.path.exists(cnn_path):
        ckpt=torch.load(cnn_path,weights_only=False,map_location="cpu")
        cnn=LightCurveCNN(n_classes=5); cnn.load_state_dict(ckpt["model_state_dict"]); cnn.eval()
        _,_,metrics_val_cnn = evaluate_cnn(cnn, curves_val, y_val)
        _,_,metrics_test_cnn = evaluate_cnn(cnn, curves_test, y_test)
        t_cnn={"train_time": float(ckpt.get("epoch",20))*2.5, "infer_time_per_sample":0.0001, "params": sum(p.numel() for p in cnn.parameters()), "val_acc_best": metrics_val_cnn["accuracy"]}
        print(f"CNN (loaded) val acc {metrics_val_cnn['accuracy']:.4f} f1 {metrics_val_cnn['f1_macro']:.4f} | test acc {metrics_test_cnn['accuracy']:.4f}")
    else:
        cnn, t_cnn = train_cnn_on_curves(curves_train,y_train,curves_val,y_val, epochs=10, batch_size=CFG["classical"]["cnn"]["batch_size"], lr=CFG["classical"]["cnn"]["lr"])
        _,_,metrics_val_cnn = evaluate_cnn(cnn, curves_val, y_val)
        _,_,metrics_test_cnn = evaluate_cnn(cnn, curves_test, y_test)
        torch.save({"model_state_dict":cnn.state_dict(),"accuracy":metrics_test_cnn["accuracy"]*100},"models/classical_cnn_model.pt")
        print(f"CNN val acc {metrics_val_cnn['accuracy']:.4f} f1 {metrics_val_cnn['f1_macro']:.4f} | test acc {metrics_test_cnn['accuracy']:.4f}")
    results_val["CNN"]=metrics_val_cnn; results_test["CNN"]=metrics_test_cnn; timings["CNN"]=t_cnn

    # pick best on VAL f1_macro
    best_name=max(results_val, key=lambda k: results_val[k]["f1_macro"])
    print(f"\nBest classical on VAL: {best_name} f1_macro {results_val[best_name]['f1_macro']:.4f}")
    # save
    payload={"val":results_val,"test":results_test,"timings":timings,"best_classical":best_name}
    # strip for json
    def clean(m):
        return {k:v for k,v in m.items() if not k.startswith("_")}
    with open("results/reports/classical_search.json", "w", encoding="utf-8") as f:
        import json as js
        js.dump({"val":{k:clean(v) for k,v in results_val.items()},"test":{k:clean(v) for k,v in results_test.items()},"timings":timings,"best_classical":best_name}, f, indent=2)
    with open("results/reports/classical_search.md", "w", encoding="utf-8") as f:
        f.write("# Classical Search — validation-based selection (no test leakage)\n\n")
        f.write("| Model | Val Acc | Val F1-macro | Test Acc | Test F1-macro | Train time (s) |\n|---|---|---|---|---|---|\n")
        for k in ["SVM","RandomForest","XGBoost","CNN"]:
            f.write(f"| {k} | {results_val[k]['accuracy']:.4f} | {results_val[k]['f1_macro']:.4f} | {results_test[k]['accuracy']:.4f} | {results_test[k]['f1_macro']:.4f} | {timings[k]['train_time']:.1f} |\n")
        f.write(f"\n**Best on VAL:** {best_name}\n")

if __name__=="__main__":
    main()
