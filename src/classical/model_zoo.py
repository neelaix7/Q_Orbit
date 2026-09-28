# model_zoo.py — unified classical baselines: SVM / RF / XGBoost / CNN with same splits & auto-selection on validation
from __future__ import annotations
import os, json, time
from typing import Dict, Any, Tuple
import numpy as np
import joblib, torch, torch.nn as nn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.classical.cnn_baseline import LightCurveCNN
from src.classical.feature_engineering import extract_all_features
from src.evaluation.compare_three import compute_all_metrics

CLASS_NAMES = ["Intact Satellite","Dead Satellite","Rocket Body","Fragmentation Debris","Spoofed Satellite"]

def train_svm(X_train, y_train, X_val=None, y_val=None):
    t0=time.time()
    clf=SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    clf.fit(X_train, y_train)
    train_time=time.time()-t0
    # inference time on val
    t1=time.time()
    proba_val=clf.predict_proba(X_val) if X_val is not None else None
    infer_time=(time.time()-t1)/len(X_val) if X_val is not None else 0
    return clf, {"train_time":train_time,"infer_time_per_sample":infer_time,"params": int(X_train.shape[1]*len(np.unique(y_train)))}

def train_rf(X_train, y_train, X_val=None, y_val=None):
    t0=time.time()
    clf=RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    train_time=time.time()-t0
    t1=time.time()
    if X_val is not None: clf.predict_proba(X_val)
    infer_time=(time.time()-t1)/len(X_val) if X_val is not None else 0
    # params approx: trees * nodes
    return clf, {"train_time":train_time,"infer_time_per_sample":infer_time,"params": int(clf.n_estimators*50)}

def train_xgb(X_train, y_train, X_val=None, y_val=None):
    t0=time.time()
    clf=XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss", n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    train_time=time.time()-t0
    t1=time.time()
    if X_val is not None: clf.predict_proba(X_val)
    infer_time=(time.time()-t1)/len(X_val) if X_val is not None else 0
    return clf, {"train_time":train_time,"infer_time_per_sample":infer_time,"params": int(300*6*10)}

def train_cnn_on_curves(curves_train, y_train, curves_val, y_val, epochs=20, batch_size=32, lr=0.001, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model=LightCurveCNN(n_classes=5, initial_channels=16, dropout=0.3)
    opt=torch.optim.Adam(model.parameters(), lr=lr)
    crit=nn.CrossEntropyLoss()
    Xtr=torch.tensor(curves_train,dtype=torch.float32).unsqueeze(1)
    ytr=torch.tensor(y_train,dtype=torch.long)
    Xva=torch.tensor(curves_val,dtype=torch.float32).unsqueeze(1)
    yva=torch.tensor(y_val,dtype=torch.long)
    t0=time.time()
    best_val=0; best_state=None
    n=len(Xtr)
    for ep in range(epochs):
        model.train()
        idx=torch.randperm(n)
        for s in range(0,n,batch_size):
            bi=idx[s:s+batch_size]
            opt.zero_grad()
            logits=model(Xtr[bi])
            loss=crit(logits, ytr[bi])
            loss.backward(); opt.step()
        # val check
        model.eval()
        with torch.no_grad():
            logits=model(Xva)
            _,pred=torch.max(logits,1)
            acc=float((pred==yva).sum().item()/len(yva))
            if acc>best_val:
                best_val=acc; best_state={k:v.cpu() for k,v in model.state_dict().items()}
    train_time=time.time()-t0
    if best_state: model.load_state_dict(best_state)
    # infer time
    model.eval()
    t1=time.time()
    with torch.no_grad():
        _=model(Xva)
    infer=(time.time()-t1)/len(Xva)
    param_count=sum(p.numel() for p in model.parameters())
    return model, {"train_time":train_time,"infer_time_per_sample":infer,"params":param_count,"val_acc_best":best_val}

def evaluate_sklearn(clf, X, y):
    proba=clf.predict_proba(X) if hasattr(clf,"predict_proba") else np.eye(5)[clf.predict(X)]
    pred=np.argmax(proba,axis=1)
    metrics=compute_all_metrics(y,pred,proba)
    return pred, proba, metrics

def evaluate_cnn(model, curves, y):
    model.eval()
    with torch.no_grad():
        X=torch.tensor(curves,dtype=torch.float32).unsqueeze(1)
        logits=model(X)
        proba=torch.softmax(logits,dim=1).numpy()
        pred=np.argmax(proba,axis=1)
    metrics=compute_all_metrics(y,pred,proba)
    return pred, proba, metrics
