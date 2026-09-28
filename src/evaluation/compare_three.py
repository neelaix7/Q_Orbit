# compare_three.py — unified fair 3-way comparison using frozen splits
from __future__ import annotations
import os, json, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

def compute_all_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4]).tolist()
    out = {"accuracy": float(acc), "precision_macro": float(prec_m), "recall_macro": float(rec_m), "f1_macro": float(f1_m), "precision_weighted": float(prec_w), "recall_weighted": float(rec_w), "f1_weighted": float(f1_w), "confusion_matrix": cm}
    if y_proba is not None:
        try:
            y_bin = label_binarize(y_true, classes=[0,1,2,3,4])
            out["roc_auc_ovr_macro"] = float(roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr"))
        except Exception:
            out["roc_auc_ovr_macro"] = None
    # per class
    prec_c, rec_c, f1_c, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2,3,4], zero_division=0)
    out["per_class"] = {i: {"precision": float(prec_c[i]), "recall": float(rec_c[i]), "f1": float(f1_c[i]), "support": int(sup[i])} for i in range(5)}
    return out

def eval_torch_classifier(model, X, y, batch=256):
    model.eval()
    preds, probas = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i+batch], dtype=torch.float32)
            # handle CNN vs feature models
            if hasattr(model, "forward"):
                try:
                    out = model(xb)
                    if isinstance(out, tuple):
                        logits, proba = out
                    else:
                        logits = out
                        proba = torch.softmax(logits, dim=1)
                    probas.append(proba.cpu().numpy())
                    preds.append(torch.argmax(proba, dim=1).cpu().numpy())
                except Exception as e:
                    raise e
    y_pred = np.concatenate(preds)
    y_proba = np.concatenate(probas) if probas else None
    return y_pred, y_proba
