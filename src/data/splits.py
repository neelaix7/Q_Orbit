# splits.py — frozen train/val/test split manager for Q-ORBIT fair comparison
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

CONFIG_PATH = Path("configs/qorbit_config.json")

def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)

def create_frozen_splits(synthetic_path="data/synthetic/lightcurves.npz", out_dir="data/splits", seed=42, train=0.70, val=0.15, test=0.15, force=False):
    os.makedirs(out_dir, exist_ok=True)
    train_p = Path(out_dir)/"train.npz"
    if train_p.exists() and not force:
        print(f"Splits already exist at {out_dir} — skip (use force=True to rebuild)")
        return
    data = np.load(synthetic_path, allow_pickle=True)
    curves, labels = data["curves"], data["labels"].astype(int)
    X_temp, X_test, y_temp, y_test = train_test_split(curves, labels, test_size=test, stratify=labels, random_state=seed)
    val_ratio = val / (train+val)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=seed)
    np.savez_compressed(Path(out_dir)/"train.npz", curves=X_train, labels=y_train)
    np.savez_compressed(Path(out_dir)/"val.npz", curves=X_val, labels=y_val)
    np.savez_compressed(Path(out_dir)/"test.npz", curves=X_test, labels=y_test)
    with open(Path(out_dir)/"manifest.json","w") as f:
        json.dump({"seed": seed, "train": len(X_train), "val": len(X_val), "test": len(X_test), "splits": {"train":train,"val":val,"test":test}}, f, indent=2)
    print(f"Created frozen splits: train {X_train.shape} val {X_val.shape} test {X_test.shape} seed={seed}")

def load_splits(split_dir="data/splits"):
    tr = np.load(f"{split_dir}/train.npz", allow_pickle=True)
    va = np.load(f"{split_dir}/val.npz", allow_pickle=True)
    te = np.load(f"{split_dir}/test.npz", allow_pickle=True)
    return (tr["curves"], tr["labels"].astype(int)), (va["curves"], va["labels"].astype(int)), (te["curves"], te["labels"].astype(int))

if __name__ == "__main__":
    cfg = load_config()
    create_frozen_splits(cfg["dataset"]["synthetic_path"], "data/splits", cfg["seed"], **cfg["dataset"]["splits"])
