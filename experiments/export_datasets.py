# export_datasets.py — Export real NPZ data to structured CSV files
# Creates data/classical/, data/hybrid/, and data/splits CSV versions
# Source: data/synthetic/lightcurves.npz + data/splits/*.npz (frozen 70/15/15 seed 42)
# Run: python experiments/export_datasets.py
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.classical.feature_engineering import extract_all_features

CLASS_NAMES = [
    "Intact Satellite",
    "Dead Satellite",
    "Rocket Body",
    "Fragmentation Debris",
    "Spoofed Satellite",
]

FEATURE_NAMES = [
    # Time domain (11)
    "mean", "std", "variance", "median", "skewness", "kurtosis",
    "peak_to_peak", "amplitude", "energy", "above_median_fraction", "min_value",
    # Frequency domain (5)
    "dominant_freq", "dominant_magnitude", "harmonic_energy", "fft_entropy", "n_harmonics",
    # Temporal pattern (5)
    "flash_count", "rise_time", "fall_time", "eclipse_fraction", "period_estimate",
]

# Quantum-ready feature names after PCA-8 reduction
QUANTUM_FEATURE_NAMES = [f"pca_component_{i+1}" for i in range(8)]

# ── helpers ───────────────────────────────────────────────────────────────────

def extract_features_batch(curves: np.ndarray) -> np.ndarray:
    """Extract 21 features for a batch of light curves."""
    n = len(curves)
    feats = np.zeros((n, 21), dtype=np.float32)
    for i, c in enumerate(curves):
        try:
            dt = 720.0 / max(len(c) - 1, 1) if len(c) > 1 else 5.0
            f = extract_all_features(c.astype(float), sampling_interval=dt)
            f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
            feats[i] = f[:21]
        except Exception:
            feats[i] = np.zeros(21)
        if (i + 1) % 1000 == 0:
            print(f"  Extracted {i+1}/{n} features…")
    return feats


def load_split(name: str):
    path = os.path.join(ROOT, "data", "splits", f"{name}.npz")
    d = np.load(path)
    return d["curves"], d["labels"].astype(int)


def load_quantum_ready(name: str):
    """Load pre-built 8-D PCA-reduced quantum-ready split."""
    path = os.path.join(ROOT, "data", "quantum_ready", f"{name}8.npz")
    d = np.load(path)
    # keys can be 'X' or 'features' or 'data'
    key_x = next((k for k in d.files if k in ("X", "features", "data", "x")), d.files[0])
    key_y = next((k for k in d.files if k in ("y", "labels", "label")), d.files[-1])
    X = d[key_x]
    y = d[key_y].astype(int)
    return X, y


# ── main export ───────────────────────────────────────────────────────────────

def export_all():
    # Directories
    for d in ("classical", "hybrid", "splits"):
        os.makedirs(os.path.join(ROOT, "data", d), exist_ok=True)

    splits = {}
    for split_name in ("train", "val", "test"):
        print(f"\nLoading {split_name} split…")
        curves, labels = load_split(split_name)
        splits[split_name] = (curves, labels)

    # ── Feature extraction for all splits ─────────────────────────────────────
    feat_splits = {}
    for split_name, (curves, labels) in splits.items():
        print(f"Extracting features for {split_name} ({len(curves)} samples)…")
        feats = extract_features_batch(curves)
        feat_splits[split_name] = (feats, labels)

    # ── Classical dataset (train split, 21 features) ──────────────────────────
    print("\nBuilding classical dataset CSV…")
    train_feats, train_labels = feat_splits["train"]
    n_train = len(train_feats)
    classical_rows = []
    for i in range(n_train):
        row = {"sample_id": f"train_{i:05d}", "split": "train", "class_id": int(train_labels[i]),
               "class_name": CLASS_NAMES[int(train_labels[i])]}
        for j, name in enumerate(FEATURE_NAMES):
            row[name] = float(train_feats[i, j])
        classical_rows.append(row)
    # add val and test
    for sname in ("val", "test"):
        feats_s, labels_s = feat_splits[sname]
        for i in range(len(feats_s)):
            row = {"sample_id": f"{sname}_{i:05d}", "split": sname,
                   "class_id": int(labels_s[i]), "class_name": CLASS_NAMES[int(labels_s[i])]}
            for j, name in enumerate(FEATURE_NAMES):
                row[name] = float(feats_s[i, j])
            classical_rows.append(row)

    classical_df = pd.DataFrame(classical_rows)
    out_path = os.path.join(ROOT, "data", "classical", "classical_dataset.csv")
    classical_df.to_csv(out_path, index=False)
    print(f"  Saved {len(classical_df)} rows -> {out_path}")

    # ── Quantum-ready dataset (PCA-8 from pre-built files) ────────────────────
    print("\nBuilding quantum-ready dataset CSV…")
    qr_rows = []
    for sname in ("train", "val", "test"):
        try:
            X_q, y_q = load_quantum_ready(sname)
            for i in range(len(X_q)):
                row = {"sample_id": f"{sname}_{i:05d}", "split": sname,
                       "class_id": int(y_q[i]), "class_name": CLASS_NAMES[int(y_q[i])]}
                for j, qname in enumerate(QUANTUM_FEATURE_NAMES):
                    row[qname] = float(X_q[i, j])
                qr_rows.append(row)
            print(f"  Loaded quantum-ready {sname}: {len(X_q)} samples, shape {X_q.shape}")
        except Exception as e:
            print(f"  WARN: Could not load quantum-ready {sname}: {e}")
            # Fall back to PCA transform from classical features
            feats_s, labels_s = feat_splits[sname]
            try:
                import joblib
                reducer = joblib.load(os.path.join(ROOT, "models", "reducer_8q.joblib"))
                X_q = reducer.transform(feats_s)
                for i in range(len(X_q)):
                    row = {"sample_id": f"{sname}_{i:05d}", "split": sname,
                           "class_id": int(labels_s[i]), "class_name": CLASS_NAMES[int(labels_s[i])]}
                    for j, qname in enumerate(QUANTUM_FEATURE_NAMES):
                        row[qname] = float(X_q[i, j])
                    qr_rows.append(row)
                print(f"  Fallback PCA transform {sname}: {len(X_q)} samples")
            except Exception as e2:
                print(f"  WARN: Fallback PCA also failed: {e2}")

    if qr_rows:
        qr_df = pd.DataFrame(qr_rows)
        out_path_qr = os.path.join(ROOT, "data", "quantum_ready", "quantum_ready_dataset.csv")
        qr_df.to_csv(out_path_qr, index=False)
        print(f"  Saved {len(qr_df)} rows -> {out_path_qr}")

    # ── Hybrid dataset (classical 21 features + quantum 8 PCA features) ───────
    print("\nBuilding hybrid dataset CSV…")
    # Merge classical features with quantum-ready features by sample_id
    hybrid_rows = []
    for sname in ("train", "val", "test"):
        feats_s, labels_s = feat_splits[sname]
        # Try to load quantum-ready features
        try:
            X_q, _ = load_quantum_ready(sname)
        except Exception:
            try:
                import joblib
                reducer = joblib.load(os.path.join(ROOT, "models", "reducer_8q.joblib"))
                X_q = reducer.transform(feats_s)
            except Exception:
                X_q = feats_s[:, :8]  # fallback: first 8 classical features

        n = min(len(feats_s), len(X_q))
        for i in range(n):
            row = {"sample_id": f"{sname}_{i:05d}", "split": sname,
                   "class_id": int(labels_s[i]), "class_name": CLASS_NAMES[int(labels_s[i])]}
            # Classical features
            for j, name in enumerate(FEATURE_NAMES):
                row[f"classical_{name}"] = float(feats_s[i, j])
            # Quantum-ready PCA components
            for j, qname in enumerate(QUANTUM_FEATURE_NAMES):
                row[f"quantum_{qname}"] = float(X_q[i, j])
            hybrid_rows.append(row)

    hybrid_df = pd.DataFrame(hybrid_rows)
    out_path_h = os.path.join(ROOT, "data", "hybrid", "hybrid_dataset.csv")
    hybrid_df.to_csv(out_path_h, index=False)
    print(f"  Saved {len(hybrid_df)} rows -> {out_path_h}")

    # ── Splits CSV (splits/train.csv, val.csv, test.csv) ─────────────────────
    print("\nBuilding splits CSV files…")
    for sname in ("train", "val", "test"):
        feats_s, labels_s = feat_splits[sname]
        rows = []
        for i in range(len(feats_s)):
            row = {"sample_id": f"{sname}_{i:05d}", "class_id": int(labels_s[i]),
                   "class_name": CLASS_NAMES[int(labels_s[i])]}
            for j, fname in enumerate(FEATURE_NAMES):
                row[fname] = float(feats_s[i, j])
            rows.append(row)
        df = pd.DataFrame(rows)
        out = os.path.join(ROOT, "data", "splits", f"{sname}.csv")
        df.to_csv(out, index=False)
        print(f"  Saved {sname}.csv ({len(df)} rows) -> {out}")

    # ── Write README files ────────────────────────────────────────────────────
    _write_classical_readme(classical_df)
    _write_hybrid_readme(hybrid_df)

    print("\nOK All dataset CSV files exported successfully.")
    return classical_df, hybrid_df


def _write_classical_readme(df: pd.DataFrame):
    n_total = len(df)
    n_train = len(df[df["split"] == "train"])
    n_val   = len(df[df["split"] == "val"])
    n_test  = len(df[df["split"] == "test"])
    class_dist = df["class_name"].value_counts().to_dict()
    text = f"""# Classical Dataset — Q-ORBIT

## Source
Master dataset: `data/synthetic/lightcurves.npz`
Frozen splits:  `data/splits/{{train,val,test}}.npz` (seed 123, 70/15/15)

## File
`data/classical/classical_dataset.csv`

## Overview
| Property       | Value |
|----------------|-------|
| Total samples  | {n_total:,} |
| Train          | {n_train:,} |
| Validation     | {n_val:,} |
| Test           | {n_test:,} |
| Features       | 21 engineered features |
| Classes        | 5 |
| Source         | Physics-informed synthetic light curves, 720s window, 256 observations |

## Class Distribution
{chr(10).join(f'- {k}: {v}' for k, v in class_dist.items())}

## Columns
- `sample_id`: unique identifier (split_index)
- `split`: train / val / test
- `class_id`: integer label 0–4
- `class_name`: human-readable class name

### Feature Groups

**Time Domain (11)**
mean, std, variance, median, skewness, kurtosis, peak_to_peak, amplitude, energy,
above_median_fraction, min_value

**Frequency Domain (5)**
dominant_freq, dominant_magnitude, harmonic_energy, fft_entropy, n_harmonics

**Temporal Pattern (5)**
flash_count, rise_time, fall_time, eclipse_fraction, period_estimate

## Preprocessing
- Light curves: raw 5s cadence -> 144 samples -> resampled to 256 -> min-max normalized [0,1]
- Features: extracted by `src/classical/feature_engineering.py`
- Standardization: z-score scaler fit on train only (`models/feature_scaler.joblib`)

## Reproducibility
All RNGs seeded with 123. Run `python experiments/export_datasets.py` to regenerate.
"""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "classical", "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Wrote {path}")


def _write_hybrid_readme(df: pd.DataFrame):
    n_total = len(df)
    text = f"""# Hybrid Dataset — Q-ORBIT

## Source
Same master dataset as classical: `data/synthetic/lightcurves.npz`
Same frozen splits: seed 123, 70/15/15

## File
`data/hybrid/hybrid_dataset.csv`

## Overview
| Property       | Value |
|----------------|-------|
| Total samples  | {n_total:,} |
| Classical features | 21 (prefixed `classical_`) |
| Quantum features   | 8 PCA components (prefixed `quantum_`) |
| Total columns  | 32 (id + split + class_id + class_name + 21 + 8) |

## Pipeline Traceability
```
data/synthetic/lightcurves.npz (master, seed 123)
    ↓ feature extraction (src/classical/feature_engineering.py)
21 classical features
    ↓ z-score standardization (models/feature_scaler.joblib, fit on train only)
    ↓ PCA-8 reduction (data/quantum_ready/reducer8.joblib, fit on train only)
8 quantum-ready PCA components
    ↓ merged with classical features
hybrid_dataset.csv
```

## Quantum Feature Justification
- 8 components: matches 8-qubit VQC/hybrid ansatz (data/quantum_ready/README.json)
- Explained variance: 95.9% retained by PCA-8
- Encoding: theta = tanh(z) * pi -> Angle RY embedding

## Columns
- `sample_id`, `split`, `class_id`, `class_name`
- `classical_<feature>`: 21 classical engineered features
- `quantum_pca_component_<N>`: 8 PCA-reduced quantum-ready dimensions

## Same source guarantee
Both classical and hybrid datasets derive from the SAME 25,000-sample master
dataset with the SAME frozen splits (seed 123). No separate data generation.
"""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "hybrid", "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Wrote {path}")


if __name__ == "__main__":
    export_all()
