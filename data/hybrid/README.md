# Hybrid Dataset — Q-ORBIT

## Source
Same master dataset as classical: `data/synthetic/lightcurves.npz`
Same frozen splits: seed 123, 70/15/15

## File
`data/hybrid/hybrid_dataset.csv`

## Overview
| Property       | Value |
|----------------|-------|
| Total samples  | 25,000 |
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
