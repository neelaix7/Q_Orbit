# Quantum-Ready Dataset — Q-ORBIT

## Source
Same master as classical: `data/synthetic/lightcurves.npz`, same frozen
70/15/15 splits (seed 123). Built by `experiments/build_quantum_ready.py`.

## Files
- `quantum_ready_dataset.csv` — 25,000 rows × (`sample_id`, `split`,
  `class_id`, `class_name`, `pca_component_1..8`).
- `train8.npz` / `val8.npz` / `test8.npz` — second-standardized 8-D splits + labels.
- `reducer8.joblib` / `norm8.joblib` — PCA + scaler pipeline, fit on train only.
- `README.json` — machine-readable meta (see below).

## Reduction justification
21 features → z-score → PCA-8 (train-fit, `random_state=42`) → second
standardization. PCA-8 retains **95.9%** of feature variance (see `README.json`
`explained_variance_ratio`). 8 dimensions chosen to match the 8-qubit
VQC/hybrid ansatz (depth 5); 4/6/8 ablations in
`experiments/search_pure_quick.py`. Redundant pairs (|r|>0.9, e.g.
mean↔median/energy) and low-info features (amplitude, eclipse_fraction, …)
are compressed away by PCA rather than removed arbitrarily — full analysis in
`results/reports/feature_analysis.json`.

## Quantum encoding
`θ = tanh(z)·π`, state `|ψ(x)⟩ = ⊗ᵢ RY(θᵢ)|0⟩` (AngleEmbedding), measured as
`⟨Z⟩×8`. Same master, same splits — not a separate dataset.
