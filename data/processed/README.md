# Processed Data — Q-ORBIT

## Source
Master: `data/synthetic/lightcurves.npz` (25,000 curves, seed 123).
Standardizer fit on **train split only**; saved params reused for val/test.

## Files
- `feature_scaler.joblib` — z-score scaler (mean/scale per 21 features, train-fit).
- `preprocessing_params.json` — target length 256, standardizer mean/scale vectors.
- `train_processed.npz` — **5-sample demo artifact** (`curves` (5,256), `X` (5,21),
  `labels` (5,)), from `experiments/phase2_preprocessing_demo.py`. Not used for
  training; full training features are extracted on the fly by
  `src/classical/feature_engineering.py` and exported to
  `data/classical/classical_dataset.csv`.

## Pipeline
Raw curve → linear interpolation of dropouts → min-max [0,1] (already in NPZ)
→ 21 engineered features → z-score with train-fit scaler.
