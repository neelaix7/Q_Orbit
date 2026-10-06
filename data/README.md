# Q-ORBIT Dataset

## Overview
**Regime A — Controlled Synthetic (primary, used for all reported metrics)**

Q-ORBIT uses a physics-informed synthetic photometric light-curve dataset. The dataset is **visibly stored in the repository** and loaded directly by the experiments and the application — no hidden generation at runtime.

## Files

```
data/
├── raw/                      # raw generator outputs (original un-split archive)
│   ├── lightcurves_raw.npz   # 25k × 256 raw archive (master copy: synthetic/lightcurves.npz)
│   ├── sample_lightcurves.csv
│   └── README.md
├── processed/                # standardized features + scaler fitted on train only
│   ├── feature_scaler.joblib
│   ├── preprocessing_params.json
│   ├── train_processed.npz   # 5-sample demo artifact (not training data)
│   └── README.md
├── classical/                # real 21-feature export (SAME master, SAME splits)
│   ├── classical_dataset.csv # 25k rows × 21 features + sample_id/split/class
│   └── README.md
├── hybrid/                   # real 21+8 export (SAME master, SAME splits)
│   ├── hybrid_dataset.csv    # 25k rows × classical_21 + quantum_pca_8
│   └── README.md
├── synthetic/                # same as raw (canonical location, legacy)
│   ├── lightcurves.npz
│   ├── lightcurves_train.npz  (legacy 80/20 split, not used for fair comparison)
│   ├── lightcurves_test.npz
│   ├── overlapping_lightcurves.npz (parameter-overlap stress experiment)
│   └── test_small.npz         (100-sample quick-test)
├── splits/                   # FROZEN 70/15/15 splits used for all 3-way comparisons
│   ├── train.npz             # 17500 curves
│   ├── val.npz               # 3750 curves
│   ├── test.npz              # 3750 curves (FROZEN, never used for tuning)
│   └── manifest.json
├── quantum_ready/            # SCIENTIFIC SUBSET of master: 21D → PCA8 (train-fit only)
│   ├── quantum_ready_dataset.csv  # 25k rows × 8 PCA components + ids/splits
│   ├── train8.npz / val8.npz / test8.npz  # second-standardized 8-D + labels
│   ├── reducer8.joblib / norm8.joblib
│   ├── README.md             # human-readable justification (why 8, variance, encoding)
│   └── README.json           # machine-readable meta: 96% variance, 8-qubit ansatz match
└── README.md                 # this file
```

Canonical frozen splits are `data/splits/{train,val,test}.npz` (seed 123). The application and `experiments/run_comprehensive_fair.py` load from these files.

## Dataset Specification

| Property | Value |
|---|---|
| **Name** | Q-ORBIT Synthetic Light Curves Regime A |
| **Number of samples** | 25,000 light curves |
| **Number of classes** | 5 |
| **Class names** | 0: Intact Satellite, 1: Dead Satellite, 2: Rocket Body, 3: Fragmentation Debris, 4: Spoofed Satellite |
| **Samples per class** | 5,000 (balanced) |
| **Observations per light curve** | 256 (resampled model input). Raw sampling: 5-second cadence → 144 raw samples over 720 s window, interpolated to 256, per-curve min-max normalized to [0,1] |
| **Observation window** | 720 seconds (12 minutes) — LEO pass, `configs/qorbit_config.json: observation_window_sec` |
| **Feature count** | 21 engineered features (11 time-domain + 5 frequency + 5 temporal pattern) OR 256 raw points for CNN |
| **Train/Val/Test split** | 70/15/15 → 17500/3750/3750, stratified, seed 123, frozen |
| **File format** | NumPy compressed `.npz` with arrays `curves` shape `(N,256)` float32 and `labels` shape `(N,)` int64 |
| **Seed** | 123 (RNG for generation + splits + training) |
| **Diversity** | per-sample photon noise U[0.01,0.05], dropout U[0.03,0.10], exposure jitter σ 0.05, ~12% class-2/4 boundary overlap |

## Class Physics Parameters (from `src/simulator/config.py`)

| Class | Tumble Period (s) | Aspect Ratio | Reflectivity | n_faces |
|---|---|---|---|---|
| Intact Satellite | 60–90 | 1.2–2.5 | 0.3–0.5 | 4 |
| Dead Satellite | 150–280 | 1.5–4.0 | 0.1–0.3 | 6 |
| Rocket Body | 350–500 | 4.0–8.0 | 0.1–0.25 | 8 |
| Fragmentation Debris | 30–50 | 0.5–1.8 | 0.05–0.15 | 3 |
| Spoofed Satellite | 350–500 | 2.0–5.0 | 0.05–0.12 | 6 |

Physics: specular + diffuse (Lambertian) reflectance, sun/observer geometry, eclipse (0.27°), random spin axis, phase, inclination/RAAN uniform [0,360), per-sample photon noise U[0.01,0.05], per-sample dropout U[0.03,0.10] linearly interpolated, exposure jitter σ 0.05, ~12% class-2/4 boundary overlap.

## Train/Validation/Test Counts

| Split | Samples | Per-class | File |
|---|---|---|---|
| Train | 17,500 | 3,500 | `data/splits/train.npz` |
| Validation | 3,750 | 750 | `data/splits/val.npz` |
| Test | 3,750 | 750 | `data/splits/test.npz` |
| **Total** | **25,000** | **5,000** | `data/synthetic/lightcurves.npz` |

Validation is used for model selection (best classical on val F1, quantum/hybrid grid on val F1). Test is frozen and evaluated once.

## Feature Details (21)

**Time Domain (11):** mean, std, variance, median, skewness, kurtosis, peak_to_peak, amplitude, energy, above_median_fraction, min_value
**Frequency Domain (5):** dominant_freq, dominant_magnitude, harmonic_energy, fft_entropy, n_harmonics
**Temporal Pattern (5):** flash_count, rise_time, fall_time, eclipse_fraction, period_estimate

Standardization: z-score fitted on train only (`data/processed/feature_scaler.joblib`, `data/processed/preprocessing_params.json`).

## Quantum Reduction (scientifically justified subset)

`experiments/feature_analysis.py` (train/val only) found ESSENTIAL `period_estimate, dominant_freq, n_harmonics, fft_entropy, fall_time, std, variance, flash_count`; REDUNDANT pairs (|r|>0.9) `mean↔median/energy/skewness, std↔variance`; LOW-INFO `amplitude, eclipse_fraction, min_value, above_median_fraction, peak_to_peak`. Full report `results/reports/feature_analysis.json`.
`experiments/build_quantum_ready.py`: master 21D → z-score → PCA8 (fit train-only, `random_state=42`) → second standardization → `data/quantum_ready/*8.npz`. **Why 8:** matches 8-qubit VQC/hybrid ansatz (depth 5); PCA-8 retains 95.9% variance; 4/6/8 grid in `search_pure_quick.py`. Encoding `θ=tanh(z)·π`, `|ψ(x)⟩=⊗Ry(xᵢ)|0⟩`. Same master, same splits — no separate dataset.

## How Dataset Was Generated

```bash
py -3 src/pipeline/generate_dataset.py --n-per-class 5000 --seed 123
py -3 -m src.data.splits   # creates frozen 70/15/15 splits from synthetic archive
# or:
py -3 src/data/splits.py
```

Generator: `src/simulator/lightcurve_generator.py:generate_dataset` → `src/simulator/physics.py:compute_light_curve_from_faces` → `np.savez_compressed`.

Splits: `src/data/splits.py:create_frozen_splits` uses `sklearn.model_selection.train_test_split` stratify, seed 123.

## Reproducibility

- All RNGs seeded with 123 (`src/simulator/config.py:SEED`)
- Deterministic splits (`data/splits/manifest.json`)
- Preprocessing fit on train only, scaler saved
- Experiments log seed/config in `results/reports/*` and `experiments/*` artifacts
- To regenerate the **complete** dataset (25k × 256): run generator above (takes several minutes). The full dataset is already committed as `data/synthetic/lightcurves.npz` (~22.8 MB) and splits — no need to regenerate unless modifying physics.

## Representative Sample

A small 100-sample subset for quick unit tests: `data/synthetic/test_small.npz` (curves shape (100,256)). Overlap stress-test dataset: `data/synthetic/overlapping_lightcurves.npz` (widens distributions to create genuine difficulty, see `experiments/parameter_overlap_experiment.py`).

## Loading Example (Python)

```python
import numpy as np
data = np.load("data/splits/test.npz")
curves, labels = data["curves"], data["labels"]  # (3750,256), (3750,)
# or full archive
full = np.load("data/synthetic/lightcurves.npz")
```

## Notes

- **Regime B (External Observational):** No suitable public labeled optical light-curve dataset for space debris with class labels was available at build time. `src/pipeline/download_real_data.py` documents ESA DISCOS/NASA ODPO/CelesTrak sources (TLE/catalogs, not labeled curves). Synthetic Regime A is primary. Pipeline accepts real curves via same 21-feature extraction when available.
- Data is loaded from files — no silent mock generation. If `data/splits/test.npz` is missing, experiments fail explicitly.
