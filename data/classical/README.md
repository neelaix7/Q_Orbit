# Classical Dataset — Q-ORBIT

## Source
Master dataset: `data/synthetic/lightcurves.npz`
Frozen splits:  `data/splits/{train,val,test}.npz` (seed 123, 70/15/15)

## File
`data/classical/classical_dataset.csv`

## Overview
| Property       | Value |
|----------------|-------|
| Total samples  | 25,000 |
| Train          | 17,499 |
| Validation     | 3,751 |
| Test           | 3,750 |
| Features       | 21 engineered features |
| Classes        | 5 |
| Source         | Physics-informed synthetic light curves, 720s window, 256 observations |

## Class Distribution
- Fragmentation Debris: 5000
- Intact Satellite: 5000
- Spoofed Satellite: 5000
- Rocket Body: 5000
- Dead Satellite: 5000

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
