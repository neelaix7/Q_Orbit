# Q-ORBIT — Quantum-Classical AI for Space Object Classification

> **Research question:** Can combining classical AI with quantum machine learning provide a measurable classification or robustness advantage over pure classical and pure quantum approaches for challenging photometric light curves — and under what observation conditions does any advantage appear?

Q-ORBIT is a research framework for experimentally comparing classical, quantum, and hybrid learning on the **same data, same task, same 70/15/15 split (seed 123), same 3,750 test samples**. It does not assume hybrid wins — it measures.

---

## Quick Start

```powershell
# 1 — Install dependencies
pip install -r requirements.txt

# 2 — Generate dataset (already committed; only needed if modifying physics)
python src/pipeline/generate_dataset.py --n-per-class 5000 --seed 123

# 3 — Run experiments (results already committed in results/reports/)
python experiments/run_comprehensive_fair.py

# 4 — Launch frontend
streamlit run app/app.py
```

The app opens at `http://localhost:8501`. Use the sidebar to navigate between pages.

---

## Architecture Overview

```
Light Curve (256 obs, 720s window)
        │
        ▼
  Preprocessing
  (interpolation + min-max normalisation)
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
Feature Extraction                       Raw Curve
(21 engineered features)                (256 points)
        │                                      │
        ├─────────┬─────────┐                  │
        ▼         ▼         ▼                  ▼
       SVM        RF      XGBoost            1D-CNN
        │         │         │                  │
        └─────────┴─────────┴──────────────────┘
                            │
                     Classical Baseline
                            │
        ┌───────────────────┘
        │
        ▼
  z-score Scaler (train-fit) → PCA-8 (train-fit)
        │
        ▼
  Quantum Layer (8 qubits, VQC)
  AngleEmbedding(RY) → Rot+CNOT×2 → ⟨Z⟩×8
        │
        ├─────────────────────────────────────┐
        ▼                                     ▼
  Pure VQC                             Hybrid Model
  (94 params)                    (quantum layer + MLP head)
                                         (1,798 params)
                                              │
                                              ▼
                                     Adaptive Fusion
                                   (signal-quality gate)
```

---

## Dataset

### Regime A — Controlled Synthetic (primary)

| Property              | Value                                      |
|-----------------------|--------------------------------------------|
| Samples               | 25,000 light curves                        |
| Classes               | 5 (balanced, 5,000 per class)              |
| Observations          | 256 per curve (resampled from 144 raw)     |
| Window                | 720 s (LEO pass)                           |
| Split                 | 70/15/15 · seed 123 · frozen               |
| Train / Val / Test    | 17,500 / 3,750 / 3,750                     |
| Features              | 21 engineered (time-domain + freq + pattern) |
| Quantum reduction     | PCA-8 (95.9% variance retained)            |
| Seed                  | 123 (generation + splits + training)       |
| Diversity             | per-sample noise U[0.01,0.05], dropout U[0.03,0.10], exposure jitter σ 0.05, ~12% class-2/4 boundary overlap |

### Class Definitions

| ID | Name                  | Tumble Period | Reflectivity |
|----|-----------------------|---------------|--------------|
| 0  | Intact Satellite      | 60–90 s       | 0.30–0.50    |
| 1  | Dead Satellite        | 150–280 s     | 0.10–0.30    |
| 2  | Rocket Body           | 350–500 s     | 0.10–0.25    |
| 3  | Fragmentation Debris  | 30–50 s       | 0.05–0.15    |
| 4  | Spoofed Satellite     | 350–500 s     | 0.05–0.12    |

### Dataset Files

```
data/
├── synthetic/
│   └── lightcurves.npz            # master (25k × 256), seed 123
├── splits/
│   ├── train.npz / val.npz / test.npz   # frozen 70/15/15 NPZ
│   ├── train.csv / val.csv / test.csv   # same splits, 21 features
│   └── manifest.json
├── classical/
│   ├── classical_dataset.csv      # 25k rows × 21 features (real)
│   └── README.md
├── hybrid/
│   ├── hybrid_dataset.csv         # 25k rows × (21 + 8) features
│   └── README.md
├── quantum_ready/
│   ├── quantum_ready_dataset.csv  # 25k rows × 8 PCA components
│   ├── train8.npz / val8.npz / test8.npz
│   ├── reducer8.joblib / norm8.joblib
│   └── README.json
├── raw/
│   └── sample_lightcurves.csv
└── README.md
```

All CSV files are real exports from the NPZ splits. Generate with:
```powershell
python experiments/export_datasets.py
```

---

## Classical Pipeline

**Models:** SVM (RBF), Random Forest (300 trees), XGBoost (300 est.), 1D-CNN

**Best (val-selected):** 1D-CNN — 87.76% test accuracy, F1 0.8750

| Model   | Test Acc | Test F1 | Params  | Input        |
|---------|----------|---------|---------|--------------|
| CNN     | 87.76%   | 0.8750  | ~280k   | 256 raw obs  |
| SVM     | 74.00%   | 0.7393  | ~1k     | 21 features  |
| RF      | 79.44%   | 0.7943  | ~5k     | 21 features  |
| XGB     | 80.51%   | 0.8052  | ~5k     | 21 features  |

Source: `results/reports/fair_comparison.json` (fresh clean regime, seed 123, 3,750 test)

---

## Quantum Pipeline

**Architecture:** PureVQC — PCA-8 input → AngleEmbedding(RY) → Rot+CNOT ansatz × 3 layers → ⟨Z⟩ × 8 → Linear(8→5)

| Property       | Value                     |
|----------------|---------------------------|
| Qubits         | 8                         |
| Layers         | 3                         |
| Circuit depth  | 7                         |
| Parameters     | 118 (circuit + readout)   |
| Encoding       | θ = tanh(z)·π → RY        |
| Test accuracy  | 62.08%                    |
| Test F1-macro  | 0.6197                    |
| Simulator      | PennyLane default.qubit   |

Continued training: 55.63% → **62.08%** (+6.45pp). Source: `results/reports/qml_continued_training.json`.

**Why 8 qubits:** Matches the hybrid ansatz; PCA-8 retains 95.9% of feature variance. Ablations over 4/6/8 qubits × 1/2/3 layers in `experiments/search_pure_quick.py`.

---

## Hybrid Pipeline

**Architecture:** 21 features → Linear(21→8) + tanh·π → AngleEmbedding(RY) → Rot+CNOT × 2 → ⟨Z⟩ × 8 → MLP head (32→32→5)

| Property       | Value      |
|----------------|------------|
| Total params   | 1,798      |
| Qubits         | 8          |
| Circuit depth  | 5          |
| Test accuracy  | 75.41%     |
| Test F1-macro  | 0.7540     |

Continued training: 72.96% → **75.41%** (+2.45pp, best epoch 28, val-selected).
Best quantum result — 156× fewer params than the CNN.

**Adaptive Fusion (ARCHIVED old regime):** `results/reports/adaptive_fusion.json` (77.01%) is from the previous diverse regime — kept for provenance, not head-to-head with fresh scores.

Source: `results/reports/adaptive_fusion.json`

---

## Research Conclusion (dynamic, from measured results)

| Comparison              | Result                                          |
|-------------------------|-------------------------------------------------|
| Hybrid vs Pure Quantum  | +13.4 pp F1 (hybrid clearly helps)             |
| Hybrid vs Best Classical| −12.1 pp F1 (CNN remains strongest on clean)   |
| Hybrid vs SVM           | +1.5 pp accuracy (beats SVM, close to RF/XGB)  |

**Honest finding:** Hybrid advantage is not demonstrated on clean data vs the best classical CNN (87.76%). Hybrid is the clear best-quantum result (75.41% vs 62.08% pure, +13.4pp F1) at 1,798 params. Check robustness results for stability comparisons.

---

## Robustness Experiments

Training weights frozen. Test-time degradation only.

| Condition             | Levels             |
|-----------------------|--------------------|
| Noise (std)           | 0%, 10%, 20%, 30%  |
| Observation fraction  | 100%, 75%, 50%, 25%|
| Missing data          | 0%, 5%, 10%, 20%   |

Results in `results/reports/robustness_*.json`. Robustness score formula:
```
RS = mean(acc_degraded / acc_clean) across all conditions
```
Source: `src/evaluation/robustness_score.py`

---

## Frontend Pages

| Page          | File                        | Content                                           |
|---------------|-----------------------------|---------------------------------------------------|
| Home          | `app/app.py`                | Hero, workflow, results summary, inference lab    |
| Data          | `app/pages/2_Data.py`       | Dataset explorer, class dist, feature analysis    |
| Classical AI  | `app/pages/3_Classical_AI.py`| Pipelines, 4 model cards, confusion matrices     |
| Quantum       | `app/pages/4_Quantum.py`    | VQC spec, circuit diagram, Bloch sphere, ablations|
| Hybrid        | `app/pages/5_Hybrid.py`     | Architecture, comparison, adaptive fusion         |
| Experiments   | `app/pages/6_Experiments.py`| Full 3-way table, radar, robustness, ablations    |
| Quantum Lab   | `app/pages/1_Quantum_Lab.py`| Interactive gates, Bloch sphere, Aer simulator    |

Shared CSS/theme: `app/space_theme.py`

---

## Repository Structure

```
Q-ORBIT/
├── app/
│   ├── app.py                  # Home page (Streamlit entrypoint)
│   ├── space_theme.py          # Shared CSS + theme helpers
│   ├── light_theme.py          # Legacy (unused)
│   └── pages/
│       ├── 1_Quantum_Lab.py
│       ├── 2_Data.py
│       ├── 3_Classical_AI.py
│       ├── 4_Quantum.py
│       ├── 5_Hybrid.py
│       └── 6_Experiments.py
├── src/
│   ├── simulator/              # Physics light-curve generator
│   ├── classical/              # CNN, SVM, RF, XGB, feature engineering
│   ├── quantum/                # VQC, hybrid model, Bloch, live circuit
│   ├── features/               # PCA reducer
│   ├── fusion/                 # Adaptive fusion, signal quality
│   ├── evaluation/             # Metrics, degradation, robustness score
│   ├── preprocessing/          # Pipeline
│   ├── agents/                 # AI analyst
│   └── pipeline/               # Training scripts
├── experiments/
│   ├── export_datasets.py      # Generate CSV files from NPZ
│   ├── run_comprehensive_fair.py
│   ├── search_classical.py
│   ├── search_hybrid_quick.py
│   ├── search_pure_quick.py
│   └── ...
├── data/                       # See Dataset section above
├── models/                     # Trained model checkpoints
│   ├── hybrid_quantum_model.pt
│   ├── classical_cnn_model.pt
│   ├── classical_svm_model.joblib
│   ├── classical_rf_model.joblib
│   ├── classical_xgb_model.joblib
│   ├── pure_quantum_vqc.pt
│   └── feature_scaler.joblib
├── results/
│   ├── reports/                # JSON/MD experiment outputs
│   └── figures/
├── configs/
│   └── qorbit_config.json      # 720s window, seed, robustness combos
├── tests/                      # Unit tests
├── notebooks/
├── requirements.txt
└── README.md
```

---

## Technology Stack

| Component          | Technology                          |
|--------------------|-------------------------------------|
| Frontend           | Streamlit ≥1.40, Plotly ≥5.0        |
| Quantum simulation | PennyLane ≥0.40 (default.qubit)     |
| Quantum circuits   | Qiskit ≥1.0, Qiskit-Aer ≥0.15      |
| Classical ML       | scikit-learn, XGBoost               |
| Deep learning      | PyTorch ≥2.2                        |
| Data               | NumPy, pandas, SciPy                |
| Visualisation      | Plotly, Matplotlib                  |
| Serialisation      | joblib                              |

---

## How to Run Experiments

```powershell
# Classical baseline search
python experiments/search_classical.py

# Pure quantum grid search
python experiments/search_pure_quick.py

# Hybrid grid search
python experiments/search_hybrid_quick.py

# Full fair 3-way comparison
python experiments/run_comprehensive_fair.py

# Adaptive fusion training
python experiments/train_adaptive_real.py

# Ablation study
python experiments/run_ablations.py

# Export dataset CSVs
python experiments/export_datasets.py
```

---

## Reproducibility

- All RNGs seeded with 123 (`configs/qorbit_config.json`, `src/simulator/config.py`)
- Frozen 70/15/15 split (`data/splits/manifest.json`)
- Preprocessing scaler fit on train only; saved to `models/feature_scaler.joblib`
- Test set evaluated once, never used for tuning
- Experiment configs and seeds logged to `results/reports/`
- Full dataset (25k × 256) committed as `data/synthetic/lightcurves.npz`

---

## Known Limitations

1. **Synthetic data only.** No publicly labelled optical light-curve dataset for space debris was available at build time. Results apply to the simulated Regime A; real-world performance is unknown.
2. **Pure quantum below classical.** VQC (62.1%) cannot match CNN (87.8%) on this dataset — expected given 118 vs 280k parameters and lower information in PCA-8 vs raw 256-point input.
3. **Hybrid below best classical on clean data.** Hybrid (75.4%) < CNN (87.8%) on noise-free test. Hybrid beats SVM (74.0%) and is close to RF/XGB. Robustness advantage may exist under degraded conditions.
4. **Simulator only.** No real QPU results. All quantum computations use PennyLane `default.qubit` or Qiskit `AerSimulator`.
5. **Class overlap.** Rocket Body (class 2) and Spoofed Satellite (class 4) have overlapping tumble periods — hardest classes for all models.

---

## Positioning

Q-ORBIT is a **research framework** for experimentally comparing classical, quantum, and hybrid learning for photometric light-curve classification. It does not claim operational deployment, guaranteed quantum advantage, or NASA/ISRO endorsement. Results are honest, reproducible, and drawn directly from measured experiments.

---

*All metrics on frozen 70/15/15 seed 123 · 3,750 test samples · PennyLane default.qubit · offline · no paid APIs · reproducible*
