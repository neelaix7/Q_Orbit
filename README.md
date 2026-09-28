# Q-ORBIT — Adaptive Physics-Informed Quantum-Classical AI Framework for Space Object Classification

**Title:** Adaptive Physics-Informed Quantum-Classical AI Framework for Space Object Classification from Photometric Light Curves

**Core Research Question:** *Can combining classical AI with quantum machine learning provide a measurable classification or robustness advantage over pure classical and pure quantum approaches for challenging photometric light curves — and under what observation conditions does any advantage appear?*

> Q-ORBIT is a **research framework** for experimentally comparing classical, quantum, and hybrid learning on the **same data, same task, same 70/15/15 split (seed 42), same 1500 test samples**. It does not assume hybrid wins — it measures. See `reports/project_audit.md` and `results/reports/fair_comparison.md` (dynamic conclusion).

This research is motivated by challenges in space situational awareness. It does not claim operational NASA/ISRO deployment, guaranteed quantum advantage, or exact collision prediction.

---

## Dataset — Controlled Synthetic, Honestly Positioned

**Regime A — Controlled Synthetic (primary, used for all reported metrics):**
- 10,000 light curves, 5 classes ×2000, seed 42, `data/synthetic/lightcurves.npz` (10000,256)
- **Observation window: 720-second window represented by 256 resampled observations** (`configs/qorbit_config.json: observation_window_sec 720`). Raw sampling 5-second cadence → 144 raw samples (`src/simulator/config.py:11`) interpolated to 256 model inputs, per-curve min-max normalized to [0,1]. Every document/UI/report now uses this definition; RAW (144) vs RESAMPLED (256) explicitly distinguished.
- Physics: specular + diffuse (Lambertian) reflectance, sun/observer geometry, eclipse (0.27°), tumble periods/aspect/reflectivity per class (`config.py:21-45`), `n_faces` 3–8, photon noise 0.02, 5% dropout interpolated.

**Regime B — External Observational (honest limitation):**
- External observational validation is limited by the availability of publicly labeled data. No suitable public optical light-curve dataset for space debris with class labels was available at build time (see `src/pipeline/download_real_data.py` — ESA DISCOS/NASA ODPO/CelesTrak provide TLE/catalogs, not labeled light curves). Synthetic Regime A is primary. Pipeline accepts real curves via same 21-feature extraction when available.

**Parameter Overlap Experiment (easy vs challenging):**
- Baseline simulator has partially overlapping periods (e.g., Rocket 350–500 vs Frag 30–50 → trivial) — audit flags this. `experiments/parameter_overlap_experiment.py` generates an **overlapping regime** where tumble/reflectivity/aspect distributions overlap significantly (e.g., shared 80–200s band), demonstrating genuine difficulty. Report: *Class separability under controlled physical overlap.*

## Physics Simulator — Overlapping Variation
Retains specular/Lambertian, eclipse, tumbling, aspect, reflectivity, noise, missing. Improves by ensuring **controlled within-class variation, overlapping between classes** — no single parameter uniquely determines class. Each class samples: tumble period uniform within range, aspect, reflectivity, random spin axis, phase, inclination/RAAN. For overlap experiment, distributions widen/overlap to prevent period memorization. See `src/simulator/config.py` and `lightcurve_generator.py:41`.

## Feature Engineering — 21 Baseline, Physics-Informed
**Baseline (retained):** 21 dims `src/classical/feature_engineering.py:1`

**Physics-Informed Signal Features** (`src/features/physics_informed.py`) — same 21 organized, clearly labeled:
- **TIME DOMAIN (8):** mean, variance, std, amplitude, peak-to-peak, skew, kurtosis, energy — statistical brightness proxies
- **FREQUENCY DOMAIN (4):** dominant frequency, harmonic energy, spectral entropy, harmonic count (FFT, `sampling_interval=720/255≈2.82s`)
- **TEMPORAL/PHYSICAL PROXIES (9):** median, above_median_fraction, min, n_harmonics, flash frequency, rise/fall behavior, period estimate, eclipse fraction, flash count — signal proxies (labeled as proxies, not direct physical measurements)

No claim of direct physical measurement unless proven.

## Classical Baselines — Fair, Capacity-Reported
Separated:
- **Classical Feature Baseline** (21 features → SVM/RF/XGBoost) `src/classical/model_zoo.py:1`
- **Classical Deep Baseline** (256 raw → 1D-CNN) `src/classical/cnn_baseline.py:22`
- Auto-selected best on **validation only** `experiments/search_classical.py:1` → CNN best (Val F1 0.8657, Test 85.20% 280k params) vs XGB/RF/SVM. Capacity table (params, input dim, epochs) reported alongside metrics to avoid huge-CNN-vs-tiny-quantum unfair claim.

## Pure Quantum — Controlled Ablations
Primary: PCA→8 dims → tanh·π → Angle RY → Rot(3)×layer + chain CNOT ×2 → ⟨Z⟩×8 → Linear 8→5 `src/quantum/vqc.py:11` (94 params, depth 5, default.qubit backprop). Ablations `experiments/search_pure_quick.py:1` compare **qubits 4/6/8 × depth 1/2/3 × encoding Angle/ZZ** on subset, select reasonable config (8q2L val-best). Results: 8q2L Test 48.07% — genuine quantum representation, no excessive circuit.

## Hybrid — A/B + C/D Frontier, Val-Selected
- **Hybrid A:** Features → PCA → Quantum map → Classical classifier
- **Hybrid B:** Features → trainable quantum layer → MLP (production `HybridQuantumClassifier` 21→8q + head 758 params)
- **Hybrid C (feasible frontier):** 1D-CNN extractor (frozen early layers) → dim reduction → quantum → classifier (`src/quantum/hybrid_variants.py`)
- **Hybrid D (feasible frontier):** dual-branch (classical Feats + quantum PCA) → fusion → classifier

Grid `experiments/search_hybrid_quick.py:1` (5 configs q4l2h16..q8l2h32, 5 epochs, subset) → production B (8q2L16) val-selected. If Hybrid below CNN, not hidden — investigated via ablations/fusion.

## Adaptive Fusion Engine — Novel Contribution (`src/fusion/`)
**Q-ORBIT Adaptive Fusion Engine:** Different curves have different signal quality — learns how much to trust each representation.
- **Signal Quality Analyzer** `src/fusion/signal_quality.py`: from light curve → noise estimate (residual std), signal std, missing %, observation coverage, periodicity strength (FFT peak ratio), spectral complexity (entropy), completeness.
- **Adaptive Gate:** classical rep (21D or CNN embedding) + quantum rep (8D ⟨Z⟩) + quality score → learned gating network (simple: concat→Linear→sigmoid) outputs weight α(classical)∈[0,1], fusion = α·classical + (1-α)·quantum or fused prob = α·p_classical + (1-α)·p_quantum, trained on **train/val only**, frozen test. Start simple gating, extensible to attention.

Conceptual: `Light Curve → Signal Quality → Classical Rep + Quantum Rep → Adaptive Gate → Fusion → Prediction` with dynamic display *Classical 63% / Quantum 37%* from actual model.

## Model Disagreement & Uncertainty
- **Agreement Engine** `src/evaluation/agreement.py`: computes prediction agreement (consensus 3/3 vs 2/3 vs 0/3), probability divergence (KL/JS), entropy, margin (top1-top2).
- **Uncertainty** `src/evaluation/uncertainty.py`: prediction + probability + **uncertainty** (entropy→Low/Moderate/High, margin) + signal quality + agreement (e.g., *Prediction Rocket 81%, Uncertainty Moderate, Agreement 2/3, Signal Low*), not just max prob.

## Robustness — Core Experiment (`src/evaluation/degradation.py:1`)
Training frozen. Only test degraded via same pipeline:
- Noise 0/5/10/20/30% (`0,0.05,0.10,0.20,0.30`)
- Observation 100/75/50/25% (truncate + last-value pad)
- Missing 0/5/10/20% (random dropout + interp)
- **Combinations:** Noise+Missing, Noise+Short observation (new)
Results auto: `robustness_*.json` + `results/figures/robustness_*.png`.

**Robustness Score** `src/evaluation/robustness_score.py`: `RS = mean( normalized accuracy across all degradation conditions / clean accuracy )` ∈[0,1]. Documented formula, shown alongside Accuracy/F1 to determine where hybrid is more robust even if not more accurate.

## Ablations (7) `experiments/run_ablations.py`
1 Classical only, 2 Quantum only, 3 Classical+Quantum (hybrid static), 4 Hybrid without adaptive fusion, 5 Hybrid with adaptive fusion, 6 Hybrid without physics-aware features, 7 Hybrid with physics-aware. Answers *which component contributes*.

## Hyperparameter Search — Validation Only
`experiments/hyperparam_search.py` searches SVM C/gamma, XGB depth/lr, quantum qubits/layers/lr, hybrid qubits/layers/head/fusion dims via val F1. Test frozen.

## Statistical Validation
Seed 42 reproducibility + multi-seed where feasible: 5 runs seeds 42-46 `experiments/statistical_validation.py` → Mean±Std, paired t-test where justified, reported in `fair_comparison.md`. Never claim significance without test.

## Final Experimental Matrix (auto)
`results/reports/experimental_matrix.csv` — rows Clean/5%/10%/20%/30% noise × 100/75/50/25% obs × 0/5/10/20% missing, cols Classical / Quantum / Hybrid / Adaptive Hybrid → accuracy. Generated by `run_comprehensive_fair.py` + `experimental_matrix.py`.

## Research Conclusion Logic (dynamic, 4-way)
`run_comprehensive_fair.py` distinguishes:
- Accuracy advantage
- F1 advantage
- Robustness advantage
- Computational trade-off
→ *Hybrid advantage on clean* / *under degraded* / *robustness only* / *no measurable* — based on measured test, never fabricated (current: NOT DEMONSTRATED on clean, +26.7pp over pure, check robustness).

## Honesty & Scope Cleanup
Removed/rewritten: guaranteed quantum advantage, perfect performance, NASA deployment claims → *This research is motivated by space situational awareness*. Removed blockchain/Web3, collision prediction, hardware dependence. Focus kept: Light Curves + Classical+Quantum+Hybrid+Robustness+Uncertainty+AI interpretation.

## AI Analyst — Traceable (`src/agents/analyst.py`)
Receives structured JSON `{classical_f1, quantum_f1, hybrid_f1, adaptive_f1, signal_quality, agreement, uncertainty, robustness}` → templated language. LLM never invents numbers. Traceable `analyst/trace.json`.

## Dashboard — Scientific Readability First (`app/app.py`)
Retains black-hole/nebula/satellite/glass but prioritizes readability. Sections: Hero (Q-ORBIT + statement + CLASSICAL vs QUANTUM vs HYBRID + Can they work better together?), Workflow viz (SAME DATA→...), 4 comparison cards (Pure Classical / Pure Quantum / Hybrid / **Adaptive Hybrid**), Research Insight (What Did Experiment Show? Best Acc/F1/Robustness/Efficiency → Hybrid Advantage), Signal Quality panel (noise/completeness/periodicity), Model Agreement viz, Quantum Lab (qubits/depth/feature map/gates/simulator), Adaptive Fusion viz (classical+quantum→gate→fusion→prediction with dynamic weights), interactive charts (bar, radar, robustness curves, confusion, agreement matrix, hybrid improvement %).

## Research Report — Auto 20 Sections
`reports/research_report.md` auto-generated from configs+results (abstract→future work), pulls actual numbers, never hard-coded. Experiment tracking `experiments/{classical,quantum,hybrid,adaptive,robustness,ablations}/` saves config/seed/metrics/time/params/checkpoint JSON/CSV.

## Testing (`tests/`)
`test_dataset.py`, `test_features.py`, `test_quantum.py`, `test_fusion.py`, `test_metrics.py`, `test_degradation.py`, `test_analyst.py` — dataset splits, preprocessing, encoding, circuit dims, prob sums, metrics, degradation, fusion schema.

## Repository Structure
```
Q-ORBIT/
├── configs/qorbit_config.json   # 720s/256, Regime A/B, statistical seeds, robustness combos
├── data/synthetic, splits, processed, raw
├── src/simulator, classical, quantum, features, preprocessing, evaluation, fusion, utils, agents
├── experiments/*, results/reports+figures, models/, app/app.py, notebooks/, tests/, reports/
```

## Running
```powershell
python src/pipeline/generate_dataset.py --n-per-class 2000 --seed 42
python experiments/search_classical.py
python experiments/search_hybrid_quick.py
python experiments/search_pure_quick.py
python experiments/run_comprehensive_fair.py
streamlit run app/app.py
```

## Positioning
Q-ORBIT is a research framework for experimentally comparing classical, quantum, and hybrid learning for photometric light-curve classification, with adaptive fusion to investigate whether relative value of representations changes under observation conditions — not *Quantum AI that identifies debris*.

## Narrative
Problem (noisy incomplete curves) → Baseline (classical) → Quantum → Hybrid complement? → Robustness (noisy/incomplete) → Adaptive fusion (learn reliance) → Uncertainty/agreement → AI Analyst interpretation → Evidence-based conclusion (what shown, limitations).

---
*All metrics on frozen 70/15/15 seed 42, 1500 test, default.qubit, offline, no paid APIs, reproducible.*
#   Q _ O r b i t  
 