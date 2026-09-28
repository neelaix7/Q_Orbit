# Q-ORBIT Project Audit — 2026-09-01 (Pre-Revision, Seed 42)

**Title (current config):** `Hybrid Quantum-Classical AI Framework for Space Object Classification from Photometric Light Curves` (`configs/qorbit_config.json:3`)
**Proposed title (post-revision):** `Adaptive Physics-Informed Quantum-Classical AI Framework for Space Object Classification from Photometric Light Curves`
**Seed:** 42 (`src/simulator/config.py:6`). Splits frozen `data/splits/` 70/15/15 (7000/1500/1500) `data/splits/manifest.json:1`.

---

## 1. Current Architecture
- `src/simulator/` — physics generator (`lightcurve_generator.py`, `physics.py`, `config.py`) → synthetic curves
- `src/preprocessing/pipeline.py` — `CurvePreprocessor` (NaN interp, resample) + `FeatureStandardizer` (z-score fit train-only)
- `src/classical/feature_engineering.py` — 21 features (11 time, 5 freq, 5 temporal)
- `src/features/reducer.py` — `QuantumReducer` PCA+Scaler for quantum
- `src/classical/` — `cnn_baseline.py` (LightCurveCNN 280k params) + SVM/RF/XGB via `model_zoo.py`
- `src/quantum/` — `feature_map.py` (angle/reuploading/Fourier), `hybrid_model.py` (QuantumLayer 8q + classical head 758 params), `vqc.py` (PureVQC 94 params), `qsvm.py` (unused kernel)
- `src/evaluation/` — `compare_three.py`, `degradation.py` (noise/truncate/missing)
- `src/agents/analyst.py` — consensus/confidence text generator
- `src/pipeline/` — generate/train_classical/train_quantum/train_pure_quantum/evaluate + `download_real_data.py`
- `experiments/` — `run_comprehensive_fair.py`, `search_classical.py`, `search_hybrid_quick.py`, `search_pure_quick.py`
- `app/app.py` (788 lines) — Streamlit NASA+quantum+black-hole UI, 4 tabs, Plotly, glassmorphism
- `assets/images/` duplicated in `assets/images/web/`
- `data/` — `synthetic/lightcurves.npz` (10k,256) + `splits/` + `processed/` + `raw/` (empty)
- `models/` — cnn/svm/rf/xgb/hybrid/pure + reducers + scaler
- `results/` — `reports/` (fair/classical/hybrid/pure/robustness) + `figures/`
- `notebooks/demo_notebook.ipynb` (8374B), `requirements.txt`, `README.md`

## 2. Current Dataset
- **Synthetic only:** 10,000 curves, 5 classes ×2000, 256 samples per curve (`configs/qorbit_config.json:9`), generated via `generate_dataset()` `lightcurve_generator.py:157`
- **Physics:** specular+diffuse, sun/observer geometry, eclipse (0.27°), tumble periods/aspect/reflectivity per class (`config.py:21-45`), `n_faces` 3–8 (`lightcurve_generator.py:41-47`), photon noise 0.02, 5% dropout
- **Regime:** Single regime, no Regime A/B distinction. No real labeled external dataset; `download_real_data.py` honestly states no public optical light-curve dataset available (ESA DISCOS/NASA ODPO/CelesTrak only TLE/facts).
- **Known inconsistency:** Documentation says “2-hour, 256 samples @5s” → would be 1440 raw samples. Code reality: `TOTAL_DURATION_SEC=720.0` (12 min), `N_SAMPLES=256`, `SAMPLE_INTERVAL_SEC=5.0`, `N_TIME_STEPS=144` (`config.py:11-13`). Gallery correctly shows 256. Inconsistent across README/config/UI/reports.

## 3. Current Models
- **Classical feature baselines:** SVM RBF C1 (`model_zoo.py:train_svm`), RF 300 trees, XGB 300×d6 lr0.05 — all on 21 standardized features.
- **Classical deep:** 1D-CNN 16→32→64 channels, 3 pools 256→32, FC 128→64→5, BN+Dropout 0.3 (`cnn_baseline.py:22`).
- **Pure Quantum:** PCA→8 dims → tanh·π → AngleEmbedding RY → Rot(3)×layer + chain CNOT ×2 → ⟨Z⟩×8 → Linear 8→5 (PureVQC `vqc.py:11`). Grid 4/6/8q × depth 1-3 tested via quick search (2000 subset, 5 epochs) `search_pure_quick.py:1`.
- **Hybrid:** Linear(21→8)+tanh·π → same quantum layer (8q2L) → MLP 16→16→5 (HybridQuantumClassifier `hybrid_model.py:93`). Grid 5 configs q4l2h16..q8l2h32 `search_hybrid_quick.py:1` (subset), production Hybrid 8q2L16 (758 params). `qsvm.py` unused/inefficient O(n²) kernel.

## 4. Current Results (Frozen Test 1500, `fair_comparison.json` 2026-09-01 09:49)
| Model | Acc | F1 macro | F1 w | ROC-AUC | Params | Qubits/Depth |
|---|---|---|---|---|---|---|
| Classical_CNN | 0.8520 | 0.8511 | 0.8511 | 0.9665 | 280389 | 0/0 |
| Classical_RF | 0.7767 | 0.7760 | 0.7760 | 0.9569 | 5000* | 0 |
| Classical_XGB | 0.7760 | 0.7756 | 0.7756 | 0.9571 | 5000* | 0 |
| Classical_SVM | 0.7060 | 0.7038 | 0.7038 | 0.9396 | ~1050 | 0 |
| Hybrid_Quantum | 0.7013 | 0.6926 | 0.6926 | 0.9326 | 758 | 8/5 |
| Pure_Quantum_VQC | 0.4807 | 0.4253 | 0.4253 | 0.8301 | 94 | 8/5 |
- Per-class F1: CNN perfect on 0/1/3, weak on Rocket 0.656, Spoofed 0.599; Hybrid weak on Spoofed 0.5205 vs SVM 0.5383.
- Robustness: `robustness_noise.json` (0,0.1,0.2,0.3) — Hybrid drops 0.701→0.282→0.170, CNN 0.852→0.525→0.290 (less robust than pure at 0.2/0.3: Pure 0.274 vs Hybrid 0.170).
- Classical search: `classical_search.json` val best CNN (val F1 0.8657). Hybrid quick subset: val best q8l3h16 F1 0.5900 (subset) but production Hybrid 70% dominates on full.

## 5. Methodological Weaknesses
- Single seed (42) reporting point estimate, no mean±std or significance (`qorbit_config.json:evaluation.statistical` defines 5 seeds but never run).
- Feature leakage previously fixed (scaler train-only) but PCA reducer still re-standardizes after PCA with train mean/std — documented but fragile.
- Hyperparameter search is ad-hoc quick subset (5 epochs, 2000 train) not full validation-based thorough search; CNN reused checkpoint not searched; no learning rate/C tuning.
- Ablations missing: no isolation of classical vs quantum vs fusion contributions; no physics-aware vs baseline feature comparison.
- Adaptive fusion absent — current Hybrid is static concatenation (quantum→MLP), no signal-quality gating/attention.
- Uncertainty is max-prob/entropy only, not calibrated; no model agreement engine.

## 6. Data Issues
- **Time inconsistency** (2h vs 720s) violates scientific rigor; downstream dt calculations use 7200/(n-1) in `app.py:364` while config says 720s — factor 10 error in frequency features (`feature_engineering.py:43` sampling_interval default 5.0 mismatched to resampled 720/256≈2.8s).
- **Class separability trivial:** Period ranges non-overlapping (Frag 30-50s vs Rocket 350-500s, `config.py:21-27` and `CLASS_PARAMS` 60/150/350 vs 30) → model can memorize period; no overlapping distributions → inflates accuracy.
- No Regime A/B; external validation honesty present in code but not in README/UI positioning (claims operational NASA/ISRO).
- No controlled overlap experiment (easy vs overlapping) to show genuine difficulty.
- No dataset versioning beyond `manifest.json`; raw synthetic `lightcurves.npz` 9131223B duplicates splits.

## 7. Fairness/Comparison Issues
- Parameter count asymmetry acknowledged in text but not visually reported alongside metrics in old UI (capacity 280k vs 758 vs 94 hidden). Now fixed in new `fair_comparison.md` but not in per-model cards originally.
- Input representation asymmetry: CNN sees 256 raw points (703× more numbers than 21 features) — fair but must be explicitly separated as Feature Baseline vs Deep Baseline (Part 7).
- Training epochs unequal (CNN 20 epochs ×7000 vs Hybrid 15×7000 vs Pure quick 5×2000) — not normalized.
- No computation trade-off table (train time CNN 22.5s vs Hybrid 60s vs Pure 12.5s hidden).
- Robustness only single-condition, no combinations (noise+missing), no robustness score.

## 8. Quantum Implementation Issues
- Only angle RY encoding tested; ZZFeatureMap/re-uploading in `feature_map.py` unused. No encoding ablation.
- CNOT chain only, no circular/full entanglement ablation.
- Qubits limited to 8 max, depth to 3 — reasonable but ablation results only on subset, not full data.
- `qsvm.py` implements invalid kernel (calls `qn.state()` twice incorrectly) and O(n²) loops — unused, misleading.
- No noisy simulator vs ideal comparison, no hardware optional small experiment.
- Circuit depth computed as 1+2*layers simplistic, not from PennyLane specs.

## 9. Hybrid-Model Weaknesses
- Hybrid below CNN not hidden — honestly reported — but root cause not investigated (capacity? representation? fusion?).
- Architectures B/C/D (CNN extractor→quantum, dual-branch fusion) not implemented; only Hybrid A/B (feature→quantum→MLP) exists.
- No trainable fusion (concatenation vs gating vs weighted prob vs attention) — static.
- No signal-quality-aware fusion (noise/missing/periodicity scores) — the proposed novel contribution missing.
- Best hybrid selected on subset val (quick) not full val; no adaptive weighting learned from data.
- No dedicated hybrid capacity reporting (quantum params vs classical head params split).

## 10. UI Issues
- Previous prototype: visually spectacular (nebula, satellite, orbital rings) but scientifically noisy — comparison cards small, workflow static, quantum viz limited to text, no adaptive fusion viz, no uncertainty/agreement panels, no signal-quality panel, no master experimental matrix, overclaim language (“Quantum Intelligence that identifies debris”).
- Hero text before fix: “Classifying Space Objects with Quantum Intelligence” not matching revised title/statement.
- Fair protocol (70/15/15, no leakage) only in tooltip, not visually central.
- Research report auto-generation only via static `run_comprehensive_fair.py` md, not structured 20-section report.
- No lazy-loading, but animations GPU-friendly; no scroll-reveal yet.

## 11. Recommended Changes (Priority Order from Brief)
1. Fix time inconsistency: adopt `720-second observation window represented by 256 resampled observations` everywhere.
2. Introduce Regime A (controlled synthetic, current) + Regime B (external, state unavailable) honesty.
3. Improve simulator to overlapping distributions (see Part 4) + parameter-overlap experiment (easy vs overlapping).
4. Keep 21 baseline, add physics-informed grouping (TIME/FREQ/PHYSICAL PROXY) labeled.
5. Separate Classical Feature vs Deep baselines + capacity table.
6. Pure quantum ablations full-data (4/6/8q ×1/2/3 depth ×2 encodings) with val selection.
7. Implement Hybrid A/B (and C/D if feasible), val-selected.
8. Build Adaptive Fusion Engine (learned gate on signal-quality + representations) — novel core.
9. Signal Quality Analyzer (noise/missing/completeness/periodicity) → fusion input.
10. Model Agreement Engine + Uncertainty (entropy/margin/calibration).
11. Robustness core: 0/5/10/20/30% noise, 100/75/50/25% observation, 0/5/10/20% missing + combos (noise+missing) + robustness score definition.
12. Ablations 7 (classical, quantum, hybrid, hybrid no fusion, adaptive, physics-off/on).
13. Validation-based hyperparam search (SVM C/gamma, XGB depth, quantum qubits/layers/lr).
14. Multi-seed (42-46) mean±std, paired t-test where justified.
15. Master matrix table (clean × conditions) auto-generated.
16. Conclusion logic 4-way (accuracy/F1/robustness/trade-off).
17. Delete overclaims, blockchain/Web3, etc.
18. Traceable AI Analyst (structured JSON → text, no invented numbers).
19. Dashboard redesign for scientific readability (hero, comparison 4 cards inc. Adaptive, research insight, signal quality, agreement, quantum lab, adaptive viz, interactive charts).
20. 20-section auto research report, experiment tracking `experiments/{classical,quantum,hybrid,adaptive,robustness,ablations}/`, frozen test final selection, optional noisy hardware, tests, rewritten README with research narrative.

## 12. Files That Should Be Deleted
- `src/quantum/qsvm.py` — broken kernel, replace with correct `qsvm_v2` or remove (currently unused)
- `test_hybrid.py` (root) — ad-hoc dev script, superseded by `experiments/search_hybrid*.py`
- `make_quantum_img.py`, `resize_images.py` — one-off asset helpers, not part of pipeline
- `quai_guard/`, `quai_guard_demo.png`, `127-Yellow-RAT.zip` etc — unrelated to Q-ORBIT (verify outside D:/Capstone?)
- `app/app_backup.py` — keep until revision verified then delete
- `assets/images/web/` duplicates? Keep one, symlink or remove duplication (currently both `assets/images/` and `assets/images/web/` contain same 11 jpgs)
- `.playwright-mcp/` logs — dev artifacts, not repo
- `data/synthetic/lightcurves.npz` duplicate of splits (optional: keep with note or regenerate)

## 13. Files That Should Be Retained
- `README.md` (to be rewritten), `requirements.txt`, `configs/qorbit_config.json`
- `data/splits/` (7000/1500/1500) — frozen truth, never regenerate without force
- `data/processed/` scaler/params — retain
- `src/simulator/` (config, physics, lightcurve_generator) — core, to be improved not removed
- `src/classical/feature_engineering.py` (21 baseline), `cnn_baseline.py`, `model_zoo.py`
- `src/features/reducer.py`, `src/preprocessing/pipeline.py`
- `src/quantum/hybrid_model.py`, `vqc.py`, `feature_map.py`
- `src/evaluation/compare_three.py`, `degradation.py`, `src/utils/metrics.py`, `src/agents/analyst.py` (to be Traceable)
- `src/pipeline/generate_dataset.py`, `train_*.py`, `evaluate.py`, `download_real_data.py`
- `experiments/search_classical.py`, `search_hybrid_quick.py`, `search_pure_quick.py`, `run_comprehensive_fair.py` (to be expanded)
- `models/` all checkpoints (especially cnn 85%, hybrid 70%, pure 48%) — retain, add adaptive
- `results/reports/fair_comparison.*`, `classical_search.*`, `hybrid_search.*`, `pure_search.*`, `robustness_*.json` — retain, expand
- `results/figures/` — retain
- `app/app.py` — retain as base for redesign, not rebuild
- `notebooks/demo_notebook.ipynb` — retain, update
- `assets/images/` — retain (dedupe)

## 14. Files That Should Be Refactored
- `src/simulator/config.py` — fix time def (720s/256 resampled), update TUMBLE_PERIOD_RANGE to overlapping distributions, clarify RAW vs RESAMPLED
- `src/simulator/lightcurve_generator.py` — CLASS_PARAMS overlapping, add variation docs, fix dt handling
- `src/simulator/physics.py` — add comments for overlap rationale
- `src/classical/feature_engineering.py` — keep but reorganize into TIME/FREQ/PHYSICAL PROXY groups, add physics-informed label
- `src/quantum/feature_map.py` — implement proper Angle vs ZZ ablation entry points
- `src/quantum/hybrid_model.py` — add Hybrid A/B/C/D variants, entanglement options, adaptive fusion hook
- `src/evaluation/` — add robustness_score, experimental matrix, statistical validation
- `src/agents/analyst.py` — make traceable (structured input JSON → templated text, no invented numbers)
- `app/app.py` — hero text, workflow viz, quantum lab, adaptive fusion viz, agreement/uncertainty panels, research insight, scientific readability over effects
- `configs/qorbit_config.json` — add regime field, observation window field, overlapping flag, statistical seeds, robustness combos
- `README.md` — rewrite around research methodology: Problem→Dataset→Classical→Quantum→Hybrid→Adaptive→Robustness→Analyst→Conclusion, positioning as research framework not deployed system
- `requirements.txt` — pin versions verified (numpy 2.5.2, torch 2.13+cpu, pennylane 0.45, xgboost 3.3, sklearn 1.9)
- New files to create: `src/fusion/adaptive_fusion.py`, `src/fusion/signal_quality.py`, `src/evaluation/robustness_score.py`, `src/evaluation/ablations.py`, `reports/research_report.md` generator, `experiments/{adaptive,robustness,ablations}/`, `tests/`
