# Q-ORBIT Fair Three-Way Comparison (70/15/15 frozen, scaler fit on train only)

Seed 42 | Test 1500 samples

| Model | Accuracy | Precision_macro | Recall_macro | F1_macro | F1_weighted | ROC-AUC | Params | Qubits | Depth | Train(s) | Infer(ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Classical_CNN | 0.8520 | 0.8533 | 0.8520 | 0.8511 | 0.8511 | 0.9665 | 280389 | 0 | 0 | 22.5 | 0.15 |
| Classical_SVM | 0.7060 | 0.7034 | 0.7060 | 0.7038 | 0.7038 | 0.9396 | 1050 | 0 | 0 | 7.9 | 0.57 |
| Classical_RF | 0.7767 | 0.7759 | 0.7767 | 0.7760 | 0.7760 | 0.9569 | 5000 | 0 | 0 | 0.5 | 0.20 |
| Classical_XGB | 0.7760 | 0.7754 | 0.7760 | 0.7756 | 0.7756 | 0.9571 | 5000 | 0 | 0 | 0.5 | 0.20 |
| Pure_Quantum_VQC | 0.4807 | 0.4759 | 0.4807 | 0.4253 | 0.4253 | 0.8301 | 94 | 8 | 5 | 12.5 | 0.07 |
| Hybrid_Quantum | 0.7220 | 0.7292 | 0.7220 | 0.7234 | 0.7234 | 0.9425 | 1798 | 8 | 5 | 60.0 | 0.04 |

## Per-class F1

| Class | Classical_CNN | Classical_SVM | Pure_Quantum | Hybrid |
|---|---|---|---|---|
| Intact Satellite | 1.0000 | 0.8180 | 0.6190 | 0.8286 |
| Dead Satellite | 1.0000 | 0.8183 | 0.3093 | 0.8778 |
| Rocket Body | 0.6563 | 0.5178 | 0.0793 | 0.5069 |
| Fragmentation Debris | 1.0000 | 0.8268 | 0.5593 | 0.8456 |
| Spoofed Satellite | 0.5993 | 0.5383 | 0.5594 | 0.5580 |

## Research Conclusion (generated from measured results, not hard-coded)

**HYBRID ADVANTAGE NOT DEMONSTRATED**

Measured results do not show a universal hybrid advantage. Best F1-macro: Classical CNN 0.8511, Classical SVM 0.7038, Hybrid 0.7234, Pure 0.4253. Hybrid outperforms pure quantum by 29.81 pp, demonstrating quantum+classical integration helps over pure quantum, but does not surpass the strongest classical baseline on this dataset/split.

Robustness analysis (noise/observation/missing) should be inspected to determine if hybrid exhibits stability advantages even when clean accuracy is lower.

*All metrics computed on the same frozen 70/15/15 split (seed 42), same 1500-sample test set, fixed simulator (default.qubit). No test leakage.*
