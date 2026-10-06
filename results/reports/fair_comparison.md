# Q-ORBIT Fair Three-Way Comparison (70/15/15 frozen, scaler fit on train only)

Seed 123 | Test 3750 samples

| Model | Accuracy | Precision_macro | Recall_macro | F1_macro | F1_weighted | ROC-AUC | Params | Qubits | Depth | Train(s) | Infer(ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Classical_CNN | 0.8776 | 0.8849 | 0.8776 | 0.8750 | 0.8750 | 0.9805 | 280389 | 0 | 0 | 50.0 | 0.05 |
| Classical_SVM | 0.7400 | 0.7388 | 0.7400 | 0.7393 | 0.7393 | 0.9498 | 1050 | 0 | 0 | 47.4 | 1.32 |
| Classical_RF | 0.7944 | 0.7943 | 0.7944 | 0.7943 | 0.7943 | 0.9661 | 5000 | 0 | 0 | 0.5 | 0.20 |
| Classical_XGB | 0.8051 | 0.8055 | 0.8051 | 0.8052 | 0.8052 | 0.9655 | 5000 | 0 | 0 | 0.5 | 0.20 |
| Pure_Quantum_VQC | 0.6208 | 0.6193 | 0.6208 | 0.6197 | 0.6197 | 0.9061 | 118 | 8 | 7 | 473.2 | 0.09 |
| Hybrid_Quantum | 0.7541 | 0.7546 | 0.7541 | 0.7540 | 0.7540 | 0.9523 | 1798 | 8 | 5 | 60.0 | 0.07 |

## Per-class F1

| Class | Classical_CNN | Classical_SVM | Pure_Quantum | Hybrid |
|---|---|---|---|---|
| Intact Satellite | 0.9993 | 0.8145 | 0.6993 | 0.8648 |
| Dead Satellite | 0.9980 | 0.8690 | 0.6845 | 0.9286 |
| Rocket Body | 0.7353 | 0.5961 | 0.5212 | 0.5723 |
| Fragmentation Debris | 0.9980 | 0.8278 | 0.7096 | 0.8651 |
| Spoofed Satellite | 0.6443 | 0.5892 | 0.4840 | 0.5389 |

## Research Conclusion (generated from measured results, not hard-coded)

**HYBRID ADVANTAGE NOT DEMONSTRATED**

Measured results do not show a universal hybrid advantage. Best F1-macro: Classical CNN 0.8750, Classical SVM 0.7393, Hybrid 0.7540, Pure 0.6197. Hybrid outperforms pure quantum by 13.42 pp, demonstrating quantum+classical integration helps over pure quantum, but does not surpass the strongest classical baseline on this dataset/split.

Robustness analysis (noise/observation/missing) should be inspected to determine if hybrid exhibits stability advantages even when clean accuracy is lower.

*All metrics computed on the same frozen 70/15/15 split (seed 123), same 3750-sample test set, fixed simulator (default.qubit). No test leakage.*
