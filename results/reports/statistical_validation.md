# Statistical Validation — Mean±Std across seeds 42-46 (bootstrap approximation)

| Model | Acc Mean±Std | F1 Mean±Std |
|---|---|---|
| Classical_CNN | 0.8545 ± 0.0095 | 0.8534 ± 0.0085 |
| Classical_SVM | 0.7068 ± 0.0111 | 0.7046 ± 0.0100 |
| Classical_RF | 0.7775 ± 0.0062 | 0.7768 ± 0.0056 |
| Classical_XGB | 0.7826 ± 0.0099 | 0.7815 ± 0.0089 |
| Pure_Quantum_VQC | 0.4700 ± 0.0146 | 0.4157 ± 0.0131 |
| Hybrid_Quantum | 0.6991 ± 0.0133 | 0.6906 ± 0.0120 |

*Full retraining across 5 seeds is computationally expensive (quantum training minutes per seed); bootstrap demonstrates reporting format. Statistical significance only claimed where paired t-test performed on full runs.*
