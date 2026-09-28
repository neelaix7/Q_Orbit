# Classical Search — validation-based selection (no test leakage)

| Model | Val Acc | Val F1-macro | Test Acc | Test F1-macro | Train time (s) |
|---|---|---|---|---|---|
| SVM | 0.6920 | 0.6910 | 0.7060 | 0.7038 | 7.9 |
| RandomForest | 0.7600 | 0.7604 | 0.7767 | 0.7760 | 1.8 |
| XGBoost | 0.7787 | 0.7793 | 0.7760 | 0.7756 | 4.5 |
| CNN | 0.8660 | 0.8657 | 0.8520 | 0.8511 | 22.5 |

**Best on VAL:** CNN
