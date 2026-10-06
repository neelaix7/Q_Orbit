# Hybrid Search (quick subset demo — validation-selected best, no test leakage)

Seed 123 | Grid 5 configs | Epochs 5 | Subset train 2000 val 500

| # | Qubits | Layers | Hidden | Val Acc | Val F1 | Test Acc | Test F1 | Params | Depth | Train(s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 2 | 16 | 0.5500 | 0.4992 | 0.5352 | 0.4848 | 582 | 5 | 2.4 |
| 2 | 6 | 2 | 16 | 0.6000 | 0.5478 | 0.5741 | 0.5183 | 670 | 5 | 3.6 |
| 3 | 8 | 2 | 16 | 0.6300 | 0.6388 | 0.6085 | 0.6116 | 758 | 5 | 6.2 |
| 4 | 8 | 3 | 16 | 0.6140 | 0.5846 | 0.5896 | 0.5603 | 782 | 7 | 8.0 |
| 5 | 8 | 2 | 32 | 0.6160 | 0.5951 | 0.6061 | 0.5749 | 1798 | 5 | 6.1 |

**Best on VAL (subset):** qubits=8 layers=2 hidden=16 -> Val F1 0.6388 Test F1 0.6116

Note: Full production hybrid (8 qubits, 2 layers, 16 hidden) trained on full 7000-sample train set achieves 70.13% test accuracy (see fair_comparison.md). This quick search demonstrates systematic validation-based selection; production model remains the best validated on full data.
